from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from dataset import Dataset
from losses.elr import elr_loss
from models.eogfd import EOGFD, EOGFDHetero  

warnings.filterwarnings("ignore")


EPS = 1e-10


@dataclass
class RunResult:
    mf1: float
    auc: float


def get_best_macro_f1_threshold(
    labels: Sequence[int],
    probs: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> Tuple[float, float]:
    """
    Find the threshold (on positive-class probability) that maximizes macro-F1.

    Args:
        labels: Ground-truth labels (0/1).
        probs: Predicted probabilities with shape (N, 2) or (N,).
               If (N, 2), probs[:, 1] is used as positive-class probability.
        thresholds: Optional threshold candidates. If None, use 0.05..0.95 step 0.05.

    Returns:
        best_macro_f1: Best macro-F1 across thresholds.
        best_threshold: Threshold that achieves best_macro_f1.
    """
    y = np.asarray(labels, dtype=np.int64).reshape(-1)

    p = np.asarray(probs, dtype=np.float32)
    if p.ndim == 2:
        if p.shape[1] != 2:
            raise ValueError(f"Expected probs shape (N, 2), got {p.shape}.")
        pos_probs = p[:, 1]
    elif p.ndim == 1:
        pos_probs = p
    else:
        raise ValueError(f"Expected probs 1D or 2D, got {p.ndim}D.")

    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)

    # Vectorized threshold evaluation
    preds = (pos_probs[:, None] > thresholds[None, :]).astype(np.int32)
    y_exp = y[:, None]

    tp = np.sum((preds == 1) & (y_exp == 1), axis=0)
    fp = np.sum((preds == 1) & (y_exp == 0), axis=0)
    fn = np.sum((preds == 0) & (y_exp == 1), axis=0)
    tn = np.sum((preds == 0) & (y_exp == 0), axis=0)

    f1_pos = 2.0 * tp / (2.0 * tp + fp + fn + EPS)
    f1_neg = 2.0 * tn / (2.0 * tn + fn + fp + EPS)
    macro_f1s = 0.5 * (f1_pos + f1_neg)

    best_idx = int(np.argmax(macro_f1s))
    return float(macro_f1s[best_idx]), float(thresholds[best_idx])


def build_masks(
    n: int,
    idx_train: Sequence[int],
    idx_valid: Sequence[int],
    idx_test: Sequence[int],
    device: torch.device | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build boolean masks for train/valid/test indices.
    """
    train_mask = torch.zeros(n, dtype=torch.bool, device=device)
    val_mask = torch.zeros(n, dtype=torch.bool, device=device)
    test_mask = torch.zeros(n, dtype=torch.bool, device=device)

    train_mask[list(idx_train)] = True
    val_mask[list(idx_valid)] = True
    test_mask[list(idx_test)] = True

    return train_mask, val_mask, test_mask


def split_indices(
    labels_np: np.ndarray,
    dataset_name: str,
    train_ratio: float,
    seed: int = 2,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Create stratified train/valid/test splits over node indices.

    For the 'amazon' dataset, follow the original indexing rule:
    only indices in [3305, N) are used for splitting.
    """
    n = len(labels_np)
    all_idx = list(range(n))
    if dataset_name.lower() == "amazon":
        all_idx = list(range(3305, n))

    idx_train, idx_rest, y_train, y_rest = train_test_split(
        all_idx,
        labels_np[all_idx],
        stratify=labels_np[all_idx],
        train_size=train_ratio,
        random_state=seed,
        shuffle=True,
    )

    idx_valid, idx_test, _, _ = train_test_split(
        idx_rest,
        y_rest,
        stratify=y_rest,
        test_size=0.67,
        random_state=seed,
        shuffle=True,
    )

    return list(idx_train), list(idx_valid), list(idx_test)


def train_one_run(model: torch.nn.Module, graph, args: argparse.Namespace) -> RunResult:
    """
    Train one run and evaluate on the test split using the best validation threshold.

    Returns:
        RunResult(mf1, auc) where both are in [0, 1].
    """
    features = graph.ndata["feature"]
    labels = graph.ndata["label"]

    labels_np = labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
    labels_np = labels_np.astype(np.int64).reshape(-1)

    idx_train, idx_valid, idx_test = split_indices(
        labels_np=labels_np,
        dataset_name=args.dataset,
        train_ratio=args.train_ratio,
        seed=2,
    )

    device = labels.device if isinstance(labels, torch.Tensor) else None
    train_mask, val_mask, test_mask = build_masks(
        n=len(labels_np),
        idx_train=idx_train,
        idx_valid=idx_valid,
        idx_test=idx_test,
        device=device,
    )

    num_classes = int(labels.max().item() + 1)
    e_loss = elr_loss(len(idx_train), num_classes, args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Class weight for imbalance: weight = (#neg / #pos) on training set
    pos = labels[train_mask].float().sum().item()
    neg = (1.0 - labels[train_mask].float()).sum().item()
    weight_ratio = float(neg / (pos + EPS))

    best_val_f1 = -1.0
    best_test_mf1 = 0.0
    best_test_auc = 0.0

    time_start = time.time()

    for epoch in range(args.epoch):
        model.train()
        logits, _ = model(graph, features)

        loss = e_loss(
            logits[train_mask],
            labels[train_mask],
            weight=torch.tensor([1.0, weight_ratio], device=logits.device),
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Validation threshold selection + test evaluation
        model.eval()
        with torch.no_grad():
            probs = logits.softmax(dim=1)

            val_probs_np = probs[val_mask].detach().cpu().numpy()
            val_f1, thres = get_best_macro_f1_threshold(labels_np[val_mask.detach().cpu().numpy()], val_probs_np)

            full_probs_np = probs.detach().cpu().numpy()
            preds = np.zeros_like(labels_np, dtype=np.int64)
            preds[full_probs_np[:, 1] > thres] = 1

            test_mask_np = test_mask.detach().cpu().numpy()
            y_test = labels_np[test_mask_np]
            y_pred = preds[test_mask_np]
            y_score = full_probs_np[test_mask_np][:, 1]

            test_rec = recall_score(y_test, y_pred)
            test_pre = precision_score(y_test, y_pred)
            test_mf1 = f1_score(y_test, y_pred, average="macro")
            test_auc = roc_auc_score(y_test, y_score)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_test_mf1 = float(test_mf1)
                best_test_auc = float(test_auc)

    time_end = time.time()
    _ = time_end - time_start  # keep for optional logging

    print(f"    -> Run Result: MF1 {best_test_mf1 * 100:.2f} AUC {best_test_auc * 100:.2f}")
    return RunResult(mf1=best_test_mf1, auc=best_test_auc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EOGFD (CPU)")

    parser.add_argument(
        "--dataset",
        type=str,
        default="tfinance",
        choices=["yelp", "amazon", "tfinance", "tsocial"],
        help="Dataset name.",
    )
    parser.add_argument("--train_ratio", type=float, default=0.4, help="Training ratio.")
    parser.add_argument("--hid_dim", type=int, default=64, help="Hidden dimension.")
    parser.add_argument("--order", type=int, default=3, help="Order C in Beta Wavelet.")
    parser.add_argument("--beta_p", type=int, default=1, help="p in Beta Wavelet.")
    parser.add_argument("--pop_size", type=int, default=10, help="Population size.")
    parser.add_argument("--mut_rate", type=float, default=0.1, help="Mutation rate.")
    parser.add_argument(
        "--homo",
        type=int,
        default=0,
        choices=[0, 1],
        help="1: EOGFD(Homo), 0: EOGFD(Hetero).",
    )

    parser.add_argument("--beta", type=float, default=0.5, help="Beta for ELR loss.")
    parser.add_argument("--lamb", type=float, default=0.0, help="Lambda for ELR loss.")

    # Fixed parameters
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--epoch", type=int, default=100, help="Number of epochs.")
    parser.add_argument("--run", type=int, default=3, help="Runs per setting.")

    # Grid search
    parser.add_argument(
        "--grid_search",
        action="store_true",
        default=True,
        help="If set, ignore pop_size/mut_rate args and use predefined lists.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.grid_search:
        pop_size_list = [20]
        mut_rate_list = [0.01, 0.05, 0.1, 0.2]
        print("Starting grid search...")
        print(f"Pop sizes: {pop_size_list}")
        print(f"Mutation rates: {mut_rate_list}")
    else:
        pop_size_list = [args.pop_size]
        mut_rate_list = [args.mut_rate]

    dataset_name = args.dataset
    homo = bool(args.homo)
    order = args.order
    beta_p = args.beta_p
    hid_dim = args.hid_dim

    graph = Dataset(dataset_name, int(homo)).graph
    in_feats = int(graph.ndata["feature"].shape[1])
    num_classes = int(graph.ndata["label"].max().item() + 1)

    all_results: List[Dict[str, float]] = []

    total = len(pop_size_list) * len(mut_rate_list)
    count = 0

    for pop_size in pop_size_list:
        for mut_rate in mut_rate_list:
            count += 1
            print(f"\n[{count}/{total}] Testing: pop_size={pop_size}, mut_rate={mut_rate}")

            args.pop_size = pop_size
            args.mut_rate = mut_rate

            mf1s: List[float] = []
            aucs: List[float] = []

            for r in range(args.run):
                if homo:
                    model = EOGFD(
                        in_feats, hid_dim, num_classes, d=order, beta_p=beta_p,
                        population_size=pop_size, mutation_rate=mut_rate
                    )
                else:
                    model = EOGFDHetero(
                        in_feats, hid_dim, num_classes, d=order, beta_p=beta_p,
                        population_size=pop_size, mutation_rate=mut_rate
                    )

                if args.run > 1:
                    print(f"  Run {r + 1}/{args.run}...", end="")

                result = train_one_run(model, graph, args)
                mf1s.append(result.mf1)
                aucs.append(result.auc)

            mf1_mean = float(np.mean(mf1s) * 100.0)
            mf1_std = float(np.std(mf1s) * 100.0)
            auc_mean = float(np.mean(aucs) * 100.0)
            auc_std = float(np.std(aucs) * 100.0)

            print(f"  >> Avg Result: MF1 {mf1_mean:.2f} ± {mf1_std:.2f}, AUC {auc_mean:.2f} ± {auc_std:.2f}")

            all_results.append(
                {
                    "pop_size": float(pop_size),
                    "mut_rate": float(mut_rate),
                    "mf1_mean": mf1_mean,
                    "mf1_std": mf1_std,
                    "auc_mean": auc_mean,
                    "auc_std": auc_std,
                }
            )

    print("\n" + "=" * 80)
    print(f"FINAL GRID SEARCH RESULTS ({args.dataset})")
    print("=" * 80)

    header = f"{'Pop Size':<10} | {'Mut Rate':<10} | {'MF1 (Mean±Std)':<25} | {'AUC (Mean±Std)':<25}"
    print(header)
    print("-" * len(header))

    for res in all_results:
        mf1_str = f"{res['mf1_mean']:.2f} ± {res['mf1_std']:.2f}"
        auc_str = f"{res['auc_mean']:.2f} ± {res['auc_std']:.2f}"
        print(f"{int(res['pop_size']):<10} | {res['mut_rate']:<10.2g} | {mf1_str:<25} | {auc_str:<25}")

    print("=" * 80)


if __name__ == "__main__":
    main()


# Example usage:
# 1. 进入脚本所在目录
# cd /root/WangXuan/EOGFD/
# 2. 激活环境
# conda activate wx
# 3. 运行脚本
# yelp/amazon/tfinance/tsocial
# python main.py --dataset yelp --train_ratio 0.4 --hid_dim 64 --order 3 --beta_p 1 --pop_size 20 --mut_rate 0.1 --homo 0 --beta 0.5 --lamb 0
# python main.py --dataset amazon --train_ratio 0.4 --hid_dim 64 --order 4 --beta_p 1 --pop_size 20 --mut_rate 0.1 --homo 0 --beta 0.9 --lamb 0
# python main.py --dataset tfinance --train_ratio 0.4 --hid_dim 64 --order 7 --beta_p 1 --pop_size 20 --mut_rate 0.1 --homo 0 --beta 0.9 --lamb 0
# python main.py --dataset tsocial --train_ratio 0.4 --hid_dim 10 --order 8 --beta_p 1 --pop_size 20 --mut_rate 0.1 --homo 0 --beta 0.5 --lamb 0