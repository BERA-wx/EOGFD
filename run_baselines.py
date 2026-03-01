"""
Noise correction baseline comparison script for EOGFD.

Compares:
- GCN
- GCN + SCE
- GCN + Forward Correction
- GCN + Co-teaching
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from dataset import Dataset
from models.noise_correction_baselines import CoTeaching, ForwardCorrectionLoss, GCN, HeteroGCN, SCELoss

warnings.filterwarnings("ignore")

EPS = 1e-10


@dataclass
class SplitMasks:
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor


@dataclass
class EvalResult:
    val_macro_f1: float
    test_macro_f1: float
    test_auc: float
    threshold: float


def get_best_macro_f1_threshold(labels: np.ndarray, probs: np.ndarray) -> Tuple[float, float]:
    """
    Find the threshold on positive-class probability that maximizes macro-F1.

    Args:
        labels: shape (N,), values in {0,1}.
        probs: shape (N, 2), softmax probabilities.

    Returns:
        best_macro_f1, best_threshold
    """
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    p = np.asarray(probs, dtype=np.float32)
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError(f"Expected probs shape (N, 2), got {p.shape}.")

    pos_probs = p[:, 1]
    thresholds = np.linspace(0.05, 0.95, 19)

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


def split_masks(labels_np: np.ndarray, dataset_name: str, train_ratio: float, seed: int = 2) -> SplitMasks:
    """
    Build train/val/test masks with stratified splits.

    For 'amazon', only indices [3305, N) are considered (to match the original setup).
    """
    n = len(labels_np)
    idx = list(range(n))
    if dataset_name.lower() == "amazon":
        idx = list(range(3305, n))

    idx_train, idx_rest, _, y_rest = train_test_split(
        idx,
        labels_np[idx],
        stratify=labels_np[idx],
        train_size=train_ratio,
        random_state=seed,
        shuffle=True,
    )
    idx_val, idx_test, _, _ = train_test_split(
        idx_rest,
        y_rest,
        stratify=y_rest,
        test_size=0.67,
        random_state=seed,
        shuffle=True,
    )

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)

    train_mask[idx_train] = True
    val_mask[idx_val] = True
    test_mask[idx_test] = True

    return SplitMasks(train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)


def compute_class_weight(labels: torch.Tensor, train_mask: torch.Tensor) -> torch.Tensor:
    """
    Compute class weight tensor [w0, w1] for binary cross-entropy-style losses.
    """
    y_train = labels[train_mask].float()
    pos = y_train.sum().item()
    neg = (1.0 - y_train).sum().item()
    w1 = float(neg / (pos + EPS))
    return torch.tensor([1.0, w1], dtype=torch.float32)


@torch.no_grad()
def evaluate_single_model(
    model: torch.nn.Module,
    graph,
    features: torch.Tensor,
    labels_np: np.ndarray,
    masks: SplitMasks,
) -> EvalResult:
    model.eval()
    logits = model(graph, features)
    probs = F.softmax(logits, dim=1).cpu().numpy()

    val_idx = masks.val_mask.cpu().numpy()
    test_idx = masks.test_mask.cpu().numpy()

    val_f1, thres = get_best_macro_f1_threshold(labels_np[val_idx], probs[val_idx])

    preds = np.zeros_like(labels_np, dtype=np.int64)
    preds[probs[:, 1] > thres] = 1

    y_test = labels_np[test_idx]
    y_pred = preds[test_idx]
    y_score = probs[test_idx][:, 1]

    test_mf1 = float(f1_score(y_test, y_pred, average="macro"))
    test_auc = float(roc_auc_score(y_test, y_score))

    return EvalResult(val_macro_f1=val_f1, test_macro_f1=test_mf1, test_auc=test_auc, threshold=thres)


@torch.no_grad()
def evaluate_coteaching(
    model1: torch.nn.Module,
    model2: torch.nn.Module,
    graph,
    features: torch.Tensor,
    labels_np: np.ndarray,
    masks: SplitMasks,
) -> EvalResult:
    model1.eval()
    model2.eval()
    logits = (model1(graph, features) + model2(graph, features)) / 2.0
    probs = F.softmax(logits, dim=1).cpu().numpy()

    val_idx = masks.val_mask.cpu().numpy()
    test_idx = masks.test_mask.cpu().numpy()

    val_f1, thres = get_best_macro_f1_threshold(labels_np[val_idx], probs[val_idx])

    preds = np.zeros_like(labels_np, dtype=np.int64)
    preds[probs[:, 1] > thres] = 1

    y_test = labels_np[test_idx]
    y_pred = preds[test_idx]
    y_score = probs[test_idx][:, 1]

    test_mf1 = float(f1_score(y_test, y_pred, average="macro"))
    test_auc = float(roc_auc_score(y_test, y_score))

    return EvalResult(val_macro_f1=val_f1, test_macro_f1=test_mf1, test_auc=test_auc, threshold=thres)


def build_gcn(args: argparse.Namespace, in_feats: int, num_classes: int) -> torch.nn.Module:
    if args.homo == 1:
        return GCN(in_feats, args.hid_dim, num_classes)
    return HeteroGCN(in_feats, args.hid_dim, num_classes)


def train_vanilla_gcn(graph, args: argparse.Namespace) -> Tuple[float, float]:
    features = graph.ndata["feature"]
    labels = graph.ndata["label"]
    in_feats = int(features.shape[1])
    num_classes = int(labels.max().item() + 1)

    labels_np = labels.detach().cpu().numpy().astype(np.int64)
    masks = split_masks(labels_np, args.dataset, args.train_ratio)

    weight = compute_class_weight(labels, masks.train_mask)
    model = build_gcn(args, in_feats, num_classes)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = -1.0
    best_test_mf1 = 0.0
    best_test_auc = 0.0

    for _ in range(args.epoch):
        model.train()
        logits = model(graph, features)
        loss = F.cross_entropy(logits[masks.train_mask], labels[masks.train_mask], weight=weight)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        ev = evaluate_single_model(model, graph, features, labels_np, masks)
        if ev.val_macro_f1 > best_val:
            best_val = ev.val_macro_f1
            best_test_mf1 = ev.test_macro_f1
            best_test_auc = ev.test_auc

    return best_test_mf1, best_test_auc


def train_gcn_sce(graph, args: argparse.Namespace) -> Tuple[float, float]:
    features = graph.ndata["feature"]
    labels = graph.ndata["label"]
    in_feats = int(features.shape[1])
    num_classes = int(labels.max().item() + 1)

    labels_np = labels.detach().cpu().numpy().astype(np.int64)
    masks = split_masks(labels_np, args.dataset, args.train_ratio)

    weight = compute_class_weight(labels, masks.train_mask)
    model = build_gcn(args, in_feats, num_classes)
    sce = SCELoss(alpha=args.sce_alpha, beta=args.sce_beta, num_classes=num_classes)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = -1.0
    best_test_mf1 = 0.0
    best_test_auc = 0.0

    for _ in range(args.epoch):
        model.train()
        logits = model(graph, features)
        loss = sce(logits[masks.train_mask], labels[masks.train_mask], weight=weight)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        ev = evaluate_single_model(model, graph, features, labels_np, masks)
        if ev.val_macro_f1 > best_val:
            best_val = ev.val_macro_f1
            best_test_mf1 = ev.test_macro_f1
            best_test_auc = ev.test_auc

    return best_test_mf1, best_test_auc


def train_gcn_forward(graph, args: argparse.Namespace) -> Tuple[float, float]:
    features = graph.ndata["feature"]
    labels = graph.ndata["label"]
    in_feats = int(features.shape[1])
    num_classes = int(labels.max().item() + 1)

    labels_np = labels.detach().cpu().numpy().astype(np.int64)
    masks = split_masks(labels_np, args.dataset, args.train_ratio)

    weight = compute_class_weight(labels, masks.train_mask)
    model = build_gcn(args, in_feats, num_classes)
    forward_loss = ForwardCorrectionLoss(num_classes=num_classes, noise_rate=args.noise_rate)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Stage 1: warm-up with CE
    warmup_epochs = max(1, args.epoch // 2)
    for _ in range(warmup_epochs):
        model.train()
        logits = model(graph, features)
        loss = F.cross_entropy(logits[masks.train_mask], labels[masks.train_mask], weight=weight)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Estimate transition matrix T
    forward_loss.estimate_T_from_model(model, graph, features, labels, masks.train_mask)

    best_val = -1.0
    best_test_mf1 = 0.0
    best_test_auc = 0.0

    # Stage 2: train with corrected loss
    for _ in range(warmup_epochs, args.epoch):
        model.train()
        logits = model(graph, features)
        loss = forward_loss(logits[masks.train_mask], labels[masks.train_mask], weight=weight)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        ev = evaluate_single_model(model, graph, features, labels_np, masks)
        if ev.val_macro_f1 > best_val:
            best_val = ev.val_macro_f1
            best_test_mf1 = ev.test_macro_f1
            best_test_auc = ev.test_auc

    return best_test_mf1, best_test_auc


def train_coteaching(graph, args: argparse.Namespace) -> Tuple[float, float]:
    features = graph.ndata["feature"]
    labels = graph.ndata["label"]
    in_feats = int(features.shape[1])
    num_classes = int(labels.max().item() + 1)

    labels_np = labels.detach().cpu().numpy().astype(np.int64)
    masks = split_masks(labels_np, args.dataset, args.train_ratio)

    weight = compute_class_weight(labels, masks.train_mask)

    model1 = build_gcn(args, in_feats, num_classes)
    model2 = build_gcn(args, in_feats, num_classes)

    coteaching = CoTeaching(
        model1,
        model2,
        forget_rate=args.forget_rate,
        num_gradual=args.num_gradual,
    )

    optim1 = torch.optim.Adam(model1.parameters(), lr=args.lr)
    optim2 = torch.optim.Adam(model2.parameters(), lr=args.lr)

    best_val = -1.0
    best_test_mf1 = 0.0
    best_test_auc = 0.0

    for epoch in range(args.epoch):
        model1.train()
        model2.train()

        step = coteaching.train_step(
            graph=graph,
            features=features,
            labels=labels,
            train_mask=masks.train_mask,
            class_weight=weight,
            epoch=epoch,
        )

        optim1.zero_grad(set_to_none=True)
        step.loss1.backward()
        optim1.step()

        optim2.zero_grad(set_to_none=True)
        step.loss2.backward()
        optim2.step()

        ev = evaluate_coteaching(model1, model2, graph, features, labels_np, masks)
        if ev.val_macro_f1 > best_val:
            best_val = ev.val_macro_f1
            best_test_mf1 = ev.test_macro_f1
            best_test_auc = ev.test_auc

    return best_test_mf1, best_test_auc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline comparison for EOGFD")

    # Dataset
    parser.add_argument("--dataset", type=str, default="yelp", choices=["yelp", "amazon", "tfinance", "tsocial"])
    parser.add_argument("--train_ratio", type=float, default=0.4)
    parser.add_argument("--homo", type=int, default=0, choices=[0, 1], help="1: homogeneous, 0: heterogeneous")

    # Model
    parser.add_argument("--hid_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--run", type=int, default=5)

    # SCE
    parser.add_argument("--sce_alpha", type=float, default=0.1)
    parser.add_argument("--sce_beta", type=float, default=1.0)

    # Forward correction
    parser.add_argument("--noise_rate", type=float, default=0.2)

    # Co-teaching
    parser.add_argument("--forget_rate", type=float, default=0.2)
    parser.add_argument("--num_gradual", type=int, default=10)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("\n" + "=" * 60)
    print(f"Dataset: {args.dataset}")
    print("=" * 60)

    graph = Dataset(args.dataset, args.homo).graph

    methods: Dict[str, Callable] = {
        "GCN (Vanilla)": train_vanilla_gcn,
        "GCN + SCE": train_gcn_sce,
        "GCN + Forward": train_gcn_forward,
        "GCN + Co-teaching": train_coteaching,
    }

    results: Dict[str, Dict[str, float]] = {}

    for name, train_fn in methods.items():
        print(f"\nTraining {name}...")
        mf1_list: List[float] = []
        auc_list: List[float] = []

        for r in range(args.run):
            print(f"  Run {r + 1}/{args.run}...", end=" ")
            mf1, auc = train_fn(graph, args)
            mf1_list.append(mf1)
            auc_list.append(auc)
            print(f"MF1: {mf1 * 100:.2f}, AUC: {auc * 100:.2f}")

        results[name] = {
            "mf1_mean": float(np.mean(mf1_list) * 100.0),
            "mf1_std": float(np.std(mf1_list) * 100.0),
            "auc_mean": float(np.mean(auc_list) * 100.0),
            "auc_std": float(np.std(auc_list) * 100.0),
        }

        print(
            f"  >> {name}: MF1 {results[name]['mf1_mean']:.2f}±{results[name]['mf1_std']:.2f}, "
            f"AUC {results[name]['auc_mean']:.2f}±{results[name]['auc_std']:.2f}"
        )

    print("\n" + "=" * 80)
    print(f"FINAL RESULTS - {args.dataset}")
    print("=" * 80)
    print(f"{'Method':<25} | {'MF1 (Mean±Std)':<20} | {'AUC (Mean±Std)':<20}")
    print("-" * 80)

    for name, res in results.items():
        mf1_str = f"{res['mf1_mean']:.2f} ± {res['mf1_std']:.2f}"
        auc_str = f"{res['auc_mean']:.2f} ± {res['auc_std']:.2f}"
        print(f"{name:<25} | {mf1_str:<20} | {auc_str:<20}")

    print("=" * 80)


if __name__ == "__main__":
    main()


# Example:
# python run_baselines.py --dataset yelp --train_ratio 0.4 --hid_dim 64 --homo 0 --run 5
# python run_baselines.py --dataset amazon --train_ratio 0.4 --hid_dim 64 --homo 0 --run 5
# python run_baselines.py --dataset tfinance --train_ratio 0.4 --hid_dim 64 --order 7 --beta_p 1 --pop_size 20 --mut_rate 0.1 --homo 0 --beta 0.9 --lamb 0
# python run_baselines.py --dataset tsocial --train_ratio 0.4 --hid_dim 10 --order 8 --beta_p 1 --pop_size 20 --mut_rate 0.1 --homo 0 --beta 0.5 --lamb 0