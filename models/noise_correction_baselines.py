"""
Baseline models and robust-learning components for comparison with EOGFD.

Included:
- GCN
- HeteroGCN
- SCELoss (Symmetric Cross Entropy)
- ForwardCorrectionLoss (loss correction with a noise transition matrix)
- CoTeaching (two-network small-loss sample selection strategy)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-7


# ---------------------------------------------------------------------
# 1) GCN
# ---------------------------------------------------------------------
class GCNLayer(nn.Module):
    """
    A single GCN layer using normalized message passing:
        H' = D^{-1/2} A D^{-1/2} H W
    """

    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        activation: Optional[callable] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph, feat: torch.Tensor) -> torch.Tensor:
        with graph.local_scope():
            degs = graph.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5).unsqueeze(-1).to(feat.device)

            graph.ndata["h"] = feat * norm
            graph.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
            h = graph.ndata["h"] * norm

            h = self.linear(h)
            h = self.dropout(h)
            if self.activation is not None:
                h = self.activation(h)
            return h


class GCN(nn.Module):
    """
    Two-layer GCN classifier for homogeneous graphs.
    """

    def __init__(self, in_feats: int, hid_feats: int, num_classes: int, dropout: float = 0.5) -> None:
        super().__init__()
        self.layer1 = GCNLayer(in_feats, hid_feats, activation=F.relu, dropout=dropout)
        self.layer2 = GCNLayer(hid_feats, num_classes, activation=None, dropout=0.0)

    def forward(self, graph, feat: torch.Tensor) -> torch.Tensor:
        h = self.layer1(graph, feat)
        return self.layer2(graph, h)


class HeteroGCN(nn.Module):
    """
    A simple heterogeneous-graph GCN baseline.

    For each canonical edge type, apply two GCN layers on the induced subgraph,
    then aggregate relation-specific node embeddings by mean pooling.
    """

    def __init__(self, in_feats: int, hid_feats: int, num_classes: int, dropout: float = 0.5) -> None:
        super().__init__()
        self.layer1 = GCNLayer(in_feats, hid_feats, activation=F.relu, dropout=dropout)
        self.layer2 = GCNLayer(hid_feats, hid_feats, activation=F.relu, dropout=dropout)
        self.classifier = nn.Linear(hid_feats, num_classes)

    def forward(self, graph, feat: torch.Tensor) -> torch.Tensor:
        # If heterogeneous, iterate relations; otherwise treat as homogeneous.
        if hasattr(graph, "canonical_etypes") and len(graph.canonical_etypes) > 1:
            rel_out = []
            for etype in graph.canonical_etypes:
                sub_g = graph[etype]
                h = self.layer1(sub_g, feat)
                h = self.layer2(sub_g, h)
                rel_out.append(h)
            h_all = torch.stack(rel_out, dim=0).mean(dim=0)
        else:
            h_all = self.layer1(graph, feat)
            h_all = self.layer2(graph, h_all)

        return self.classifier(h_all)


# ---------------------------------------------------------------------
# 2) SCE losses
# ---------------------------------------------------------------------
class SCELoss(nn.Module):
    """
    Symmetric Cross Entropy (SCE) loss.

    Paper: "Symmetric Cross Entropy for Robust Learning with Noisy Labels" (ICCV 2019)
    SCE = alpha * CE + beta * RCE
    """

    def __init__(self, alpha: float = 0.1, beta: float = 1.0, num_classes: int = 2) -> None:
        super().__init__()
        if alpha < 0.0 or beta < 0.0:
            raise ValueError("alpha and beta must be non-negative.")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.num_classes = int(num_classes)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=weight)

        probs = F.softmax(logits, dim=1)
        probs = torch.clamp(probs, EPS, 1.0 - EPS)

        target_onehot = F.one_hot(targets, num_classes=self.num_classes).float()
        target_onehot = torch.clamp(target_onehot, 1e-4, 1.0)

        # Reverse cross entropy: -E[ sum_i p_i log(q_i) ]
        rce = -(probs * torch.log(target_onehot)).sum(dim=1).mean()

        return self.alpha * ce + self.beta * rce


class ForwardCorrectionLoss(nn.Module):
    """
    Forward loss correction with a (fixed or estimated) transition matrix T,
    where T[i, j] = P(observed=j | true=i).

    Paper: "Making Deep Neural Networks Robust to Label Noise: a Loss Correction Approach" (CVPR 2017)
    """

    def __init__(self, num_classes: int = 2, noise_rate: float = 0.2) -> None:
        super().__init__()
        if not (0.0 <= noise_rate < 1.0):
            raise ValueError(f"noise_rate must be in [0, 1), got {noise_rate}.")
        self.num_classes = int(num_classes)
        self.noise_rate = float(noise_rate)

        t = torch.eye(self.num_classes) * (1.0 - self.noise_rate)
        t += (1.0 - torch.eye(self.num_classes)) * (self.noise_rate / (self.num_classes - 1))
        self.register_buffer("T", t)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        probs_corrected = probs @ self.T.t()
        probs_corrected = torch.clamp(probs_corrected, EPS, 1.0 - EPS)

        return F.nll_loss(torch.log(probs_corrected), targets, weight=weight)

    @torch.no_grad()
    def estimate_T_from_model(
        self,
        model: nn.Module,
        graph,
        features: torch.Tensor,
        labels: torch.Tensor,
        train_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate transition matrix T from a trained model using mean predicted
        probabilities within each observed class in the training set.

        This is a simple estimator; more advanced estimators can be plugged in.
        """
        model.eval()
        logits = model(graph, features)
        probs = F.softmax(logits[train_mask], dim=1)
        y = labels[train_mask]

        t_est = torch.zeros(self.num_classes, self.num_classes, device=probs.device)
        for c in range(self.num_classes):
            mask = (y == c)
            if mask.any():
                t_est[c] = probs[mask].mean(dim=0)

        t_est = t_est / t_est.sum(dim=1, keepdim=True).clamp(min=EPS)
        self.T = t_est.to(self.T.device)
        return self.T


# ---------------------------------------------------------------------
# 3) Co-teaching
# ---------------------------------------------------------------------
@dataclass
class CoTeachingStepOutput:
    loss1: torch.Tensor
    loss2: torch.Tensor
    logits1: torch.Tensor
    logits2: torch.Tensor


class CoTeaching:
    """
    Co-teaching strategy: two networks teach each other by selecting small-loss samples.

    Paper: "Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels" (NeurIPS 2018)
    """

    def __init__(
        self,
        model1: nn.Module,
        model2: nn.Module,
        forget_rate: float = 0.2,
        num_gradual: int = 10,
        exponent: float = 1.0,
    ) -> None:
        if not (0.0 <= forget_rate < 1.0):
            raise ValueError(f"forget_rate must be in [0, 1), got {forget_rate}.")
        self.model1 = model1
        self.model2 = model2
        self.forget_rate = float(forget_rate)
        self.num_gradual = int(num_gradual)
        self.exponent = float(exponent)

    def get_forget_rate(self, epoch: int) -> float:
        if epoch < self.num_gradual:
            return self.forget_rate * min(1.0, (epoch / max(1, self.num_gradual)) ** self.exponent)
        return self.forget_rate

    @staticmethod
    def select_small_loss_indices(loss_per_sample: torch.Tensor, forget_rate: float) -> torch.Tensor:
        num = int((1.0 - forget_rate) * loss_per_sample.numel())
        num = max(num, 1)
        _, idx = torch.sort(loss_per_sample)
        return idx[:num]

    def train_step(
        self,
        graph,
        features: torch.Tensor,
        labels: torch.Tensor,
        train_mask: torch.Tensor,
        class_weight: Optional[torch.Tensor],
        epoch: int,
    ) -> CoTeachingStepOutput:
        """
        Perform one co-teaching step (forward + sample selection).

        Note: this method does not call backward() or optimizer.step().
        The caller should:
            - compute loss1/loss2,
            - backprop for each model with its own optimizer.
        """
        fr = self.get_forget_rate(epoch)

        y = labels[train_mask]
        logits1 = self.model1(graph, features)
        logits2 = self.model2(graph, features)

        out1 = logits1[train_mask]
        out2 = logits2[train_mask]

        loss1_each = F.cross_entropy(out1, y, weight=class_weight, reduction="none")
        loss2_each = F.cross_entropy(out2, y, weight=class_weight, reduction="none")

        idx_from_2 = self.select_small_loss_indices(loss2_each, fr)
        idx_from_1 = self.select_small_loss_indices(loss1_each, fr)

        loss1 = F.cross_entropy(out1[idx_from_2], y[idx_from_2], weight=class_weight)
        loss2 = F.cross_entropy(out2[idx_from_1], y[idx_from_1], weight=class_weight)

        return CoTeachingStepOutput(loss1=loss1, loss2=loss2, logits1=logits1, logits2=logits2)