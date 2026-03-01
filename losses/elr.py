from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Wrapper for torch.nn.functional.cross_entropy.

    Args:
        logits: Model outputs of shape (N, C).
        targets: Integer class labels of shape (N,).
        weight: Optional class weights of shape (C,).

    Returns:
        Cross-entropy loss (scalar).
    """
    return F.cross_entropy(logits, targets, weight=weight)


class ELRLoss(nn.Module):
    """
    Early-Learning Regularization (ELR) loss.

    The target distribution is updated as an EMA of normalized predictions:
        target <- beta * target + (1 - beta) * normalize(pred)

    The final objective is:
        CE(logits, labels) + lamb * mean(log(1 - sum(target * pred)))

    Notes:
    - `num_nodes` should match the number of training instances whose targets are tracked.
    - This implementation tracks targets on CPU by default; it will be moved to the
      correct device on first forward call.
    """

    def __init__(self, num_nodes: int, num_classes: int, beta: float, lamb: float) -> None:
        super().__init__()
        if not (0.0 <= beta <= 1.0):
            raise ValueError(f"beta must be in [0, 1], got {beta}.")
        if lamb < 0.0:
            raise ValueError(f"lamb must be non-negative, got {lamb}.")

        self.beta = float(beta)
        self.lamb = float(lamb)

        # Registered buffer so it moves with .to(device) and is saved in state_dict
        self.register_buffer("target", torch.zeros(num_nodes, num_classes))

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        clamp_min: float = 1e-4,
    ) -> torch.Tensor:
        """
        Args:
            logits: Shape (N, C).
            labels: Shape (N,).
            weight: Optional class weights of shape (C,).
            clamp_min: Clamp lower bound for probabilities; upper bound is 1 - clamp_min.

        Returns:
            Loss scalar tensor.
        """
        probs = F.softmax(logits, dim=1)
        probs = torch.clamp(probs, clamp_min, 1.0 - clamp_min)

        # Ensure buffer is on the same device/dtype as probs
        if self.target.device != probs.device:
            self.target = self.target.to(device=probs.device)

        # Detach to avoid backprop through target update
        probs_detached = probs.detach()
        probs_norm = probs_detached / probs_detached.sum(dim=1, keepdim=True)

        self.target = self.beta * self.target + (1.0 - self.beta) * probs_norm

        ce = F.cross_entropy(logits, labels, weight=weight)

        # ELR regularizer: log(1 - <target, probs>)
        # Note: log argument is in (0, 1], so log() is <= 0.
        inner = (self.target * probs).sum(dim=1)
        elr_reg = torch.log(1.0 - inner).mean()

        return ce + self.lamb * elr_reg


def elr_loss(num_nodes: int, num_classes: int, args) -> ELRLoss:
    """
    Backward-compatible factory to match the original code usage.

    Expects `args` to have attributes: `beta` and `lamb`.
    """
    return ELRLoss(num_nodes=num_nodes, num_classes=num_classes, beta=args.beta, lamb=args.lamb)