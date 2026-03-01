from __future__ import annotations

from typing import List, Tuple

import dgl.function as fn
import numpy as np
import scipy.special
import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-10


class EvolutionStrategyThetaOptimizer:
    """
    Evolution-strategy optimizer for theta that cooperates with gradient descent.
    """

    def __init__(self, population_size: int, mutation_rate: float, crossover_rate: float = 0.8) -> None:
        if population_size <= 0:
            raise ValueError(f"population_size must be positive, got {population_size}.")
        if not (0.0 <= mutation_rate <= 1.0):
            raise ValueError(f"mutation_rate must be in [0, 1], got {mutation_rate}.")
        if not (0.0 <= crossover_rate <= 1.0):
            raise ValueError(f"crossover_rate must be in [0, 1], got {crossover_rate}.")

        self.population_size = int(population_size)
        self.mutation_rate = float(mutation_rate)
        self.crossover_rate = float(crossover_rate)

    def initialize_population(self, base_theta: torch.Tensor) -> torch.Tensor:
        """
        Initialize a population around current theta (vectorized).

        Args:
            base_theta: Tensor of shape (N, K) or (K,).

        Returns:
            population: Tensor of shape (P, N, K) or (P, K).
        """
        noise = torch.randn(self.population_size, *base_theta.shape, device=base_theta.device)
        return base_theta.unsqueeze(0) + noise * 0.1

    def mutate(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Scale-aware masked mutation (vectorized).

        Args:
            theta: Tensor of shape (P, N, K) or (P, K).

        Returns:
            mutated: Same shape as theta.
        """
        mask = (torch.rand_like(theta) < self.mutation_rate).to(theta.dtype)
        mutation = torch.randn_like(theta) * (0.1 + 0.05 * torch.abs(theta))
        return theta + mask * mutation

    def crossover(self, parent: torch.Tensor, donor: torch.Tensor) -> torch.Tensor:
        """
        Binomial crossover (vectorized).

        Args:
            parent: Tensor of shape (P, ...).
            donor: Tensor of shape (P, ...).

        Returns:
            trial: Tensor of shape (P, ...).
        """
        cross_mask = torch.rand_like(parent) < self.crossover_rate
        return torch.where(cross_mask, donor, parent)

    def select(
        self,
        population: torch.Tensor,
        fitness: torch.Tensor,
        elite_num: int = 3,
        tournament_k: int = 3,
    ) -> torch.Tensor:
        """
        Elitist tournament selection (lower fitness is better).

        Args:
            population: Tensor of shape (P, ...).
            fitness: Tensor of shape (P,).
            elite_num: Number of elites preserved.
            tournament_k: Tournament size.

        Returns:
            selected: Tensor of shape (population_size, ...).
        """
        if fitness.ndim != 1:
            raise ValueError(f"fitness must be 1D, got shape {fitness.shape}.")
        if population.size(0) != fitness.size(0):
            raise ValueError("population and fitness must have matching first dimension.")

        p = int(population.size(0))
        elite_num = max(0, min(int(elite_num), p))
        tournament_k = max(1, int(tournament_k))

        sorted_idx = torch.argsort(fitness)  # ascending: best first
        elites = population[sorted_idx[:elite_num]]

        num_to_select = self.population_size - elite_num
        if num_to_select <= 0:
            return elites

        device = population.device
        contestants_idx = torch.randint(0, p, (num_to_select, tournament_k), device=device)
        contestants_fit = fitness[contestants_idx]
        winner_col_idx = torch.argmin(contestants_fit, dim=1)
        winner_indices = torch.gather(contestants_idx, 1, winner_col_idx.unsqueeze(1)).squeeze(1)

        selected = population[winner_indices]
        return torch.cat([elites, selected], dim=0)


class DynamicPolyConv(nn.Module):
    """
    Polynomial convolution with dynamic coefficients.

    Given bases [H, L H, L^2 H, ...], aggregate them using theta and then apply a linear projection.
    """

    def __init__(self, in_feats: int, out_feats: int) -> None:
        super().__init__()
        self.in_feats = int(in_feats)
        self.out_feats = int(out_feats)
        self.linear = nn.Linear(self.in_feats, self.out_feats)

    @staticmethod
    def _unnormalized_laplacian(
        graph,
        feat: torch.Tensor,
        d_invsqrt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute: feat - D^{-1/2} A D^{-1/2} feat
        """
        graph.ndata["h"] = feat * d_invsqrt
        graph.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
        agg = graph.ndata.pop("h") * d_invsqrt
        return feat - agg

    def compute_bases(self, graph, feat: torch.Tensor, max_k: int) -> torch.Tensor:
        """
        Precompute polynomial bases: [H, L H, L^2 H, ..., L^{K} H].

        Args:
            graph: DGLGraph
            feat: Node features, shape (N, F)
            max_k: Maximum polynomial order K

        Returns:
            bases: Tensor of shape (N, K+1, F)
        """
        if max_k < 0:
            raise ValueError(f"max_k must be non-negative, got {max_k}.")

        with graph.local_scope():
            degs = graph.in_degrees().float().clamp(min=1)
            d_invsqrt = torch.pow(degs, -0.5).unsqueeze(-1).to(feat.device)

            bases = [feat]
            current = feat
            for _ in range(max_k):
                current = self._unnormalized_laplacian(graph, current, d_invsqrt)
                bases.append(current)

            return torch.stack(bases, dim=1)

    def forward(self, graph, feat: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        Args:
            theta: Either
                - shared coefficients: shape (K+1,)
                - node-specific coefficients: shape (N, K+1)

        Returns:
            output features: shape (N, out_feats)
        """
        bases = self.compute_bases(graph, feat, int(theta.shape[-1] - 1))

        if theta.ndim == 1:
            h = torch.einsum("nkf,k->nf", bases, theta)
        elif theta.ndim == 2:
            h = torch.einsum("nkf,nk->nf", bases, theta)
        else:
            raise ValueError(f"Invalid theta shape: {tuple(theta.shape)}")

        return self.linear(h)


def _init_beta_wavelet_bias(c: int, p: int, q: int) -> torch.Tensor:
    """
    Initialize Beta-wavelet base coefficients as bias for theta_generator.
    """
    base_theta = torch.zeros(c + 1)
    beta_norm = 1.0 / (2.0 * scipy.special.beta(p + 1, q + 1))
    for k in range(c + 1):
        binom_coeff = scipy.special.comb(c, k)
        base_theta[k] = beta_norm * binom_coeff * (0.5 ** k) * ((-0.5) ** (c - k))
    return base_theta


class EOGFDModel(nn.Module):
    """
    EOGFD model for homogeneous graphs.
    """

    def __init__(
        self,
        in_feats: int,
        hid_feats: int,
        num_classes: int,
        d: int = 3,
        iterations: int = 3,
        beta_p: int = 2,
        population_size: int = 20,
        mutation_rate: float = 0.1,
        adaptive_alpha_init: float = 0.5,
    ) -> None:
        super().__init__()

        self.d = int(d)
        self.iterations = int(iterations)

        self.theta_generator = nn.Linear(hid_feats, self.d + 1)
        self._init_theta_generator(self.d, beta_p, self.d + 1 - beta_p)

        self.conv = DynamicPolyConv(hid_feats, hid_feats)
        self.gru = nn.GRUCell(hid_feats, hid_feats)

        self.input_encoder = nn.Sequential(
            nn.Linear(in_feats, hid_feats),
            nn.ReLU(),
            nn.LayerNorm(hid_feats),
        )
        self.output_decoder = nn.Sequential(
            nn.Linear(hid_feats, hid_feats),
            nn.ReLU(),
            nn.LayerNorm(hid_feats),
            nn.Linear(hid_feats, num_classes),
        )

        self.res_weights = nn.Parameter(torch.ones(self.iterations))
        self.es_optimizer = EvolutionStrategyThetaOptimizer(
            population_size=population_size,
            mutation_rate=mutation_rate,
        )

        # Keep theta from the last iteration for logging/analysis.
        self.register_buffer("theta_last", torch.empty(0))

        self.adaptive_alpha = nn.Parameter(torch.tensor([adaptive_alpha_init], dtype=torch.float32))

    def _init_theta_generator(self, c: int, p: int, q: int) -> None:
        base_theta = _init_beta_wavelet_bias(c, p, q)
        with torch.no_grad():
            nn.init.orthogonal_(self.theta_generator.weight)
            self.theta_generator.bias.copy_(base_theta)

    def evolutionary_step(self, graph, h: torch.Tensor, base_theta: torch.Tensor) -> torch.Tensor:
        """
        One ES step that returns the selected theta (node-specific).
        """
        bases = self.conv.compute_bases(graph, h, self.d)

        population = self.es_optimizer.initialize_population(base_theta)  # (P, N, K+1)

        with torch.no_grad():
            poly_agg = torch.einsum("nkf,pnk->pnf", bases, population)
            conv_feats = self.conv.linear(poly_agg)
            target = h.unsqueeze(0)
            fitness = F.mse_loss(conv_feats, target.expand_as(conv_feats), reduction="none").mean(dim=(1, 2))

        donors = self.es_optimizer.mutate(population)
        perm = torch.randperm(population.size(0), device=population.device)
        trials = self.es_optimizer.crossover(population, donors[perm])

        with torch.no_grad():
            poly_agg_trials = torch.einsum("nkf,pnk->pnf", bases, trials)
            conv_feats_trials = self.conv.linear(poly_agg_trials)
            fitness_trials = F.mse_loss(conv_feats_trials, target.expand_as(conv_feats_trials), reduction="none").mean(
                dim=(1, 2)
            )

        # Include base theta as a candidate.
        combined_pop = torch.cat([trials, base_theta.unsqueeze(0)], dim=0)
        combined_fit = torch.cat([fitness_trials, fitness.mean().unsqueeze(0)], dim=0)

        best_theta = self.es_optimizer.select(combined_pop, combined_fit, elite_num=1)[0]
        return best_theta

    def forward(self, graph, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.input_encoder(feat)
        res_feats: List[torch.Tensor] = []

        for _ in range(self.iterations):
            base_theta = F.softmax(self.theta_generator(h), dim=1)

            with torch.no_grad():
                delta_theta = self.evolutionary_step(graph, h, base_theta)

            theta = self.adaptive_alpha * base_theta + (1.0 - self.adaptive_alpha) * delta_theta
            self.theta_last = theta.detach()

            conv_feat = self.conv(graph, h, theta)
            h = self.gru(conv_feat, h)
            res_feats.append(h)

        stacked = torch.stack(res_feats, dim=0) * self.res_weights.view(-1, 1, 1)
        h_out = torch.mean(stacked, dim=0)

        logits = self.output_decoder(h_out)
        return logits, self.theta_last


class EOGFDHeteroModel(nn.Module):
    """
    EOGFD model for heterogeneous graphs.
    """

    def __init__(
        self,
        in_feats: int,
        hid_feats: int,
        num_classes: int,
        d: int = 3,
        iterations: int = 1,
        beta_p: int = 2,
        population_size: int = 20,
        mutation_rate: float = 0.1,
        adaptive_alpha_init: float = 0.5,
    ) -> None:
        super().__init__()

        self.d = int(d)
        self.iterations = int(iterations)

        self.theta_generator = nn.Linear(hid_feats, self.d + 1)
        self._init_theta_generator(self.d, beta_p, self.d + 1 - beta_p)

        self.conv = DynamicPolyConv(hid_feats, hid_feats)
        self.gru = nn.GRUCell(hid_feats, hid_feats)

        self.input_encoder = nn.Sequential(
            nn.Linear(in_feats, hid_feats),
            nn.ReLU(),
            nn.LayerNorm(hid_feats),
        )
        self.output_decoder = nn.Sequential(
            nn.Linear(hid_feats, hid_feats),
            nn.ReLU(),
            nn.LayerNorm(hid_feats),
            nn.Linear(hid_feats, num_classes),
        )

        self.res_weights = nn.Parameter(torch.ones(self.iterations))
        self.es_optimizer = EvolutionStrategyThetaOptimizer(
            population_size=population_size,
            mutation_rate=mutation_rate,
        )

        self.register_buffer("theta_last", torch.empty(0))
        self.adaptive_alpha = nn.Parameter(torch.tensor([adaptive_alpha_init], dtype=torch.float32))

    def _init_theta_generator(self, c: int, p: int, q: int) -> None:
        base_theta = _init_beta_wavelet_bias(c, p, q)
        with torch.no_grad():
            nn.init.orthogonal_(self.theta_generator.weight)
            self.theta_generator.bias.copy_(base_theta)

    def evolutionary_step(self, graph, h: torch.Tensor, base_theta: torch.Tensor) -> torch.Tensor:
        bases = self.conv.compute_bases(graph, h, self.d)
        population = self.es_optimizer.initialize_population(base_theta)

        with torch.no_grad():
            poly_agg = torch.einsum("nkf,pnk->pnf", bases, population)
            conv_feats = self.conv.linear(poly_agg)
            target = h.unsqueeze(0)
            fitness = F.mse_loss(conv_feats, target.expand_as(conv_feats), reduction="none").mean(dim=(1, 2))

        donors = self.es_optimizer.mutate(population)
        perm = torch.randperm(population.size(0), device=population.device)
        trials = self.es_optimizer.crossover(population, donors[perm])

        with torch.no_grad():
            poly_agg_trials = torch.einsum("nkf,pnk->pnf", bases, trials)
            conv_feats_trials = self.conv.linear(poly_agg_trials)
            fitness_trials = F.mse_loss(conv_feats_trials, target.expand_as(conv_feats_trials), reduction="none").mean(
                dim=(1, 2)
            )

        combined_pop = torch.cat([trials, base_theta.unsqueeze(0)], dim=0)
        combined_fit = torch.cat([fitness_trials, fitness.mean().unsqueeze(0)], dim=0)

        best_theta = self.es_optimizer.select(combined_pop, combined_fit, elite_num=1)[0]
        return best_theta

    def forward(self, graph, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.input_encoder(feat)
        rel_reprs: List[torch.Tensor] = []

        for relation in graph.canonical_etypes:
            sub_g = graph[relation]
            h_rel = h
            res_feats: List[torch.Tensor] = []

            for _ in range(self.iterations):
                base_theta = F.softmax(self.theta_generator(h_rel), dim=1)

                with torch.no_grad():
                    delta_theta = self.evolutionary_step(sub_g, h_rel, base_theta)

                theta = self.adaptive_alpha * base_theta + (1.0 - self.adaptive_alpha) * delta_theta
                self.theta_last = theta.detach()

                conv_feat = self.conv(sub_g, h_rel, theta)
                h_rel = self.gru(conv_feat, h_rel)
                res_feats.append(h_rel)

            stacked = torch.stack(res_feats, dim=0) * self.res_weights.view(-1, 1, 1)
            rel_reprs.append(torch.mean(stacked, dim=0))

        h_all = torch.stack(rel_reprs, dim=0).sum(dim=0)
        logits = self.output_decoder(h_all)
        return logits, self.theta_last