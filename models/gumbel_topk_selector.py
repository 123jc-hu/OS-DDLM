"""Straight-through Gumbel-Top-K channel selection without replacement."""

from __future__ import annotations

import math
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.eeg_models import model_dict


class StraightThroughGumbelTopKSelector(nn.Module):
    """Select exactly K distinct physical EEG channels with masked ST samples."""

    def __init__(
        self,
        C: int,
        K: int,
        beta_start: float = 5.0,
        beta_end: float = 0.1,
        T_anneal: int = 25,
    ):
        super().__init__()
        if not 0 < K <= C:
            raise ValueError(f"Gumbel-Top-K requires 0 < K <= C, got K={K}, C={C}.")

        self.C = int(C)
        self.K = int(K)
        self.alpha = nn.Parameter(torch.zeros(self.C))
        nn.init.uniform_(self.alpha, -1e-2, 1e-2)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.T_anneal = max(1, int(T_anneal))
        self._t = 0

    def step_epoch(self, n: int = 1) -> None:
        self._t += int(n)

    def _beta(self) -> float:
        progress = min(self._t / self.T_anneal, 1.0)
        beta = self.beta_start * (self.beta_end / self.beta_start) ** progress
        return float(max(beta, 1e-6))

    @torch.no_grad()
    def hard_indices(self) -> list[int]:
        return torch.topk(self.alpha, self.K, largest=True, sorted=True).indices.tolist()

    def _masked_soft_steps(self) -> torch.Tensor:
        """Return K deterministic soft rows for diagnostics and optional losses."""
        available = torch.ones(self.C, device=self.alpha.device, dtype=torch.bool)
        rows = []
        for _ in range(self.K):
            logits = self.alpha.masked_fill(~available, -torch.inf)
            rows.append(torch.softmax(logits / self._beta(), dim=0))
            available[torch.argmax(logits)] = False
        return torch.stack(rows, dim=0)

    def current_p_soft(self) -> torch.Tensor:
        return self._masked_soft_steps()

    @staticmethod
    def normalized_entropy(p_soft: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        eps = 1e-12
        probabilities = p_soft.clamp_min(eps)
        entropy = -(probabilities * probabilities.log()).sum(dim=1) / math.log(probabilities.size(1))
        return entropy, entropy.mean()

    def _sample_without_replacement(self, batch_size: int) -> torch.Tensor:
        """Sample K ST one-hot rows per trial while masking already chosen channels."""
        available = torch.ones(batch_size, self.C, device=self.alpha.device, dtype=torch.bool)
        samples = []
        for _ in range(self.K):
            logits = self.alpha.unsqueeze(0).expand(batch_size, -1).masked_fill(~available, -torch.inf)
            selected = F.gumbel_softmax(logits, tau=self._beta(), hard=True, dim=1)
            samples.append(selected)
            # The forward hard choice alone controls availability; gradients stay
            # on the straight-through sample rather than on the discrete mask.
            available = available & ~selected.detach().bool()
        return torch.stack(samples, dim=1)

    def forward(self, x: torch.Tensor, current_epoch: int = None):
        batch_size, _, channels, _ = x.shape
        if channels != self.C:
            raise ValueError(f"Selector expects C={self.C}, got C={channels}.")

        if self.training:
            sampled_weights = self._sample_without_replacement(batch_size)
            selected = torch.einsum("bkc,bct->bkt", sampled_weights, x.squeeze(1)).unsqueeze(1)
            return selected, sampled_weights.mean(dim=0), sampled_weights

        indices = torch.tensor(self.hard_indices(), device=x.device, dtype=torch.long)
        hard_weights = torch.zeros(self.K, self.C, device=x.device, dtype=x.dtype)
        hard_weights[torch.arange(self.K, device=x.device), indices] = 1.0
        return x[:, :, indices, :], hard_weights, hard_weights


class GumbelTopKSelectorModel(nn.Module):
    """Attach exact-K, without-replacement selection to the configured backbone."""

    def __init__(
        self,
        backbone_cls,
        backbone_config: Dict[str, Any],
        C: int,
        K: int,
        beta_start: float,
        beta_end: float,
        T_anneal: int,
    ):
        super().__init__()
        if backbone_config["n_channels"] != K:
            raise ValueError("backbone n_channels must equal selector K.")

        self.selector = StraightThroughGumbelTopKSelector(
            C=C,
            K=K,
            beta_start=beta_start,
            beta_end=beta_end,
            T_anneal=T_anneal,
        )
        self.backbone = backbone_cls(backbone_config)

    def forward(self, x: torch.Tensor, current_epoch: int = None, **kwargs):
        selected, weights, _ = self.selector(x, current_epoch=current_epoch)
        output = self.backbone(selected, **kwargs)
        return (*output, weights) if isinstance(output, tuple) else (output, weights)

    def step_epoch(self, n: int = 1) -> None:
        self.selector.step_epoch(n)

    @torch.no_grad()
    def selected_indices(self) -> list[int]:
        return self.selector.hard_indices()


def build_model_gumbel_topk(config: Dict[str, Any]) -> nn.Module:
    """Build the exact-K Gumbel-Top-K baseline on the configured backbone."""
    backbone_name = str(config.get("backbone_model", config["model"]))
    backbone_cls = model_dict()[backbone_name].Model
    if not bool(config.get("use_selector", False)):
        return backbone_cls(config)

    channels = int(config["n_channels"])
    selected_k = int(config.get("selector_K", max(1, channels // 2)))
    backbone_config = dict(config)
    backbone_config["n_channels"] = selected_k
    return GumbelTopKSelectorModel(
        backbone_cls=backbone_cls,
        backbone_config=backbone_config,
        C=channels,
        K=selected_k,
        beta_start=float(config.get("selector_beta_start", 5.0)),
        beta_end=float(config.get("selector_beta_end", 0.1)),
        T_anneal=int(config.get("selector_T", config.get("epochs", 200))),
    )
