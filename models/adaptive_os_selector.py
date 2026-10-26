from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import math
from typing import Any, Dict

import torch
import torch.nn as nn

from models.eeg_models import model_dict


class ConcreteMultiSelector(nn.Module):
    """
    DEL4RSVP-style concrete channel selector over the real EEG channels only.

    The selector chooses K channels from C physical channels.
    There is no extra null / zero channel in this implementation.
    """

    def __init__(
        self,
        C: int,
        K: int,
        beta_start: float = 10.0,
        beta_end: float = 0.1,
        T_anneal: int = 150,
    ):
        super().__init__()
        self.C, self.K = int(C), int(K)
        self.alpha = nn.Parameter(torch.zeros(K, C))
        nn.init.uniform_(self.alpha, -1e-2, 1e-2)

        self.beta_start, self.beta_end = float(beta_start), float(beta_end)
        self.T_anneal = max(1, int(T_anneal))
        self._t = 0

    def step_epoch(self, n: int = 1):
        self._t += int(n)

    def _beta(self) -> float:
        t = min(self._t / self.T_anneal, 1.0)
        val = self.beta_start * (self.beta_end / self.beta_start) ** t
        return float(max(val, 1e-6))

    @torch.no_grad()
    def hard_indices(self):
        return torch.argmax(self.alpha, dim=1).tolist()

    @staticmethod
    def _row_softmax(logits: torch.Tensor, beta: float) -> torch.Tensor:
        return torch.softmax(logits / beta, dim=1)

    def current_p_soft(self) -> torch.Tensor:
        beta = self._beta()
        return self._row_softmax(self.alpha, beta)

    def forward(self, x: torch.Tensor, current_epoch: int = None):
        """
        x: [B, 1, C, T]
        """
        _, _, C, _ = x.shape
        assert C == self.C, f"Selector expects C={self.C}, got C={C}"
        beta = self._beta()

        if self.training:
            # DEL4RSVP uses deterministic temperature-annealed softmax rather
            # than stochastic Gumbel-noise exploration.
            logits = self.alpha
            P_soft = self._row_softmax(logits, beta)

            with torch.no_grad():
                idx = torch.argmax(logits, dim=1)
                W_hard = torch.zeros_like(P_soft)
                W_hard[torch.arange(self.K), idx] = 1.0
            W = P_soft + (W_hard - P_soft).detach()

            z = torch.einsum("kc,bct->bkt", W, x.squeeze(1)).unsqueeze(1)
            return z, W, P_soft

        idx = torch.argmax(self.alpha, dim=1)
        W_hard = torch.zeros(self.K, self.C, device=x.device, dtype=x.dtype)
        W_hard[torch.arange(self.K), idx] = 1.0
        z = x[:, :, idx, :]
        return z, W_hard, W_hard

    def orth_regularizer(self, P_soft: torch.Tensor) -> torch.Tensor:
        G = P_soft @ P_soft.t()
        I = torch.eye(self.K, device=G.device, dtype=G.dtype)
        return ((G - I) ** 2).sum()

    @staticmethod
    def normalized_entropy(P_soft: torch.Tensor):
        eps = 1e-12
        pk = P_soft.clamp_min(eps)
        Hk = -(pk * pk.log()).sum(dim=1) / math.log(pk.size(1))
        return Hk, Hk.mean()


class DeterministicTopKMultiSelector(ConcreteMultiSelector):
    """DEL4RSVP selector with deterministic, without-replacement hard choices."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.K > self.C:
            raise ValueError(f"Top-K selection requires K <= C, got K={self.K}, C={self.C}.")

    def _selection_weights(self, beta: float, use_straight_through: bool):
        available = torch.ones(self.C, device=self.alpha.device, dtype=torch.bool)
        weights = []
        soft_rows = []
        for head in range(self.K):
            logits = self.alpha[head].masked_fill(~available, -torch.inf)
            p_soft = torch.softmax(logits / beta, dim=0)
            index = torch.argmax(logits)
            hard = torch.zeros_like(p_soft)
            hard[index] = 1.0
            weights.append(p_soft + (hard - p_soft).detach() if use_straight_through else hard)
            soft_rows.append(p_soft)
            available[index] = False
        return torch.stack(weights), torch.stack(soft_rows)

    @torch.no_grad()
    def hard_indices(self):
        weights, _ = self._selection_weights(self._beta(), use_straight_through=False)
        return torch.argmax(weights, dim=1).tolist()

    def current_p_soft(self) -> torch.Tensor:
        _, p_soft = self._selection_weights(self._beta(), use_straight_through=False)
        return p_soft

    def forward(self, x: torch.Tensor, current_epoch: int = None):
        _, _, channels, _ = x.shape
        assert channels == self.C, f"Selector expects C={self.C}, got C={channels}"

        if self.training:
            weights, p_soft = self._selection_weights(self._beta(), use_straight_through=True)
            selected = torch.einsum("kc,bct->bkt", weights, x.squeeze(1)).unsqueeze(1)
            return selected, weights, p_soft

        indices = torch.as_tensor(self.hard_indices(), device=x.device, dtype=torch.long)
        hard_weights = torch.zeros(self.K, self.C, device=x.device, dtype=x.dtype)
        hard_weights[torch.arange(self.K, device=x.device), indices] = 1.0
        return x[:, :, indices, :], hard_weights, hard_weights


class ResidualGumbelMultiSelector(ConcreteMultiSelector):
    """Fixed-budget ResGS-style selector over the C physical EEG channels."""

    def __init__(self, *args, residual_weight: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.residual_weight = float(residual_weight)

    @staticmethod
    def _sample_gumbel_like(values: torch.Tensor) -> torch.Tensor:
        uniform = torch.rand_like(values).clamp_(1e-6, 1.0 - 1e-6)
        return -torch.log(-torch.log(uniform))

    def forward(self, x: torch.Tensor, current_epoch: int = None):
        batch_size, _, channels, _ = x.shape
        assert channels == self.C, f"Selector expects C={self.C}, got C={channels}"

        if not self.training:
            indices = torch.argmax(self.alpha, dim=1)
            hard_weights = torch.zeros(self.K, self.C, device=x.device, dtype=x.dtype)
            hard_weights[torch.arange(self.K), indices] = 1.0
            return x[:, :, indices, :], hard_weights, hard_weights

        beta = self._beta()
        global_soft = self._row_softmax(self.alpha, beta)
        logits = self.alpha.unsqueeze(0).expand(batch_size, -1, -1)
        sampled_soft = self._row_softmax(logits + self._sample_gumbel_like(logits), beta)
        with torch.no_grad():
            sampled_indices = torch.argmax(sampled_soft, dim=2)
            sampled_hard = torch.zeros_like(sampled_soft)
            sampled_hard.scatter_(2, sampled_indices.unsqueeze(-1), 1.0)
        sampled_st = sampled_soft + (sampled_hard - sampled_soft).detach()
        weights = (
            (1.0 - self.residual_weight) * sampled_st
            + self.residual_weight * global_soft.unsqueeze(0)
        )
        selected = torch.einsum("bkc,bct->bkt", weights, x.squeeze(1)).unsqueeze(1)
        return selected, weights.mean(dim=0), global_soft


class ModelWithSelector(nn.Module):
    """
    Prepend the concrete channel selector before a K-channel backbone.
    """

    def __init__(
        self,
        backbone_cls,
        backbone_config: Dict[str, Any],
        C: int,
        K: int,
        beta_start: float = 10.0,
        beta_end: float = 0.1,
        T_anneal: int = 150,
        selector_variant: str = "ours",
        residual_weight: float = 0.1,
    ):
        super().__init__()
        assert backbone_config["n_channels"] == K, "backbone n_channels must equal K"

        selector_cls = {
            "deterministic_topk": DeterministicTopKMultiSelector,
            "resgs": ResidualGumbelMultiSelector,
        }.get(selector_variant, ConcreteMultiSelector)
        selector_kwargs = {"residual_weight": residual_weight} if selector_variant == "resgs" else {}
        self.selector = selector_cls(
            C=C,
            K=K,
            beta_start=beta_start,
            beta_end=beta_end,
            T_anneal=T_anneal,
            **selector_kwargs,
        )
        self.backbone = backbone_cls(backbone_config)
        self.post_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor, current_epoch: int = None, **kwargs):
        z, W, _ = self.selector(x, current_epoch=current_epoch)
        z = z * self.post_scale
        out = self.backbone(z, **kwargs)
        return (*out, W) if isinstance(out, tuple) else (out, W)

    def step_epoch(self, n: int = 1):
        self.selector.step_epoch(n)

    @torch.no_grad()
    def selected_indices(self):
        return self.selector.hard_indices()

    @torch.no_grad()
    def project_channel_attn(self, full_attn: torch.Tensor) -> torch.Tensor:
        assert full_attn.dim() == 4 and full_attn.size(2) == self.selector.C
        idx = torch.argmax(self.selector.alpha, dim=1)
        return full_attn[:, :, idx, :]


def build_model(config: Dict[str, Any]) -> nn.Module:
    """
    Unified factory:
    - use_selector=False -> backbone only
    - use_selector=True  -> selector + backbone
    """

    backbone_name = str(config.get("backbone_model", config["model"]))
    BaseCls = model_dict()[backbone_name].Model
    use_selector = bool(config.get("use_selector", False))
    if not use_selector:
        return BaseCls(config)

    C = int(config["n_channels"])
    K = int(config.get("selector_K", max(1, C // 2)))
    beta_start = float(config.get("selector_beta_start", 10.0))
    beta_end = float(config.get("selector_beta_end", 0.1))
    T_anneal = int(config.get("selector_T", config.get("epochs", 100)))
    selector_variant = str(config.get("selector_variant", "ours")).lower()
    backbone_cfg = dict(config)
    backbone_cfg["n_channels"] = K

    return ModelWithSelector(
        backbone_cls=BaseCls,
        backbone_config=backbone_cfg,
        C=C,
        K=K,
        beta_start=beta_start,
        beta_end=beta_end,
        T_anneal=T_anneal,
        selector_variant=selector_variant,
        residual_weight=float(config.get("resgs_residual_weight", 0.1)),
    )
