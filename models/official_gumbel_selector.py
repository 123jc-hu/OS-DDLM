from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from typing import Any, Dict

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.eeg_models import model_dict


class OfficialGumbelSelector(nn.Module):
    """Faithful selector semantics for Strypsteen et al. using per-sample Gumbel sampling."""

    def __init__(
        self,
        C: int,
        K: int,
        beta_start: float = 10.0,
        beta_end: float = 0.1,
        beta_anneal_epochs: int = 150,
        tau_start: float = 3.0,
        tau_end: float = 1.1,
        tau_anneal_epochs: int = 200,
    ):
        super().__init__()
        self.C = int(C)
        self.K = int(K)

        self.alpha = nn.Parameter(torch.randn(self.K, self.C) / 100.0)

        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.beta_anneal_epochs = max(1, int(beta_anneal_epochs))

        self.tau_start = float(tau_start)
        self.tau_end = float(tau_end)
        self.tau_anneal_epochs = max(1, int(tau_anneal_epochs))

        self._epoch = 0

    def step_epoch(self, n: int = 1):
        self._epoch += int(n)

    def _exp_anneal(self, start: float, end: float, current: int, total: int) -> float:
        frac = min(max(current / max(total, 1), 0.0), 1.0)
        value = float(start) * ((float(end) / float(start)) ** frac)
        return float(max(value, 1e-6))

    def _beta(self) -> float:
        return self._exp_anneal(self.beta_start, self.beta_end, self._epoch, self.beta_anneal_epochs)

    def _tau(self) -> float:
        return self._exp_anneal(self.tau_start, self.tau_end, self._epoch, self.tau_anneal_epochs)

    def current_p_soft(self) -> torch.Tensor:
        # Official duplicate regularizer uses selector probabilities without the temperature term.
        return torch.softmax(self.alpha, dim=1)

    @staticmethod
    def _sample_gumbel_like(x: torch.Tensor) -> torch.Tensor:
        u = torch.rand_like(x).clamp_(1e-6, 1 - 1e-6)
        return -torch.log(-torch.log(u))

    @staticmethod
    def normalized_entropy(P_soft: torch.Tensor):
        eps = 1e-12
        pk = P_soft.clamp_min(eps)
        Hk = -(pk * pk.log()).sum(dim=1) / math.log(pk.size(1))
        return Hk, Hk.mean()

    @torch.no_grad()
    def hard_indices(self):
        return torch.argmax(self.alpha, dim=1).tolist()

    def forward(self, x: torch.Tensor):
        batch_size, _, channels, _ = x.shape
        assert channels == self.C, f"Selector expects C={self.C}, got C={channels}"

        x_squeezed = x.squeeze(1)  # [B, C, T]

        if self.training:
            beta = self._beta()
            logits = self.alpha.unsqueeze(0).expand(batch_size, -1, -1)
            logits = (logits + self._sample_gumbel_like(logits)) / beta
            sampled_weights = torch.softmax(logits, dim=2)  # [B, K, C]
            z = torch.einsum("bkc,bct->bkt", sampled_weights, x_squeezed).unsqueeze(1)
            return z, sampled_weights.mean(dim=0), sampled_weights

        indices = torch.argmax(self.alpha, dim=1)
        hard_weights = torch.zeros(self.K, self.C, device=x.device, dtype=x.dtype)
        hard_weights[torch.arange(self.K), indices] = 1.0
        z = torch.einsum("kc,bct->bkt", hard_weights, x_squeezed).unsqueeze(1)
        return z, hard_weights, hard_weights


class OfficialGumbelSelectorModel(nn.Module):
    """Attach the official selector semantics in front of an arbitrary K-channel backbone."""

    def __init__(
        self,
        backbone_cls,
        backbone_config: Dict[str, Any],
        C: int,
        K: int,
        beta_start: float = 10.0,
        beta_end: float = 0.1,
        beta_anneal_epochs: int = 150,
        tau_start: float = 3.0,
        tau_end: float = 1.1,
        tau_anneal_epochs: int = 200,
    ):
        super().__init__()
        assert backbone_config["n_channels"] == K, "backbone n_channels must match K"

        self.selector = OfficialGumbelSelector(
            C=C,
            K=K,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_anneal_epochs=beta_anneal_epochs,
            tau_start=tau_start,
            tau_end=tau_end,
            tau_anneal_epochs=tau_anneal_epochs,
        )
        self.backbone = backbone_cls(backbone_config)

    def forward(self, x: torch.Tensor, current_epoch: int = None, **kwargs):
        z, W, P = self.selector(x)
        out = self.backbone(z, **kwargs)
        return (*out, W) if isinstance(out, tuple) else (out, W)

    def step_epoch(self, n: int = 1):
        self.selector.step_epoch(n)

    @torch.no_grad()
    def selected_indices(self):
        return self.selector.hard_indices()


def build_model_official(config: Dict[str, Any]) -> nn.Module:
    """Build the official Gumbel-Softmax baseline on top of the configured backbone."""
    backbone_name = str(config.get("backbone_model", config["model"]))
    BaseCls = model_dict()[backbone_name].Model
    if not bool(config.get("use_selector", False)):
        return BaseCls(config)

    C = int(config["n_channels"])
    K = int(config.get("selector_K", max(1, C // 2)))

    backbone_cfg = dict(config)
    backbone_cfg["n_channels"] = K

    return OfficialGumbelSelectorModel(
        backbone_cls=BaseCls,
        backbone_config=backbone_cfg,
        C=C,
        K=K,
        beta_start=float(config.get("selector_beta_start", 10.0)),
        beta_end=float(config.get("selector_beta_end", 0.1)),
        beta_anneal_epochs=int(config.get("selector_T", max(1, int(config.get("epochs", 200) * 3 / 4)))),
        tau_start=float(config.get("selector_tau_start", 3.0)),
        tau_end=float(config.get("selector_tau_end", 1.1)),
        tau_anneal_epochs=int(config.get("selector_tau_T", config.get("epochs", 200))),
    )
