"""Paper-guided residual-Gumbel components for an isolated baseline pilot.

This module is intentionally separate from the production selector paths.  It
implements the two-stage idea described for ResGS: residual training uses a
full-channel backbone, whereas the final model consumes only the frozen hard
selection from the original EEG signal.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F


class ResidualGumbelSelector(nn.Module):
    """Shared K-head Gumbel selector with a transparent C-channel projection."""

    def __init__(self, n_channels: int, k: int, residual_weight: float = 0.1):
        super().__init__()
        self.n_channels = int(n_channels)
        self.k = int(k)
        self.residual_weight = float(residual_weight)
        self.alpha = nn.Parameter(torch.empty(self.k, self.n_channels))
        nn.init.normal_(self.alpha, mean=0.0, std=0.01)

    def hard_indices(self) -> torch.Tensor:
        return self.alpha.detach().argmax(dim=1)

    def probabilities(self, temperature: float) -> torch.Tensor:
        return torch.softmax(self.alpha / max(float(temperature), 1e-6), dim=1)

    def _weights(self, temperature: float) -> torch.Tensor:
        if self.training:
            # ResGS uses the soft Gumbel output during residual training. At
            # high temperature this is intentionally close to a channel
            # average, which is why the residual path is needed at all.
            return F.gumbel_softmax(self.alpha, tau=max(float(temperature), 1e-6), hard=False, dim=1)
        hard = self.hard_indices()
        return F.one_hot(hard, num_classes=self.n_channels).to(dtype=self.alpha.dtype)

    def forward(self, x: torch.Tensor, temperature: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Return residual full-channel input and the K selected signal streams.

        ``x`` is ``[B, C, T]``. The K Gumbel outputs are zero-padded along
        their channel dimension to C, exactly matching the ``Padding(z)``
        operation in the ResGS residual-training equation. Their order is the
        selector-head order, not an electrode-location remapping.
        """
        weights = self._weights(temperature)
        selected = torch.einsum("kc,bct->bkt", weights, x)
        padded = F.pad(selected, (0, 0, 0, self.n_channels - self.k))
        mixed = (1.0 - self.residual_weight) * x + self.residual_weight * padded
        return mixed, selected


class PaperGuidedResidualGumbelModel(nn.Module):
    """A full-channel backbone preceded by the residual-Gumbel input path."""

    def __init__(self, backbone: nn.Module, n_channels: int, k: int, residual_weight: float = 0.1):
        super().__init__()
        self.selector = ResidualGumbelSelector(n_channels, k, residual_weight)
        self.backbone = backbone

    def forward(self, x: torch.Tensor, temperature: float) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4 or x.size(1) != 1:
            raise ValueError(f"Expected [B, 1, C, T] input, got {tuple(x.shape)}.")
        mixed, _ = self.selector(x[:, 0], temperature)
        logits, features = self.backbone(mixed.unsqueeze(1))
        return logits, features


def transfer_full_backbone_to_selected(
    full_backbone: nn.Module,
    selected_backbone: nn.Module,
    selected_channels: Sequence[int],
    group_dim: int = 4,
) -> None:
    """Warm-start the K-channel DDLM by slicing its channel-dependent layers.

    Non-channel-dependent tensors are copied verbatim. The depthwise temporal
    convolution and its batch normalization are sliced in selector-head order;
    the spatial low-rank projection is sliced along the corresponding physical
    electrode axis. This is a DDLM-specific adaptation: the ResGS paper's
    BASEN has a different encoder and does not specify this weight transfer.
    """
    indices = torch.as_tensor(selected_channels, dtype=torch.long, device=full_backbone.channel_tcb_block[1].weight.device)
    expanded = (indices[:, None] * group_dim + torch.arange(group_dim, device=indices.device)).reshape(-1)
    source = full_backbone.state_dict()
    target = selected_backbone.state_dict()

    for name, value in target.items():
        if name in source and source[name].shape == value.shape:
            target[name] = source[name].detach().clone()

    conv_name = "channel_tcb_block.1.weight"
    target[conv_name] = source[conv_name].detach().index_select(0, expanded).clone()

    bn_prefix = "channel_tcb_block.2."
    for suffix in ("weight", "bias", "running_mean", "running_var"):
        name = bn_prefix + suffix
        target[name] = source[name].detach().index_select(0, expanded).clone()
    target[bn_prefix + "num_batches_tracked"] = source[bn_prefix + "num_batches_tracked"].detach().clone()

    spatial_name = "spatiotemporal_fusion_block.1.proj.weight"
    target[spatial_name] = source[spatial_name].detach().index_select(2, indices).clone()
    selected_backbone.load_state_dict(target, strict=True)
