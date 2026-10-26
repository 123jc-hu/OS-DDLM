from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict
from models.eeg_models import model_dict
import math


# =========================
#    通道选择器（Concrete, Duplicate-penalty）
# =========================
class ConcreteMultiSelectorDup(nn.Module):
    """
    与你的 ConcreteMultiSelector 完全同形同口径：
    - 训练期：batch 共享一次 Gumbel（per_sample=True → alpha + gumbel；False → 只用 alpha）
    - 推理期：argmax → one-hot
    - 返回: z [B,1,K,T], W [K,C], P [K,C]
    不同点：orth_regularizer() 改为论文 Eq.(5) 的“重复通道惩罚”：
        L(P) = Σ_c ReLU( Σ_k P[k,c] - τ )
    并提供 τ 的指数退火（与 β 同步风格）。
    """
    def __init__(
        self,
        C: int,
        K: int,
        beta_start: float = 10.0,
        beta_end: float = 0.1,
        T_anneal: int = 150,
        per_sample: bool = True,
        train_soft_only: bool = False,
        # ↓ 新增但有默认值，不破坏外部接口
        tau_start: float = 3.0,
        tau_end: float = 1.1,
    ):
        super().__init__()
        self.C, self.K = int(C), int(K)
        self.per_sample = bool(per_sample)
        self.train_soft_only = bool(train_soft_only)

        # α logits
        self.alpha = nn.Parameter(torch.zeros(K, C))
        nn.init.uniform_(self.alpha, -1e-2, 1e-2)

        # 退火参数
        self.beta_start, self.beta_end = float(beta_start), float(beta_end)
        self.tau_start,  self.tau_end  = float(tau_start),  float(tau_end)
        self.T_anneal = max(1, int(T_anneal))
        self._t = 0

    # ---------- 调度 ----------
    def step_epoch(self, n: int = 1):
        self._t += int(n)

    def _beta(self) -> float:
        t = min(self._t / self.T_anneal, 1.0)
        val = self.beta_start * (self.beta_end / self.beta_start) ** t
        return float(max(val, 1e-6))

    def _tau(self) -> float:
        t = min(self._t / self.T_anneal, 1.0)
        # 与论文一致：τ 指数退火到 ~1
        return float(self.tau_start * (self.tau_end / self.tau_start) ** t)

    @torch.no_grad()
    def hard_indices(self):
        return torch.argmax(self.alpha, dim=1).tolist()

    # ---------- 工具 ----------
    @staticmethod
    def _gumbel_like(x: torch.Tensor) -> torch.Tensor:
        u = torch.rand_like(x).clamp_(1e-6, 1 - 1e-6)
        return -torch.log(-torch.log(u))

    @staticmethod
    def _row_softmax(logits: torch.Tensor, beta: float) -> torch.Tensor:
        return torch.softmax(logits / beta, dim=1)

    def current_p_soft(self) -> torch.Tensor:
        beta = self._beta()
        return self._row_softmax(self.alpha, beta)  # (K,C)

    # ---------- 前向 ----------
    def forward(self, x: torch.Tensor):
        """
        x: [B,1,C,T]
        """
        B, _, C, T = x.shape
        assert C == self.C, f"Selector expects C={self.C}, got C={C}"
        beta = self._beta()

        if self.training:
            if self.per_sample:
                logits = self.alpha + self._gumbel_like(self.alpha)   # batch 共享
            else:
                logits = self.alpha

            P_soft = self._row_softmax(logits, beta)  # (K,C)

            if self.train_soft_only:
                W = P_soft
            else:
                with torch.no_grad():
                    idx = torch.argmax(logits, dim=1)
                    W_hard = torch.zeros_like(P_soft)
                    W_hard[torch.arange(self.K), idx] = 1.0
                W = P_soft + (W_hard - P_soft).detach()  # ST

            z = torch.einsum("kc,bct->bkt", W, x.squeeze(1)).unsqueeze(1)
            return z, W, P_soft

        # eval：硬 one-hot
        idx = torch.argmax(self.alpha, dim=1)
        W_hard = torch.zeros(self.K, self.C, device=x.device, dtype=x.dtype)
        W_hard[torch.arange(self.K), idx] = 1.0
        z = x[:, :, idx, :]
        return z, W_hard, W_hard

    # ---------- 正则（论文式重复通道惩罚） ----------
    def orth_regularizer(self, P_soft: torch.Tensor) -> torch.Tensor:
        """
        用论文 Eq.(5) 替代“近似正交”：
        L(P) = Σ_c ReLU( Σ_k p_{k,c} - τ )
        其中 p_{k,c} 为第 k 个选择器在通道 c 的概率。
        实现：对列（通道维）求和，再做 ReLU-τ，再求和。
        """
        tau = self._tau()
        # 列和：每个通道被 K 个选择器“选择”的总概率
        col_sum = P_soft.sum(dim=0)            # (C,)
        return F.relu(col_sum - tau).sum()

    @staticmethod
    def normalized_entropy(P_soft: torch.Tensor):
        """
        与你原实现一致：对每一行（选择器）求归一化熵，再取均值。
        """
        eps = 1e-12
        pk = P_soft.clamp_min(eps)
        Hk = -(pk * pk.log()).sum(dim=1) / math.log(pk.size(1))
        return Hk, Hk.mean()


# =========================
#   包装器：选择器 + 主干（Dup 版）
# =========================
class ModelWithSelectorDup(nn.Module):
    """
    与 ModelWithSelector 完全一致，只是选择器换成 ConcreteMultiSelectorDup。
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
        per_sample: bool = True,
        tau_start: float = 3.0,
        tau_end: float = 1.1,
    ):
        super().__init__()
        assert backbone_config["n_channels"] == K, "backbone 的 n_channels 必须等于 K"

        self.selector = ConcreteMultiSelectorDup(
            C=C,
            K=K,
            beta_start=beta_start,
            beta_end=beta_end,
            T_anneal=T_anneal,
            per_sample=per_sample,
            train_soft_only=False,
            tau_start=tau_start,
            tau_end=tau_end,
        )
        self.backbone = backbone_cls(backbone_config)

        self.post_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor, current_epoch: int = None, **kwargs):
        z, W, P = self.selector(x)
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
        idx = torch.argmax(self.selector.alpha, dim=1)  # (K,)
        return full_attn[:, :, idx, :]


# =========================
#       模型工厂（Dup 版）
# =========================
def build_model_dup(config: Dict[str, Any]) -> nn.Module:
    """
    与原 build_model 相同，只是内部选择器换成“重复通道惩罚”版本。
    依旧读取相同的配置键；另外可选：
      - selector_tau_start (默认 3.0)
      - selector_tau_end   (默认 1.1)
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
    per_sample = bool(config.get("selector_per_sample", False))  # 仍为 batch 共享语义
    tau_start = float(config.get("selector_tau_start", 3.0))
    tau_end   = float(config.get("selector_tau_end",   1.1))

    backbone_cfg = dict(config)
    backbone_cfg["n_channels"] = K

    return ModelWithSelectorDup(
        backbone_cls=BaseCls,
        backbone_config=backbone_cfg,
        C=C,
        K=K,
        beta_start=beta_start,
        beta_end=beta_end,
        T_anneal=T_anneal,
        per_sample=per_sample,
        tau_start=tau_start,
        tau_end=tau_end,
    )
