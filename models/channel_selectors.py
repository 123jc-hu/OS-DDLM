import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import math


# =========================
#    Correlation-based channel selection 
#    from 《Correlation-based channel selection and regularized feature optimization for MI-based BCI》
# =========================
class _CS_ChannelSubsetDataset(Dataset):
    """惰性抽取通道的包装器，避免拷贝整数据到内存。"""
    def __init__(self, base_dataset: Dataset, channels_idx: np.ndarray, keepdim: bool = True):
        self.base = base_dataset
        # 关键修复：将 numpy 索引转为 torch.LongTensor，防止张量索引报错
        self.channels_np = np.asarray(channels_idx, dtype=int)
        self.channels = torch.as_tensor(self.channels_np, dtype=torch.long)
        self.keepdim = keepdim

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        out = self.base[idx]

        # 保证至少有 x, y
        if isinstance(out, (tuple, list)):
            x, y, *rest = out   # 拆出前两个，其余放到 rest
        else:
            raise ValueError(f"Dataset item must be tuple/list, got {type(out)}")

        # ---- 下面是通道筛选逻辑 ----
        if x.ndim == 3:  # (1, C, T) 或 (C, T)
            if x.shape[0] == 1:           # (1, C, T)
                x = x[:, self.channels, :]         # (1, K, T)
            else:                          # (C, T)
                x = x[self.channels, :]            # (K, T)
                if self.keepdim:
                    x = x.unsqueeze(0)             # (1, K, T)
        elif x.ndim == 2:  # (C, T)
            x = x[self.channels, :]
            if self.keepdim:
                x = x.unsqueeze(0)                 # (1, K, T)
        else:
            raise ValueError(f"Unsupported x shape in dataset: {tuple(x.shape)}")

        # ---- 返回时把 rest 拼回去，避免丢失信息 ----
        if rest:
            return (x, y, *rest)
        return (x, y)


class CorrelationBasedChannelSelector:
    """
    Correlation-Based Channel Selection (Cox & Savoy, 2003; Li et al., 2009)
    """
    def __init__(self, dataloader, K: int, device: str = "cpu"):
        self.dataloader = dataloader
        self.K = int(K)
        self.device = device
        self.selected_channels_ = None  # 拟合后填充

    @torch.no_grad()
    def _compute_channel_importance(self):
        channel_votes = []
        n_channels_ref = None

        for batch in self.dataloader:
            x = batch[0]
            # 统一到 CPU，便于 numpy 计算
            if x.device.type != "cpu":
                x = x.cpu()

            # 统一到 (B, C, T)
            if x.ndim == 4:         # (B, 1, C, T)
                x = x.squeeze(1)    # (B, C, T)
            elif x.ndim != 3:
                raise ValueError(f"Unsupported input shape: {tuple(x.shape)}")

            B, C, T = x.shape
            if n_channels_ref is None:
                n_channels_ref = C
            elif C != n_channels_ref:
                raise ValueError(f"Inconsistent channel count across batches: {C} vs {n_channels_ref}")

            if self.K > C:
                raise ValueError(f"K={self.K} exceeds number of channels C={C}")

            x_np = x.numpy()
            for i in range(B):
                trial = x_np[i]  # (C, T)
                R = np.corrcoef(trial)                      # (C, C)
                R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
                np.fill_diagonal(R, 0.0)
                mean_corr = R.mean(axis=1)
                top_idx = np.argsort(mean_corr)[::-1][:self.K]
                channel_votes.extend(top_idx.tolist())

        counts = np.bincount(channel_votes, minlength=n_channels_ref)
        top_channels = np.argsort(counts)[::-1][:self.K]
        return top_channels

    def fit(self):
        self.selected_channels_ = self._compute_channel_importance()
        return self

    def transform(self, dataloader, batch_size=None, shuffle: bool = True):
        if self.selected_channels_ is None:
            raise RuntimeError("You must call fit() before transform().")

        wrapped_ds = _CS_ChannelSubsetDataset(
            base_dataset=dataloader.dataset,
            channels_idx=self.selected_channels_,
            keepdim=True
        )

        # 关键改进：透传性能相关参数
        new_loader = DataLoader(
            wrapped_ds,
            batch_size=batch_size or dataloader.batch_size,
            shuffle=shuffle,
            drop_last=False,
            num_workers=getattr(dataloader, "num_workers", 0),
            pin_memory=getattr(dataloader, "pin_memory", False),
            persistent_workers=getattr(dataloader, "persistent_workers", False),
        )
        return new_loader

    def get_new_dataloader(self, batch_size=None, shuffle: bool = True):
        if self.selected_channels_ is None:
            self.fit()
        return self.transform(self.dataloader, batch_size=batch_size, shuffle=shuffle)
    
"""
使用方法：
# 1) 仅用训练集计算所选通道
selector = CorrelationBasedChannelSelector(dataloader=train_loader, K=16, device="cuda").fit()
print("Selected channels:", selector.selected_channels_)

# 2) 分别生成新的 train/val/test dataloader
train_new = selector.transform(train_loader, batch_size=train_loader.batch_size, shuffle=True)
val_new   = selector.transform(val_loader,   batch_size=val_loader.batch_size,   shuffle=False)
test_new  = selector.transform(test_loader,  batch_size=test_loader.batch_size,  shuffle=False)
"""

# =========================
#    SparseEA channel selection 
#    from 《Multi-objective optimization approach for channel selection and cross-subject generalization in RSVP-based BCIs》
# =========================
# -*- coding: utf-8 -*-

# ---------- 小工具 ----------

def _to_BCT(x: torch.Tensor) -> torch.Tensor:
    """兼容 (B,1,C,T) 或 (B,C,T) -> 统一为 (B,C,T)"""
    if x.ndim == 4:
        return x.squeeze(1)
    elif x.ndim == 3:
        return x
    else:
        raise ValueError(f"Unsupported x shape: {tuple(x.shape)}")

def _windows(T: int, fs: int, win_ms: int = 100):
    """按100ms划窗，返回[(s,e), ...]，e为开区间索引"""
    win_len = max(1, int(round(fs * win_ms / 1000.0)))
    K = T // win_len
    idx = [(i * win_len, (i + 1) * win_len) for i in range(K)]
    if K < 1:
        raise ValueError(f"T={T} 太短，无法以 {win_ms}ms 划窗（fs={fs}, win_len={win_len}）")
    return idx, K, win_len

def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    p = max(1, np.sum(y_true == 1))
    n = max(1, np.sum(y_true == 0))
    tpr = tp / p
    tnr = tn / n
    return 0.5 * (tpr + tnr)

def _lda_fit(X: np.ndarray, y: np.ndarray, reg: float = 1e-3):
    """
    Fisher LDA（两类）：返回 w, thresh
    X: (N, d), y in {0,1}
    thresh = 0.5 * w^T (mu1 + mu0)  （等先验）
    """
    y = y.astype(int)
    X0 = X[y == 0]
    X1 = X[y == 1]
    mu0 = X0.mean(axis=0)
    mu1 = X1.mean(axis=0)
    # pooled covariance
    def _cov(M):
        M = M - M.mean(axis=0, keepdims=True)
        return (M.T @ M) / max(1, (M.shape[0] - 1))
    Sw = _cov(X0) + _cov(X1) + reg * np.eye(X.shape[1], dtype=X.dtype)
    # solve Sw w = (mu1 - mu0)
    w = np.linalg.solve(Sw, (mu1 - mu0))
    thresh = 0.5 * float(w @ (mu1 + mu0))
    return w, thresh

def _lda_decision(X: np.ndarray, w: np.ndarray, thresh: float):
    z = X @ w
    yhat = (z >= thresh).astype(int)
    return z, yhat

def _stack_loader_to_numpy(loader: DataLoader):
    """
    将 DataLoader 全量堆叠到 CPU/NumPy：
      - 兼容 item 形如 (x, y), (x, y, *rest) 或 dict({'x':..., 'y':...})
      - x 支持 (B, 1, C, T) 或 (B, C, T)；输出统一为 X:(N, C, T)
      - y 统一为一维整型数组 (N,)
    """
    xs, ys = [], []

    for item in loader:
        # 解包：支持 tuple/list 或 dict
        if isinstance(item, (tuple, list)):
            if len(item) < 2:
                raise ValueError(f"Dataset item needs at least (x, y), got length {len(item)}")
            x, y = item[0], item[1]
        elif isinstance(item, dict):
            # 常见键名兜底
            x = item.get("x", item.get("inputs", None))
            y = item.get("y", item.get("labels", None))
            if x is None or y is None:
                raise ValueError(f"Dict item must contain 'x' and 'y' (or 'inputs'/'labels'), keys={list(item.keys())}")
        else:
            raise ValueError(f"Unsupported dataset item type: {type(item)}")

        # 搬到 CPU
        if isinstance(x, torch.Tensor) and x.device.type != "cpu":
            x = x.cpu()
        if isinstance(y, torch.Tensor) and y.device.type != "cpu":
            y = y.cpu()

        # 统一 x 形状到 (B, C, T)
        if isinstance(x, torch.Tensor):
            if x.ndim == 4:  # (B,1,C,T)
                x = x.squeeze(1)
            elif x.ndim != 3:
                raise ValueError(f"Unsupported x shape: {tuple(x.shape)} (expect (B,1,C,T) or (B,C,T))")
            xs.append(x)
        else:
            # 若上游返回 numpy，可转成 tensor 再处理
            x = torch.as_tensor(x)
            if x.ndim == 4:
                x = x.squeeze(1)
            elif x.ndim != 3:
                raise ValueError(f"Unsupported x numpy shape: {tuple(x.shape)}")
            xs.append(x)

        # 处理 y -> (B,)
        if not isinstance(y, torch.Tensor):
            y = torch.as_tensor(y)
        if y.ndim > 1:
            # 兼容 one-hot 或额外维度的标签
            y = y.view(y.shape[0], -1)
            if y.shape[1] > 1:  # one-hot -> argmax
                y = y.argmax(dim=1)
            else:
                y = y.squeeze(1)
        ys.append(y)

    # 堆叠并转 numpy
    X = torch.cat(xs, dim=0).numpy()                 # (N, C, T)
    y = torch.cat(ys, dim=0).to(torch.int64).numpy() # (N,)
    return X, y


def _apply_channels_numpy(X: np.ndarray, channels: np.ndarray):
    return X[:, channels, :]

# ---------- HDCA 分类器（Algorithm 1） ----------

class HDCAClassifier:
    """
    HDCA (Hierarchical Discriminant Component Analysis) for RSVP/P300
    - 每个100ms窗内做 FLD 得到空间权重 u_k，得到标量 y_k = u_k^T x_k
    - 将 y = [y_1,...,y_K] 用 LDA 得到时间维权重 v 作线性判别

    参数
    ----
    fs : 采样率 (Hz)
    win_ms : 窗宽毫秒（默认100）
    reg_spatial : 每窗FLD的协方差岭化
    reg_temporal: y特征上的LDA岭化
    """
    def __init__(self, fs: int, win_ms: int = 100, reg_spatial: float = 1e-3, reg_temporal: float = 1e-3):
        self.fs = int(fs)
        self.win_ms = int(win_ms)
        self.reg_spatial = float(reg_spatial)
        self.reg_temporal = float(reg_temporal)
        self.us_ = None       # list of u_k, 每个 shape (C,)
        self.v_ = None        # shape (K,)
        self.th_temporal_ = None
        self.win_idx_ = None  # 列表[(s,e),...]
        self.C_ = None
        self.K_ = None

    def _window_mean(self, X: np.ndarray):
        """X:(N,C,T) -> Xm:(N,C,K)  取各窗的平均"""
        N, C, T = X.shape
        idx, K, _ = _windows(T, self.fs, self.win_ms)
        Xm = np.empty((N, C, K), dtype=X.dtype)
        for k, (s, e) in enumerate(idx):
            Xm[:, :, k] = X[:, :, s:e].mean(axis=2)
        return Xm, idx, K

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        X:(N,C,T), y:(N,)
        """
        X = np.asarray(X)
        y = np.asarray(y).astype(int)
        N, C, T = X.shape
        Xm, idx, K = self._window_mean(X)  # (N,C,K)
        self.us_ = []
        # 窗内FLD：每窗特征 (N,C) -> u_k
        for k in range(K):
            Xk = Xm[:, :, k]  # (N,C)
            u_k, _ = _lda_fit(Xk, y, reg=self.reg_spatial)
            self.us_.append(u_k)
        self.us_ = np.stack(self.us_, axis=0)  # (K,C)

        # 生成 y 特征 (N,K)
        Y = np.empty((N, K), dtype=X.dtype)
        for k in range(K):
            Y[:, k] = Xm[:, :, k] @ self.us_[k]  # y_k

        # 时间维 LDA
        v, th = _lda_fit(Y, y, reg=self.reg_temporal)
        self.v_ = v
        self.th_temporal_ = th
        self.win_idx_ = idx
        self.C_ = C
        self.K_ = K
        return self

    def predict(self, X: np.ndarray):
        """返回 logits(=z), y_pred；X:(N,C,T)"""
        assert self.us_ is not None, "Call fit() first"
        Xm, _, K = self._window_mean(X)  # (N,C,K)
        Y = np.empty((Xm.shape[0], K), dtype=X.dtype)
        for k in range(K):
            Y[:, k] = Xm[:, :, k] @ self.us_[k]
        z, yhat = _lda_decision(Y, self.v_, self.th_temporal_)
        return z, yhat

    def score_balanced_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        z, yhat = self.predict(X)
        return _balanced_accuracy(y, yhat)

# ---------- SparseEA 通道选择（Algorithm 2 框架的工程化实现） ----------

class SparseEAChannelSelector:
    """
    多目标 SparseEA 通道选择
    目标: minimize f1 = 1-BA, minimize f2 = |S|（通道数）
    评价器: HDCAClassifier（可替换）
    """
    def __init__(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        fs: int,
        target_K: int | None = None,
        population_size: int = 60,
        generations: int = 40,
        win_ms: int = 100,
        reg_spatial: float = 1e-3,
        reg_temporal: float = 1e-3,
        seed: int = 42,
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.fs = int(fs)
        self.win_ms = int(win_ms)
        self.reg_spatial = float(reg_spatial)
        self.reg_temporal = float(reg_temporal)
        self.pop_size = int(population_size)
        self.generations = int(generations)
        self.rng = np.random.default_rng(seed)
        self.target_K = int(target_K) if target_K is not None else None

        # 缓存数据（CPU, numpy）
        self.Xtr_, self.ytr_ = _stack_loader_to_numpy(train_loader)  # (N,C,T)
        self.Xva_, self.yva_ = _stack_loader_to_numpy(val_loader)
        self.C_ = self.Xtr_.shape[1]

        self.population_ = None      # list of masks (C,)
        self.objs_ = None            # array (P,2): [1-BA, |S|]
        self.selected_channels_ = None

        # 单通道精度作为引导（论文初始化思想）
        self.channel_scores_ = self._single_channel_scores()

    def _single_channel_scores(self):
        scores = np.zeros(self.C_)
        for c in range(self.C_):
            hdca = HDCAClassifier(fs=self.fs, win_ms=self.win_ms,
                                  reg_spatial=self.reg_spatial, reg_temporal=self.reg_temporal)
            hdca.fit(self.Xtr_[:, [c], :], self.ytr_)
            scores[c] = hdca.score_balanced_accuracy(self.Xva_[:, [c], :], self.yva_)
        return scores  # BA per channel

    def _evaluate_mask(self, mask: np.ndarray) -> tuple[float, float]:
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            # f1=1-BA -> 1.0(最差)，f2=与K的偏差（若无K则子集大小）
            if self.target_K is None:
                return 1.0, 0.0
            else:
                return 1.0, float(self.target_K)
        hdca = HDCAClassifier(fs=self.fs, win_ms=self.win_ms,
                            reg_spatial=self.reg_spatial, reg_temporal=self.reg_temporal)
        hdca.fit(_apply_channels_numpy(self.Xtr_, idx), self.ytr_)
        ba = hdca.score_balanced_accuracy(_apply_channels_numpy(self.Xva_, idx), self.yva_)
        f1 = 1.0 - float(ba)
        k  = float(len(idx))
        if self.target_K is None:
            f2 = k                      # 原版：越少越好
        else:
            f2 = abs(k - self.target_K) # 贴近K越好
        return f1, f2

    # --- NSGA-II 相关 ---

    def _init_population(self):
        pop = []
        C = self.C_
        order = np.argsort(self.channel_scores_)[::-1]

        seeds = [1, 2, 4, 8, max(1, C // 4), max(1, C // 2)]
        if self.target_K is not None:
            seeds.extend([
                max(1, self.target_K - 4),
                self.target_K,
                min(C, self.target_K + 4),
            ])
        seeds = [min(C, s) for s in seeds]

        added = set()
        for s in seeds:
            m = np.zeros(C, dtype=np.uint8)
            m[order[:s]] = 1
            t = tuple(m.tolist())
            if t not in added:
                pop.append(m); added.add(t)

        rng = self.rng
        while len(pop) < self.pop_size:
            # 若有 target_K，随机 k 在 [max(1,K-8), min(C,K+8)] 内均匀采样，有助贴近K
            if self.target_K is not None:
                lo = max(1, self.target_K - 8)
                hi = min(C, self.target_K + 8)
                k = int(rng.integers(low=lo, high=hi+1))
            else:
                k = int(rng.integers(low=1, high=C+1))
            m = np.zeros(C, dtype=np.uint8)
            idx = rng.choice(C, size=k, replace=False)
            m[idx] = 1
            t = tuple(m.tolist())
            if t not in added:
                pop.append(m); added.add(t)
        return pop

    @staticmethod
    def _nondominated_sort(F):
        """F:(P,2) -> ranks(list of fronts as lists of indices)"""
        P = F.shape[0]
        dominates = [set() for _ in range(P)]
        dominated_count = np.zeros(P, dtype=int)
        fronts = [[]]
        for p in range(P):
            for q in range(P):
                if p == q: continue
                if (F[p][0] <= F[q][0] and F[p][1] <= F[q][1]) and (F[p][0] < F[q][0] or F[p][1] < F[q][1]):
                    dominates[p].add(q)
                elif (F[q][0] <= F[p][0] and F[q][1] <= F[p][1]) and (F[q][0] < F[p][0] or F[q][1] < F[p][1]):
                    dominated_count[p] += 1
            if dominated_count[p] == 0:
                fronts[0].append(p)
        i = 0
        while fronts[i]:
            Q = []
            for p in fronts[i]:
                for q in dominates[p]:
                    dominated_count[q] -= 1
                    if dominated_count[q] == 0:
                        Q.append(q)
            i += 1
            fronts.append(Q)
        if not fronts[-1]:
            fronts.pop()
        return fronts

    @staticmethod
    def _crowding_distance(F, idxs):
        """F:(P,2), idxs:list -> distances np.array"""
        if len(idxs) == 0:
            return np.array([])
        if len(idxs) == 1:
            return np.array([np.inf])
        subF = F[idxs]
        d = np.zeros(len(idxs))
        for m in range(2):
            order = np.argsort(subF[:, m])
            d[order[0]] = d[order[-1]] = np.inf
            minv, maxv = subF[order[0], m], subF[order[-1], m]
            rng = max(1e-12, maxv - minv)
            for j in range(1, len(idxs) - 1):
                d[order[j]] += (subF[order[j+1], m] - subF[order[j-1], m]) / rng
        return d

    def _tournament(self, pop, F, fronts):
        """二元锦标赛（先比较前沿等级，再比较拥挤距离）"""
        i, j = self.rng.integers(0, len(pop), size=2)
        # 找到 i,j 分别所在的前沿和在该前沿的拥挤距离
        def key(a):
            # 返回 (front_rank, -crowd) 便于最小化
            for r, fr in enumerate(fronts):
                if a in fr:
                    # crowding distance 需要在该前沿内计算
                    cd = self._crowding_distance(F, fr)
                    crowd = cd[fr.index(a)]
                    return (r, -crowd)
            return (np.inf, -np.inf)
        return i if key(i) < key(j) else j

    def _crossover_mutation(self, p1: np.ndarray, p2: np.ndarray, mut_prob: float = 0.2):
        """均匀交叉 + 稀疏变异（以相同概率翻转一个0或1）"""
        C = len(p1)
        mask = self.rng.integers(0, 2, size=C, dtype=np.uint8)
        child = (p1 & mask) | (p2 & (1 - mask))
        if self.rng.random() < mut_prob:
            ones = np.flatnonzero(child == 1)
            zeros = np.flatnonzero(child == 0)
            if len(ones) == 0 and len(zeros) == 0:
                pass
            elif len(ones) == 0:
                # 只能从0->1
                j = self.rng.choice(zeros)
                child[j] = 1
            elif len(zeros) == 0:
                # 只能从1->0
                j = self.rng.choice(ones)
                child[j] = 0
            else:
                if self.rng.random() < 0.5:
                    j = self.rng.choice(zeros)
                    child[j] = 1
                else:
                    j = self.rng.choice(ones)
                    child[j] = 0
        return child

    def fit(self):
        # 初始化种群
        pop = self._init_population()
        F = np.array([self._evaluate_mask(m)[0:2] for m in pop], dtype=float)  # (P,2)

        for gen in range(self.generations):
            fronts = self._nondominated_sort(F)
            # 产生子代
            children = []
            while len(children) < self.pop_size:
                # 选父母（按前沿+拥挤距离）
                fronts = self._nondominated_sort(F)
                p_idx = self._tournament(pop, F, fronts)
                q_idx = self._tournament(pop, F, fronts)
                child = self._crossover_mutation(pop[p_idx], pop[q_idx], mut_prob=0.3)
                children.append(child)

            # 合并 & 选择下一代
            union = pop + children
            F_union = np.array([self._evaluate_mask(m)[0:2] for m in union], dtype=float)
            fronts = self._nondominated_sort(F_union)

            new_pop = []
            new_F = []
            for fr in fronts:
                if len(new_pop) + len(fr) <= self.pop_size:
                    new_pop.extend([union[i] for i in fr])
                    new_F.extend([F_union[i] for i in fr])
                else:
                    # 需要在该前沿里按拥挤距离截断
                    cd = self._crowding_distance(F_union, fr)
                    order = np.argsort(-cd)  # 先选拥挤距离大的
                    for j in order:
                        if len(new_pop) < self.pop_size:
                            new_pop.append(union[fr[j]])
                            new_F.append(F_union[fr[j]])
                        else:
                            break
                    break
            pop, F = new_pop, np.array(new_F)

        # --- 最终选择解（强制贴近 target_K，并兜底保证“恰好K个通道”） ---
        self.population_ = pop
        self.objs_ = F  # 每行: [f1=1-BA, f2(贴近K或子集大小)]

        ks  = np.array([int(m.sum()) for m in pop])   # 每个体的通道数
        bas = 1.0 - F[:, 0]                           # BA = 1 - f1

        if self.target_K is not None:
            # 先找恰好等于 K 的
            exact = np.where(ks == self.target_K)[0]
            if len(exact) > 0:
                best_i = exact[np.argmax(bas[exact])]
            else:
                # 找 |k-K| 最小的一批，再选 BA 最高
                gap = np.abs(ks - self.target_K)
                near = np.where(gap == gap.min())[0]
                best_i = near[np.argmax(bas[near])]
        else:
            # 未指定K：选 BA 最大，若并列通道更少者优先
            best_i = np.lexsort((ks, -bas))[0]

        best_mask = pop[best_i].astype(bool)
        selected = np.flatnonzero(best_mask)

        # ---- 兜底：确保“恰好 K 个” ----
        if self.target_K is not None:
            K = self.target_K
            if selected.size > K:
                # 过多 -> 在已选中里按单通道评分截断到前K
                order_in = np.argsort(self.channel_scores_[selected])[::-1]
                selected = selected[order_in[:K]]
            elif selected.size < K:
                # 过少 -> 从未选中的里按单通道评分补齐
                remaining = np.setdiff1d(np.arange(self.C_), selected, assume_unique=False)
                add_order = np.argsort(self.channel_scores_[remaining])[::-1]
                to_add = remaining[add_order[:K - selected.size]]
                selected = np.concatenate([selected, to_add])
            # 为了整洁，升序排序
            selected = np.unique(selected)
            if selected.size != K:
                # 再次强制截断到K（极端情况下）
                selected = selected[:K]

        self.selected_channels_ = selected
        return self

    def transform(self, loader: DataLoader, batch_size: int | None = None, shuffle: bool = False) -> DataLoader:
        if self.selected_channels_ is None:
            raise RuntimeError("Call fit() first.")
        return _ChannelSubsetDataLoader.from_base_loader(loader, self.selected_channels_, batch_size, shuffle)

class _ChannelSubsetDataLoader:
    @staticmethod
    def from_base_loader(base_loader: DataLoader, channels_idx: np.ndarray, batch_size=None, shuffle=False) -> DataLoader:
        ds = _CS_ChannelSubsetDataset(base_loader.dataset, channels_idx, keepdim=True)
        return DataLoader(
            ds,
            batch_size=batch_size or base_loader.batch_size,
            shuffle=shuffle,
            drop_last=False,
            num_workers=getattr(base_loader, "num_workers", 0),
            pin_memory=getattr(base_loader, "pin_memory", False),
            persistent_workers=getattr(base_loader, "persistent_workers", False),
        )

"""
使用方法：
# 1) 用 train/val 来进化选择通道（target_K 可设为你想要的 K）
selector = SparseEAChannelSelector(
    train_loader=train_dataloader,
    val_loader=val_dataloader,
    fs=config["fs"],          # 必填
    target_K=16,              # 你的目标通道数；也可设为 None 得到Pareto解里BA最高者
    population_size=60,
    generations=40,
    win_ms=100,
).fit()

print("Selected channels:", selector.selected_channels_)  # numpy array

# 2) 将相同通道集应用到 train/val/test
train_new = selector.transform(train_dataloader, batch_size=train_dataloader.batch_size, shuffle=True)
val_new   = selector.transform(val_dataloader,   batch_size=val_dataloader.batch_size,   shuffle=False)
test_new  = selector.transform(test_dataloader,  batch_size=test_dataloader.batch_size,  shuffle=False)
"""

# =========================
#    EEG channel selection based on frequency domain weighting 
#    from 《Depression detection based on the temporal-spatial-frequency feature fusion of EEG》
# =========================
# -*- coding: utf-8 -*-

# ===================== 基础工具 =====================

def _normalize_cov(trial: np.ndarray) -> np.ndarray:
    """
    trial: (C,T), 返回 归一化协方差 Cx / trace(Cx)
    """
    Cx = trial @ trial.T
    tr = np.trace(Cx)
    if tr <= 1e-12:
        return np.eye(Cx.shape[0], dtype=trial.dtype)
    return Cx / tr

def _mean_cov(X: np.ndarray) -> np.ndarray:
    """
    X: (N,C,T) -> 归一化协方差的均值 (C,C)
    """
    N, C, T = X.shape
    acc = np.zeros((C, C), dtype=X.dtype)
    for i in range(N):
        acc += _normalize_cov(X[i])
    acc /= max(1, N)
    # 对称化
    return 0.5 * (acc + acc.T)

# ===================== 纯 PyTorch FIR 带通（可选） =====================

def _design_fir_bandpass(fs: int, f_lo: float, f_hi: float, numtaps: int = 129) -> torch.Tensor:
    """
    窗函数法设计对称FIR (Hamming)。返回 1D kernel (numtaps,)
    """
    assert 0 < f_lo < f_hi < fs / 2.0, "频带必须在(0, fs/2)内"
    n = torch.arange(numtaps, dtype=torch.float32)
    m = (numtaps - 1) / 2.0
    # 避免除零：sinc(x) = sin(pi x)/(pi x)
    def sinc(x): 
        y = torch.where(x == 0, torch.ones_like(x), torch.sin(np.pi * x) / (np.pi * x))
        return y
    # 理想带通 = 高通(f_lo) + 低通(f_hi) - 全通
    h_lp = 2 * (f_hi / fs) * sinc(2 * (f_hi / fs) * (n - m))
    h_hp = -2 * (f_lo / fs) * sinc(2 * (f_lo / fs) * (n - m))
    h_id = h_lp + h_hp
    # Hamming 窗
    w = 0.54 - 0.46 * torch.cos(2 * np.pi * n / (numtaps - 1))
    h = h_id * w
    # 归一化 DC/能量
    h = h / torch.sum(h)
    return h.float()

def _apply_bandpass_torch(X: np.ndarray, fs: int, band: tuple[float, float], numtaps: int = 129) -> np.ndarray:
    """
    X: (N,C,T) numpy -> 经过带通滤波后的 numpy (N,C,T)
    逐通道逐 trial 用 Conv1d 实现，padding='same'。
    """
    N, C, T = X.shape
    kernel = _design_fir_bandpass(fs, band[0], band[1], numtaps=numtaps)  # (K,)
    k = kernel.view(1, 1, -1)  # (out_ch=1, in_ch=1, K)
    pad = (k.shape[-1] - 1) // 2

    Xt = torch.from_numpy(X).float()  # (N,C,T)
    Xt = Xt.reshape(N * C, 1, T)      # 合并 batch 和通道
    k = k.to(Xt.device)

    Y = torch.nn.functional.conv1d(torch.nn.functional.pad(Xt, (pad, pad), mode="reflect"), k, groups=1)
    Y = Y.reshape(N, C, T - 0)  # 长度保持一致（reflect+same设计）
    return Y.numpy()

# ===================== CSP（白化解法，无需 SciPy） =====================

def _csp_filters(C1: np.ndarray, C2: np.ndarray, n_components: int | None = None):
    """
    输入两类的平均协方差矩阵 C1, C2 (C,C)，返回 CSP 投影矩阵 W (C,C) 及特征值向量 lambda
    白化法：C=C1+C2 -> EΛE^T -> P=Λ^{-1/2}E^T -> S1=P C1 P^T -> S1=UΣU^T
    W = P U (symmetric whitening backprojection)
    """
    C = C1 + C2
    # 对称本征分解
    evals, E = np.linalg.eigh(C)
    # 数值稳定
    evals = np.clip(evals, 1e-12, None)
    P = (E @ np.diag(evals ** -0.5) @ E.T)  # 白化
    S1 = P @ C1 @ P.T
    s, U = np.linalg.eigh(S1)
    # s 从小到大，CSP 常取两端分量
    W = P @ U
    # 返回所有投影分量（列为滤波器），与 s 对应
    order = np.argsort(s)[::-1]  # 也可以保留未翻转，下面会用到两端
    W = W[:, order]
    s = s[order]
    if n_components is not None:
        W = W[:, :n_components]
        s = s[:n_components]
    return W, s

def _csp_channel_weights_w1w2(W_full: np.ndarray) -> np.ndarray:
    """
    取“最显著”的两端空间滤波器：第一列(最大特征值)作为 w1，最后一列(最小特征值)作为 w2，
    然后按 a_i = sqrt(w1_i^2 + w2_i^2) 计算通道权重（式(3)）。
    """
    w1 = W_full[:, 0]           # 对应一类最有判别力
    w2 = W_full[:, -1]          # 另一端
    a = np.sqrt(w1 ** 2 + w2 ** 2)  # (C,)
    # 归一化为 b_i（式(4)）
    s = np.sum(a)
    if s <= 1e-12:
        b = np.ones_like(a) / len(a)
    else:
        b = a / s
    return b  # (C,)


def _wrap_loader(base_loader: DataLoader, channels_idx: np.ndarray, batch_size=None, shuffle=False) -> DataLoader:
    ds = _CS_ChannelSubsetDataset(base_loader.dataset, channels_idx, keepdim=True)
    return DataLoader(
        ds,
        batch_size=batch_size or base_loader.batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=getattr(base_loader, "num_workers", 0),
        pin_memory=getattr(base_loader, "pin_memory", False),
        persistent_workers=getattr(base_loader, "persistent_workers", False),
    )

# ===================== 频域加权 CSP 通道选择 =====================

class FrequencyWeightedCSPChannelSelector:
    """
    EEG 通道选择（基于频域加权的 CSP）
    - 对每个频带：用 CSP 求两端空间滤波器 w1, w2
    - 按 a_i = sqrt(w1_i^2 + w2_i^2)，b_i = a_i / sum(a_i) 得到通道权重
    - 多频带时对 b_i 做聚合（mean/max/median），选出 Top-K

    参数
    ----
    train_loader : 仅用于估计通道权重（建议 drop_last=True）
    fs           : 采样率 (Hz)
    bands        : List[Tuple[float, float]]，例如 [(1,4),(4,8),(8,13),(13,30)]；None=不分频带
    K            : 选择通道数
    aggregate    : 多频带聚合策略: "mean" | "max" | "median"
    fir_taps     : FIR 滤波器 taps 数（奇数，默认129）
    use_filter   : 是否开启带通滤波（True=对每个频带做 FIR；False=直接在原始宽带做 CSP；若 bands 多个但 use_filter=False，则仅按宽带算一次）
    """

    def __init__(
        self,
        train_loader: DataLoader,
        fs: int,
        K: int,
        bands: list[tuple[float, float]] | None = None,
        aggregate: str = "mean",
        fir_taps: int = 129,
        use_filter: bool = True,
    ):
        self.train_loader = train_loader
        self.fs = int(fs)
        self.K = int(K)
        self.bands = bands if bands is not None and len(bands) > 0 else None
        self.aggregate = aggregate
        self.fir_taps = int(fir_taps)
        self.use_filter = bool(use_filter)

        self.selected_channels_ = None         # np.ndarray (K,)
        self.band_weights_ = {}                # dict: band-> b_i (C,)
        self.global_weights_ = None            # 聚合后的 b_i (C,)

        # 缓存训练数据（numpy）
        self.Xtr_, self.ytr_ = _stack_loader_to_numpy(train_loader)
        self.C_ = self.Xtr_.shape[1]

    def _compute_band_weights(self, X: np.ndarray, y: np.ndarray, band: tuple[float, float] | None):
        """
        计算一个频带的通道权重 b_i；若 band=None，则直接宽带 CSP。
        """
        if band is not None and self.use_filter:
            Xb = _apply_bandpass_torch(X, self.fs, band, numtaps=self.fir_taps)  # (N,C,T)
        else:
            Xb = X

        # 计算两类平均协方差
        cls0 = Xb[y == 0]
        cls1 = Xb[y == 1]
        if len(cls0) == 0 or len(cls1) == 0:
            raise ValueError("需要两类样本以计算 CSP（y=0 和 y=1 均需存在）。")
        C0 = _mean_cov(cls0)  # (C,C)
        C1 = _mean_cov(cls1)  # (C,C)

        # CSP 滤波器
        W, s = _csp_filters(C1, C0)  # 注意：与文中 C2^{-1}C1 一致（这里把C1视为“第一类”）
        # 取两端滤波器，按式(3)(4)计算通道权重
        b = _csp_channel_weights_w1w2(W)  # (C,)
        return b

    def fit(self):
        # 单频带 / 多频带
        if self.bands is None or not self.use_filter:
            # 宽带一次
            b = self._compute_band_weights(self.Xtr_, self.ytr_, band=None)
            self.band_weights_ = {("broadband",): b}
            b_global = b
        else:
            # 对每个频带求 b_i
            per_band = []
            for band in self.bands:
                b = self._compute_band_weights(self.Xtr_, self.ytr_, band=band)
                self.band_weights_[tuple(band)] = b
                per_band.append(b[None, :])  # (1,C)
            B = np.concatenate(per_band, axis=0)  # (nbands, C)

            if self.aggregate.lower() == "mean":
                b_global = B.mean(axis=0)
            elif self.aggregate.lower() == "max":
                b_global = B.max(axis=0)
            elif self.aggregate.lower() == "median":
                b_global = np.median(B, axis=0)
            else:
                raise ValueError(f"Unknown aggregate: {self.aggregate}")

        # 最终归一化 + 选 Top-K
        s = np.sum(b_global)
        if s > 1e-12:
            b_global = b_global / s
        self.global_weights_ = b_global  # (C,)

        order = np.argsort(self.global_weights_)[::-1]
        if self.K > len(order):
            raise ValueError(f"K={self.K} exceeds number of channels C={len(order)}")
        self.selected_channels_ = order[:self.K]
        return self

    def transform(self, loader: DataLoader, batch_size: int | None = None, shuffle: bool = False) -> DataLoader:
        if self.selected_channels_ is None:
            raise RuntimeError("Call fit() first.")
        return _wrap_loader(loader, self.selected_channels_, batch_size=batch_size, shuffle=shuffle)

"""
使用方法：
# 例：按 (theta, alpha, beta) 三个频带求权重并取 Top-K=16
bands = [(4,8), (8,13), (13,30)]  # 自行按数据频谱定义
selector = FrequencyWeightedCSPChannelSelector(
    train_loader=train_dataloader,
    fs=config["fs"],
    K=16,
    bands=bands,
    aggregate="mean",   # "mean"/"max"/"median"
    fir_taps=129,
    use_filter=True,    # 若你预先已分频，可设 False 直接宽带
).fit()

print("Selected channels:", selector.selected_channels_)      # numpy array (K,)
print("Global weights shape:", selector.global_weights_.shape)  # (C,)

# 将相同通道集应用到 train/val/test
train_new = selector.transform(train_dataloader, batch_size=train_dataloader.batch_size, shuffle=True)
val_new   = selector.transform(val_dataloader,   batch_size=val_dataloader.batch_size,   shuffle=False)
test_new  = selector.transform(test_dataloader,  batch_size=test_dataloader.batch_size,  shuffle=False)
"""

