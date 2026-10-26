import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from scipy.signal import cheby2, sosfiltfilt
from joblib import Parallel, delayed
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import hilbert


def load_from_checkpoint(checkpoint_path: Path):
    """从检查点加载模型参数"""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return checkpoint


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

    def reset(self):
        """Reset the early stopping counter"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf


class SaveBestValBA:
    """Saves the model when validation accuracy (val_ba) is improved."""
    def __init__(self, verbose=False, delta=0, path='best_model.pt', trace_func=print):
        """
        Args:
            verbose (bool): If True, prints a message for each validation accuracy improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'best_model.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.verbose = verbose
        self.best_score = None
        self.best_val_ba = 0
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_ba, model):
        """Call this method every epoch to check if val_ba has improved."""
        if self.best_score is None:
            self.best_score = val_ba
            self.save_checkpoint(val_ba, model)
        elif val_ba > self.best_score + self.delta:
            self.best_score = val_ba
            self.save_checkpoint(val_ba, model)

    def save_checkpoint(self, val_ba, model):
        """Saves model when validation accuracy (val_ba) improves."""
        if self.verbose:
            self.trace_func(f'Validation accuracy (val_ba) improved ({self.best_val_ba:.6f} --> {val_ba:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.best_val_ba = val_ba

def class_aware_mixup_data(x, y, alpha: float = 1.0, use_cuda: bool = True):
    """
    Class‑aware Mixup: 生成 (mixed_x, y_a, y_b, lam) ，使得每个 (y_a, y_b) 标签必然不同。
    x : Tensor => (B, 1, C, T)
    y : LongTensor => (B,)
    """
    if alpha > 0.0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    B = x.size(0)
    device = x.device

    # 找到各类别索引
    idx_cls0 = torch.nonzero(y == 0, as_tuple=False).squeeze(1)
    idx_cls1 = torch.nonzero(y == 1, as_tuple=False).squeeze(1)

    # 如果有类别缺失，直接退化为普通 mixup
    if len(idx_cls0) == 0 or len(idx_cls1) == 0:
        rand_index = torch.randperm(B, device=device)
    else:
        # 为每个样本挑选一个“异类伙伴”
        rand_index = torch.empty(B, dtype=torch.long, device=device)

        # 对 y==0，从 idx_cls1 里采样（可能重复）
        rand_index[idx_cls0] = idx_cls1[torch.randint(
            len(idx_cls1), size=(len(idx_cls0),), device=device)]

        # 对 y==1，从 idx_cls0 里采样（可能重复）
        rand_index[idx_cls1] = idx_cls0[torch.randint(
            len(idx_cls0), size=(len(idx_cls1),), device=device)]

    # 线性混合
    mixed_x = lam * x + (1.0 - lam) * x[rand_index]
    y_a, y_b = y, y[rand_index]

    return mixed_x, y_a, y_b, lam

def plot_average_target_eeg(data_loader, fs=128):
    """
    取前两个batch中所有y==1的EEG样本，做叠加平均，并绘制前10个通道的时间序列图。

    参数:
    - data_loader: PyTorch DataLoader，输出 (x, y)，其中 x.shape = (batch_size, C, T)
    - fs: 采样率（每秒的采样点数，默认128）
    """
    target_eegs = []
    nontarget_eegs = []
    batch_count = 0

    for x, y, _ in data_loader:
        if (y == 1).any():
            if batch_count <= 3:
                target_eegs.append(x[y == 1])
        if (y == 0).any():
            nontarget_eegs.append(x[y == 0])

        batch_count += 1
        if batch_count >= 10:
            break

    if not target_eegs or not nontarget_eegs:
        print("前3个 batch 中缺少 target 或 non-target 数据。")
        return

    # 叠加平均
    target_eeg = torch.cat(target_eegs, dim=0).mean(dim=0)  # shape: (C, T)
    nontarget_eeg = torch.cat(nontarget_eegs, dim=0).mean(dim=0)  # shape: (C, T)

    # 指定通道（例如 Pz, POz 等）：18, 20, 26, 30
    ch_indices = [18, 20, 26, 30]
    target_plot = target_eeg[ch_indices]  # shape: (4, T)
    nontarget_plot = nontarget_eeg[ch_indices]  # shape: (4, T)

    time = torch.arange(target_plot.shape[1]) / fs

    # 创建子图
    fig, axes = plt.subplots(target_plot.size(0), 1, figsize=(10, 8), sharex=True)
    for i in range(target_plot.size(0)):
        axes[i].plot(time, target_plot[i], color='b')
        axes[i].set_ylabel(f'Ch {i + 1}')
        axes[i].grid(True)
        axes[i].tick_params(labelsize=8)

    axes[-1].set_xlabel('Time (s)')
    # fig.suptitle(f'Average EEG for y=1 (First {eeg_10.size(0)} Channels)', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('different_channels_target.png', dpi=300, bbox_inches='tight')
    plt.show(block=True)

    # 绘制 non-target EEG
    fig, axes = plt.subplots(nontarget_plot.size(0), 1, figsize=(10, 8), sharex=True)
    for i in range(nontarget_plot.size(0)):
        axes[i].plot(time, nontarget_plot[i], color='b')
        axes[i].set_ylabel(f'Ch {i + 1}')
        axes[i].grid(True)
        axes[i].tick_params(labelsize=8)
    axes[-1].set_xlabel('Time (s)')
    # fig.suptitle(f'Average EEG for y=0 (First {eeg_10.size(0)} Channels)', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('different_channels_nontarget.png', dpi=300, bbox_inches='tight')
    plt.show(block=True)

def split_source_target_indices(yd):
    """
    根据域标签yd划分源域和目标域索引。

    参数:
        yd (Tensor or list or ndarray): 1D，表示每个样本的域标签，比如 [0, 1, 0, 2, 2]

    返回:
        source_indices (List[int]): 源域样本的索引（标签不是最大值）
        target_indices (List[int]): 目标域样本的索引（标签是最大值）
    """
    max_label = yd.max().item()

    target_indices = (yd == max_label).nonzero(as_tuple=True)[0].tolist()
    source_indices = (yd != max_label).nonzero(as_tuple=True)[0].tolist()

    return source_indices, target_indices


class preprocess_for_SDDA:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.device = next(iter(dataloader))[0].device  # 保留原数据所在设备

    def trial_normalization(self, x):
        # x shape: (N, C, T)
        abs_max = torch.amax(torch.abs(x), dim=(1, 2), keepdim=True)  # (N, 1, 1)
        x_norm = x / (abs_max + 1e-6)
        return x_norm

    def euclidean_alignment(self, x):
        # x shape: (N, C, T)
        N, C, T = x.shape
        cov_sum = torch.zeros((C, C), device=x.device)

        # 计算平均协方差矩阵
        for i in range(N):
            xi = x[i]
            cov_i = xi @ xi.T / T  # shape: (C, C)
            cov_sum += cov_i
        R_bar = cov_sum / N

        # 计算 R_bar^(-1/2)
        eigvals, eigvecs = torch.linalg.eigh(R_bar)
        eigvals = torch.clamp(eigvals, min=1e-6)
        R_inv_sqrt = eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals)) @ eigvecs.T  # shape: (C, C)

        # 应用 EA
        x_aligned = torch.matmul(R_inv_sqrt.unsqueeze(0), x)  # shape: (N, C, T)
        return x_aligned

    def process(self):
        all_x, all_y, all_yd = [], [], []

        for x, y, yd in self.dataloader:
            all_x.append(x)
            all_y.append(y)
            all_yd.append(yd)

        x_all = torch.cat(all_x, dim=0).to(self.device)  # (N, C, T)
        y_all = torch.cat(all_y, dim=0)
        yd_all = torch.cat(all_yd, dim=0)

        # Trial Normalization
        x_norm = self.trial_normalization(x_all)

        # Euclidean Alignment
        x_aligned = self.euclidean_alignment(x_norm)

        # 返回新的 DataLoader（保持原来的参数）
        new_dataset = TensorDataset(x_aligned.cpu(), y_all, yd_all)
        new_loader = DataLoader(
            new_dataset,
            batch_size=self.dataloader.batch_size,
            shuffle=getattr(self.dataloader, 'shuffle', False),
            num_workers=self.dataloader.num_workers,
            drop_last=self.dataloader.drop_last,
            sampler=self.dataloader.sampler if hasattr(self.dataloader, 'sampler') else None
        )
        return new_loader


class BandpassFilterbankTransform:
    def __init__(self, sfreq=128):
        """
        Chebyshev Type II filter bank transform for EEG.

        Args:
            sfreq (int): Sampling frequency in Hz.
        """
        self.sfreq = sfreq
        self.bands = [
            (0.3, 4), (4, 8), (8, 12), (12, 16), (16, 20),
            (20, 24), (24, 28), (28, 32), (32, 36), (36, 40)
        ]

    def _filter(self, x):
        x_np = x.cpu().numpy()
        batch, C, T = x_np.shape
        filtered = []

        num_cores = multiprocessing.cpu_count()

        for band in self.bands:
            low, high = band
            nyq = 0.5 * self.sfreq
            low /= nyq
            high /= nyq
            sos = cheby2(N=4, rs=20, Wn=[low, high], btype='bandpass', output='sos')

            # 并行处理每个样本和通道
            def filter_one(b, c):
                return sosfiltfilt(sos, x_np[b, c, :])

            band_filtered = np.zeros_like(x_np)

            results = Parallel(n_jobs=num_cores)(
                delayed(filter_one)(b, c)
                for b in range(batch)
                for c in range(C)
            )

            # 重新填充 band_filtered
            idx = 0
            for b in range(batch):
                for c in range(C):
                    band_filtered[b, c, :] = results[idx]
                    idx += 1

            filtered.append(band_filtered)

        filtered_x = np.stack(filtered, axis=1)  # (batch, 10, C, T)
        return torch.tensor(filtered_x, dtype=x.dtype, device=x.device)

    def transform_dataloader(self, dataloader: DataLoader) -> DataLoader:
        """
        Transform input dataloader by applying bandpass filterbank on x.

        Args:
            dataloader (DataLoader): Input dataloader yielding (x, y, yd)

        Returns:
            DataLoader: New dataloader with filtered x of shape (10, C, T)
        """
        all_x, all_y, all_yd = [], [], []
        for batch in dataloader:
            if len(batch) == 2:
                x, y = batch
                yd = None
            elif len(batch) == 3:
                x, y, yd = batch
            else:
                raise ValueError(f"Unsupported batch format with {len(batch)} elements.")
            x_filtered = self._filter(x)  # (batch, 10, C, T)
            all_x.append(x_filtered)
            all_y.append(y)
            if yd is not None:
                all_yd.append(yd)

        all_x = torch.cat(all_x, dim=0)
        all_y = torch.cat(all_y, dim=0)

        if all_yd:
            all_yd = torch.cat(all_yd, dim=0)
            dataset = TensorDataset(all_x, all_y, all_yd)
        else:
            dataset = TensorDataset(all_x, all_y)

        new_dataloader = DataLoader(dataset, batch_size=dataloader.batch_size,
                                    shuffle=getattr(dataloader, 'shuffle', False),
                                    num_workers=dataloader.num_workers,
                                    drop_last=dataloader.drop_last,
                                    sampler=dataloader.sampler if hasattr(dataloader, 'sampler') else None
                                    )

        return new_dataloader


def bandpass_filterbank_for_SCLDGN(x, sfreq=128, num_threads=8):
    """
    Apply Chebyshev Type II bandpass filters to input EEG signal to get 10 frequency bands.
    
    Parameters:
    - x: torch.Tensor, shape (batch, C, T)
    - sfreq: int, sampling frequency (Hz)
    - num_threads: int, number of threads to use for parallel filtering

    Returns:
    - filtered_x: torch.Tensor, shape (batch, 10, C, T)
    """
    bands = [
        (0.3, 4), (4, 8), (8, 12), (12, 16), (16, 20),
        (20, 24), (24, 28), (28, 32), (32, 36), (36, 40)
    ]
    x_np = x.cpu().numpy()
    batch, C, T = x_np.shape

    def filter_one_band(band):
        low, high = band
        nyq = 0.5 * sfreq
        low /= nyq
        high /= nyq
        sos = cheby2(N=4, rs=20, Wn=[low, high], btype='bandpass', output='sos')

        # Prepare a function for parallel processing of each (b, c)
        def filter_channel(b, c):
            return sosfiltfilt(sos, x_np[b, c, :])

        filtered_band = np.zeros((batch, C, T), dtype=x_np.dtype)
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {
                executor.submit(filter_channel, b, c): (b, c)
                for b in range(batch) for c in range(C)
            }
            for future in as_completed(futures):
                b, c = futures[future]
                filtered_band[b, c, :] = future.result()
        return filtered_band

    # Loop over bands (sequentially), inside each band the (b,c) filters are parallel
    filtered = [filter_one_band(band) for band in bands]
    filtered_x = np.stack(filtered, axis=1)  # (batch, 10, C, T)
    return torch.tensor(filtered_x, dtype=x.dtype, device=x.device)


def compute_hilbert_components(E):
    """
    对 EEG 实值信号进行 Hilbert 变换，输出复值信号的实部与虚部。

    支持输入类型:
        - numpy.ndarray, shape=(B, C, T) 或 (B, 1, C, T)
        - torch.Tensor, shape=(B, C, T) 或 (B, 1, C, T)

    返回:
        real_part, imag_part: torch.FloatTensor, shape=(B, C, T)
    """
    device = E.device if isinstance(E, torch.Tensor) else 'cpu'
    # 如果输入是 torch.Tensor，转为 numpy
    is_tensor = isinstance(E, torch.Tensor)
    if is_tensor:
        E = E.detach().cpu().numpy()

    # 判断输入 shape，转为 (B, C, T)
    if E.ndim == 4 and E.shape[1] == 1:
        E = E[:, 0, :, :]  # 去除单通道维度
    elif E.ndim != 3:
        raise ValueError(f"Unsupported input shape: {E.shape}, expected (B,C,T) or (B,1,C,T)")

    B, C, T = E.shape
    real_part = np.zeros((B, 1, C, T), dtype=np.float32)
    imag_part = np.zeros((B, 1, C, T), dtype=np.float32)

    for b in range(B):
        # 对每个 trial 的所有通道做 Hilbert（对每行进行）
        X_complex = hilbert(E[b], axis=-1)  # shape: (C, T)
        real_part[b, 0] = X_complex.real
        imag_part[b, 0] = X_complex.imag

    # 转回 PyTorch Tensor
    real_part = torch.tensor(real_part, dtype=torch.float32, device=device)
    imag_part = torch.tensor(imag_part, dtype=torch.float32, device=device)

    return real_part, imag_part


def compute_plv_attention_vector(dataloader):
    """
    计算静态通道注意力向量，基于所有试次的 PLV。
    """
    all_phases = []

    for batch in dataloader:
        x = batch[0]
        # x: (B, C, T)

        # Apply Hilbert Transform on each channel
        analytic = hilbert(x, axis=-1)  # (B, C, T), complex
        phase = np.angle(analytic)  # Phase: (B, C, T)

        all_phases.append(phase)

    all_phases = np.concatenate(all_phases, axis=0)  # (N, C, T)
    N, C, T = all_phases.shape

    plv_matrix = np.zeros((C, C))
    for i in range(C):
        for j in range(C):
            delta_phase = all_phases[:, i, :] - all_phases[:, j, :]
            plv = np.abs(np.mean(np.exp(1j * delta_phase)))
            plv_matrix[i, j] = plv

    # 对每一行求平均，即每个通道与其它通道的相位一致性
    channel_attn = plv_matrix.mean(axis=1)
    channel_attn = channel_attn / np.max(channel_attn)  # normalize to [0, 1]

    return torch.tensor(channel_attn, dtype=torch.float32)  # (C,)