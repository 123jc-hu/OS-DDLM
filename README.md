# OS-DDLM

论文 **"Orthogonal Hard Channel Selection with a Low-Arithmetic Decoder for Budget-Constrained
RSVP-EEG Decoding"** 的代码。

OS-DDLM 由两个模块组成：

- **OS-Selector**（Orthogonal Softmax Selector）：确定性的硬通道选择器。训练和推理都前向传播 argmax
  选出的电极，通过 softmax 代理梯度学习，并用正交惩罚约束软分配，减少重复选择。
- **DDLM-Extractor**（Depthwise-Dilated and Low-Rank Mixer Extractor）：低计算量解码器，由分组空洞时间卷积、
  低秩空间混合和 ECA 通道注意力组成。

## 目录结构

```text
configs/config.yaml          模型、数据集和训练参数
main.py                      单次实验入口（由启动脚本调用）
scripts/run.py               解码器、OS-Selector 及通道选择基线的统一启动脚本
scripts/run_resgs.py         ResGS 基线（三阶段，按原论文思路改写）
scripts/analysis/            被试配对统计检验
scripts/benchmark/           解码器参数量、MACs 和推理时延测试
scripts/smoke_test.py        用随机输入检查所有模型能否构建和前向
models/                      OS-Selector、DDLM-Extractor 以及各基线模型
training/trainer.py          训练与测试流程（所有方法共用）
data/datamodule.py           被试内 / 跨被试数据加载
splits/                      论文使用的固定欠采样索引
```

论文中的方法名与代码的对应关系：

| 论文 | `--model` 参数 / 脚本 | 实现 |
|---|---|---|
| DDLM-Extractor | `DDLM-Extractor` | `models/Ours_light.py`（全通道）、`models/Ours_light_with_K.py`（K 通道输入） |
| OS-Selector (Ours) | `OS-Selector (Ours)` | `models/adaptive_os_selector.py` + DDLM-Extractor |
| w/o Orthogonality | `OS-Selector w/o Orthogonality` | 同上，`selector_orth: 0` |
| Deterministic masking | `OS-Selector w/ Deterministic Top-K` | `models/adaptive_os_selector.py` |
| Correlation-based | `Correlation-based Selection` | `models/channel_selectors.py` |
| Frequency-domain | `Frequency-Domain Weighting` | `models/channel_selectors.py` |
| SparseEA | `SparseEA` | `models/channel_selectors.py` |
| Gumbel-Softmax | `Gumbel-Softmax Selector` | `models/official_gumbel_selector.py` |
| Gumbel-Top-K | `Gumbel-Top-K Selector` | `models/gumbel_topk_selector.py` |
| ResGS (adapted) | `scripts/run_resgs.py` | `models/resgs_paper_guided.py` |
| EEGNet、PLNet、PPNN、LMDA-Net、LENet、1D-CNN-BiLSTM、Lightweight-ERP | `EEGNet`、`PLNet`、`PPNN`、`LMDANet`、`LENet`、`1DCNN_BiLSTM`、`lightweight_erp` | `models/` |

## 1. 环境配置

先按本机 CUDA 版本安装 PyTorch 2.3.1（见 https://pytorch.org/get-started/locally/ ），再安装其余依赖并自检：

```bash
pip install -r requirements.txt
python scripts/smoke_test.py      # 不需要数据，输出 "Smoke test passed" 即可
```

## 2. 数据准备

三个数据集均可从原作者处公开获取：

- **THU**：S. Zhang et al., "A benchmark dataset for RSVP-based brain-computer interfaces," *Front. Neurosci.*, 2020.
- **GIST**：K. Won et al., "EEG dataset for RSVP and P300 speller brain-computer interfaces," *Sci. Data*, 2022.
- **TCTR**：W. Wei et al., "Preliminary study on rapid serial visualization presentation multi-class target EEG classification," *IEEE Trans. Biomed. Eng.*, 2025.

按论文描述进行预处理：带通滤波、共平均参考、截取刺激后 0–1000 ms、基线校正、降采样到 128 Hz，
并在每个 epoch 内对各通道标准化。每个被试保存为一个 `.npz` 文件，**保持原始 epoch 顺序**
（训练/测试按存储顺序划分）：

```text
<data-root>/THU/Standard_128Hz/sub1.npz
<data-root>/GIST/Standard_128Hz/sub1.npz
<data-root>/TCTR/Standard_128Hz/task1/sub1.npz      # task1、task2、taskA、taskB
```

每个文件包含 `x_data`（形状为 `(epoch 数, 通道数, 128)`）和二分类标签 `y_data`。
然后把固定的欠采样索引复制到数据目录，并指定数据路径：

```bash
cp -r splits/* <data-root>/
export RSVP_DATASET_ROOT=<data-root>      # 或在每条命令后加 --data-root <data-root>
```

## 3. 基本用法

所有实验都通过 `scripts/run.py` 启动，一条命令对应"一个方法 × 一个数据集 × 一种协议"，会依次跑完所有被试：

```bash
python scripts/run.py --model "OS-Selector (Ours)" --dataset THU --train-mode single-subject --k 32
```

常用参数：

| 参数 | 说明 |
|---|---|
| `--model` | 方法名，见上表或 `configs/config.yaml` |
| `--dataset` | `THU`、`GIST`、`TCTR_1`、`TCTR_2`、`TCTR_A`、`TCTR_B` |
| `--train-mode` | `single-subject`（被试内，默认）或 `cross-subject`（跨被试，留一被试） |
| `--k` | 通道预算，仅通道选择方法需要 |
| `--subjects 1 5` | 只跑第 1–5 号被试（闭区间），默认全部 |
| `--seed` | 随机种子，默认 2025 |
| `--epochs` | 最大训练轮数，默认 200（早停） |
| `--orth` | 覆盖正交损失权重 λ，默认 1.0 |

OS-Selector、w/o Orthogonality、Deterministic masking 和 Gumbel-Top-K 在不同数据集上的选择器学习率和退火参数
（见补充材料 S8、S15）由脚本自动设置，无需手动指定。若要完全复现论文数值，运行前设置 `PYTHONHASHSEED=2025`。

结果保存在 `experiments/<运行名>/<数据集>/<协议>/results.csv`，每行一个被试（AUC、BA、F1、TPR、TNR），
末尾两行为均值和标准差；日志在 `logs/`，所选通道和运行时间统计保存在同一结果目录下。

## 4. 复现主要结果

**全通道解码器对比**（每个数据集、两种协议各跑一遍）

```bash
for model in DDLM-Extractor EEGNet PLNet PPNN LMDANet LENet 1DCNN_BiLSTM lightweight_erp; do
  python scripts/run.py --model "$model" --dataset THU --train-mode single-subject
done
```

**参考预算下的通道选择对比**（THU、TCTR 用 K = 32，GIST 用 K = 16）

```bash
for model in "OS-Selector (Ours)" "OS-Selector w/o Orthogonality" "Correlation-based Selection" \
             "Frequency-Domain Weighting" SparseEA "Gumbel-Softmax Selector" "Gumbel-Top-K Selector"; do
  python scripts/run.py --model "$model" --dataset THU --train-mode single-subject --k 32
done
python scripts/run_resgs.py --dataset THU --mode single-subject --k 32
python scripts/run.py --model "OS-Selector w/ Deterministic Top-K" --dataset THU --k 32   # 确定性掩码对照
```

其他预算用 `--k` 指定（THU、TCTR：8、16、32、48；GIST：4、8、16、24）。

**DDLM-Extractor 组件消融**

```bash
for variant in late_downsampling no_dilation full_spatial no_eca; do
  python scripts/run.py --model DDLM-Extractor --ddlm-ablation $variant --dataset THU
done
```

**OS-Selector 与其他解码器的兼容性**（THU，被试内，K = 32）

```bash
for model in EEGNet_OS EEGNet_Gumbel EEGNet_OS_wo_Orth PLNet_OS PLNet_Gumbel PLNet_OS_wo_Orth; do
  python scripts/run.py --model $model --dataset THU --k 32
done
```

**解码器效率测试**（CPU/GPU 推理时延、参数量、MACs；需要 GPU，不需要数据）

```bash
python scripts/benchmark/bench_backbone_efficiency.py --output-dir experiments/efficiency
```

## 5. 统计检验

`compare_methods.py` 对一个方法和若干对比方法做被试配对的 Wilcoxon 符号秩检验（Holm 校正），
并给出秩双列相关效应量和 bootstrap 95% 置信区间。用逗号连接多个目录，可先在被试内对多次训练取平均。

```bash
python scripts/analysis/compare_methods.py \
  --reference "experiments/OS-Selector (Ours)_K32_s2025/THU/single-subject" \
  --others "experiments/SparseEA_K32_s2025/THU/single-subject" \
           "experiments/Gumbel-Top-K Selector_K32_s2025/THU/single-subject" \
           experiments/ResGS/THU/single-subject/K32_s2025
```

## 引用

论文发表后补充引用信息。

## 许可证

见 [LICENSE](LICENSE)。
