"""Serial decoder-only efficiency benchmark (Supplementary Section S10)."""

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.eeg_models import model_dict
from utils.config import build_config
from utils.run_manifest import file_sha256

DATASETS = ['THU', 'GIST', 'TCTR_1']
MODELS = ['EEGNet', 'PLNet', 'PPNN', 'LMDANet', '1DCNN_BiLSTM',
          'LENet', 'lightweight_erp', 'DDLM-Extractor']
ROUNDS, WARMUP, ITERATIONS = 5, 100, 500
IDENTITY = ['dataset', 'mode', 'model', 'n_channels']


def select_configs():
    configs = []
    for dataset in DATASETS:
        for mode in ['single-subject', 'cross-subject']:
            for model in MODELS:
                configs.append({**build_config(model, dataset), 'train_mode': mode})
    return configs


def latency_summary(values):
    q25, median, q75, p95 = np.percentile(values, [25, 50, 75, 95])
    return dict(median_ms=float(median), q25_ms=float(q25), q75_ms=float(q75), p95_ms=float(p95))


def gpu_snapshot():
    return subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,utilization.gpu,memory.used',
                                    '--format=csv'], text=True).strip()


def count_macs(model, x):
    """Per-sample Conv/Linear/LSTM matrix MACs; other operations are excluded."""
    total = 0

    def hook(layer, inputs, output):
        nonlocal total
        batch = inputs[0].shape[0]
        if isinstance(layer, (torch.nn.Conv1d, torch.nn.Conv2d)):
            total += output.numel() // batch * (layer.in_channels // layer.groups) * math.prod(layer.kernel_size)
        elif isinstance(layer, torch.nn.Linear):
            total += output.numel() // batch * layer.in_features
        elif isinstance(layer, torch.nn.LSTM):
            steps = inputs[0].shape[1 if layer.batch_first else 0]
            directions = 2 if layer.bidirectional else 1
            for index in range(layer.num_layers):
                features = layer.input_size if index == 0 else layer.hidden_size * directions
                total += steps * directions * 4 * layer.hidden_size * (features + layer.hidden_size)

    handles = [layer.register_forward_hook(hook) for layer in model.modules()
               if isinstance(layer, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Linear, torch.nn.LSTM))]
    try:
        with torch.inference_mode():
            model(x)
    finally:
        for handle in handles:
            handle.remove()
    return total


def measure(model, x, device):
    samples = []
    with torch.inference_mode():
        output = model(x)
        logits = output[0] if isinstance(output, (tuple, list)) else output
        if logits.shape != (1, 2) or not torch.isfinite(logits).all():
            raise ValueError('Invalid decoder output')
        del output, logits
        for _ in range(WARMUP):
            model(x)
        for index in range(ITERATIONS):
            if device == 'cuda':
                torch.cuda.synchronize()
            started = time.perf_counter_ns()
            model(x)
            if device == 'cuda':
                torch.cuda.synchronize()
            samples.append(dict(iteration=index, timer='wall', latency_ms=(time.perf_counter_ns()-started)/1e6))
        baseline = peak = extra = None
        if device == 'cuda':
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            model(x)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            extra = peak - baseline
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            for index in range(ITERATIONS):
                start.record()
                model(x)
                end.record()
                end.synchronize()
                samples.append(dict(iteration=index, timer='cuda_event', latency_ms=start.elapsed_time(end)))
    if any(not math.isfinite(r['latency_ms']) or r['latency_ms'] <= 0 for r in samples):
        raise ValueError('Invalid timing')
    return samples, dict(cuda_baseline_bytes=baseline, cuda_peak_allocated_bytes=peak,
                         cuda_forward_extra_peak_bytes=extra)


def append_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False, mode='a' if path.exists() else 'w', header=not path.exists())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the GPU measurements')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    configs = select_configs()
    assert len(configs) == 48
    rng = np.random.default_rng(2025)
    configs = [configs[i] for i in rng.permutation(len(configs))]
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), ROOT / 'configs/config.yaml', *sorted((ROOT / 'models').glob('*.py'))]
    metadata = dict(status='running', started_at=datetime.now().astimezone().isoformat(),
                    python=sys.executable, torch=torch.__version__, cuda=torch.version.cuda,
                    platform=platform.platform(), cpu=platform.processor(), gpu=torch.cuda.get_device_name(0),
                    gpu_before=gpu_snapshot(), pid=os.getpid(), dtype='float32', batch=1, seq_len=128,
                    cpu_threads=1, rounds=ROUNDS, warmup=WARMUP, iterations=ITERATIONS,
                    seed=2025, configs=configs, argv=sys.argv,
                    git_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    git_status=subprocess.check_output(['git', 'status', '--porcelain'], text=True),
                    source_sha256={str(p): file_sha256(p) for p in sources},
                    scope='Initialized decoder weights; synthetic inputs; no training, data loading, selector or preprocessing',
                    counted_operations='Conv/Linear/LSTM matrix MACs, per sample; counted FLOPs=2*MACs, not total forward FLOPs',
                    excluded_operations='Bias additions, normalization, activation, pooling, recurrent gates and functional operations such as einsum/attention',
                    repetitions='Technical timing repeats, not independent participants',
                    gpu_exclusivity='No training; Windows desktop/background GPU processes remain')
    manifest = out / 'manifest.json'
    manifest.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    current = ''
    try:
        for case, cfg in enumerate(configs, 1):
            current = f"dataset={cfg['dataset']} mode={cfg['train_mode']} K={cfg['n_channels']} method={cfg['model']}"
            print(f'RUN START {current} case={case}/48', flush=True)
            torch.manual_seed(2025)
            model = model_dict()[cfg['model']].Model(cfg).eval()
            x = torch.randn(1, 1, cfg['n_channels'], cfg['fs'])
            static = dict(params=sum(p.numel() for p in model.parameters()), macs=count_macs(model, x))
            static.update(counted_flops=2*static['macs'], parameter_bytes_fp32=4*static['params'])
            identity = dict(dataset=cfg['dataset'], mode=cfg['train_mode'], model=cfg['model'], n_channels=cfg['n_channels'])
            for repeat in range(ROUNDS):
                for device in ['cpu', 'cuda']:
                    model = model.to(device)
                    x = x.to(device)
                    samples, memory = measure(model, x, device)
                    fields = dict(**identity, device=device, round=repeat)
                    append_csv(out / 'raw_latency.csv', [{**fields, **r} for r in samples])
                    append_csv(out / 'round_cost.csv', [{**fields, **static, **memory}])
                    if device == 'cuda':
                        model = model.cpu()
                        x = x.cpu()
                        torch.cuda.empty_cache()
                print(f"MEASURE {current} round={repeat+1}/{ROUNDS} done", flush=True)
            del model, x
            print(f'RUN DONE {current} case={case}/48', flush=True)
        raw = pd.read_csv(out / 'raw_latency.csv')
        keys = IDENTITY + ['device', 'timer']
        summary = [{**dict(zip(keys, key)), **latency_summary(g.latency_ms), 'samples':len(g)}
                   for key, g in raw.groupby(keys)]
        assert len(summary) == 144 and all(r['samples'] == ROUNDS*ITERATIONS for r in summary)
        pd.DataFrame(summary).to_csv(out / 'latency_summary.csv', index=False)
        cost = pd.read_csv(out / 'round_cost.csv')
        cost.groupby(IDENTITY + ['device'], as_index=False).agg(
            params=('params','first'), macs=('macs','first'), counted_flops=('counted_flops','first'),
            parameter_bytes_fp32=('parameter_bytes_fp32','first'),
            cuda_peak_allocated_bytes=('cuda_peak_allocated_bytes','max'),
            cuda_forward_extra_peak_bytes=('cuda_forward_extra_peak_bytes','max')).to_csv(out / 'cost_summary.csv', index=False)
        metadata.update(status='completed', finished_at=datetime.now().astimezone().isoformat(), gpu_after=gpu_snapshot())
        metadata['output_sha256'] = {p.name:file_sha256(p) for p in out.glob('*.csv')}
        print(f'SUMMARY SAVED output={out}', flush=True)
    except Exception as error:
        metadata.update(status='failed', error=repr(error), finished_at=datetime.now().astimezone().isoformat())
        print(f'RUN FAIL {current} error={error!r}', flush=True)
        raise
    finally:
        manifest.write_text(json.dumps(metadata, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
