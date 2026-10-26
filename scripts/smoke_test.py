"""Build every configured model and run one forward pass on random input (no EEG data needed)."""

from pathlib import Path
import os
import sys
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from training.trainer import OptimizedExperimentRunner
from utils.config import build_config, load_config


def main() -> None:
    names = list(load_config("configs/config.yaml")["models"])
    for dataset in ["THU", "GIST", "TCTR_1"]:
        for name in names:
            config = build_config(name, dataset)
            torch.manual_seed(0)
            model = OptimizedExperimentRunner._build_model_from_config(SimpleNamespace(config=config)).eval()
            with torch.no_grad():
                # External selectors pick channels in the data loader, so their decoder sees K inputs.
                channels = config["selector_K"] if config.get("apply_channel_selector") else config["n_channels"]
                output = model(torch.randn(2, 1, channels, config["fs"]))
            logits = output[0] if isinstance(output, (tuple, list)) else output
            assert logits.shape == (2, 2), (name, dataset, tuple(logits.shape))
    print(f"Smoke test passed: {len(names)} configured models on THU, GIST and TCTR input shapes.")


if __name__ == "__main__":
    main()
