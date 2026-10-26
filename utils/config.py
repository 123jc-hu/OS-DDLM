import logging
import os
import random
import re

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml


ARTIFACT_NAME_OVERRIDES = {
    "OS-Selector w/o Orthogonality": "OS-Selector wo Orthogonality",
    "OS-Selector w/ Duplicate Penalty": "OS-Selector with Duplicate Penalty",
}


def load_config(config_path):
    """Load a YAML configuration file into a plain dictionary."""
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def set_random_seed(seed=2024):
    """Set global random seeds for reproducible experiments."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def get_artifact_safe_name(name: str) -> str:
    """Return a filesystem-safe artifact name while preserving readable labels."""
    safe_name = ARTIFACT_NAME_OVERRIDES.get(name, name)
    safe_name = re.sub(r'[<>:"/\\|?*]+', " ", safe_name)
    safe_name = re.sub(r"\s+", " ", safe_name).strip().rstrip(".")
    return safe_name or "unnamed"


class Logger:
    """Small logger wrapper that avoids duplicate handlers."""

    def __init__(self, model_name):
        self.logger = logging.getLogger(model_name)
        self.logger.setLevel(logging.INFO)
        self.setup_handlers()

    def setup_handlers(self):
        """Attach fresh console and file handlers for the current logger."""
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        log_dir = os.path.join("logs")
        os.makedirs(log_dir, exist_ok=True)
        log_name = get_artifact_safe_name(self.logger.name)
        log_path = os.path.join(log_dir, f"{log_name}.log")
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.INFO)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def get_logger(self):
        """Return the configured logger instance."""
        return self.logger


def build_config(model_name, dataset_name=None):
    config_path = os.path.join("configs", "config.yaml")
    full_config = load_config(config_path)

    supported_models = set(full_config.get("models", {}))
    if model_name not in supported_models:
        supported_list = ", ".join(sorted(supported_models))
        raise KeyError(f"Unknown model '{model_name}'. Supported models: {supported_list}")

    # Keep both paper-facing names and legacy experiment aliases valid.
    # We intentionally preserve the user-provided model name so existing
    # log filenames and experiment directories remain unchanged.
    if dataset_name is None:
        dataset_name = full_config["dataset"]
    elif dataset_name not in full_config.get("datasets", {}):
        supported_datasets = ", ".join(sorted(full_config.get("datasets", {})))
        raise KeyError(
            f"Unknown dataset '{dataset_name}'. Supported datasets: {supported_datasets}"
        )

    shared_config = {
        key: value
        for key, value in full_config.items()
        if key not in ("models", "datasets")
    }
    model_specific = full_config.get("models", {}).get(model_name, {})
    dataset_specific = full_config.get("datasets", {}).get(dataset_name, {})

    merged_config = {**shared_config, **model_specific, **dataset_specific}
    merged_config["model"] = model_name
    merged_config["model_artifact_name"] = get_artifact_safe_name(model_name)
    merged_config["dataset"] = dataset_name

    n_fold_value = merged_config.get("n_fold", None)
    if n_fold_value in (None, "None", "", "null", "NULL"):
        merged_config["n_fold"] = None
    else:
        merged_config["n_fold"] = int(n_fold_value)

    # Preserve the legacy `seed` key for code paths that still read it.
    if "seed" not in merged_config:
        merged_config["seed"] = merged_config.get("random_seed", 2024)

    int_fields = ["sub_num", "n_channels", "fs", "n_class", "batch_size", "epochs"]
    for field_name in int_fields:
        if field_name in merged_config:
            merged_config[field_name] = int(merged_config[field_name])

    float_fields = [
        "learning_rate",
        "weight_decay",
        "selector_beta_start",
        "selector_beta_end",
        "selector_orth",
    ]
    for field_name in float_fields:
        if field_name in merged_config:
            merged_config[field_name] = float(merged_config[field_name])

    bool_fields = [
        "is_training",
        "use_gpu",
        "data_mix",
        "log_runtime",
        "use_selector",
        "selector_per_sample",
    ]
    for field_name in bool_fields:
        if field_name in merged_config:
            merged_config[field_name] = bool(merged_config[field_name])

    if merged_config["model"] == "LMDANet":
        if merged_config["train_mode"] == "single-subject":
            merged_config["lmda_avgpool"] = merged_config["fs"] // 10
        elif merged_config["train_mode"] == "cross-subject":
            merged_config["lmda_avgpool"] = 1

    return merged_config
