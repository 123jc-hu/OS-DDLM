from shutil import rmtree
import gc
import logging
import os
import re
import time as _t
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

from data.datamodule import EEGDataModuleSingleSubject, EEGDataModuleCrossSubject
from utils.config import set_random_seed
from utils.run_manifest import file_sha256
from utils.utils import load_from_checkpoint, EarlyStopping, SaveBestValBA
from utils.metrics import calculate_metrics, cal_F1_score
from models.adaptive_os_selector import build_model
from models.official_gumbel_selector import build_model_official
from models.gumbel_topk_selector import build_model_gumbel_topk
from models.regularized_gumbel import build_model_dup
from utils.Loss import *  # Import all loss functions
from models.channel_selectors import (
    CorrelationBasedChannelSelector,
    FrequencyWeightedCSPChannelSelector,
    SparseEAChannelSelector,
)


class MemoryManager:
    """Centralized memory management for CUDA operations"""
    
    @staticmethod
    @contextmanager
    def cuda_memory_context():
        """Context manager for automatic GPU memory cleanup"""
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
    
    @staticmethod
    def cleanup_tensors(*tensors):
        for t in tensors:
            if isinstance(t, torch.Tensor):
                del t
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class OptimizedExperimentRunner:
    METRIC_NAMES = ("AUC", "BA", "F1", "TPR", "TNR")
    # Coordinates subject-wise training, evaluation, and summary export.

    """
    End-to-end runner for subject-wise EEG experiments.
    - Builds models through the configured factory entry point.
    - Reuses one cached initialization template so every subject starts from
      the same model state.
    - Supports optional training followed by evaluation in one pipeline.
    - Uses the standard subject data path for all runs.
    - Delegates epoch-level optimization and logging to `OptimizedTrainer`.
    """

    def __init__(self, config: Dict[str, Any], main_logger: logging.Logger):
        self.config = config
        self.log = main_logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.exp_dir = self._setup_exp_dir()
        self.datamodule = EEGDataModuleSingleSubject(self.config) if self.config["train_mode"] == "single-subject" else EEGDataModuleCrossSubject(self.config)
        # Cache one clean initialization state per model signature.

        # Cached template state used to recreate identical fresh models.
        self._init_state_cache: Dict[tuple, Dict[str, torch.Tensor]] = {}
        # One row per subject/config: {subject, K, uniq, reg, dataset}

        self.selection_records = []  # One selector summary row per subject.

        self.runtime_records: List[Dict[str, Any]] = []  # One runtime summary row per subject.
        self.run_provenance: Dict[str, Any] = {}

    # ---------------- experiment directory ----------------
    # Experiment artifacts are grouped by model / dataset / train mode.
    def _setup_exp_dir(self) -> Path:
        """Create `experiments/<model>/<dataset>/<train_mode>` if needed."""
        model_dir = self.config.get("model_artifact_name", self.config["model"])
        root = Path("experiments") / model_dir / self.config["dataset"] / self.config["train_mode"]
        root.mkdir(parents=True, exist_ok=True)
        return root

    # Centralized config access keeps fallback seed handling consistent.
    def _random_seed(self) -> int:
        """Centralize the default random seed lookup used across subjects."""
        return int(self.config.get("random_seed", 2024))

    # File discovery helpers.
    def _extract_subject_id(self, file_name: str):
        match = re.search(r"sub(\d+)", file_name)
        return int(match.group(1)) if match else None

    def _list_subject_files(self, base_dir: Path) -> List[str]:
        return sorted(
            file_name
            for file_name in os.listdir(base_dir)
            if file_name.endswith(".npz") and "_10band" not in file_name
        )

    def _build_subject_file_map(self, base_dir: Path, subject_files: List[str]) -> Dict[str, str]:
        subject_file_map: Dict[str, str] = {}
        for file_name in subject_files:
            subject_id = self._extract_subject_id(file_name)
            if subject_id is None:
                continue
            subject_file_map[f"sub{subject_id}"] = str(base_dir / file_name)
        return subject_file_map

    def _subject_in_requested_range(self, subject_id: int) -> bool:
        subject_start = self.config.get("subject_start_id", None)
        subject_end = self.config.get("subject_end_id", None)

        unset_values = (None, "None", "", "null", "NULL")
        if subject_start not in unset_values and subject_id < int(subject_start):
            return False
        if subject_end not in unset_values and subject_id > int(subject_end):
            return False
        return True

    def _prepare_subject_data(
        self,
        subject_id: int,
        subject_file_name: str,
        base_dir: Path,
        indices_path: Path,
        subject_file_map: Dict[str, str],
    ) -> None:
        if self.config["train_mode"] == "single-subject":
            self.datamodule.prepare_data(
                test_subject_id=subject_id,
                data_path=str(base_dir / subject_file_name),
                indices_path=indices_path,
                mode="normal",
            )
            return

        self.datamodule.prepare_data(
            test_subject_id=subject_id,
            subject_file_map=subject_file_map,
            indices_path=indices_path,
            mode="normal",
            train_range=self.config.get("train_range", "train"),
        )

    # Aggregate selector diversity statistics across all processed subjects.
    def _log_selection_summary(self) -> None:
        if not self.selection_records:
            return

        ratios = [record["uniq"] / record["K"] for record in self.selection_records if record["K"] > 0]
        mean_ratio = float(np.mean(ratios)) if ratios else float("nan")
        std_ratio = float(np.std(ratios)) if ratios else float("nan")
        num_subjects = len(ratios)
        total_unique = int(sum(record["uniq"] for record in self.selection_records))
        unique_k_values = sorted({record["K"] for record in self.selection_records})

        if len(unique_k_values) == 1:
            shared_k = unique_k_values[0]
            total_ratio = total_unique / (shared_k * num_subjects) if (shared_k > 0 and num_subjects > 0) else float("nan")
            self.log.info(
                f"[RUN-UNIQ] subjects={num_subjects} K={shared_k} "
                f"mean(uniq/K)={mean_ratio:.4f}+/-{std_ratio:.4f} "
                f"sum(uniq)={total_unique} total_ratio={total_ratio:.4f}"
            )
            return

        self.log.info(
            f"[RUN-UNIQ] subjects={num_subjects} Ks={unique_k_values} "
            f"mean(uniq/K)={mean_ratio:.4f}+/-{std_ratio:.4f} sum(uniq)={total_unique}"
        )

    def _save_selection_records(self) -> None:
        """Export per-subject hard selections when the model exposes a selector."""
        if not self.selection_records:
            return

        selector_k = self.config.get("selector_K")
        suffix = f"_K={int(selector_k)}" if selector_k is not None else ""
        pd.DataFrame(self.selection_records).to_csv(
            self.exp_dir / f"selected_channels{suffix}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    def _record_selection(self, subject_id: int, indices, regularized: bool = False) -> None:
        """Record one subject's hard channel set in a common format."""
        channel_indices = [int(index) for index in indices]
        selected_k = int(self.config.get("selector_K", len(channel_indices)))
        unique_count = len(set(channel_indices))
        record = {
            "subject": subject_id,
            "K": selected_k,
            "uniq": unique_count,
            "reg": bool(regularized),
            "dataset": self.config.get("dataset", "NA"),
            "selected_channels": ",".join(str(index) for index in channel_indices),
        }
        self.selection_records.append(record)
        print(
            f"[SEL-STAT] subject={subject_id} K={selected_k} "
            f"uniq={unique_count} reg={bool(regularized)}"
        )
        print(f"  indices: {channel_indices}")
        self.log.info(f"subject={subject_id} selected channels: {channel_indices}")

    # Aggregate runtime metrics across all processed subjects.
    def _save_runtime_dataset_summary(self) -> None:
        if not self.runtime_records:
            return

        runtime_df = pd.DataFrame(self.runtime_records)
        summary_columns = [
            "Model p50 (ms)",
            "Model p95 (ms)",
            "Model jitter (ms)",
            "E2E p50 (ms)",
            "E2E p95 (ms)",
            "E2E jitter (ms)",
            "Throughput (samples/s)",
            "Peak CUDA Mem (MB)",
            "Miss Rate",
        ]
        summary_columns = [column for column in summary_columns if column in runtime_df.columns]

        mean_row = runtime_df[summary_columns].mean(numeric_only=True).to_dict()
        std_row = runtime_df[summary_columns].std(numeric_only=True).to_dict()

        mean_summary = {"subject": "AVG"}
        mean_summary.update({key: float(mean_row[key]) for key in mean_row})
        std_summary = {"subject": "STD"}
        std_summary.update({key: float(std_row[key]) for key in std_row})

        runtime_df_out = pd.concat([runtime_df, pd.DataFrame([mean_summary, std_summary])], ignore_index=True)
        out_csv = self.exp_dir / "runtime_dataset_summary.csv"
        runtime_df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")

        pretty = {
            key: f"{mean_row[key]:.3f}+/-{std_row[key]:.3f}"
            for key in summary_columns
            if key in mean_row and key in std_row
        }
        print("[RUNTIME-DATASET] ", pretty)

    # ---------------- model signature ----------------
    # Model construction helpers.
    def _model_signature(self) -> tuple:
        """
        Return a hashable signature for the cached initialization template.
        Any config change that affects model construction should be reflected
        in this tuple.
        """
        cfg = self.config
        return (
            cfg["model"],
            bool(cfg.get("use_selector", False)),
            int(cfg.get("selector_K", -1)),
            float(cfg.get("selector_beta_start", 5.0)),
            float(cfg.get("selector_beta_end", 0.1)),
            float(cfg.get("selector_T", cfg.get("epochs", 100))),
            bool(cfg.get("selector_per_sample", True)),
            int(cfg.get("n_channels", -1)),
            int(cfg.get("n_class", -1)),
            int(cfg.get("fs", -1)),
        )

    # ---------------- initialization cache ----------------
    def _build_model_from_config(self) -> nn.Module:
        """Route selector variants to the historical builder that matches the paper setup."""
        model_name = str(self.config.get("model", ""))
        selector_variant = str(self.config.get("selector_variant", "")).lower()
        algorithm = str(self.config.get("train_algorithm", "")).lower()

        if model_name == "Gumbel-Softmax Selector" or selector_variant == "official_gumbel":
            return build_model_official(self.config)
        if model_name == "Gumbel-Top-K Selector" or selector_variant == "gumbel_topk":
            return build_model_gumbel_topk(self.config)
        if model_name in {
            "OS-Selector w/o Orthogonality",
            "OS-Selector w/ Deterministic Top-K",
            "OS-Selector (Ours)",
        } or selector_variant in {"ours", "wo_orth", "deterministic_topk", "resgs"}:
            return build_model(self.config)
        if algorithm in {"reg", "dup"}:
            return build_model_dup(self.config)
        return build_model(self.config)

    def _get_init_state(self) -> Dict[str, torch.Tensor]:
        """
        Lazily build and cache one CPU copy of the initial `state_dict`.
        Later subject runs clone from this cache so initialization stays
        identical across runs.
        """
        sig = self._model_signature()
        if sig not in self._init_state_cache:
            model = self._build_model_from_config()
            self._init_state_cache[sig] = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            del model
        return self._init_state_cache[sig]

    def _new_model(self) -> nn.Module:
        """Instantiate a fresh model from the cached initialization template."""
        model = self._build_model_from_config()
        init_state = self._get_init_state()
        model.load_state_dict({k: v.clone() for k, v in init_state.items()})
        return model.to(self.device)

    # ---------------- training components ----------------
    # Bundle optimizer, scheduler, and checkpoint criteria in one place.
    def _configure_training_components(self, model: nn.Module):
        """
        Build the optimizer stack for one subject run.
        Backbone and selector parameters are separated so they can use
        different learning rates or weight decay values.
        """
        cfg = self.config

        # Keep selector and backbone parameters in separate optimizer groups.
        sel_params, bb_params = [], []
        for n, p in model.named_parameters():
            (sel_params if n.startswith("selector.") else bb_params).append(p)

        param_groups = []
        if bb_params:
            param_groups.append({
                "params": bb_params,
                "lr": float(cfg["learning_rate"]),
                "weight_decay": float(cfg.get("weight_decay", 0.0)),
            })
        if sel_params:
            param_groups.append({
                "params": sel_params,
                "lr": float(cfg.get("selector_lr", cfg["learning_rate"])),
                "weight_decay": float(cfg.get("selector_weight_decay", cfg.get("weight_decay", 0.0))),
                "name": "selector",
            })

        optimizer = torch.optim.Adam(param_groups)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=int(cfg.get("lr_patience", 5))
        )

        stage_tag = cfg.get("train_mode", "train")
        early = EarlyStopping(
            patience=int(cfg.get("patience", 10)),
            verbose=False,
            delta=float(cfg.get("es_delta", 0.0)),
            path=f"best_model-{stage_tag}.pth",
        )
        save_ba = SaveBestValBA(
            verbose=False,
            delta=float(cfg.get("ba_delta", 0.0)),
            path=f"best_ba_model-{stage_tag}.pth",
        )
        return optimizer, scheduler, early, save_ba

    # ---------------- subject loop ----------------
    # Top-level experiment loop over all discovered subjects.
    def run_experiment(self):
        set_random_seed(self._random_seed())

        # Reset per-run summaries before iterating over subjects.
        self.runtime_records = []
        self.selection_records = []

        base_dir = self._dataset_base()
        indices_path = base_dir / f"balanced_indices_for_{self.config['train_mode']}.pkl"

        # Build a stable subject file list and a subject-to-path lookup table.
        subject_files = self._list_subject_files(base_dir)
        subject_file_map = self._build_subject_file_map(base_dir, subject_files)
        self.run_provenance = {
            "data_split": {
                "path": str(indices_path),
                "sha256": file_sha256(indices_path),
            },
            "subjects": subject_files,
        }

        # Warm up the cached initialization template once for this run.
        self._get_init_state()

        results = {metric_name: [] for metric_name in self.METRIC_NAMES}

        with MemoryManager.cuda_memory_context():
            for subject_file_name in subject_files:
                set_random_seed(self._random_seed())

                subject_id = self._extract_subject_id(subject_file_name)
                if subject_id is None:
                    continue
                if not self._subject_in_requested_range(subject_id):
                    continue

                self._prepare_subject_data(
                    subject_id=subject_id,
                    subject_file_name=subject_file_name,
                    base_dir=base_dir,
                    indices_path=indices_path,
                    subject_file_map=subject_file_map,
                )

                # Training and evaluation for the current subject.
                metrics = self._run_subject(subject_id)
                self._log_subject(subject_id, metrics)
                for metric_name in results:
                    results[metric_name].append(metrics[metric_name])

        # Summarize selector diversity over all processed subjects.
        self._log_selection_summary()
        self._save_selection_records()

        self._save_results(results)

        # Save the dataset-level runtime summary collected during evaluation.
        self._save_runtime_dataset_summary()

    # ---------------- dataset root ----------------
    # Resolve the dataset root while preserving the original folder layout.
    def _dataset_base(self) -> Path:
        """
        Build the dataset root path while preserving the original folder layout.
        Expected shape: <root>/<dataset>/Standard_<fs>Hz[/taskX]
        """
        cfg = self.config
        configured_root = cfg.get("data_root") or os.environ.get("RSVP_DATASET_ROOT")
        if not configured_root:
            raise ValueError("Set `data_root` in configs/config.yaml or RSVP_DATASET_ROOT before running an experiment.")
        root = Path(configured_root)
        if "_" in cfg["dataset"]:
            dataset, task = cfg["dataset"].split("_")
            return root / dataset / f"Standard_{cfg['fs']}Hz" / f"task{task}"
        return root / cfg["dataset"] / f"Standard_{cfg['fs']}Hz"

    # ---------------- single-subject flow ----------------
    # Per-subject wrapper that restores a fresh model, runs train/eval, and
    # returns rounded metrics in a stable key order.
    def _run_subject(self, subject_id: int) -> Dict[str, float]:
        """
        Run one subject end-to-end:
        optional training, then evaluation, then metric collection.
        """
        subject_start_time = _t.perf_counter()

        # Each subject starts from the same cached initialization point.
        model = self._new_model()
        metric_values = self._run_single_subject(model, subject_id)

        del model
        elapsed_seconds = _t.perf_counter() - subject_start_time
        print(f"Subject {subject_id} completed in {elapsed_seconds:.2f}s")

        # Return the rounded metric bundle in the canonical metric order.
        return {metric_name: round(value, 4) for metric_name, value in zip(self.METRIC_NAMES, metric_values)}

    # Single-subject execution path: logging setup, datamodule setup, optional
    # training, and final evaluation.
    def _run_single_subject(self, model: nn.Module, subject_id: int) -> Tuple[float, float, float, float, float]:
        """Single-stage workflow for one subject: setup, optional training, evaluation."""
        # Build training utilities for this subject.
        optimizer, scheduler, early, save_ba = self._configure_training_components(model)

        # Create the subject-local trainer wrapper.
        trainer = OptimizedTrainer(
            config=self.config,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            earlystopping=early,
            save_other_model=save_ba,
            device=self.device,
            algorithm=self.config.get("train_algorithm", "ce"),
        )

        # Prepare checkpoint and logging directories for this subject.
        ckpt_dir, log_dir = self._prepare_dirs(subject_id)
        if self.config.get("is_training", True):
            self._clean_subject_checkpoints(ckpt_dir)
        # Only clear stale TensorBoard runs when a new training pass will start.
        if self.config.get("is_training", True):
            self._clean_subject_logs(log_dir, run_prefix="metrics")
        trainer.setup_logging(log_dir, ckpt_dir, stage=2, run_name=f"metrics_{self._run_tag()}_sub{subject_id}")

        # Prepare train/val/test splits for the current subject.
        self.datamodule.setup(stage=2, is_split_domains=True)

        # Training is optional and can be skipped from config.
        if self.config.get("is_training", True):
            self._train(trainer)

        # Evaluate on the held-out split at the end of the subject run.
        return self._eval(trainer, ckpt_dir)

    def _fit_external_channel_selector(self, selector_name: str, train_loader, val_loader):
        """Fit one classical selector using the current experiment seed."""
        selector_k = int(self.config.get("selector_K", 32))
        if selector_name == "Correlation-based Selection":
            selector = CorrelationBasedChannelSelector(
                dataloader=train_loader,
                K=selector_k,
                device=str(self.device),
            )
        elif selector_name == "SparseEA":
            selector = SparseEAChannelSelector(
                train_loader=train_loader,
                val_loader=val_loader,
                fs=self.config["fs"],
                target_K=selector_k,
                population_size=60,
                generations=40,
                win_ms=100,
                seed=self._random_seed(),
            )
        elif selector_name == "Frequency-Domain Weighting":
            selector = FrequencyWeightedCSPChannelSelector(
                train_loader=train_loader,
                fs=self.config["fs"],
                K=selector_k,
                bands=[(1, 4), (4, 8), (8, 12), (12, 30), (30, 40)],
                aggregate="mean",
                fir_taps=129,
                use_filter=True,
            )
        else:
            raise ValueError(f"Unsupported external channel selector: {selector_name}")
        return selector.fit()

    # Training helper kept separate so future ablations can swap loaders or
    # channel-selection behavior without touching the higher-level workflow.
    def _train(self, trainer):
        """Run the training phase for the current subject."""
        cfg = self.config

        batch_size = int(cfg["batch_size"])

        # Legacy domain-aware sampling experiments are kept here for reference.
        # sampler = None
        # if cfg["train_mode"] == "cross-subject" or stage == 2:
        # labels_np, doms_np = self.act_labels_domains(self.datamodule.train_dataset)
        # sampler = BalancedRandomDomainSampler(
        #     batch_size=batch_size,
        #     n_domains_per_batch=int(self.config.get("domains_per_batch", 32)),
        #     labels=labels_np,
        #     domain_labels=doms_np
        # )

        train_loader = self.datamodule.train_dataloader(batch_size=batch_size)
        val_loader = self.datamodule.val_dataloader(batch_size=batch_size)
        # print(f"  Training samples: {len(ds)} val samples: {len(self.datamodule.val_dataset)}")

        # val_loader = make_corrupted_dataloader(
        #     dataloader=val_loader,
        #     mask_channels=self.mask_channels,
        #     mode="noise",
        #     shuffle=False,
        #     noise_std=1.0,
        #     n_channels=62,
        #     noise_seed=2025,
        # )

        # Optional baseline channel-selection experiments kept for reuse.
        # self.channel_selector = CorrelationBasedChannelSelector(dataloader=train_loader, K=self.config.get("selector_K", 32), device="cuda").fit()
        # self.channel_selector = SparseEAChannelSelector(
        #     train_loader=train_loader,
        #     val_loader=val_loader,
        #     fs=self.config["fs"],
        #     target_K=self.config.get("selector_K", 32),
        #     population_size=60,
        #     generations=40,
        #     win_ms=100,
        # ).fit()
        if bool(self.config.get("apply_channel_selector", False)):
            model_name = str(self.config.get("model", ""))
            selector_name = str(self.config.get("channel_selector_name", model_name))
            self.channel_selector = self._fit_external_channel_selector(
                selector_name,
                train_loader,
                val_loader,
            )
            print("Selected channels:", self.channel_selector.selected_channels_)
            train_loader = self.channel_selector.transform(train_loader, batch_size=train_loader.batch_size, shuffle=True)
            val_loader   = self.channel_selector.transform(val_loader,   batch_size=val_loader.batch_size,   shuffle=False)

        # Launch subject-level optimization.
        trainer.fit(train_loader, val_loader, stage=2, epochs=int(cfg["epochs"]))

        # Release datasets that are no longer needed after training.
        del self.datamodule.train_dataset, self.datamodule.val_dataset

        # Save the final subject checkpoint for debugging or retrospective checks.
        torch.save(trainer.model.state_dict(), trainer.checkpoint_dir / "last_model--2.pth")

    # Evaluation helper that restores the best checkpoint, computes held-out
    # metrics, and appends selection/runtime summaries.
    def _eval(self, trainer, ckpt_dir: Path) -> Tuple[float, float, float, float, float]:
        """Load the best checkpoint, evaluate, and record selection/runtime summaries."""
        best_path = ckpt_dir / "best_model--2.pth"
        if not best_path.exists():
            fallback_path = ckpt_dir / "best_ba_model--2.pth"
            if fallback_path.exists():
                best_path = fallback_path
        trainer.model.load_state_dict(load_from_checkpoint(Path.cwd() / best_path))

        # Record selector usage statistics for this subject when available.
        if hasattr(trainer.model, "selector"):
            indices = trainer.model.selector.hard_indices()  # List[int], length = K
            self._record_selection(
                subject_id=self.datamodule.subject_id,
                indices=indices,
                regularized=float(self.config.get("selector_orth", 0.0)) > 0.0,
            )
        elif bool(self.config.get("apply_channel_selector", False)):
            self._record_selection(
                subject_id=self.datamodule.subject_id,
                indices=self.channel_selector.selected_channels_,
            )

        # Build the held-out evaluation dataloader.
        test_loader = self.datamodule.test_dataloader(batch_size=int(self.config.get("test_batch_size", 1000)))


        # Apply the externally fitted selector during evaluation when requested.
        if bool(self.config.get("apply_channel_selector", False)):
            test_loader = self.channel_selector.transform(
                test_loader,
                batch_size=test_loader.batch_size,
                shuffle=False,
            )

        metrics = trainer._run_epoch(test_loader, training=False, loss_fn=trainer.loss_fn)

        # Keep the latest runtime summary at the runner level for aggregation.
        runtime_stats = getattr(trainer, "last_runtime_stats", None)
        # Reuse subject parsing from the datamodule test path when possible.
        try:
            runtime_subject_id = int(re.search(r"sub(\d+)", str(self.datamodule.test_path)).group(1)) \
                if hasattr(self.datamodule, "test_path") else None
        except Exception:
            runtime_subject_id = None

        if runtime_stats is not None:
            # Append the current subject summary to the run-level table.
            row = {"subject": runtime_subject_id}
            row.update(runtime_stats)
            self.runtime_records.append(row)

            # Print a compact runtime summary to the console.
            print("[RUNTIME] ", " | ".join(
                f"{k}: {runtime_stats[k]:.3f}" if isinstance(runtime_stats[k], (int, float)) else f"{k}: {runtime_stats[k]}"
                for k in [
                    "Model p50 (ms)",
                    "Model p95 (ms)",
                    "E2E p50 (ms)",
                    "E2E p95 (ms)",
                    "Peak CUDA Mem (MB)",
                    "Throughput (samples/s)",
                    "Miss Rate",
                ] if k in runtime_stats
            ))

            # Append the subject summary to the runtime CSV.
            try:
                runtime_row_df = pd.DataFrame([{"subject": runtime_subject_id, **runtime_stats}])
                out_csv = self.exp_dir / "runtime_metrics.csv"
                header = (not out_csv.exists())
                runtime_row_df.to_csv(out_csv, mode="a", index=False, encoding="utf-8-sig", header=header)
            except Exception as _e:
                print(f"[WARN] runtime metrics save failed: {_e}")
        del self.datamodule.test_dataset
        return metrics[1:]  # AUC, BA, F1, TPR, TNR

    # ---------------- directories, logs, and results ----------------
    # Filesystem helpers below keep per-subject artifacts isolated.
    def _run_tag(self) -> str:
        """Return a stable per-run tag so different K settings never share artifacts."""
        selector_k = self.config.get("selector_K", None)
        if selector_k in (None, "", "None", "null", "NULL"):
            return "full"
        return f"K{int(selector_k)}"

    def _prepare_dirs(self, subject_id: int) -> Tuple[Path, Path]:
        """Create per-subject checkpoint and log directories."""
        run_tag = self._run_tag()
        base = self.exp_dir / "checkpoints" / run_tag / f"sub{subject_id}"
        logs = self.exp_dir / "logs" / run_tag / f"sub{subject_id}"
        base.mkdir(parents=True, exist_ok=True)
        logs.mkdir(parents=True, exist_ok=True)
        return base, logs

    def _maybe_clean_dirs(self, *paths: Path):
        # Only create missing directories; keep populated folders intact.
        for p in paths:
            if p.exists() and any(p.iterdir()):
                continue
            p.mkdir(parents=True, exist_ok=True)
    
    def _clean_subject_logs(self, log_root: Path, *, run_prefix: str = "metrics"):
        """
        Remove TensorBoard subdirectories for one subject that match the given
        run prefix. This is only used before a fresh training run.
        """
        if not log_root.exists():
            return
        for child in log_root.iterdir():
            if child.is_dir() and child.name.startswith(run_prefix):
                rmtree(child, ignore_errors=True)

    def _clean_subject_checkpoints(self, ckpt_root: Path):
        """Remove stale subject checkpoints before a fresh run of the same K setting."""
        if not ckpt_root.exists():
            return
        for file_name in ("best_model--2.pth", "best_ba_model--2.pth", "last_model--2.pth"):
            checkpoint_path = ckpt_root / file_name
            if checkpoint_path.exists():
                checkpoint_path.unlink()

    def _log_subject(self, subject_id: int, metrics: Dict[str, float]):
        self.log.info(
            f"Subject {subject_id} | AUC: {metrics['AUC']:.4f} | BA: {metrics['BA']:.4f} "
            f"| F1: {metrics['F1']:.4f} | TPR: {metrics['TPR']:.4f} | TNR: {metrics['TNR']:.4f}"
        )

    # Persist per-subject metrics and append AVG/STD rows for quick review.
    def _save_results(self, results: Dict[str, List[float]]):
        """Save subject metrics to CSV and append AVG/STD summary rows."""
        df = pd.DataFrame(results)
        df.insert(0, "SUB", [f"SUB{i+1}" for i in range(len(df))])

        metrics = ["AUC", "BA", "F1", "TPR", "TNR"]
        avg = df[metrics].mean().round(4)
        std = df[metrics].std().round(4)

        # Append AVG and STD summary rows for quick inspection.
        df = pd.concat([
            df,
            pd.Series({"SUB": "AVG", **avg}).to_frame().T,
            pd.Series({"SUB": "STD", **std}).to_frame().T,
        ], ignore_index=True)

        csv_path = self.exp_dir / "results.csv"
        df.to_csv(csv_path, index=False, float_format="%.4f")

        selector_k = self.config.get("selector_K", None)
        if selector_k not in (None, "", "None", "null", "NULL"):
            k_csv_path = self.exp_dir / f"results_K={int(selector_k)}.csv"
            df.to_csv(k_csv_path, index=False, float_format="%.4f")

        formatted = {k: f"{avg[k]:.4f}+/-{std[k]:.4f}" for k in metrics}
        self.log.info(f"Average metrics: {formatted}")
        self.log.info(f"Results saved to {csv_path}")
    
    def _extract_labels_domains(ds):
        # Support both ConcatDataset-like inputs and a single dataset instance.
        if hasattr(ds, "datasets"):  # Concat
            ys, doms = [], []
            for dom_id, part in enumerate(ds.datasets):
                # Read labels directly from cached arrays without loading signals.
                y = np.asarray(part.y_data[part.indices], dtype=np.int64)
                ys.append(y)
                doms.append(np.full(y.shape, dom_id, dtype=np.int64))
            return np.concatenate(ys), np.concatenate(doms)
        else:
            y = np.asarray(ds.y_data[ds.indices], dtype=np.int64)
            return y, np.zeros_like(y)


class OptimizedTrainer:
    """
    Trainer wrapper with a unified epoch core and configurable loss routing.
    `loss_fn` is expected to return:
    - loss_total: Tensor
    - loss_items: OrderedDict[str, float-like], in logging order
    """

    def __init__(self, config: Dict, model: nn.Module, optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler._LRScheduler,
                 earlystopping: EarlyStopping, save_other_model: SaveBestValBA,
                 device: torch.device, algorithm: str = "ce"):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.earlystopping = earlystopping
        self.save_other_model = save_other_model
        self.device = device

        # ---- loss routing ----
        self.loss_fn = self._resolve_loss_fn(algorithm)

        # ---- infra ----
        self.scaler = GradScaler()
        self.tb_logger = None
        self.checkpoint_dir = None
        self.current_epoch = 0

        # Runtime profiling configuration and last-epoch summary.
        self.log_runtime: bool = bool(self.config.get("log_runtime", True))
        self.rt_deadline_ms: float = float(self.config.get("rt_deadline_ms", 200.0))  # Real-time deadline threshold in milliseconds.
        self.last_runtime_stats: dict | None = None  # Updated after each completed `_run_epoch`.

    # ---------- public ----------
    def setup_logging(self, log_dir: Path, checkpoint_dir: Path, stage: int, *, run_name: str):
        self.checkpoint_dir = checkpoint_dir
        # Keep one isolated TensorBoard run per subject.
        self.tb_logger = TensorBoardLogger(
            save_dir=log_dir,          # e.g. experiments/.../logs/sub11
            name=run_name,             # e.g. f"metrics_stage{stage}"
            version=None,              # Let TensorBoardLogger create a fresh version.
            default_hp_metric=False,
        )
        stage_suffix = f"-{stage}"
        self.earlystopping.path = checkpoint_dir / f"best_model-{stage_suffix}.pth"
        self.save_other_model.path = checkpoint_dir / f"best_ba_model-{stage_suffix}.pth"

    def fit(self, train_loader, val_loader, stage: int, epochs: int):
        # Configure external selector annealing and dup-specific early-stop gating.
        base_external_beta = bool(self.config.get("selector_beta_external", False))

        mode = str(self.config.get("train_algorithm", "")).lower()
        # The original Gumbel baseline activates early stopping only after its
        # selector entropy reaches the configured convergence threshold.
        is_dup = mode == "dup"
        # Preserve the original behavior: `reg` forces external annealing, while other modes follow config.
        use_external_beta = (self.config["train_algorithm"] == "reg") or base_external_beta

        beta_s = float(self.config.get("selector_beta_start", 10.0))
        beta_e = float(self.config.get("selector_beta_end",   0.1))
        T      = int(self.config.get("selector_T", epochs))

        tau_s  = float(self.config.get("selector_tau_start", 3.0))
        tau_e  = float(self.config.get("selector_tau_end",   1.1))

        # Entropy-triggered early-stop parameters used by dup mode.
        H_thresh         = float(self.config.get("selector_entropy_thresh", 0.70))  # Default threshold for arming early stopping.
        entropy_patience = int(self.config.get("selector_entropy_patience", 3))
        # Reset the gating state at the start of every fit call.
        self._es_armed = False                          # Delay early stopping until the gate is armed.
        self._h_below_count = 0                         # Count consecutive epochs below the entropy threshold.

        def _exp_anneal(start: float, end: float, t: int, T: int) -> float:
            frac = min(max(t / max(T, 1), 0.0), 1.0)
            return float(start) * ((float(end) / float(start)) ** frac)

        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch

            # Apply external annealing only when the current configuration requests it.
            if use_external_beta and hasattr(self.model, "selector"):
                if hasattr(self.model.selector, "set_temperature"):
                    beta_t = _exp_anneal(beta_s, beta_e, epoch, T)
                    self.model.selector.set_temperature(beta_t)
                # Keep external threshold updates restricted to the original reg path.
                if (self.config["train_algorithm"] == "reg") and hasattr(self.model.selector, "set_thresh"):
                    tau_t = _exp_anneal(tau_s, tau_e, epoch, T)
                    self.model.selector.set_thresh(tau_t)

            with MemoryManager.cuda_memory_context():
                # ---- train ----
                train_metrics = self._run_epoch(train_loader, training=True, loss_fn=self.loss_fn, epoch=epoch)
                train_loss_avg = train_metrics[0][0][0]  # total is first

                # ---- val (same loss recipe, eval mode) ----
                val_metrics = self._run_epoch(val_loader, training=False, loss_fn=self.loss_fn, epoch=epoch)
                val_loss_avg = val_metrics[0][0][0]
                val_ba = val_metrics[2]

                self._display_progress(epoch, epochs, train_loss_avg, val_loss_avg, val_ba)

                self._log_training_metrics(train_metrics, mode="train", stage=stage)
                self._log_training_metrics(val_metrics,   mode="val",   stage=stage)

                if (self.current_epoch % 5 == 0) and hasattr(self.model, "selector"):
                    d = self._selector_diag()
                    if d is not None:
                        tau_show = (tau_t if (use_external_beta and (self.config["train_algorithm"] == "reg") and hasattr(self.model.selector, "set_thresh")) else None)
                        extra = (f" tau={tau_show:.3f}" if tau_show is not None else "")
                        print(
                            f"[SEL] ep={self.current_epoch} "
                            f"beta={d['beta']:.3f}{extra} H_mean={d['H_mean']:.3f} Top1={d['Top1_mean']:.3f} "
                            f"Kuniq={d['K_unique']} | ||alpha||={d['alpha_norm']:.2f} "
                            f"||grad||={d['alpha_grad_norm'] if d['alpha_grad_norm'] is not None else 'NA'} "
                            f"indices={d['indices_head']}"
                        )
                    with torch.no_grad():
                        P = self.model.selector.current_p_soft()      # (K,C)
                        col_sum = P.sum(dim=0)                        # (C,)
                        tau = (self.model.selector._tau() if hasattr(self.model.selector, "_tau")
                            else float(self.config.get("selector_tau_end", 1.1)))
                        violating = (col_sum > tau).sum().item()
                        margin = float(col_sum.max().item() - tau)
                        dup_rate = 1.0 - (P.argmax(dim=1).unique().numel() / P.size(0))
                        print(f"[dup] tau={tau:.3f} max={col_sum.max():.3f} mean={col_sum.mean():.3f} "
                            f"viol={violating} margin={margin:+.3f} dup_rate={dup_rate:.3f}")

                # In dup mode, arm early stopping only after entropy stays low for long enough.
                if is_dup and hasattr(self.model, "selector"):                 # Switched from the old reg gate to dup.
                    d2 = self._selector_diag()
                    Hm = d2.get("H_mean") if d2 is not None else None
                    if Hm is not None:
                        if Hm <= H_thresh:
                            self._h_below_count += 1
                        else:
                            self._h_below_count = 0
                        if (not self._es_armed) and (self._h_below_count >= entropy_patience):
                            self._es_armed = True
                            print(f"[EarlyStop] ARMED at epoch {epoch}: H_mean={Hm:.4f} <= {H_thresh} "
                                f"(for {self._h_below_count} consecutive epochs)")
                            # Reset the early-stopping state so older counters do not leak into the armed phase.
                            if hasattr(self.earlystopping, "reset") and callable(self.earlystopping.reset):
                                self.earlystopping.reset()
                            else:
                                if hasattr(self.earlystopping, "counter"):
                                    self.earlystopping.counter = 0
                                if hasattr(self.earlystopping, "early_stop"):
                                    self.earlystopping.early_stop = False
                                if hasattr(self.earlystopping, "best_score"):
                                    self.earlystopping.best_score = None
                                if hasattr(self.earlystopping, "val_loss_min"):
                                    self.earlystopping.val_loss_min = float("inf")

                # early stop / save / sched
                if (not is_dup) or (self._es_armed is True):                   # Switched from the old reg gate to dup.
                    self.earlystopping(val_loss_avg, self.model)
                    self.save_other_model(val_ba, self.model)
                    if self.earlystopping.early_stop:
                        print("Early stopping triggered\n")
                        break
                else:
                    # In dup mode before arming, skip early stopping but still save the BA checkpoint.
                    self.save_other_model(val_ba, self.model)

                self.scheduler.step(val_loss_avg)

                # Avoid stacking internal epoch-based annealing on top of external annealing.
                if hasattr(self.model, "step_epoch"):
                    if not use_external_beta:
                        self.model.step_epoch(1)
                    else:
                        pass


    # ---------- epoch core ----------
    def _run_epoch(self, dataloader, *, training: bool, loss_fn: Callable, epoch: int = None):
        """
        Run one full pass over a dataloader.
        - `training=True`: model.train(), backward, and optimizer step.
        - `training=False`: model.eval() under `no_grad()`.
        Returns:
            [
                (loss_avgs_list, loss_names_list),
                AUC, BA, F1, TPR, TNR,
            ]
        """
        self.model.train() if training else self.model.eval()

        # Collect runtime statistics only during evaluation to avoid training overhead.
        collect_runtime = (self.log_runtime and (not training))

        # Runtime buffers, all measured in milliseconds.
        pre_times_ms, model_times_ms, e2e_times_ms = [], [], []

        # Reset CUDA peak-memory tracking before evaluation starts.
        if collect_runtime and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device=self.device)

        # Dynamically track the active loss terms returned by the current loss function.
        loss_names: List[str] = []       # ["Total", "ce", "Func2", ...]
        loss_sums:  List[float] = []     # Running sums aligned with `loss_names`.
        n_batches = 0

        all_probs, all_preds, all_targets = [], [], []
        maybe_no_grad = torch.enable_grad if training else torch.no_grad

        sample_count = 0  # Number of evaluated samples in this epoch.
        with maybe_no_grad():
            for batch in dataloader:
                n_batches += 1

                # Start timing before host-to-device transfer so E2E latency includes I/O.
                t0 = _t.perf_counter()  # End-to-end timing start, including transfer and preprocessing.

                x = batch[0].to(self.device, non_blocking=True)
                y = batch[1].to(self.device, non_blocking=True)
                sample_count += x.size(0)  # Track the number of samples seen in this batch.

                if len(x.shape) == 3:
                    x = x.unsqueeze(1)
                
                # Synchronize once before the forward pass to isolate model compute time.
                if torch.cuda.is_available(): torch.cuda.synchronize()
                t1 = _t.perf_counter()

                if training:
                    self.optimizer.zero_grad(set_to_none=True)

                with autocast():
                    logits, extras = self._forward_unpack(x, current_epoch=epoch)
                    loss_total, items = loss_fn(logits, y, extras)
                    # `items` is an OrderedDict of scalar loss terms, usually starting with `ce`.

                    probs = torch.softmax(logits, dim=1)
                    preds = logits.argmax(1)
                
                # Synchronize after the forward pass so timing excludes queued CUDA work.
                if torch.cuda.is_available(): torch.cuda.synchronize()
                t2 = _t.perf_counter()

                # Store runtime measurements only for evaluation passes.
                if collect_runtime:
                    pre_ms   = (t1 - t0) * 1000.0
                    model_ms = (t2 - t1) * 1000.0
                    e2e_ms   = (t2 - t0) * 1000.0
                    pre_times_ms.append(pre_ms)
                    model_times_ms.append(model_ms)
                    e2e_times_ms.append(e2e_ms)

                if training:
                    self.scaler.scale(loss_total).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                # Keep `Total` first, then append loss terms in the order returned by `loss_fn`.
                if not loss_names:
                    loss_names = ["Total"] + list(items.keys())
                    loss_sums  = [0.0 for _ in loss_names]

                # total
                loss_sums[0] += float(loss_total.detach().item())
                # sub-items
                for i, k in enumerate(items.keys(), start=1):
                    loss_sums[i] += float(items[k])

                all_probs.append(probs.detach().cpu())
                all_preds.append(preds.detach().cpu())
                all_targets.append(y.detach().cpu())

                MemoryManager.cleanup_tensors(x, y, logits)

        # metrics
        probabilities = torch.cat(all_probs) if len(all_probs) else torch.empty(0)
        predictions  = torch.cat(all_preds)  if len(all_preds)  else torch.empty(0, dtype=torch.long)
        targets      = torch.cat(all_targets) if len(all_targets) else torch.empty(0, dtype=torch.long)

        if len(all_probs):
            ba, _, tpr, tnr, auc = calculate_metrics(targets, predictions, y_prob=probabilities)
            f1 = cal_F1_score(targets, predictions)
            AUC = auc.numpy().round(4) if auc is not None else None
            BA  = ba.numpy().round(4)
            F1  = f1.numpy().round(4)
            TPR = tpr.numpy().round(4)
            TNR = tnr.numpy().round(4)
        else:
            AUC = BA = F1 = TPR = TNR = None

        # Average each tracked loss term over the processed batches.
        denom = max(n_batches, 1)
        loss_avgs = [s / denom for s in loss_sums] if loss_sums else [0.0]
        metrics = [
            (loss_avgs, loss_names),  # Dynamic loss names aligned with their averaged values.
            AUC, BA, F1, TPR, TNR
        ]

        if len(all_probs):
            MemoryManager.cleanup_tensors(probabilities, predictions, targets)
        
        # ---------- runtime summary aggregation ----------
        if collect_runtime and len(e2e_times_ms) > 0:
            def _p50(a): 
                b = sorted(a); 
                return b[len(b)//2]
            def _p95(a): 
                b = sorted(a); 
                return b[int(len(b)*0.95)]
            def _jitter(a): 
                return _p95(a) - _p50(a)

            p50_model = _p50(model_times_ms); p95_model = _p95(model_times_ms); jit_model = _jitter(model_times_ms)
            p50_e2e   = _p50(e2e_times_ms);   p95_e2e   = _p95(e2e_times_ms);   jit_e2e   = _jitter(e2e_times_ms)

            # Throughput is computed as total samples divided by total end-to-end time.
            total_samples = int(sample_count)
            total_time_s = sum(e2e_times_ms) / 1000.0
            throughput = (total_samples / total_time_s) if total_time_s > 1e-9 else float('nan')

            # Peak CUDA memory in MB.
            peak_mb = float('nan')
            if torch.cuda.is_available():
                peak_mb = torch.cuda.max_memory_allocated(device=self.device) / 1e6

            # Deadline Miss Rate
            ddl = float(self.rt_deadline_ms)
            miss_rate = sum(1 for t in e2e_times_ms if t > ddl) / len(e2e_times_ms)

            # Store the most recent runtime summary so the runner can persist it later.
            self.last_runtime_stats = {
                "Preproc p50 (ms)": _p50(pre_times_ms),
                "Preproc p95 (ms)": _p95(pre_times_ms),
                "Model p50 (ms)": p50_model,
                "Model p95 (ms)": p95_model,
                "Model jitter (ms)": jit_model,
                "E2E p50 (ms)": p50_e2e,
                "E2E p95 (ms)": p95_e2e,
                "E2E jitter (ms)": jit_e2e,
                "Throughput (samples/s)": throughput,
                "Peak CUDA Mem (MB)": peak_mb,
                "Miss Rate": miss_rate,
                "Deadline (ms)": ddl,
                "#Samples": total_samples,
                "#Batches": len(e2e_times_ms),
            }
        else:
            # Clear runtime stats for training passes or empty evaluation sets.
            self.last_runtime_stats = None
        # ---------- end runtime summary aggregation ----------
        return metrics

    # ---------- loss functions ----------
    def loss_ce(self, logits, y, extras):
        """Plain cross-entropy loss."""
        ce = F.cross_entropy(logits, y)
        return ce, OrderedDict([("ce", float(ce.detach().item()))])

    def loss_gumbel(self, logits, y, extras):
        assert hasattr(self.model, "selector"), "loss_gumbel requires `model.selector`."
        ce = F.cross_entropy(logits, y)

        beta   = self._selector_beta()
        P_soft = self._selector_psoft(beta)  # (K, C), keep gradients enabled.

        selector = self.model.selector
        if hasattr(selector, "orth_regularizer"):
            orth_t = selector.orth_regularizer(P_soft)
        else:
            # Official Gumbel baseline has no orthogonality regularizer.
            orth_t = ce.new_zeros(())

        orth_coef = float(self.config.get("selector_orth", 0.0))

        total = ce + orth_coef * orth_t

        items = OrderedDict([
            ("ce",   float(ce.detach().item())),
            ("orth", float(orth_t.detach().item())),
        ])
        return total, items
    
    def differentiable_sparsity_loss(self, P_soft, target_sparsity=0.7):
        """Differentiable sparsity penalty based on expected channel usage."""
        # Estimate how many real channels are selected in expectation.
        # `P_soft` has shape (K, C) over the real EEG channels only.
        real_channel_probs = P_soft.sum(dim=0)  # Each real channel's selection probability.
        expected_real_count = real_channel_probs.sum()   # Expected number of selected channels.
        
        # Convert the expected real-channel count into a sparsity ratio.
        current_sparsity = 1 - (expected_real_count / P_soft.size(0))  # Normalize by K.
        
        # Penalize deviation from the target sparsity level.
        sparsity_loss = (current_sparsity - target_sparsity) ** 2
        
        return sparsity_loss
    
    def loss_dup(self, logits, y, extras):
        """
        Duplicate-penalty loss from the paper:
        L(P) = sum_c ReLU(sum_k P[k, c] - tau)
        total = CE + lambda * L(P)

        This path uses only the duplicate penalty and does not add the
        orthogonality regularizer.
        """
        assert hasattr(self.model, "selector"), "loss_dup requires `model.selector`."
        # Supervised classification loss.
        ce = F.cross_entropy(logits, y)

        # Soft selector probabilities at the current temperature; gradients must flow.
        beta   = self._selector_beta()
        P_soft = self._selector_psoft(beta)  # (K, C)

        # Read the duplicate threshold from the selector when available, otherwise from config.
        sel = self.model.selector
        if hasattr(sel, "_tau"):
            tau = float(sel._tau())  # Dup selector path with internal threshold annealing.
        elif hasattr(sel, "layer") and hasattr(sel.layer, "thresh"):
            tau = float(sel.layer.thresh)     # Legacy selector threshold path.
        else:
            tau = float(self.config.get("selector_tau_end", 1.1))

        # Paper penalty: sum probabilities per channel, then apply ReLU over the threshold.
        col_sum = P_soft.sum(dim=0)          # (C,)
        dup_t   = F.relu(col_sum - tau).sum()

        # Use an explicit config key so this weight is not confused with orthogonality.
        dup_coef = float(self.config.get("selector_dup", 0.1))  # Suggested config key: `selector_dup`.

        total = ce + dup_coef * dup_t

        items = OrderedDict([
            ("ce",  float(ce.detach().item())),
            ("dup", float(dup_t.detach().item())),
            ("tau", float(tau)),
        ])
        return total, items

    # ---------- helpers ----------
    def _resolve_loss_fn(self, algorithm: str) -> Callable:
        mode = (algorithm or "ce").lower()
        if mode == "ce":
            return self.loss_ce
        if mode in ("gumbel", "gumbel_selector", "ce_selreg"):
            if hasattr(self.model, "selector"):
                return self.loss_gumbel
            else:
                print("[WARN] model has no selector, fallback to 'ce'.")
                return self.loss_ce
        if mode in ("dup", "duplicate", "dup_penalty", "ce_dup"):
            if hasattr(self.model, "selector"):
                return self.loss_dup
            else:
                print("[WARN] model has no selector, fallback to 'ce'.")
                return self.loss_ce
        print(f"[WARN] Unknown loss_algorithm '{algorithm}', fallback to 'ce'.")
        return self.loss_ce
    
    def _selector_diag(self):
        m = getattr(self.model, "selector", None)
        if m is None:
            return None

        with torch.no_grad():
            P = m.current_p_soft()                 # (K, C) soft selection distribution at the current temperature.
            Hk, Hm = m.normalized_entropy(P)       # (K,), scalar
            top1 = P.max(dim=1).values.mean()      # Mean top-1 selection probability.
            idx = torch.as_tensor(m.hard_indices(), device=P.device)
            k_unique = idx.unique().numel()
            alpha_norm = m.alpha.norm(p=2).item()

        # Gradient norm is only available after backward during training.
        grad_norm = None
        if m.alpha.grad is not None:
            grad_norm = m.alpha.grad.norm(p=2).item()

        return {
            "beta": getattr(m, "_beta")() if hasattr(m, "_beta") else None,
            "H_mean": float(Hm),
            "Top1_mean": float(top1),
            "K_unique": int(k_unique),
            "alpha_norm": float(alpha_norm),
            "alpha_grad_norm": (float(grad_norm) if grad_norm is not None else None),
            "indices_head": idx[:8].tolist(),
        }

    def _selector_beta(self) -> float:
        return float(self.model.selector._beta()) if hasattr(self.model, "selector") and hasattr(self.model.selector, "_beta") else 1.0

    def _selector_psoft(self, beta: float) -> torch.Tensor:
        selector = getattr(self.model, "selector", None)
        if selector is not None and hasattr(selector, "current_p_soft"):
            return selector.current_p_soft()
        return torch.softmax(self.model.selector.alpha / max(beta, 1e-6), dim=1)

    def _forward_unpack(self, x, current_epoch: int = None):
        out = self.model(x, current_epoch)
        if isinstance(out, tuple):
            logits = out[0]
            extras = out[1:]
        else:
            logits = out
            extras = None
        return logits, extras

    def _display_progress(self, epoch: int, epochs: int, train_loss, val_loss: float, val_ba: float):
        print(f"Epoch {epoch:03d}/{epochs:3d} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f} | Val BA: {val_ba:.4f}")

    def _log_training_metrics(self, metrics, mode, stage: int):
        if self.tb_logger is None:
            return
        assert mode in ["train", "val"]
        prefix = f"{mode}/stage{stage}"

        # Log all reported loss terms to TensorBoard.
        loss_avgs, loss_names = metrics[0]
        # Example: names = ["Total", "ce", "Func2", "Func3"]
        for name, value in zip(loss_names, loss_avgs):
            if value is not None:
                self.tb_logger.experiment.add_scalar(f"{prefix}/Loss/{name}", value, self.current_epoch)

        # Validation-only scalar metrics.
        if mode == "val":
            AUC, BA, F1, TPR, TNR = metrics[1:]
            val_names = ["AUC", "BA", "F1", "TPR", "TNR"]
            for name, value in zip(val_names, [AUC, BA, F1, TPR, TNR]):
                if value is not None:
                    self.tb_logger.experiment.add_scalar(f"{prefix}/{name}", value, self.current_epoch)
