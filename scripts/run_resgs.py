"""Run the paper-guided ResGS baseline with a DDLM-Extractor decoder.

The implementation follows three explicit phases per subject:
1. train a C-channel DDLM backbone;
2. fine-tune the C-channel backbone jointly with residual-Gumbel selection;
3. freeze hard selections, slice the original EEG to K channels, transfer
   compatible C-channel weights, and fine-tune the K-channel DDLM.

It is a transparent adaptation for DEL4RSVP, not an official reproduction of
the speech-enhancement ResGS implementation. In particular, the C-to-K DDLM
weight transfer is necessary because DDLM is channel-structured, but is not
described in the source paper's BASEN architecture.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import re
import sys
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.datamodule import EEGDataModuleCrossSubject, EEGDataModuleSingleSubject
from models.Ours_light import Model as FullDDLM
from models.Ours_light_with_K import Model as SelectedDDLM
from models.resgs_paper_guided import (
    PaperGuidedResidualGumbelModel,
    transfer_full_backbone_to_selected,
)
from utils.config import build_config, set_random_seed
from utils.metrics import cal_F1_score, calculate_metrics
from utils.run_manifest import file_sha256


class ChannelSubsetDataset(Dataset):
    """Expose the fixed hard-selected physical channels without padding."""

    def __init__(self, dataset: Dataset, indices: Iterable[int]):
        self.dataset = dataset
        self.indices = torch.as_tensor(list(indices), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        x, y = self.dataset[index][:2]
        return x.index_select(0, self.indices), y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=os.environ.get("RSVP_DATASET_ROOT"))
    parser.add_argument("--dataset", choices=("THU", "GIST", "TCTR_1", "TCTR_2", "TCTR_A", "TCTR_B"), default="THU")
    parser.add_argument("--mode", choices=("single-subject", "cross-subject"), default="single-subject")
    parser.add_argument("--subjects", nargs="+", type=int, default=None, help="Defaults to all available subjects.")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--pretrain-epochs", type=int, default=100)
    parser.add_argument("--residual-epochs", type=int, default=50)
    parser.add_argument("--finetune-epochs", type=int, default=50)
    parser.add_argument("--residual-weight", type=float, default=0.1)
    parser.add_argument("--selector-lr", type=float, default=None, help="Defaults to 0.01 on THU and 0.005 otherwise.")
    parser.add_argument("--log-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None, help="Dedicated non-existing result directory for one configuration.")
    return parser.parse_args()


def log_message(log_path: Path, message: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def as_4d(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    x = x.to(device, non_blocking=True)
    return x.unsqueeze(1) if x.ndim == 3 else x


def annealed_temperature(epoch: int, total_epochs: int, start: float = 10.0, end: float = 0.1) -> float:
    fraction = min(max(epoch / max(total_epochs, 1), 0.0), 1.0)
    return start * ((end / start) ** fraction)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, temperature: float | None = None) -> dict[str, float]:
    model.eval()
    probs, preds, targets = [], [], []
    with torch.no_grad():
        for batch in loader:
            x, y = batch[:2]
            x = as_4d(x, device)
            y = y.to(device, non_blocking=True)
            output = model(x, temperature) if temperature is not None else model(x)
            logits = output[0] if isinstance(output, tuple) else output
            probs.append(torch.softmax(logits, dim=1).cpu())
            preds.append(logits.argmax(dim=1).cpu())
            targets.append(y.cpu())
    probability = torch.cat(probs)
    prediction = torch.cat(preds)
    target = torch.cat(targets)
    ba, _, tpr, tnr, auc = calculate_metrics(target, prediction, y_prob=probability)
    f1 = cal_F1_score(target, prediction)
    return {
        "AUC": float(auc) if auc is not None else float("nan"),
        "BA": float(ba),
        "F1": float(f1),
        "TPR": float(tpr),
        "TNR": float(tnr),
    }


def fit_phase(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    selector_lr: float | None,
    residual_phase: bool,
    log_path: Path,
    stage_name: str,
) -> dict[str, torch.Tensor]:
    parameter_groups = [{"params": [p for n, p in model.named_parameters() if not n.startswith("selector.")], "lr": learning_rate}]
    if selector_lr is not None:
        parameter_groups.append({"params": model.selector.parameters(), "lr": selector_lr})
    optimizer = torch.optim.Adam(parameter_groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=8)
    best_ba = -float("inf")
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, epochs + 1):
        model.train()
        temperature = annealed_temperature(epoch, epochs) if residual_phase else None
        loss_sum = 0.0
        batches = 0
        for batch in train_loader:
            x, y = batch[:2]
            x = as_4d(x, device)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = model(x, temperature) if residual_phase else model(x)
            logits = output[0] if isinstance(output, tuple) else output
            loss = F.cross_entropy(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
            optimizer.step()
            loss_sum += float(loss.detach())
            batches += 1
        metrics = evaluate(model, val_loader, device, temperature)
        scheduler.step(metrics["BA"])
        if metrics["BA"] > best_ba:
            best_ba = metrics["BA"]
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            log_message(log_path, f"{stage_name} epoch={epoch}/{epochs} loss={loss_sum / max(batches, 1):.4f} val_BA={metrics['BA']:.4f} temp={temperature if temperature is not None else 'fixed'}")

    if best_state is None:
        raise RuntimeError(f"{stage_name} produced no checkpoint.")
    model.load_state_dict(best_state)
    return best_state


def make_loaders(datamodule: EEGDataModuleSingleSubject | EEGDataModuleCrossSubject, batch_size: int, indices: list[int] | None = None) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = datamodule.train_dataset
    val_dataset = datamodule.val_dataset
    test_dataset = datamodule.test_dataset
    if indices is not None:
        train_dataset = ChannelSubsetDataset(train_dataset, indices)
        val_dataset = ChannelSubsetDataset(val_dataset, indices)
        test_dataset = ChannelSubsetDataset(test_dataset, indices)
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0),
        DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=0),
    )


def dataset_base(data_root: Path, dataset: str, fs: int) -> Path:
    if "_" in dataset:
        dataset_name, task = dataset.split("_", maxsplit=1)
        return data_root / dataset_name / f"Standard_{fs}Hz" / f"task{task}"
    return data_root / dataset / f"Standard_{fs}Hz"


def subject_file_map(base_dir: Path) -> dict[int, Path]:
    files: dict[int, Path] = {}
    for path in base_dir.glob("sub*.npz"):
        if "_10band" in path.name:
            continue
        match = re.fullmatch(r"sub(\d+)\.npz", path.name)
        if match:
            files[int(match.group(1))] = path
    if not files:
        raise FileNotFoundError(f"No subject files found in {base_dir}")
    return dict(sorted(files.items()))


def main() -> None:
    args = parse_args()
    if args.selector_lr is None:
        args.selector_lr = 0.01 if args.dataset == "THU" else 0.005
    if not args.data_root:
        raise ValueError("Provide --data-root or set RSVP_DATASET_ROOT.")
    data_root = Path(args.data_root).resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {data_root}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or PROJECT_ROOT / "experiments" / "ResGS" / args.dataset / args.mode / f"K{args.k}_s{args.seed}"
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    log_path = (args.log_path or PROJECT_ROOT / "logs" / f"resgs_{args.dataset}_{args.mode}_{timestamp}.log").resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = build_config("DDLM-Extractor", args.dataset)
    config.update({"data_root": str(data_root), "train_mode": args.mode, "random_seed": args.seed, "seed": args.seed})
    base_dir = dataset_base(data_root, args.dataset, int(config["fs"]))
    index_path = base_dir / f"balanced_indices_for_{args.mode}.pkl"
    if not index_path.exists():
        raise FileNotFoundError(index_path)
    files = subject_file_map(base_dir)
    subjects = args.subjects or list(files)
    unknown_subjects = sorted(set(subjects) - set(files))
    if unknown_subjects:
        raise ValueError(f"Missing requested subject files: {unknown_subjects}")
    cross_file_map = {f"sub{subject_id}": str(path) for subject_id, path in files.items()}

    metadata = vars(args) | {
        "device": str(device),
        "data_root": str(data_root),
        "subjects": subjects,
        "method": "paper-guided-residual-gumbel-adaptation",
        "training_epoch_budget": args.pretrain_epochs + args.residual_epochs + args.finetune_epochs,
        "implementation_revision": "channel-major-transfer-v2",
        "source_sha256": {name: file_sha256(PROJECT_ROOT / name) for name in (
            "scripts/run_resgs.py", "models/resgs_paper_guided.py",
            "models/Ours_light.py", "models/Ours_light_with_K.py", "data/datamodule.py",
            "utils/config.py", "utils/metrics.py",
        )},
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    log_message(log_path, f"ResGS RUN START | dataset={args.dataset} mode={args.mode} device={device} subjects={subjects} K={args.k} seed={args.seed} SERIAL=true")

    rows = []
    for subject_id in subjects:
        set_random_seed(args.seed)
        if args.mode == "single-subject":
            datamodule = EEGDataModuleSingleSubject(config)
            datamodule.prepare_data(subject_id, str(files[subject_id]), str(index_path), mode="normal")
        else:
            datamodule = EEGDataModuleCrossSubject(config)
            datamodule.prepare_data(subject_id, cross_file_map, str(index_path), mode="normal", train_range=config.get("train_range", "train"))
        datamodule.setup(stage=2, is_split_domains=True)
        full_train, full_val, full_test = make_loaders(datamodule, int(config["batch_size"]))
        subject_dir = output_dir / f"sub{subject_id}"
        subject_dir.mkdir()

        log_message(log_path, f"SUBJECT START subject={subject_id} stage=pretrain")
        full_model = FullDDLM(config).to(device)
        pretrain_state = fit_phase(full_model, full_train, full_val, device, args.pretrain_epochs, float(config["learning_rate"]), None, False, log_path, "pretrain")
        torch.save(pretrain_state, subject_dir / "pretrain_full_ddlm.pt")

        log_message(log_path, f"SUBJECT subject={subject_id} stage=residual")
        residual_model = PaperGuidedResidualGumbelModel(full_model, config["n_channels"], args.k, args.residual_weight).to(device)
        residual_state = fit_phase(residual_model, full_train, full_val, device, args.residual_epochs, float(config["learning_rate"]), args.selector_lr, True, log_path, "residual")
        torch.save(residual_state, subject_dir / "residual_selector_full_ddlm.pt")
        indices = residual_model.selector.hard_indices().cpu().tolist()
        unique_count = len(set(indices))
        (subject_dir / "selected_channels.json").write_text(json.dumps({"indices": indices, "unique": unique_count, "K": args.k}, indent=2), encoding="utf-8")

        log_message(log_path, f"SUBJECT subject={subject_id} stage=finetune unique={unique_count}/{args.k} indices={indices}")
        selected_config = deepcopy(config)
        selected_config["selector_K"] = args.k
        selected_model = SelectedDDLM(selected_config).to(device)
        transfer_full_backbone_to_selected(residual_model.backbone, selected_model, indices, selected_config.get("group_dim", 4))
        selected_train, selected_val, selected_test = make_loaders(datamodule, int(config["batch_size"]), indices)
        final_state = fit_phase(selected_model, selected_train, selected_val, device, args.finetune_epochs, float(config["learning_rate"]), None, False, log_path, "finetune")
        torch.save(final_state, subject_dir / "finetuned_selected_ddlm.pt")
        metrics = evaluate(selected_model, selected_test, device)
        rows.append({
            "dataset": args.dataset,
            "mode": args.mode,
            "seed": args.seed,
            "subject": subject_id,
            "K": args.k,
            "unique_channels": unique_count,
            "unique_ratio": unique_count / args.k,
            **metrics,
        })
        log_message(log_path, f"SUBJECT DONE subject={subject_id} BA={metrics['BA']:.4f} unique={unique_count}/{args.k}")

    pd.DataFrame(rows).to_csv(output_dir / "subject_metrics.csv", index=False, encoding="utf-8-sig")
    log_message(log_path, f"RESULTS={output_dir / 'subject_metrics.csv'}")
    log_message(log_path, "ResGS RUN END")


if __name__ == "__main__":
    main()
