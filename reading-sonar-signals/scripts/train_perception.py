#!/usr/bin/env python3
"""Run the bounded P1/P2 scene-only weak-label CUDA baseline experiment.

Example: python scripts/train_perception.py --project-root .
The script caches deterministic unnormalized CUDA features, then trains all
models using only train/validation labels. Test labels are consulted only after
every checkpoint has been selected. This is not a real localization benchmark.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import soundfile as sf
import torch
from torch.nn import functional as F

from sonar.perception import (FEATURE_CONFIG, CompactSceneCNN, assert_model_cuda,
                              waveform_features, training_normalization,
                              confusion_counts, metric_tensors,
                              grouped_confusions, bootstrap_methods)
from sonar.signal import require_cuda


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def plain_metrics(metrics):
    return {k: v.detach().tolist() for k, v in metrics.items()}


@torch.inference_mode()
def logits_for(model, features, indices, batch_size):
    model.eval()
    return torch.cat([model(features[index]) for index in indices.split(batch_size)])


def prepare_features(rows, project_root, cache_path, manifest_hash, device):
    signature = {
        "manifest_sha256": manifest_hash,
        "feature_config": FEATURE_CONFIG,
        "scene_ids": [row["scene_id"] for row in rows],
        "source_sha256": [row["sha256"] for row in rows],
        "signal_code_sha256": sha256(project_root / "src/sonar/signal.py"),
        "perception_code_sha256": sha256(project_root / "src/sonar/perception.py"),
    }
    if cache_path.exists():
        cached = torch.load(cache_path, map_location=device, weights_only=True)
        if cached["signature"] != signature:
            raise RuntimeError("Feature cache provenance differs; use a fresh output directory")
        features = cached["features"]
        require_cuda(features)
        if not torch.isfinite(features).all():
            raise RuntimeError("Nonfinite cached features")
        return features, dict(cached["timing"], cache_reused=True)
    started = time.perf_counter()
    decode_seconds = 0.
    gpu_ms = 0.
    feature_list = []
    for index, row in enumerate(rows):
        path = project_root / row["path"]
        if sha256(path) != row["sha256"]:
            raise RuntimeError(f"Audio source hash mismatch: {row['scene_id']}")
        decode_start = time.perf_counter()
        data, rate = sf.read(path, dtype="float32", always_2d=True)
        decode_seconds += time.perf_counter() - decode_start
        if rate != row["sample_rate"]:
            raise RuntimeError(f"Manifest sampling-rate mismatch: {row['scene_id']}")
        x = torch.as_tensor(data, device=device).mean(dim=1)
        if x.numel() != round(rate * row["duration_seconds"]):
            raise RuntimeError(f"Manifest duration mismatch: {row['scene_id']}")
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.inference_mode():
            feature_list.append(waveform_features(x, rate, row["source_start_seconds"]))
        end.record(); end.synchronize(); gpu_ms += start.elapsed_time(end)
        if (index + 1) % 15 == 0:
            print(json.dumps({"phase": "features", "complete": index + 1,
                              "total": len(rows)}), flush=True)
    features = torch.stack(feature_list)
    timing = {"wall_seconds": time.perf_counter() - started,
              "cpu_decode_seconds": decode_seconds,
              "cuda_feature_seconds": gpu_ms / 1000,
              "cache_reused": False}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # Host transfer is serialization only, not a numerical CPU fallback.
    torch.save({"features": features.cpu(), "signature": signature,
                "timing": timing}, cache_path)
    return features, timing


def label_indices(rows, selected, classes, device):
    """Called on test rows only after all validation-selected checkpoints exist."""
    mapping = {name: index for index, name in enumerate(classes)}
    labels = []
    for index in selected:
        label = rows[index]["source_class"]
        if label not in mapping:
            raise ValueError(f"Unknown held-out class {label}; no silent class mapping")
        labels.append(mapping[label])
    return torch.tensor(labels, device=device, dtype=torch.long)


def train_one(method, seed, features, train_idx, val_idx, train_y, val_y,
              classes, args, output_dir, normalization, device):
    torch.cuda.manual_seed_all(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    model = CompactSceneCNN(len(classes), dual_view=method == "P2", device=device)
    assert_model_cuda(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.01)
    counts = torch.bincount(train_y, minlength=len(classes)).float()
    if (counts == 0).any():
        raise ValueError("All frozen classes require training examples")
    weights = (counts.sum() / (len(classes) * counts)).detach()
    history = []
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize(); wall_start = time.perf_counter()
    cuda_start, cuda_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    cuda_start.record()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(train_idx.numel(), generator=generator, device=device)
        losses = []
        for positions in order.split(args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            logits = model(features[train_idx[positions]])
            loss = F.cross_entropy(logits, train_y[positions], weight=weights)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite training loss for {method} seed {seed}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(torch.stack([loss.detach(), loss.new_tensor(positions.numel())]))
        with torch.inference_mode():
            validation_logits = logits_for(model, features, val_idx, args.batch_size)
            validation_loss = F.cross_entropy(validation_logits, val_y).item()
            validation_metrics = metric_tensors(confusion_counts(
                val_y, validation_logits.argmax(dim=-1), len(classes)))
            loss_pairs = torch.stack(losses)
            train_loss = (loss_pairs[:, 0] * loss_pairs[:, 1]).sum() / loss_pairs[:, 1].sum()
        history.append({"epoch": epoch, "training_weighted_ce": train_loss.item(),
                        "validation_ce": validation_loss,
                        "validation_metrics": plain_metrics(validation_metrics)})
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    cuda_end.record(); cuda_end.synchronize()
    wall_seconds = time.perf_counter() - wall_start
    checkpoint = output_dir / f"{method}_seed{seed}.pt"
    torch.save({"model_state_dict": best_state,
                "normalization": {name: value.detach().cpu() for name, value in normalization.items()},
                "method": method, "seed": seed, "class_names": classes,
                "best_epoch": best_epoch, "selection_metric": "minimum_validation_unweighted_cross_entropy",
                "feature_config": FEATURE_CONFIG}, checkpoint)
    record = {
        "method": method, "task": "weak_scene_classification_only", "seed": seed,
        "architecture": "three_convolutions_per_view_groupnorm_gelu_4x4_pool_128unit_head",
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "dual_view": method == "P2", "epochs_completed": len(history),
        "selected_epoch": best_epoch, "selected_validation_ce": best_loss,
        "history": history, "wall_seconds": wall_seconds,
        "cuda_elapsed_seconds": cuda_start.elapsed_time(cuda_end) / 1000,
        "cuda_timing_note": "CUDA event elapsed span includes gaps between queued work; not an occupancy-derived active time",
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "checkpoint": checkpoint.name, "checkpoint_sha256": sha256(checkpoint),
        "parameters_and_buffers_cuda": True,
    }
    save_json(output_dir / f"{method}_seed{seed}_training.json", record)
    print(json.dumps({"phase": "trained", "method": method, "seed": seed,
                      "best_epoch": best_epoch, "epochs": len(history),
                      "seconds": wall_seconds}), flush=True)
    del model, optimizer, best_state
    torch.cuda.empty_cache()
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--manifest", default="data/manifests/real_scenes.jsonl")
    parser.add_argument("--output", default="results/perception")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--seeds", default="1729,2718,3141")
    parser.add_argument("--methods", default="P1,P2")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--features-only", action="store_true")
    args = parser.parse_args()
    args.project_root = args.project_root.resolve()
    device = require_cuda()
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1 or args.bootstrap < 100:
        raise ValueError("Positive training bounds and >=100 bootstrap draws are required")
    seeds = [int(value) for value in args.seeds.split(",")]
    methods = args.methods.split(",")
    if len(set(seeds)) != len(seeds) or not set(methods) <= {"P1", "P2"}:
        raise ValueError("Seeds must be distinct and methods must be P1/P2")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    started = utc()
    manifest_path = args.project_root / args.manifest
    manifest_hash = sha256(manifest_path)
    manifest_rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    rows = [row for row in manifest_rows if row["domain"] == "deepship"]
    if not rows:
        raise ValueError("No DeepShip scenes in manifest")
    rows.sort(key=lambda row: row["scene_id"])
    split_indices = {name: [i for i, row in enumerate(rows)
                           if ("val" if row["split"] == "validation" else row["split"]) == name]
                     for name in ("train", "val", "test")}
    if sum(map(len, split_indices.values())) != len(rows) or not all(split_indices.values()):
        raise ValueError("Every scene requires a recognized nonempty split")
    group_sets = {name: {rows[i]["group_id"] for i in indices}
                  for name, indices in split_indices.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if group_sets[a] & group_sets[b]:
            raise ValueError(f"Recording-group leakage between {a} and {b}")
    classes = sorted({rows[i]["source_class"] for i in split_indices["train"]})
    output_dir = args.project_root / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "summary.json").exists():
        raise RuntimeError("Completed experiment exists; choose a fresh --output rather than overwrite")
    protocol = {
        "created_at": started, "task": "weak_scene_classification_only",
        "manifest_sha256": manifest_hash, "classes_from_training_only": classes,
        "split_scenes": {name: len(values) for name, values in split_indices.items()},
        "split_independent_groups": {name: len(values) for name, values in group_sets.items()},
        "seeds": seeds, "methods": methods, "feature_config": FEATURE_CONFIG,
        "max_epochs": args.epochs, "patience": args.patience, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "optimizer": "AdamW", "weight_decay": .01,
        "training_loss": "inverse_training_class_frequency_weighted_cross_entropy",
        "checkpoint_selection": "minimum_validation_unweighted_cross_entropy_with_1e-5_improvement",
        "test_labels": "not used until every model checkpoint has been selected",
        "bootstrap_draws": args.bootstrap,
        "macro_f1_definition": "macro_f1 aliases macro_f1_all_frozen_classes; also report macro_f1_present_reference_classes. Missing test classes do not have a generalization estimate.",
        "excluded_domains": sorted({row["domain"] for row in manifest_rows if row["domain"] != "deepship"}),
        "limitations": ["Recording-disjoint, not proven vessel-disjoint.",
                       "No component localization head; these do not fully instantiate the plan's localization baselines.",
                       "Weak scene labels do not identify the source of individual acoustic components.",
                       "P2 has more parameters; architecture is not parameter-matched to P1.",
                       "Some held-out classes can lack reference support; report both all-vocabulary and present-reference-class macro F1, not a four-class generalization claim.",
                       "No unseen-domain vessel classification claim; external domain lacks a shared verified label set."],
        "device": torch.cuda.get_device_name(device), "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "code_sha256": {str(path.relative_to(args.project_root)): sha256(path)
                        for path in (args.project_root / "src/sonar/perception.py",
                                     args.project_root / "src/sonar/signal.py", Path(__file__))},
    }
    existing_protocol = output_dir / "protocol.json"
    if existing_protocol.exists():
        existing = json.loads(existing_protocol.read_text())
        if {k:v for k,v in existing.items() if k != "created_at"} != {k:v for k,v in protocol.items() if k != "created_at"}:
            raise RuntimeError("Existing frozen protocol differs; choose a fresh output directory")
        protocol = existing
    else:
        save_json(existing_protocol, protocol)
    features, feature_timing = prepare_features(rows, args.project_root,
                                               output_dir / "features.pt", manifest_hash, device)
    if args.features_only:
        save_json(output_dir / "feature_timing.json", feature_timing)
        return
    indices = {name: torch.tensor(values, device=device, dtype=torch.long)
               for name, values in split_indices.items()}
    mean, std = training_normalization(features, indices["train"])
    features = (features - mean) / std
    train_y = label_indices(rows, split_indices["train"], classes, device)
    val_y = label_indices(rows, split_indices["val"], classes, device)
    training_records = []
    for method in methods:
        for seed in seeds:
            training_records.append(train_one(method, seed, features,
                                               indices["train"], indices["val"], train_y, val_y,
                                               classes, args, output_dir, {"mean": mean, "std": std}, device))
    save_json(output_dir / "checkpoint_selection_complete.json", {
        "at": utc(), "selected_before_test_label_access": True,
        "checkpoints": [{"file": r["checkpoint"], "sha256": r["checkpoint_sha256"]}
                        for r in training_records],
    })
    # First use of test labels: every checkpoint has already been selected.
    test_y = label_indices(rows, split_indices["test"], classes, device)
    test_rows = [rows[i] for i in split_indices["test"]]
    groups = sorted(group_sets["test"])
    group_mapping = {name: index for index, name in enumerate(groups)}
    group_idx = torch.tensor([group_mapping[row["group_id"]] for row in test_rows], device=device)
    predictions_path = output_dir / "predictions.jsonl"
    method_counts = {name: [] for name in methods}
    with predictions_path.open("w") as handle:
        for record in training_records:
            checkpoint = torch.load(output_dir / record["checkpoint"], map_location=device, weights_only=True)
            model = CompactSceneCNN(len(classes), record["dual_view"], device=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            assert_model_cuda(model)
            torch.cuda.synchronize(); eval_start = time.perf_counter()
            with torch.inference_mode():
                logits = logits_for(model, features, indices["test"], args.batch_size)
                probabilities = logits.softmax(dim=-1)
                predictions = logits.argmax(dim=-1)
                counts = confusion_counts(test_y, predictions, len(classes))
                metrics = plain_metrics(metric_tensors(counts))
                grouped = grouped_confusions(test_y, predictions, group_idx, len(groups), len(classes))
                method_counts[record["method"]].append(grouped)
            torch.cuda.synchronize()
            record["test_metrics"] = metrics
            record["test_confusion"] = counts.tolist()
            record["test_wall_seconds"] = time.perf_counter() - eval_start
            # Conversion here is output serialization; metric arithmetic stayed on CUDA.
            prediction_list = predictions.tolist(); probability_list = probabilities.tolist(); label_list = test_y.tolist()
            for row, predicted, probability, label in zip(test_rows, prediction_list, probability_list, label_list):
                handle.write(json.dumps({"scene_id": row["scene_id"], "group_id": row["group_id"],
                    "domain": row["domain"], "method": record["method"], "seed": record["seed"],
                    "task": "weak_scene_classification_only", "true_scene_label": classes[label],
                    "predicted_scene_label": classes[predicted], "class_names": classes,
                    "class_probabilities_uncalibrated": probability}) + "\n")
            del model, checkpoint
    grouped = {name: torch.stack(values) for name, values in method_counts.items()}
    bootstrap = bootstrap_methods(grouped, draws=args.bootstrap)
    seed_summary = {}
    for method in methods:
        per_seed = metric_tensors(grouped[method].sum(dim=1))
        seed_summary[method] = {
            name: {"values": per_seed[name].tolist(), "mean": per_seed[name].mean().item(),
                   "sample_sd": per_seed[name].std(unbiased=True).item() if len(seeds) > 1 else None}
            for name in ("accuracy", "macro_f1", "macro_f1_all_frozen_classes", "macro_f1_present_reference_classes", "balanced_accuracy")}
    summary = {
        "started_at": started, "completed_at": utc(), "status": "complete",
        "task": "weak_scene_classification_only", "protocol": "protocol.json",
        "feature_timing": feature_timing, "classes": classes,
        "split_scenes": protocol["split_scenes"], "split_independent_groups": protocol["split_independent_groups"],
        "test_group_order": groups,
        "test_class_support": dict(zip(classes, torch.bincount(test_y, minlength=len(classes)).tolist())),
        "test_label_interpretation": "Present-class macro F1 averages only classes with reference support in the evaluated sample. All-vocabulary macro F1 also includes unsupported classes as zero. Neither establishes generalization for a class absent from test truth.",
        "runs": [{k:v for k,v in r.items() if k != "history"} for r in training_records],
        "seed_variability": seed_summary, "group_bootstrap": bootstrap,
        "predictions_sha256": sha256(predictions_path),
        "limitations": protocol["limitations"],
    }
    save_json(output_dir / "summary.json", summary)
    print(json.dumps({"phase": "complete", "output": str(output_dir),
                      "seed_variability": seed_summary}), flush=True)


if __name__ == "__main__":
    main()
