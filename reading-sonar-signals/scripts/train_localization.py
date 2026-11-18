#!/usr/bin/env python3
"""Three-seed CUDA morphology training and frozen held-out localization.

Auxiliary exact synthetic training is a separately reported development pool; it
does not change the declared real-data fraction of the dense benchmark. Real AI
annotations are not used for training or negative labels. Exported real predictions
can be evaluated as AI-reference agreement by the separate benchmark evaluator.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from sonar.data import load_waveform, read_manifest
from sonar.synthetic import generate_scene, KINDS
from sonar.localization import (CONFIG, DenseMorphologyCNN, localizer_features, dense_targets,
                                segmentation_loss, decode_components, auxiliary_broadband_scene, MAX_COMPONENTS)
from sonar.metrics import event_metrics, aggregate_event_metrics
from sonar.signal import require_cuda


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_development(count, split, seed_start):
    features, targets, valid, truths, views_all = [], [], [], [], []
    for index in range(count):
        snr = (-5., 0., 5., 10., 20.)[index % 5]
        pool=(*KINDS,"auxiliary_broadband")
        kind=pool[index%len(pool)]
        if kind=="auxiliary_broadband":
            waveform,truth=auxiliary_broadband_scene(seed_start+index,split,snr)
        else:
            waveform, truth = generate_scene(seed_start + index, split, kind, snr_db=snr)
        with torch.inference_mode():
            feature, views = localizer_features(waveform, 16000)
            target, known = dense_targets(views, truth["events"], truth.get("modulation_exhaustive", False))
        features.append(feature); targets.append(target); valid.append(known)
        truths.append(truth)
        # Validation keeps native measured grids for threshold selection. Training
        # does not need to retain its native PSD tensors once target masks exist.
        if split == "validation":
            views_all.append(views)
        if (index + 1) % 24 == 0:
            print(json.dumps(dict(phase="auxiliary_features", split=split, complete=index + 1, total=count)), flush=True)
    return torch.stack(features), torch.stack(targets), torch.stack(valid), truths, views_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-manifest", default="data/manifests/main_test.jsonl")
    parser.add_argument("--output", default="results/localization")
    parser.add_argument("--train-scenes", type=int, default=192)
    parser.add_argument("--validation-scenes", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--seeds", default="17,29,43")
    args = parser.parse_args()
    device = require_cuda()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    if (output / "FROZEN_MODELS.json").exists():
        raise RuntimeError("This output already contains frozen models; choose a new run directory")
    started = time.time()
    config = dict(**vars(args), features=CONFIG, training_seed_start=10000,
                  validation_seed_start=20000, auxiliary_training_outside_dense_benchmark=True,
                  train_labels="exact_simulator_masks", real_AI_labels_used_for_training=False,
                  auxiliary_kind_pool=[*KINDS,"auxiliary_broadband"],max_output_components=MAX_COMPONENTS,
                  normalization="train_only_branch_global_mean_and_std", threshold_candidates=[.25, .5, .75],
                  checkpoint_selection="minimum_validation_weighted_BCE_plus_Dice",
                  threshold_selection="maximum_validation_required_event_microF1",
                  hypotheses="secondary_supervised_localization_baselines_not_confirmatory_main_comparison",
                  source_sha256={str(p.relative_to(ROOT)): sha(p) for p in
                                 [ROOT / "src/sonar/localization.py", ROOT / "src/sonar/signal.py",
                                  ROOT / "src/sonar/synthetic.py", ROOT / "src/sonar/metrics.py", Path(__file__)]},
                  started_at_unix=started)
    save_json(output / "configuration.json", config)
    train_x, train_y, train_valid, train_truth, _ = make_development(args.train_scenes, "train", 10000)
    val_x, val_y, val_valid, val_truth, val_views = make_development(args.validation_scenes, "validation", 20000)
    save_json(output / "auxiliary_scene_manifest.json", {
        "train": [{k: row[k] for k in ("scene_id", "generator_seed", "kind", "split", "snr_db")} for row in train_truth],
        "validation": [{k: row[k] for k in ("scene_id", "generator_seed", "kind", "split", "snr_db")} for row in val_truth]})
    mean = train_x.mean(dim=(0, 2, 3), keepdim=True)
    std = train_x.std(dim=(0, 2, 3), unbiased=False, keepdim=True).clamp_min(1e-6)
    train_x = (train_x - mean) / std
    val_x = (val_x - mean) / std
    positives = (train_y * train_valid).sum(dim=(0, 2, 3))
    negatives = ((1 - train_y) * train_valid).sum(dim=(0, 2, 3))
    positive_weight = (negatives / positives.clamp_min(1)).clamp(1, 40)
    if not bool((positives>0).all()):
        raise RuntimeError(f"Unavailable intended training channels: positive valid mask counts={positives.tolist()}")
    save_json(output / "training_label_accounting.json", dict(
        positive_pixels=positives.tolist(), negative_pixels=negatives.tolist(),
        ignored_pixels=(1 - train_valid).sum(dim=(0, 2, 3)).tolist(),
        positive_weight=positive_weight.tolist(), exact_truth_scope="declared_simulated_observables",
        intended_channels=["tone","chirp","transient","broadband","modulation_500_2000Hz","modulation_2000_5000Hz"],
        all_intended_channels_have_positive_valid_masks=True,
        auxiliary_kind_counts={split:{kind:sum(t["kind"]==kind for t in truths)
            for kind in [*KINDS,"auxiliary_broadband"]} for split,truths in [("train",train_truth),("validation",val_truth)]}))
    frozen, training_records = [], []
    for seed in map(int, args.seeds.split(",")):
        for method, dual in (("P1_localizer", False), ("P2_localizer", True)):
            torch.cuda.manual_seed_all(seed)
            generator = torch.Generator(device=device).manual_seed(seed)
            model = DenseMorphologyCNN(dual_view=dual)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.01)
            best_loss, best_epoch, stale = float("inf"), 0, 0
            best_state = None
            history = []
            cuda_start, cuda_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(); begin = time.perf_counter(); cuda_start.record()
            for epoch in range(1, args.epochs + 1):
                model.train()
                order = torch.randperm(train_x.shape[0], device=device, generator=generator)
                train_losses = []
                for indices in order.split(args.batch_size):
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(train_x[indices])
                    loss = segmentation_loss(logits, train_y[indices], train_valid[indices], positive_weight)
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite localizer training loss")
                    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
                    train_losses.append(loss.detach())
                model.eval()
                with torch.inference_mode():
                    val_losses = [segmentation_loss(model(val_x[i:i + args.batch_size]),
                                                    val_y[i:i + args.batch_size], val_valid[i:i + args.batch_size],
                                                    positive_weight) for i in range(0, val_x.shape[0], args.batch_size)]
                    val_loss = torch.stack(val_losses).mean().item()
                history.append(dict(epoch=epoch, training_loss=torch.stack(train_losses).mean().item(), validation_loss=val_loss))
                if val_loss < best_loss - 1e-5:
                    best_loss, best_epoch, stale = val_loss, epoch, 0
                    best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
                else:
                    stale += 1
                if epoch % 5 == 0:
                    print(json.dumps(dict(phase="localizer_training", method=method, seed=seed, **history[-1])), flush=True)
                if stale >= args.patience:
                    break
            model.load_state_dict(best_state); model.eval()
            with torch.inference_mode():
                probabilities = torch.cat([model(val_x[i:i + args.batch_size]).sigmoid()
                                           for i in range(0, val_x.shape[0], args.batch_size)])
                threshold_results = []
                for threshold in (.25, .5, .75):
                    scores = []
                    for index in range(val_x.shape[0]):
                        predictions = decode_components(probabilities[index], val_views[index], threshold)
                        if not val_truth[index].get("modulation_exhaustive", False):
                            predictions = [p for p in predictions if p.get("frequency_domain") != "modulation"]
                        scores.append(event_metrics(predictions, val_truth[index]["events"]))
                    threshold_results.append(dict(threshold=threshold, **aggregate_event_metrics(scores)))
                selector = torch.tensor([row["f1"] for row in threshold_results], device=device)
                selected = threshold_results[int(selector.argmax())]
            cuda_end.record(); cuda_end.synchronize()
            name = f"{method}_seed{seed}"
            checkpoint = output / "checkpoints" / f"{name}.pt"; checkpoint.parent.mkdir(exist_ok=True)
            torch.save(dict(model_state={k: v.cpu() for k, v in best_state.items()},
                            mean=mean.cpu(), std=std.cpu(), dual_view=dual, threshold=selected["threshold"],
                            method=method, seed=seed, config=CONFIG,max_output_components=MAX_COMPONENTS), checkpoint)
            record = dict(method=method, seed=seed, best_epoch=best_epoch, history=history,
                          parameter_count=sum(p.numel() for p in model.parameters()),
                          threshold_results=threshold_results, selected_threshold=selected["threshold"],
                          wall_seconds=time.perf_counter() - begin,
                          cuda_event_elapsed_seconds=cuda_start.elapsed_time(cuda_end) / 1000,
                          cuda_timing_note="elapsed CUDA event span includes host launch gaps, not occupancy-based active time",
                          checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint))
            training_records.append(record); save_json(output / "training_records.json", training_records)
            frozen.append(dict(name=name, model=model, threshold=selected["threshold"], record=record))
            print(json.dumps(dict(phase="localizer_frozen", method=method, seed=seed, threshold=selected["threshold"],
                                  validation_f1=selected["f1"])), flush=True)
    # All test labels remain unopened until every seed/method checkpoint and its
    # threshold has been selected and the complete frozen manifest is persisted.
    save_json(output / "FROZEN_MODELS.json", dict(frozen_at_unix=time.time(), models=[f["record"] for f in frozen],
                                                 evaluation_manifest_sha256=sha(args.evaluation_manifest)))
    records = read_manifest(args.evaluation_manifest)
    save_json(output / "predictions" / "run.json", dict(
        manifest=args.evaluation_manifest, manifest_sha256=sha(args.evaluation_manifest),
        scene_count=len(records), methods=[entry["name"] for entry in frozen],
        started_at_unix=time.time(), task="supervised_component_localization", generative=False,
        reference_annotations_visible_to_model=False,
        source_sha256=config["source_sha256"],
        checkpoint_sha256={entry["name"]:entry["record"]["checkpoint_sha256"] for entry in frozen},
        explanation_quality="not_applicable; these models produce no natural-language explanation",
        training_truth="auxiliary_exact_synthetic_only"))
    exact_scores = {f["name"]: [] for f in frozen}
    for index, row in enumerate(records):
        waveform, sr, origin, truth = load_waveform(row)
        if truth is not None:
            saved_truth=dict(truth)
            if saved_truth.get("scene_id") not in (None,row["scene_id"]):
                saved_truth["generator_scene_id"]=saved_truth["scene_id"]
            saved_truth["scene_id"]=row["scene_id"]
            save_json(output / "predictions" / "sealed_truth" / f"{row['scene_id']}.json",saved_truth)
        with torch.inference_mode():
            pre_begin, pre_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            pre_begin.record()
            feature, views = localizer_features(waveform, sr, origin)
            normalized = (feature[None] - mean) / std
            pre_end.record(); pre_end.synchronize()
            preprocessing_seconds = pre_begin.elapsed_time(pre_end) / 1000
            for entry in frozen:
                torch.cuda.synchronize(); start = time.perf_counter()
                inference_begin, inference_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                inference_begin.record()
                probabilities = entry["model"](normalized).sigmoid()[0]
                predictions = decode_components(probabilities, views, entry["threshold"])
                inference_end.record(); inference_end.synchronize()
                report=dict(components=predictions, relations=[], claims=[], unresolved_questions=[],
                            abstained=not bool(predictions))
                result = dict(scene_id=row["scene_id"], method=entry["name"],
                              report=report, raw_report=report,
                              schema_valid=True, generative=False, training_truth="auxiliary_exact_synthetic",
                              explanation_quality="not_applicable; no natural-language explanations generated",
                              signal_processing_device=str(device), wall_seconds=time.perf_counter() - start,
                              model_gpu_seconds=inference_begin.elapsed_time(inference_end) / 1000,
                              tool_gpu_seconds=0., preprocess_gpu_seconds=preprocessing_seconds,
                              input_tokens=0, output_tokens=0, actions=[], generations=[],
                              model_and_decoding_gpu_elapsed_seconds=inference_begin.elapsed_time(inference_end) / 1000,
                              shared_preprocessing_gpu_elapsed_seconds=preprocessing_seconds,
                              preprocessing_accounting="one common feature pass reused across all six models; do not sum this repeated field",
                              localization_checkpoint_sha256=entry["record"]["checkpoint_sha256"])
                if truth and truth.get("stratum") == "synthetic":
                    scored_predictions = predictions if truth.get("modulation_exhaustive", False) else [
                        p for p in predictions if p.get("frequency_domain") != "modulation"]
                    score = event_metrics(scored_predictions, truth["events"])
                    exact_scores[entry["name"]].append(score)
                    result["synthetic_exact_metrics"] = score
                save_json(output / "predictions" / row["scene_id"] / entry["name"] / "episode.json", result)
        if (index + 1) % 10 == 0:
            print(json.dumps(dict(phase="localizer_evaluation", complete=index + 1, total=len(records))), flush=True)
    save_json(output / "synthetic_summary.json", {name: aggregate_event_metrics(rows) for name, rows in exact_scores.items()})
    save_json(output / "COMPLETED.json", dict(started_at_unix=started, finished_at_unix=time.time(),
                                              models=len(frozen), evaluation_scenes=len(records),
                                              real_reference_correctness="Not assessed here; use separate AI-reference agreement evaluation"))
    print("LOCALIZATION_FINISHED", flush=True)


if __name__ == "__main__":
    main()
