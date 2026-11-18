#!/usr/bin/env python3
"""Package an offline interactive viewer from recorded research artifacts.

The bundle uses local files and embedded JSON; it requires no web service. CUDA
export is optional only when reusing already exported rasters. No experiment
metrics, annotations, or model outputs are recomputed by this packaging step.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).resolve().parent / "viewer_assets"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def export_view_cuda(view, directory, name):
    """GPU display transformations, followed only by host raster rendering."""
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(ROOT / "src"))
    from sonar.signal import require_cuda
    require_cuda(view.psd)
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    power = view.log_power()
    # This is a labelled display-only contrast transform, never a PSD or score.
    contrast = power - power.median(dim=0, keepdim=True).values
    maximum, minimum = power.max(), power.min()
    raw_limits = torch.stack((maximum - 60., maximum))
    if bool(maximum == minimum):
        raw_limits = torch.stack((maximum, maximum + 60.))
    contrast_limits = torch.quantile(contrast.flatten(), power.new_tensor([.01, .995]))
    if bool(contrast_limits[0] == contrast_limits[1]):
        contrast_limits[1] = contrast_limits[0] + 1.
    native_shape = list(power.shape)
    output_shape = (min(512, power.shape[0]), min(512, power.shape[1]))
    paths = {}
    for mode, values, limits in (("raw", power, raw_limits), ("contrast", contrast, contrast_limits)):
        if values.shape != output_shape:
            values = torch.nn.functional.adaptive_max_pool2d(values[None, None], output_shape)[0, 0]
        path = directory / f"{name}_{mode}.png"
        plt.imsave(path, values.cpu().numpy(), cmap="magma", origin="lower",
                   vmin=limits[0].item(), vmax=limits[1].item())
        paths[mode] = {"path": str(path.resolve()), "sha256": digest(path), "limits_db": limits.tolist()}
    df = view.frequencies_hz.new_tensor(view.metadata["bin_spacing_hz"])
    dt = view.times_s.new_tensor(view.metadata["hop_s"])
    frequency_edges = torch.stack(((view.frequencies_hz[0] - df / 2).clamp_min(0), view.frequencies_hz[-1] + df / 2))
    time_edges = torch.stack((view.times_s[0] - dt / 2, view.times_s[-1] + dt / 2))
    result = {"name": name, "rasters": paths, "time_range_s": time_edges.tolist(),
              "frequency_range_hz": frequency_edges.tolist(), "metadata": view.metadata,
              "frequency_domain": view.metadata["frequency_domain"],
              "carrier_band_hz": view.metadata.get("carrier_band_hz"),
              "native_shape": native_shape, "display_shape": list(output_shape),
              "display_aggregation": "CUDA adaptive maximum pooling; native measurements unchanged",
              "raw_units": "dB re digital amplitude squared/Hz",
              "contrast_units": "dB relative to each frame's median over frequency; not absolute power",
              "numeric_device": str(power.device)}
    write_json(directory / f"{name}_view.json", result)
    return result


def copy_asset(source, output, destination):
    source = Path(source).resolve()
    target = (output / destination).resolve()
    if output != target and output not in target.parents:
        raise ValueError("Asset target escaped viewer directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    if source != target:
        shutil.copy2(source, target)
    return {"url": target.relative_to(output).as_posix(), "sha256": digest(target)}


def build_viewer(project_root, output, run, selection, references_path, manifest_path,
                 views_by_scene=None, method="F", allow_unmapped_replay=False):
    root, output, run = Path(project_root).resolve(), Path(output).resolve(), Path(run).resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = {r["scene_id"]: r for r in (json.loads(line) for line in Path(manifest_path).read_text().splitlines() if line.strip())}
    references = {r["scene_id"]: r for r in (json.loads(line) for line in Path(references_path).read_text().splitlines() if line.strip())} if references_path and Path(references_path).exists() else {}
    source_files = {str(Path(manifest_path).resolve()): digest(manifest_path)}
    if references_path and Path(references_path).exists():
        source_files[str(Path(references_path).resolve())] = digest(references_path)
    cases = []
    chosen_rows = [(key, value) for key, value in selection.items() if isinstance(value, dict) and "scene_id" in value]
    for key, chosen in chosen_rows:
        sid = chosen["scene_id"]
        if sid not in records:
            raise ValueError(f"Unknown selected scene {sid}")
        record = records[sid]
        folder = run / sid / method
        episode_path = folder / "episode.json"
        failure_path = folder / "failure.json"
        if episode_path.exists():
            episode = read_json(episode_path); source_files[str(episode_path)] = digest(episode_path)
        elif failure_path.exists():
            episode = {"scene_id": sid, "method": method, "failure": read_json(failure_path), "actions": []}
            source_files[str(failure_path)] = digest(failure_path)
        else:
            raise FileNotFoundError(f"No recorded episode/failure for {sid}/{method}")
        asset_dir = Path("assets") / sid
        views = []
        if views_by_scene and sid in views_by_scene:
            for name, view in views_by_scene[sid].items():
                views.append(export_view_cuda(view, output / asset_dir, name))
        else:
            views = [read_json(path) for path in sorted((output / asset_dir).glob("*_view.json"))]
        views.sort(key=lambda view: (view["frequency_domain"] != "acoustic", (view.get("carrier_band_hz") or [0])[0], view["name"]))
        for view in views:
            for mode, raster in view["rasters"].items():
                source = Path(raster["path"])
                if not source.exists():
                    source = output / asset_dir / source.name
                view["rasters"][mode] = dict(raster, **copy_asset(source, output, asset_dir / source.name))
        if not views and not allow_unmapped_replay:
            raise RuntimeError(f"No GPU-exported coordinate-mapped rasters for {sid}; render figures first")
        audio = None
        if record.get("path") and (root / record["path"]).is_file():
            audio = copy_asset(root / record["path"], output, asset_dir / "audio.wav")
        actions = []
        for action in episode.get("actions", []):
            images = [copy_asset(path, output, asset_dir / path.name) for path in sorted(folder.glob(f"action_{action.get('step')}_*.png"))]
            actions.append(dict(action, saved_images=images))
        global_images = [dict(copy_asset(path, output, asset_dir / path.name), name=path.stem)
                         for path in sorted(folder.glob("global_*.png"))]
        diagnostic_annotations = []
        if references_path:
            for annotator in ("internvl4b", "internvl8b"):
                annotation_path = Path(references_path).parent / annotator / (sid + ".json")
                if annotation_path.exists():
                    source_files[str(annotation_path.resolve())] = digest(annotation_path)
                    diagnostic_annotations.append({"annotator": annotator, "scope": "unqualified_diagnostic_only",
                        "record": read_json(annotation_path), "sha256": digest(annotation_path)})
        cases.append({"key": key, "selection": chosen, "scene_id": sid, "method": method,
                      "domain": record.get("domain", record.get("stratum")), "record": record,
                      "time_s": episode.get("final_state", {}).get("time_s", [record.get("source_start_seconds", 0), record.get("source_start_seconds", 0) + record.get("duration_seconds", 30)]),
                      "views": views, "audio": audio, "recorded_global_images": global_images,
                      "actions": actions, "report": episode.get("report"), "raw_report": episode.get("raw_report"),
                      "measurements": episode.get("final_state", {}).get("measurements", []),
                      "measurement_components": episode.get("final_state", {}).get("components", []),
                      "verification_audit": episode.get("verification_audit", []),
                      "reference": references.get(sid), "diagnostic_annotations": diagnostic_annotations,
                      "failure": episode.get("failure"),
                      "mapped_view_status": "GPU_exported" if views else "recorded_images_only_no_coordinate_mapping"})
    annotation_completion = Path(references_path).parent / "COMPLETED.json" if references_path else None
    if annotation_completion and annotation_completion.exists():
        source_files[str(annotation_completion.resolve())] = digest(annotation_completion)
    payload = {"version": "interactive-case-viewer-v1", "created_at": datetime.now(timezone.utc).isoformat(),
               "cases": cases, "selection": selection, "source_files": source_files,
               "reference_meaning": "Eligibility-limited AI agreement; shared model ancestry, no independent real physical truth.",
               "interaction_scope": "Browser cursor, geometric selection, replay and separate local corrections only. No scientific metrics computed in browser.",
               "annotation_completion_marker_present": bool(annotation_completion and annotation_completion.exists()),
               "offline": True, "public_hosting": False}
    write_json(output / "viewer_bundle.json", payload)
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    page = (ASSETS / "index.html").read_text().replace("__CASE_DATA__", encoded)
    (output / "index.html").write_text(page)
    for filename in ("app.js", "style.css"):
        shutil.copy2(ASSETS / filename, output / filename)
    (output / "README.txt").write_text(
        "Open index.html in a browser. All case data, images and audio are local; no server or GPU is needed for replay.\n"
        "The cursor links original-recording seconds across LOFAR/DEMON. Select a component to inspect overlapping intervals and cited evidence.\n"
        "Drag across a plot to select an interval. Raw/contrast switches change display only. Recorded action images remain unmodified.\n"
        "Local correction notes are stored separately in this browser and may be exported; they never overwrite benchmark annotations.\n"
        "Optional new measurements require scripts/serve_viewer_gpu.py on an available CUDA machine, bound to 127.0.0.1 only.\n"
        "Exploratory reinspection is excluded from the frozen benchmark. This local audio package does not assert redistribution permission.\n")
    # Subsequent software checks and exploratory measurements are separate from
    # this frozen replay package; adding them must not stale its source hashes.
    excluded = {"case_manifest.json", "browser_check.json", "browser_check.png"}
    inventory = {str(path.relative_to(output)): digest(path) for path in sorted(output.rglob("*"))
                 if path.is_file() and path.name not in excluded
                 and "reinspections" not in path.relative_to(output).parts}
    write_json(output / "case_manifest.json", {"version": payload["version"], "selection": selection,
        "files_sha256": inventory, "sources": source_files, "publication": "local_only_not_hosted",
        "excluded_from_frozen_bundle": ["browser_check.json", "browser_check.png", "reinspections/**"],
        "complete_coordinate_mapping": all(case["views"] for case in cases),
        "builder_sha256": digest(Path(__file__))})
    return {"path": str(output / "index.html"), "cases": len(cases), "coordinate_mapped": all(case["views"] for case in cases)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=Path("viewer"))
    parser.add_argument("--run", type=Path, default=Path("results/main"))
    parser.add_argument("--selected-cases", type=Path, default=Path("figures/selected_cases.json"))
    parser.add_argument("--references", type=Path, default=Path("results/ai_annotations/primary_reference.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/main_test.jsonl"))
    parser.add_argument("--method", default="F")
    parser.add_argument("--scene-id")
    parser.add_argument("--allow-unmapped-replay", action="store_true")
    parser.add_argument("--export-cuda-views", action="store_true",
                        help="Compute bounded fixed views from selected source waveforms on CUDA before packaging")
    args = parser.parse_args()
    root = args.project_root.resolve()
    chosen = {"pilot": {"scene_id": args.scene_id, "purpose": "Explicit development replay", "criterion_met": False}} if args.scene_id else read_json(root / args.selected_cases)
    prepared = None
    if args.export_cuda_views:
        import os
        os.chdir(root)
        sys.path.insert(0, str(root / "src"))
        from sonar.data import load_waveform
        from sonar.signal import require_cuda, lofar, demon
        require_cuda()
        records = {r["scene_id"]: r for r in (json.loads(line) for line in (root / args.manifest).read_text().splitlines() if line.strip())}
        prepared = {}
        for sid in sorted({row["scene_id"] for row in chosen.values() if isinstance(row, dict) and "scene_id" in row}):
            waveform, rate, origin, _ = load_waveform(records[sid])
            prepared[sid] = {"lofar": lofar(waveform, rate, window_s=1., hop_s=.25, offset_s=origin)}
            for name, band in (("demon_low", (500., 2000.)), ("demon_high", (2000., 5000.))):
                try:
                    prepared[sid][name] = demon(waveform, rate, band, offset_s=origin)
                except ValueError:
                    pass  # Unavailable carriers never become a fabricated view.
    print(json.dumps(build_viewer(root, root / args.output, root / args.run, chosen, root / args.references,
                                 root / args.manifest, views_by_scene=prepared, method=args.method,
                                 allow_unmapped_replay=args.allow_unmapped_replay)))


if __name__ == "__main__":
    main()
