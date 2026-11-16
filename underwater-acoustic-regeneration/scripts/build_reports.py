"""Validate completed artifacts and derive the claim, experiment, and compute ledgers."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = "results/pilot_pools_clipped_seed17"
DIAG = RUN + "_diagnostics"
TRANSFER = "results/transfer_with_waveforms"


def read(path):
    return json.loads((ROOT / path).read_text())


def rows(path):
    return [json.loads(line) for line in (ROOT / path).read_text().splitlines() if line.strip()]


def digest(path):
    checksum = hashlib.sha256()
    with (ROOT / path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def save(path, value):
    (ROOT / path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    pilot = read(RUN + "/pilot_summary.json")
    frozen = read(RUN + "/frozen_predictors.json")
    diagnostics = read(DIAG + "/summary.json")
    transfer = read(TRANSFER + "/summary.json")
    assert diagnostics["status"] == "completed" and transfer["status"] == "completed"
    assert diagnostics["completed_cases"] == 800 and transfer["completed_cases"] == 8
    assert not pilot["actual_test_opened"] and not pilot["actual_calibration_opened"]
    cases = rows(RUN + "/cases.jsonl")
    with gzip.open(ROOT / "manifests/sources.jsonl.gz", "rt") as stream:
        source_manifest = [json.loads(line) for line in stream]
    source_counts = Counter(row["partition"] for row in source_manifest)
    group_partitions = defaultdict(set)
    for row in source_manifest:
        group_partitions[row["group_id"]].add(row["partition"])
    assert all(len(value) == 1 for value in group_partitions.values())
    group_counts = Counter(next(iter(value)) for value in group_partitions.values())
    audit = read("manifests/audit.json")
    assert dict(source_counts) == audit["sources"] and dict(group_counts) == audit["independent_groups"]
    assert Counter(row["pilot_role"] for row in cases) == {"D": 200, "V": 64, "dev_C": 64, "dev_test": 64}
    assert all(row["source_partition"] == ("D" if row["pilot_role"] == "D" else "V") for row in cases)
    assert len({row["group_id"] for row in cases}) == len(cases)
    opportunity = sum(min(candidate["Q"] for candidate in row["candidate_metrics"]) <= row["fallback_metrics"]["Q"] - .02
                      for row in cases if row["pilot_role"] == "V")
    assert opportunity == 5 == frozen["oracle_opportunity_cases"]
    dry = [row for row in cases if row["pilot_role"] == "dev_test"]
    baselines = {}
    for method in ("zero", "linear"):
        values = [row[method + "_metrics"] for row in dry]
        baselines[method] = {"mean_Q": float(np.mean([r["Q"] for r in values])),
                             "mean_gap_nmse": float(np.mean([r["gap_nmse"] for r in values])),
                             "mean_gap_spectral_convergence": float(np.mean([r["gap_spectral_convergence"] for r in values])),
                             "cases": len(values), "scope": "64 development V groups; no C/test outcomes"}
    save(RUN + "/classical_dry_test_summary.json", baselines)

    # Verify all saved arrays against the per-case hashes, including derived data.
    checked = []
    for path, artifact_key, hash_key in [(RUN + "/cases.jsonl", "pool_path", "pool_sha256"),
                                          (DIAG + "/cases.jsonl", "artifact_path", "artifact_sha256"),
                                          (TRANSFER + "/cases.jsonl", "artifact_path", "artifact_sha256")]:
        for row in rows(path):
            assert digest(row[artifact_key]) == row[hash_key], row[artifact_key]
            checked.append(row[artifact_key])
    velocity_rows = []
    if (ROOT / "results/velocity_followup/summary.json").exists():
        velocity_summary = read("results/velocity_followup/summary.json")
        assert velocity_summary["status"] == "completed" and velocity_summary["groups"] == 192
        velocity_rows = rows("results/velocity_followup/cases.jsonl")
        assert len(velocity_rows) == 192 and all(row["source_partition"] == "V" for row in velocity_rows)
        for row in velocity_rows:
            assert digest(row["artifact_path"]) == row["artifact_sha256"], row["artifact_path"]
            checked.append(row["artifact_path"])
        assert read("reports/velocity_training_match.json")["status"] == "passed"
        assert read("reports/velocity_followup_validation.json")["status"] == "passed"
    pairs = read(DIAG + "/hidden_event_pairs.json")
    assert len(pairs) == 400 and all(row["maximum_output_difference"] == 0 for row in pairs)
    assert all(row["false_event_on_absent"] != row["missed_event_on_present"] for row in pairs)
    pair_counts = {}
    for method in ("B", "C1", "C3", "P"):
        subset = [row for row in pairs if row["method"] == method]
        pair_counts[method] = {"pairs": len(subset), "false_events": sum(row["false_event_on_absent"] for row in subset),
                               "missed_events": sum(row["missed_event_on_present"] for row in subset)}
    save(DIAG + "/hidden_event_summary.json", {"projection_threshold": .5, "methods": pair_counts,
                                               "not_a_physical_event_detector": True})
    original_transfer = rows("results/transfer/cases.jsonl")
    replay = rows(TRANSFER + "/cases.jsonl")
    assert len(original_transfer) == len(replay) == 8
    max_metric_difference = 0.0
    differences_by_metric = defaultdict(float)
    for old, new in zip(original_transfer, replay):
        assert old["source_id"] == new["source_id"] and old["sampler_seed"] == new["sampler_seed"]
        for key, value in old["metrics"].items():
            difference = abs(value - new["metrics"][key])
            max_metric_difference = max(max_metric_difference, difference)
            differences_by_metric[key] = max(differences_by_metric[key], difference)
        with np.load(ROOT / new["artifact_path"], allow_pickle=False) as arrays:
            known = arrays["observed_mask"]
            assert np.array_equal(arrays["imputed_mask"], ~known)
            assert np.array_equal(arrays["repaired"][known], arrays["reference"][known])
            assert np.all(arrays["observation"][~known] == 0)
    save("reports/transfer_replay_verification.json", {"cases": 8, "maximum_metric_difference": max_metric_difference,
         "maximum_difference_by_metric": dict(differences_by_metric),
         "bitwise_reproducible": max_metric_difference == 0,
         "limitation": "Fixed Torch seeds do not establish deterministic execution for the legacy transfer network. The first completed run is the primary descriptive transfer score; the artifact replay is retained separately, not averaged as independent evidence.",
         "purpose": "Repeat identical fixed cases/seeds to retain reconstructed waveforms and complete-pipeline timings; not an additional independent experiment",
         "original": "results/transfer", "replay": TRANSFER})
    save("reports/artifact_validation.json", {"status": "completed", "candidate_and_diagnostic_artifacts_checked": len(checked),
         "unique_paths": len(set(checked)), "groups_in_pilot": len(cases), "actual_C_test_opened": False,
         "paired_hidden_target_comparisons": len(pairs), "transferred_waveform_projection_verified": len(replay),
         "targeted_velocity_followup_groups": len(velocity_rows),
         "script_sha256": digest("scripts/build_reports.py")})

    registry = rows("results/run_registry.jsonl")
    completed = [row for row in registry if row["status"] == "completed"]
    failures = [row for row in registry if row["status"] == "failed"]
    by_run = defaultdict(list)
    for row in registry:
        by_run[row["run_id"]].append(row)
    ledger = []
    for run, entries in by_run.items():
        terminal = [row for row in entries if row["status"] in {"completed", "failed", "budget_stopped"}]
        ledger.append({"run_id": run, "final_status": terminal[-1]["status"] if terminal else "running",
                       "all_statuses": [row["status"] for row in entries],
                       "completed_attempts": sum(row["status"] == "completed" for row in terminal),
                       "recorded_failure_entries": sum(row["status"] == "failed" for row in terminal),
                       "measured_attempt_seconds": sum(row.get("elapsed_seconds", 0) for row in terminal),
                       "registry_lines_1based": [i + 1 for i, row in enumerate(registry) if row["run_id"] == run],
                       "output_paths": sorted({row["output_path"] for row in entries if "output_path" in row})})
    for run_id, prefix in [('gpu_agent_benchmark_smoke','results/gpu_agent_benchmark/smoke'),
                           ('gpu_agent_benchmark_evaluation','results/gpu_agent_benchmark/evaluation'),
                           ('gpu_agent_guard_followup','results/gpu_agent_guard/evaluation'),
                           ('gpu_agent_budget_followup','results/gpu_agent_budget/evaluation')]:
        if (ROOT / prefix / 'completion.json').exists():
            finish = read(prefix+'/completion.json')
            start = read(prefix+'/run_definition.json')
            seconds = (datetime.fromisoformat(finish['completed_utc'])-datetime.fromisoformat(start['started_utc'])).total_seconds()
            ledger.append({'run_id':run_id,'final_status':'completed','all_statuses':['running','completed'],
                           'completed_attempts':1,'recorded_failure_entries':0,'measured_attempt_seconds':seconds,
                           'registry_lines_1based':[],'source_registry':prefix+'/run_definition.json',
                           'output_paths':[prefix+'/completion.json'],'episodes':finish['episodes'],
                           'scope':'Completed batch includes unsuccessful agent tasks; task success is reported separately'})
    unrun = ["main three-seed training", "actual C calibration", "actual held-out test", "C4 six-draw medoid",
             "C5 one 192-step draw", "C6 waveform RePaint", "independent ResNet18 utility", "VLIZ external audio",
             "original restoration primary Holm-adjusted comparisons", "core no-projection ablation", "pre-frozen tonal/onset structure metrics",
             "general-audio fine-tuning"]
    save("reports/experiment_ledger.json", {"runs": ledger, "unrun": unrun,
         "unrun_reason": "The 5000-update pilot failed the predeclared opportunity gate. Full three-seed training also exceeds the ten-hour session cap at observed throughput.",
         "failures_retained": True, "source_registry": "results/run_registry.jsonl",
         "local_validation_and_posthoc_registry": "results/local_run_registry.jsonl",
         "provenance_corrections": "reports/provenance_corrections.json"})
    conditional = next(row for row in completed if row["run_id"] == "pilot_conditional_seed17")
    deterministic = next(row for row in completed if row["run_id"] == "pilot_deterministic_seed17")
    cap_training = 3 * (conditional["elapsed_seconds"] * 60000 / 5000 + deterministic["elapsed_seconds"] * 30000 / 2500) / 3600
    step = read(RUN + "/step_comparison.json")
    now = datetime.now(timezone.utc)
    authorization = read("configs/run_authorization.json")
    status = read("reports/status.json")
    completion = datetime.fromisoformat(status["updated_utc"]) if status.get("pilot_complete") else now
    elapsed = (completion - datetime.fromisoformat(authorization["run_start_utc"])).total_seconds()
    assert elapsed < authorization["maximum_run_hours"] * 3600
    compute = {"recorded_utc": now.isoformat(), "completion_utc": completion.isoformat(),
               "session_elapsed_hours_at_completion": elapsed / 3600,
               "authorized_maximum_hours": 10, "gpu": "NVIDIA GeForce RTX 4090",
               "completed_recorded_job_hours": sum(row.get("elapsed_seconds", 0) for row in completed) / 3600,
               "failed_attempt_seconds_recorded": sum(row.get("elapsed_seconds", 0) for row in failures),
               "duration_limitations": "Recorded job sum excludes uninstrumented data/setup checks, some failed startups, and idle time; it is not billed GPU time. Transfer replay is included in compute but is not a second independent sample.",
               "verified_hourly_usd": None, "billed_total_usd": None,
               "cost_status": "Actual booked rate and storage charges unavailable through supplied SSH; no dollar amount inferred from public list prices",
               "pod_instruction": "leave running", "charges_continue_after_jobs_finish": True,
               "projected_maximum_main_conditional_and_deterministic_training_hours_3seeds": cap_training,
               "projection_assumptions": "Linear scaling from measured 5000/2500-update pilots to configured 60000/30000 caps, same hardware, batch size, and I/O behavior; no early stopping assumed",
               "projected_primary_pool_hours_400groups_4gaps_3seeds": 400 * 4 * 3 * step["mean_pipeline_seconds"]["32"] / 3600,
               "projection_not_included": ["D/V/C candidate pools", "unconditional RePaint training/inference", "C4/C5 inference", "classifier", "external archive", "storage and data preparation"],
               "full_matrix_fits_ten_hours": False,
               "recorded_training_forward_batches": sum(row.get("training_forward_batches", 0) for row in completed),
               "recorded_training_backward_batches": sum(row.get("training_backward_batches", 0) for row in completed),
               "saved_main_pilot_pool_denoiser_evaluations": sum(row["denoiser_calls"] for row in cases),
               "step64_comparison_denoiser_evaluations": sum(row["denoiser_calls_64"] for row in rows(RUN + "/step_comparison_cases.jsonl")),
               "saved_synthetic_shift_denoiser_evaluations": sum(row["denoiser_calls"] for row in rows(DIAG + "/cases.jsonl")),
               "velocity_followup_denoiser_evaluations": sum(row.get("forward_evaluations_in_this_invocation", 0) for row in completed if row["run_id"] == "velocity_followup_evaluation"),
               "transfer_forward_evaluations_including_replay": sum(row["denoiser_forward_evaluations"] for row in original_transfer + replay),
               "transfer_backward_evaluations_including_replay": sum(row["backward_evaluations"] for row in original_transfer + replay),
               "call_count_units": "Training counts are microbatches; sampling counts are per-draw denoiser evaluations. These exclude some smoke, resume, validation, and G-only numerical diagnostics and are not a complete billing proxy.",
               "utilization_log_scope": "Per-minute training snapshots; inference peak allocation and duration recorded separately. Continuous inference utilization was not logged.",
               "container_digest": None, "container_digest_limitation": "Image digest not exposed through supplied SSH; package, CUDA, Python, and driver versions recorded"}
    if (ROOT / 'results/cpu_agent_demo_audited/summary.json').exists():
        local_agent = read('results/cpu_agent_demo_audited/summary.json')
        compute['additional_local_CPU_agent_after_GPU_study'] = {
            'model': local_agent['model'], 'quantization': 'Q4_0',
            'episode_and_continuation_seconds': local_agent['total_episode_seconds_excluding_researcher_audit_time'],
            'model_turns': 4, 'tool_calls': 7, 'remote_GPU_used': False, 'new_training': False,
            'scope': 'Separate local follow-up after user reported the pod unavailable; original GPU deadline and completion remain unchanged'}
    if (ROOT / 'results/gpu_agent_benchmark/summary.json').exists():
        gpu_agent = read('results/gpu_agent_benchmark/summary.json')
        gpu_auth = read('configs/gpu_agent_authorization.json')
        gpu_finish = datetime.fromisoformat(gpu_agent['gpu_completion']['completed_utc'])
        gpu_hours = (gpu_finish-datetime.fromisoformat(gpu_auth['run_start_utc'])).total_seconds()/3600
        assert gpu_hours < gpu_auth['maximum_run_hours']
        compute['replacement_pod_GPU_agent_comparison'] = {
            'gpu': gpu_auth['observed_gpu'], 'model': 'Qwen/Qwen3-8B', 'dtype': 'bfloat16',
            'cases': gpu_agent['cases'], 'neural_episodes': gpu_agent['neural_episodes'],
            'session_hours_through_evaluation': gpu_hours,
            'neural_inference_seconds': sum(m['total_inference_seconds'] for m in gpu_agent['metrics'].values()),
            'peak_allocated_bytes': gpu_agent['gpu_completion']['peak_allocated_bytes'],
            'utilization_log': 'logs/gpu_agent_utilization.jsonl',
            'utilization_log_scope': 'Minute-spaced snapshots starting during primary evaluation; setup, smoke, and earlier primary inference are not covered',
            'local_model_inference': False, 'new_training': False, 'pod_shutdown': False,
            'scope': 'Separate user-authorized replacement-pod run; original study and CPU demonstration preserved'}
        if (ROOT / 'results/gpu_agent_guard/summary.json').exists():
            guarded = read('results/gpu_agent_guard/summary.json')
            guard_finish = datetime.fromisoformat(guarded['completion']['completed_utc'])
            total_hours = (guard_finish-datetime.fromisoformat(gpu_auth['run_start_utc'])).total_seconds()/3600
            assert total_hours < gpu_auth['maximum_run_hours']
            compute['replacement_pod_GPU_agent_comparison']['guarded_followup'] = {
                'new_groups':16,'neural_episodes':32,
                'inference_seconds':sum(m['total_inference_seconds'] for m in guarded['metrics'].values()),
                'peak_allocated_bytes':guarded['completion']['peak_allocated_bytes'],
                'session_hours_through_all_agent_jobs':total_hours,
                'completed_utc':guarded['completion']['completed_utc']}
        if (ROOT / 'results/gpu_agent_budget/summary.json').exists():
            budgeted = read('results/gpu_agent_budget/summary.json')
            budget_finish = datetime.fromisoformat(budgeted['completion']['completed_utc'])
            final_hours = (budget_finish-datetime.fromisoformat(gpu_auth['run_start_utc'])).total_seconds()/3600
            assert final_hours < gpu_auth['maximum_run_hours']
            compute['replacement_pod_GPU_agent_comparison']['reasoning_budget_followup'] = {
                'reused_primary_tasks':16,'new_neural_episodes':16,
                'inference_seconds':budgeted['metrics']['reasoning_long']['total_inference_seconds'],
                'peak_allocated_bytes':budgeted['completion']['peak_allocated_bytes'],
                'utilization_log':'logs/gpu_agent_budget_utilization.csv',
                'session_hours_through_all_agent_jobs':final_hours,
                'completed_utc':budgeted['completion']['completed_utc']}
            compute['replacement_pod_GPU_agent_comparison']['total_scored_neural_episodes'] = 176
            compute['replacement_pod_GPU_agent_comparison']['total_scored_episodes_including_rules'] = 208
            compute['replacement_pod_GPU_agent_comparison']['unique_recording_groups'] = 48
            compute['replacement_pod_GPU_agent_comparison']['session_hours_to_last_job'] = final_hours
            compute['replacement_pod_GPU_agent_comparison']['total_scored_neural_inference_seconds'] = (
                sum(m['total_inference_seconds'] for m in gpu_agent['metrics'].values())
                + sum(m['total_inference_seconds'] for m in guarded['metrics'].values())
                + budgeted['metrics']['reasoning_long']['total_inference_seconds'])
    save("reports/compute_ledger.json", compute)

    claims = []
    def claim(identifier, text, verdict, paths, scope, run_ids=()):
        claims.append({"id": identifier, "claim": text, "verdict": verdict, "scope": scope,
                       "completed_run_ids": list(run_ids),
                       "evidence": [{"path": path, "sha256": digest(path)} for path in paths]})
    claim("data_counts", "43,208 mono 32 kHz PCM32 sources; Table 1 partition counts", "verified_metadata",
          ["manifests/audit.json", "manifests/sources.jsonl.gz", "reports/data_summary.json", "reports/temporal_audit.json", "reports/temporal_unknown_device_audit.json"],
          "Metadata/decoded-hash audit; not evidence of independent sessions or held-out quality", ["local_artifact_and_table_validation"])
    claim("pilot_feasibility", "5/64 tuning groups pass the oracle improvement criterion at both 32 and 64 steps", "failed_trigger",
          [RUN + "/frozen_predictors.json", RUN + "/step_comparison.json"], "One pilot seed, G-only sampler amendment; no general impossibility claim",
          ["pilot_pools_clipped_seed17_fit", "pilot_pools_clipped_seed17_compare_steps"])
    claim("dry_policy", "No eligible threshold; zero acceptance and P equals B", "verified_development_result",
          [RUN + "/calibration.json", RUN + "/pilot_summary.json"], "Reserved V groups only; accepted-sample risk undefined",
          ["pilot_pools_clipped_seed17_calibration", "pilot_pools_clipped_seed17_analysis"])
    claim("dry_quality", "Table 2 and development tails/paired intervals", "descriptive_only",
          [RUN + "/pilot_summary.json", RUN + "/classical_dry_test_summary.json", RUN + "/secondary_metrics.json", RUN + "/date_sensitivity.json"],
          "64 V groups; exploratory intervals, no primary confirmatory test", ["pilot_pools_clipped_seed17_test", "pilot_pools_clipped_seed17_analysis", "pilot_posthoc_descriptive"])
    acoustic_examples = ROOT / "figures/acoustic_example_selection.json"
    if acoustic_examples.exists():
        claim("acoustic_examples", "Three hydrophone anecdotes with spectrograms, recomputed gap scores, and fifteen listening clips including the fixed first velocity draw", "illustrative_only",
              ["figures/acoustic_example_selection.json"],
              "Cases selected using original C3-minus-B score extremes and a tonal 500-ms V example before the velocity correction; different candidate-selection rules are shown; not representative or a listening study",
              ["pilot_pools_clipped_seed17_test"])
    if (ROOT / "figures/operator_configuration.json").exists():
        claim("operator_review_views", "Recorded cues, supplied labels, two analysis settings, and the 64-case release result are shown separately", "descriptive_display_only",
              ["figures/operator_configuration.json", "figures/operator_decisions.json"],
              "Original offline display artifacts; object identification and physical equipment control are not evaluated; the later CPU agent episode has separate evidence",
              ["acoustic_example_presentation"])
    claim("synthetic_identity", "400 fixed-input method/pair comparisons have exactly equal output", "passed_implementation_check",
          [DIAG + "/summary.json", DIAG + "/hidden_event_summary.json"], "100 distinct hidden-event realizations; same inputs/seeds cannot identify invisible target content",
          ["pilot_pools_clipped_seed17_diagnostics"])
    claim("shift_behavior", "P uses B on all 100 shifted development cases", "verified_development_result",
          [DIAG + "/summary.json"], "No transferred calibration or external generalization claim", ["pilot_pools_clipped_seed17_diagnostics"])
    claim("published_transfer", "Eight fixed V cases run with hash-verified published weights", "compatibility_only",
          [TRANSFER + "/summary.json", TRANSFER + "/cases.jsonl", "provenance/transfer_numpy_compat.json", "reports/transfer_replay_verification.json"],
          "No tuning/fine-tuning; adapted rate/mask handling; not an exact reproduction or superiority comparison",
          ["moliner_zero_shot_8V_artifact_replay"])
    claim("resume", "CUDA next batch/RNG/loss/parameters reproduce exactly", "passed_implementation_check",
          ["reports/cuda_resume_check.json"], "Controlled G verification, not convergence proof")
    claim("sampler", "Unbounded terminal sampler fails; G-only correction is a disclosed approximation", "limitation",
          ["reports/sampler_numerics.json", "reports/sampler_numerics_bounded.json", "reports/sampler_numerics_truncated.json", "preregistration/sampler_amendment.json"],
          "No V-quality endpoint sweep; Gaussian initialization at nonterminal time remains mismatched")
    if velocity_rows:
        claim("velocity_training_control", "A single 5000-update velocity intervention matches the original source manifest, architecture, optimizer settings, and ending training RNG states", "verified_control",
              ["preregistration/velocity_followup.json", "reports/velocity_training_match.json"],
              "One seed; prediction coordinates and implied loss weighting both change", ["pilot_velocity_seed17"])
        claim("velocity_numerical_correction", "The measured local error propagation matches the epsilon and velocity conversion equations; paired G trajectories document the effect", "mechanistic_development_evidence",
              ["reports/velocity_followup_validation.json", "reports/velocity_numerics_terminal.json", "reports/velocity_numerics_amended.json"],
              "Four training sources; numerical stability is distinct from reconstruction utility", ["velocity_followup_evaluation"])
        claim("velocity_acoustic_comparison", "Fixed-draw and medoid comparisons on the same 192 V groups, separated into three 64-group roles", "exploratory_followup",
              ["results/velocity_followup/summary.json", "results/velocity_followup/cases.jsonl", "reports/velocity_followup_validation.json", "reports/velocity_followup_date_sensitivity.json"],
              "Intervention motivated by previously inspected V results; actual C/test untouched; no recalibrated release policy or object-identification claim", ["velocity_followup_evaluation"])
    if (ROOT / 'reports/cpu_agent_validation.json').exists():
        claim('CPU_agent_episode', 'One local Llama 3.1 8B episode executes acoustic tools; an explicit post hoc audit continuation completes missing actions while narrative limitations remain',
              'illustrative_tool_use_and_trace_audit',
              ['preregistration/cpu_agent_demo.json', 'preregistration/cpu_agent_audit_continuation.json',
               'results/cpu_agent_demo_run02/summary.json', 'results/cpu_agent_demo_audited/summary.json',
               'reports/cpu_agent_validation.json', 'reports/cpu_agent_response_audit.json', 'reports/cpu_agent_operator_card.json'],
              'One preselected V case, two display choices, corrective feedback; neither autonomous agent success rate nor identification accuracy; model prose is not a verified operator report',
              ['cpu_agent_demo_run02_seed17', 'cpu_agent_demo_audit_continuation'])
    if (ROOT / 'reports/gpu_agent_validation.json').exists():
        claim('GPU_agent_comparison', 'Five approaches compared on 32 frozen acoustic tasks; every executed measurement and structured report is audited',
              'bounded_task_comparison',
              ['preregistration/gpu_agent_benchmark.json', 'preregistration/gpu_agent_scoring_amendment.json',
               'preregistration/gpu_agent_evaluation_freeze.json', 'results/gpu_agent_benchmark/summary.json',
               'reports/gpu_agent_validation.json', 'provenance/gpu_agent_environment.json', 'figures/gpu_agent_comparison.json'],
              'One Qwen3 model and decoding seed; reused V recordings; equal ceilings with unequal actual cost; rule-defined cue labels, not vessel identification or physical hardware outcomes',
              ['gpu_agent_benchmark_evaluation'])
    if (ROOT / 'reports/gpu_agent_guard_validation.json').exists():
        claim('GPU_agent_configuration_gate', 'A separate same-model comparison tests a feasibility gate on 16 new agent source groups/dates',
              'bounded_followup_comparison',
              ['preregistration/gpu_agent_guard_followup.json','preregistration/gpu_agent_guard_freeze.json',
               'results/gpu_agent_guard/summary.json','reports/gpu_agent_guard_validation.json','reports/gpu_agent_failure_audit.json'],
              'Motivated by the initial trace audit; new agent groups within V, no oracle labels in tools; gate rejection follows from construction and complete-task benefit is evaluated separately',
              ['gpu_agent_guard_followup'])
    if (ROOT / 'reports/gpu_agent_budget_validation.json').exists():
        claim('GPU_agent_reasoning_budget', 'A separate larger-budget control reuses the first sixteen primary tasks and their original reasoning outputs',
              'exploratory_budget_sensitivity',
              ['preregistration/gpu_agent_budget_followup.json','preregistration/gpu_agent_budget_freeze.json',
               'results/gpu_agent_budget/summary.json','reports/gpu_agent_budget_validation.json'],
              'Motivated by frequent forced reasoning closure during primary execution; larger per-turn and total caps change together; no new independent source groups or model; all outcomes retained',
              ['gpu_agent_budget_followup'])
    for identifier, text in [("main_coverage", "Held-out generative coverage >=25%"),
                             ("main_harm", ">=25% relative harm reduction with confidence interval excluding zero"),
                             ("main_gate", "Diagnostics improve harm at matched coverage over duration and dispersion gates"),
                             ("main_noninferiority", "Held-out Q margin 0.01 and macro-F1 margin 0.02"),
                             ("main_projection", "Held-out observed drift <=1e-6"),
                             ("external_generalization", "Zero-adaptation benefit on VLIZ")]:
        claim(identifier, text, "untested", ["preregistration/protocol.json", "reports/pilot_decision.json"], "Required main experiment not run")
    save("reports/claim_to_evidence.json", {"research_complete": False, "submission_ready": False, "claims": claims})

    cells = []
    audit = read("manifests/audit.json")
    for partition in ["G", "D", "V", "C", "test"]:
        for column, pointer in [("Sources", "sources"), ("Groups", "independent_groups")]:
            cells.append({"table": 1, "row": partition, "column": column, "value": audit[pointer][partition],
                          "source": "manifests/audit.json", "json_pointer": "/" + pointer + "/" + partition})
        cells.append({"table": 1, "row": partition, "column": "Pilot use",
                      "value": {"G": 1440, "D": 200, "V": 200, "C": "Unopened", "test": "Unopened"}[partition],
                      "source": "manifests/pilot_sources.jsonl.gz", "derivation": "Count sources by partition; C/test absent"})
    for method, values in {**pilot["policies"], **baselines}.items():
        if method in {"no_probes", "duration_only", "dispersion_only"}:
            continue
        for column, key in [("Error Q", "mean_Q"), ("NMSE", "mean_gap_nmse"), ("AI use", "generative_coverage"), ("Worse", "all_case_harm_rate")]:
            if key in values:
                source = RUN + ("/classical_dry_test_summary.json" if method in baselines else "/pilot_summary.json")
                pointer = "/" + method + "/" + key if method in baselines else "/policies/" + method + "/" + key
                cells.append({"table": 2, "row": method, "column": column, "value": values[key], "source": source, "json_pointer": pointer})
    for method, calls in {"zero": 0, "linear": 0, "B": 0, "C1": 32, "C2": 128, "C3": 192, "full": 192}.items():
        cells.append({"table": 2, "row": method, "column": "Calls", "value": calls,
                      "source": RUN + "/pool_config.json", "derivation": "0/1/4/6 diffusion draws × 32 steps; excludes deterministic restoration pass and feature computation"})
    for method in ("zero", "linear", "B"):
        cells.append({"table": 2, "row": method, "column": "Worse", "value": "—", "derivation": "Baseline row; generative harmful-acceptance rate is not defined"})
    for method in ("zero", "linear"):
        cells.append({"table": 2, "row": method, "column": "AI use", "value": 0, "derivation": "No diffusion output is accepted by this baseline"})
    if velocity_rows:
        reserved = [row for row in velocity_rows if row["pilot_role"] == "dev_test"]
        for label, method, use_medoid in [("Velocity: old sampler", "velocity_amended", False),
                                          ("Velocity: one draw", "velocity_terminal", False),
                                          ("Velocity: central of four", "velocity_terminal", True)]:
            metrics = [row["methods"][method]["candidate_metrics"][row["methods"][method]["medoid"] if use_medoid else 0] for row in reserved]
            values = {"Error Q": float(np.mean([m["Q"] for m in metrics])),
                      "NMSE": float(np.mean([m["gap_nmse"] for m in metrics])), "AI use": 1.0,
                      "Worse": float(np.mean([m["Q"] > row["baseline_metrics"]["Q"] + .02 for m, row in zip(metrics, reserved)])),
                      "Calls": 128 if use_medoid else 32}
            for column, value in values.items():
                cells.append({"table": 2, "row": label, "column": column, "value": value,
                              "source": "results/velocity_followup/cases.jsonl",
                              "derivation": "Mean over the same 64 dev_test V groups; " + method + "; fixed first draw or spectral medoid; no release gate applied"})
    if (ROOT / 'results/gpu_agent_benchmark/summary.json').exists():
        agent = read('results/gpu_agent_benchmark/summary.json')
        labels={'rules':'Fixed rules','direct':'Direct agent','reasoning':'Reasoning agent',
                'self_review':'Self-review','two_agents':'Planner--reviewer'}
        columns={'Setting':'setting_correct','Cue':'cue_correct','Complete':'successes',
                 'Tokens':'mean_generated_tokens','Seconds':'median_elapsed_seconds'}
        for arm,label in labels.items():
            for column,key in columns.items():
                cells.append({'table':3,'row':label,'column':column,'value':agent['metrics'][arm][key],
                              'source':'results/gpu_agent_benchmark/summary.json','json_pointer':'/metrics/'+arm+'/'+key})
        if (ROOT / 'results/gpu_agent_guard/summary.json').exists():
            guarded = read('results/gpu_agent_guard/summary.json')
            for arm,label in [('reasoning','Follow-up: reasoning'),('reasoning_guarded','Follow-up: guarded')]:
                for column,key in columns.items():
                    cells.append({'table':3,'row':label,'column':column,'value':guarded['metrics'][arm][key],
                                  'source':'results/gpu_agent_guard/summary.json','json_pointer':'/metrics/'+arm+'/'+key})
        if (ROOT / 'results/gpu_agent_budget/summary.json').exists():
            budgeted = read('results/gpu_agent_budget/summary.json')
            for arm,label in [('reasoning','Budget: original'),('reasoning_long','Budget: longer')]:
                for column,key in columns.items():
                    cells.append({'table':3,'row':label,'column':column,'value':budgeted['metrics'][arm][key],
                                  'source':'results/gpu_agent_budget/summary.json','json_pointer':'/metrics/'+arm+'/'+key})
    else:
        for mechanism, item in diagnostics["mechanisms"].items():
            if item["cases"] == 20:
                for column, key in [("Cases", "cases"), ("AI use", "coverage"), ("Error Q(P)", "mean_P_Q")]:
                    cells.append({"table": 3, "row": mechanism, "column": column, "value": item[key], "source": DIAG + "/summary.json", "json_pointer": "/mechanisms/" + mechanism + "/" + key})
    save("reports/manuscript_audit/table_cells.json", cells)
    with (ROOT / "reports/experiment_ledger.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["run_id", "final_status", "completed_attempts", "measured_attempt_seconds", "recorded_failure_entries"])
        for row in ledger:
            writer.writerow([row[key] for key in ["run_id", "final_status", "completed_attempts", "measured_attempt_seconds", "recorded_failure_entries"]])
    local_record = {"run_id": "local_artifact_and_table_validation", "status": "completed", "command": sys.argv,
                    "recorded_utc": now.isoformat(), "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain", "--", "."], cwd=ROOT, text=True).strip()),
                    "source_identity_note": "git_commit names base HEAD; script_sha256 identifies the exact executed postprocessing source",
                    "script_sha256": digest("scripts/build_reports.py"), "source_manifest_sha256": digest("manifests/sources.jsonl.gz"),
                    "artifacts_verified": len(checked), "source_rows_verified": len(source_manifest),
                    "output_paths": ["reports/artifact_validation.json", "reports/claim_to_evidence.json", "reports/manuscript_audit/table_cells.json"],
                    "scope": "Verification of existing artifacts and descriptive aggregation; no new model or independent sample"}
    with (ROOT / "results/local_run_registry.jsonl").open("a") as stream:
        stream.write(json.dumps(local_record, sort_keys=True) + "\n")
    print(json.dumps({"verified_artifacts": len(checked), "claims": len(claims), "table_numeric_cells": len(cells),
                      "main_training_projection_hours": cap_training, "transfer_replay_max_difference": max_metric_difference}, indent=2))


if __name__ == "__main__":
    main()
