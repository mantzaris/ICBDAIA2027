"""Analytic fixtures only; no experiment predictions or held-out scene labels."""
import importlib.util
from pathlib import Path
import pytest
import torch

PATH = Path(__file__).resolve().parents[1] / "scripts/evaluate_relation_supplement.py"
SPEC = importlib.util.spec_from_file_location("relation_supplement", PATH)
supplement = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(supplement)


def event(name, frequency, time=(0., 10.), kind="tone", track=None, carrier=None):
    value = dict(id=name, type=kind, time_s=list(time),
                 frequency_hz=list(frequency) if isinstance(frequency, tuple) else [frequency, frequency],
                 frequency_domain="modulation" if kind == "modulation" else "acoustic")
    if track is not None:
        value["frequency_track"] = track
    if carrier is not None:
        value["carrier_band_hz"] = list(carrier)
    return value


def mapping(n):
    return torch.arange(n, device="cuda")


def states(*ids):
    return {name: {"status": "scored", "prediction_index": index} for index, name in enumerate(ids)}


def test_crossing_between_endpoints_has_exact_positive_overlap_duration():
    rising = event("a", (100., 200.), kind="chirp", track=[[0., 100.], [10., 200.]])
    falling = event("b", (100., 200.), kind="chirp", track=[[0., 200.], [10., 100.]])
    assert supplement.tf_overlap_duration(rising, falling).item() == pytest.approx(.2, abs=1e-12)
    result = supplement.geometric_tf_scores([rising, falling], [rising, falling], mapping(2))
    assert (result["tp"], result["fp"], result["fn"]) == (1, 0, 0)


def test_piecewise_knots_and_narrow_crossing_are_not_sampled_away():
    bent = event("a", (90., 110.), kind="chirp",
                 track=[[0., 90.], [3.333, 110.], [3.334, 90.], [10., 90.]])
    fixed = event("b", 100.)
    # Every knot changes slope. Rising duration 0.2*3.333, falling 0.2*0.001.
    assert supplement.tf_overlap_duration(bent, fixed).item() == pytest.approx(.6668, abs=1e-12)
    assert supplement.tf_overlap_duration(fixed, bent).item() == pytest.approx(.6668, abs=1e-12)


def test_disjoint_frequency_time_and_boundary_contacts_are_not_overlap():
    a = event("a", 100.)
    assert supplement.tf_overlap_duration(a, event("b", 102.)).item() == 0.
    assert supplement.tf_overlap_duration(a, event("b", 103.)).item() == 0.
    assert supplement.tf_overlap_duration(a, event("b", 100., time=(10., 12.))).item() == 0.
    assert supplement.tf_overlap_duration(a, event("b", 101.999)).item() == 10.
    assert supplement.temporal_adjacency([a, event("b", 999.)])[0, 1].item()


def test_broadband_declared_band_and_zero_width_region():
    band = event("a", (95., 105.), kind="broadband")
    chirp = event("b", (90., 110.), kind="chirp", track=[[0., 90.], [10., 110.]])
    assert supplement.tf_overlap_duration(band, chirp).item() == pytest.approx(6., abs=1e-12)
    assert supplement.tf_overlap_duration(event("z", (100., 100.), kind="transient"), chirp).item() == 0.


def test_domains_and_modulation_carriers_must_be_compatible():
    a = event("a", 12., kind="modulation", carrier=(500., 2000.))
    b = event("b", 12., kind="modulation", carrier=(2000., 5000.))
    c = event("c", 12., kind="modulation", carrier=(1250., 2750.))
    d = event("d", 12., kind="modulation", carrier=(1251., 2751.))
    assert supplement.tf_overlap_duration(a, b).item() == 0.
    assert supplement.tf_overlap_duration(a, c).item() == 10.
    assert supplement.tf_overlap_duration(a, d).item() == 0.
    assert supplement.tf_overlap_duration(a, event("acoustic", 12.)).item() == 0.


def test_explicit_temporal_edges_are_not_invented_from_geometry():
    gold = [event("a", 100.), event("b", 200.)]
    result = supplement.explicit_temporal_scores({"relations": []}, gold, mapping(2), states("a", "b"))
    assert (result["tp"], result["fp"], result["fn"]) == (0, 0, 1)
    result = supplement.explicit_temporal_scores({"relations": [
        {"type": "overlaps_time", "component_ids": ["b", "a"]}]}, gold, mapping(2), states("a", "b"))
    assert (result["tp"], result["fp"], result["fn"]) == (1, 0, 0)


def test_dangling_unmatched_duplicate_self_and_malformed_edges_are_false_positives():
    gold = [event("a", 100.), event("b", 200.)]
    node_map = torch.tensor([0, 1, -1], device="cuda")
    pairs = [["a", "b"], ["b", "a"], ["a", "missing"], ["a", "unmatched"], ["a", "a"], ["a"]]
    report = {"relations": [{"type": "overlaps_time", "component_ids": pair} for pair in pairs]}
    result = supplement.explicit_temporal_scores(report, gold, node_map, states("a", "b", "unmatched"))
    assert (result["tp"], result["fp"], result["fn"]) == (1, 5, 0)


def test_optional_unenumerated_and_ambiguous_endpoint_handling():
    gold = [event("a", 100.), event("b", 200.)]
    id_states = states("a", "b") | {"optional": {"status": "unscored"},
                                     "unmodeled": {"status": "unscored"},
                                     "duplicate_id": {"status": "ambiguous"}}
    pairs = [["a", "optional"], ["a", "unmodeled"], ["a", "duplicate_id"], ["missing", "optional"]]
    report = {"relations": [{"type": "overlaps_time", "component_ids": pair} for pair in pairs]}
    result = supplement.explicit_temporal_scores(report, gold, mapping(2), id_states)
    assert (result["tp"], result["fp"], result["fn"]) == (0, 2, 1)
    assert result["ignored_out_of_scope_edges"] == 2


def test_multi_endpoint_relation_expands_as_declared_clique():
    gold = [event("a", 100.), event("b", 200.), event("c", 300.)]
    report = {"relations": [{"type": "overlaps_time", "component_ids": ["a", "b", "c"]}]}
    result = supplement.explicit_temporal_scores(report, gold, mapping(3), states("a", "b", "c"))
    assert (result["tp"], result["fp"], result["fn"]) == (3, 0, 0)


def test_geometry_unmatched_edges_are_false_positives_optional_nodes_are_ignored():
    gold = [event("a", 100.), event("b", 100.)]
    predicted = gold + [event("extra", 100.)]
    node_map = torch.tensor([0, 1, -1], device="cuda")
    result = supplement.geometric_tf_scores(predicted, gold, node_map)
    assert (result["tp"], result["fp"], result["fn"]) == (1, 2, 0)
    result = supplement.geometric_tf_scores(predicted, gold, node_map, ignored_predictions=[2])
    assert (result["tp"], result["fp"], result["fn"]) == (1, 0, 0)


def test_id_states_preserve_raw_ids_and_frozen_scoring_scope():
    raw = [event("a", 100.), event("b", 110.),
           event("m", 12., kind="modulation", carrier=(500., 2000.)),
           event("duplicate", 150.), event("duplicate", 160.),
           dict(id="invalid", type="chirp", time_s=[0., 10.], frequency_hz=[100., 200.])]
    report = {"components": raw}
    reference = {"modulation_exhaustive": False}
    predictions, invalid, _ = supplement.common.report_events(report)
    scoped, _, _ = supplement.common.score_scope(predictions, invalid, report, reference)
    id_states = supplement.report_id_states(report, reference, predictions, invalid, scoped,
                                            {"ignored_predictions": [1]})
    assert id_states["a"] == {"status": "scored", "prediction_index": 0}
    assert id_states["b"]["status"] == "unscored"
    assert id_states["m"]["status"] == "unscored"
    assert id_states["duplicate"]["status"] == "ambiguous"
    assert id_states["invalid"]["status"] == "invalid"


def test_empty_graph_is_zero_f1_not_perfect_accuracy():
    result = supplement.explicit_temporal_scores({}, [], mapping(0), {})
    assert (result["tp"], result["fp"], result["fn"], result["f1"]) == (0, 0, 0, 0.)
    assert result["precision"] is None and result["recall"] is None
