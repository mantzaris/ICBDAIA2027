"""Reference validation protects original coordinates and modulation semantics."""
from copy import deepcopy
from sonar.annotators import validate_annotation,annotation_prompt,parse_complete_event_rows


def example():
    return dict(events=[dict(id="a0",type="tone",frequency_domain="acoustic",
                            time_s=[1000.,1025.],frequency_hz=[120.,120.],
                            frequency_track=[[1000.,120.],[1025.,120.]],
                            frequency_uncertainty_hz=2.,boundary_uncertainty_s=1.)],
                relations=[],reviewed_regions=[],inspected_no_events=False)


def test_original_coordinates_do_not_silently_reset():
    annotation=example()
    assert validate_annotation(annotation,1000.,30.)["annotation_status"]=="complete"
    assert validate_annotation(annotation,0.,30.)["annotation_status"]=="partial_or_invalid"


def test_modulation_needs_domain_and_carrier():
    annotation=example();annotation["events"][0]["type"]="modulation"
    assert validate_annotation(annotation,1000.,30.)["invalid_events"]
    annotation["events"][0]["frequency_domain"]="modulation"
    assert validate_annotation(annotation,1000.,30.)["invalid_events"]
    annotation["events"][0]["carrier_band_hz"]=[500,2000]
    assert validate_annotation(annotation,1000.,30.)["annotation_status"]=="complete"


def test_pending_empty_is_not_negative_truth():
    result=validate_annotation(dict(events=[]),0.,30.)
    assert result["annotation_status"]!="complete"
    assert result["independently_exhaustive"] is False
    assert result["independent_ground_truth"] is False


def test_duplicate_ids_are_retained_as_errors():
    annotation=example();annotation["events"].append(deepcopy(annotation["events"][0]))
    result=validate_annotation(annotation,1000.,30.)
    assert len(result["events"])==1 and len(result["invalid_events"])==1


def test_prompt_whitelists_view_metadata():
    prompt=annotation_prompt(0.,30.,[dict(name="LOFAR",source_class="SECRET_CLASS",source_id="SECRET_FILE",frequency_domain="acoustic",available=True)])
    assert "SECRET_CLASS" not in prompt and "SECRET_FILE" not in prompt
    assert "ORIGINAL RECORDING" in prompt


def test_syntax_repair_closes_only_complete_outer_containers():
    value,log=parse_complete_event_rows('{"events":[[1,2,100,100,"tone"]]')
    assert value=={"events":[[1,2,100,100,"tone"]]}
    assert log['syntax_repair_applied'] and not log['numeric_values_changed']
    for raw in ['{"events"}', '{"events":[[1,2,100,', '{"events":[']:
        value,log=parse_complete_event_rows(raw)
        assert value is None and not log['syntax_repair_applied']
