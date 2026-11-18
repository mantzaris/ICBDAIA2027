"""Fixed prompt-only projection; complete numeric/tool records remain on disk."""
from __future__ import annotations

PROMPT_PROTOCOL = dict(version="compact_measurements_v2", max_components=16,
    max_measurements=12, max_relations=24, track_points=3, max_peaks_per_measurement=6,
    geometry_decimal_places=3, other_float_significant_digits=6,
    sampling="track_first_middle_last; first_six_peaks_in_detector_return_order; no outcome_based_selection",
    meaning="detector_proposals_and_internal_measurements_not_independent_truth")

COMPONENT_FIELDS = ("id", "type", "frequency_domain", "time_s", "frequency_hz", "frequency_track",
    "carrier_band_hz", "bin_spacing_hz", "boundary_uncertainty_s", "frequency_uncertainty_hz",
    "slope_hz_per_s", "fit_residual_hz", "contrast_db", "evidence_ids")
MEASUREMENT_FIELDS = ("id", "tool", "time_s", "frame_center_time_s", "intervals_s", "frequency_hz",
    "frequency_domain", "carrier_band_hz", "component_ids", "input_component_ids", "measured",
    "band_power", "band_powers", "received_level_change_db", "relations", "source_association",
    "graph_memory_used", "configuration", "peaks")
CONFIGURATION_FIELDS = ("sample_rate_hz", "window_s", "hop_s", "bin_spacing_hz", "edge_guard_s")
RELATION_FIELDS = ("type", "component_ids", "evidence_ids", "overlap_fraction", "overlap_seconds")
GEOMETRY_FIELDS = frozenset(("time_s", "frame_center_time_s", "intervals_s", "frequency_hz", "frequency_track",
    "carrier_band_hz", "boundary_uncertainty_s", "frequency_uncertainty_hz", "bin_spacing_hz",
    "slope_hz_per_s", "fit_residual_hz", "window_s", "hop_s", "edge_guard_s", "overlap_seconds"))


def prompt_numbers(value, field=None):
    """Format already-measured scalar metadata, without scientific recomputation."""
    if isinstance(value,dict):
        return {key:prompt_numbers(item,key) for key,item in value.items()}
    if isinstance(value,list):
        return [prompt_numbers(item,field) for item in value]
    if isinstance(value,float):
        return float(format(value,".3f" if field in GEOMETRY_FIELDS else ".6g"))
    return value


def _pick(value,fields):
    return {key:value[key] for key in fields if key in value}


def _relation(value):
    return _pick(value,RELATION_FIELDS)


def _band_power(value):
    return _pick(value,("mean_square","level_db","units","calibrated"))


def compact_measurements(state, quantitative=True, graph=True):
    result = {key:state.get(key) for key in ["scene_id","time_s","sample_rate_hz","limitations"]}
    if quantitative:
        result["components"] = []
        for component in state.get("components",[])[:16]:
            item = _pick(component,COMPONENT_FIELDS)
            track = item.get("frequency_track",[])
            if len(track)>3:
                item["frequency_track"] = [track[0],track[len(track)//2],track[-1]]
            result["components"].append(item)
        measurements = state.get("measurements",[])
        if isinstance(measurements,dict): measurements=list(measurements.values())
        result["measurement_index"] = []
        for measurement in measurements[-12:]:
            item = _pick(measurement,MEASUREMENT_FIELDS)
            if "configuration" in item:
                item["configuration"] = _pick(item["configuration"],CONFIGURATION_FIELDS)
            if "band_power" in item: item["band_power"]=_band_power(item["band_power"])
            if "band_powers" in item: item["band_powers"]=[_band_power(p) for p in item["band_powers"]]
            if "peaks" in item:
                item["peaks"]=[_pick(p,("frequency_hz","contrast_db","bin_spacing_hz")) for p in item["peaks"][:6]]
            if "relations" in item: item["relations"]=[_relation(r) for r in item["relations"][:24]]
            if not graph:
                for key in ["component_ids","input_component_ids","relations"]: item.pop(key,None)
            result["measurement_index"].append(item)
    if graph:
        result["relations"] = [_relation(r) for r in state.get("relations",[])[:24]]
    if not quantitative:
        measurements=state.get("measurements",[])
        if isinstance(measurements,dict): measurements=list(measurements.values())
        result["available_evidence_ids"] = [m["id"] for m in measurements if "id" in m]
    return prompt_numbers(result)
