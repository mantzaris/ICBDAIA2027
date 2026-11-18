#!/usr/bin/env python3
"""Build blinded, explicitly pending AI-reference templates without labels."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    source=ROOT/"data/manifests/real_scenes.jsonl"
    scenes=[json.loads(line) for line in source.read_text().splitlines()]
    blind=[]
    templates=[]
    for scene in scenes:
        blind.append({key:scene[key] for key in ["scene_id","path","sha256","duration_seconds","sample_rate","channels"]})
        templates.append(dict(scene_id=scene["scene_id"],audio_sha256=scene["sha256"],
                              annotation_status="pending",annotator_id=None,
                              annotation_origin="ai_reference",model_id=None,model_revision=None,
                              prompt_sha256=None,completed_at_utc=None,
                              listening_conditions=None,display_settings=[],
                              inspected_no_events=False,events=[],relations=[],
                              exhaustive_regions=[],notes="Blank template; not a negative reference."))
    directory=ROOT/"annotations"
    (directory/"blind_scene_manifest.jsonl").write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in blind))
    for name in ["annotator_a","annotator_b"]:
        folder=directory/name
        folder.mkdir(exist_ok=True)
        target=folder/"templates.jsonl"
        if target.exists():
            old=[json.loads(x) for x in target.read_text().splitlines()]
            if any(x.get("annotation_status")!="pending" or x.get("events") or x.get("annotator_id") for x in old):
                raise RuntimeError("Refusing to overwrite completed annotations: "+str(target))
        target.write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in templates))
    status=dict(scene_count=len(scenes),reference_method="AI-generated annotations, user-directed protocol amendment",
                annotation_passes_planned=2,completed_ai_annotations=0,reconciliation_status="not started",
                human_annotations_required=False,independent_real_ground_truth=False,
                ai_reference_agreement_evaluation_ready=False,
                manifest_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                warning="Pending or empty templates cannot be scored as reference negatives.")
    (directory/"status.json").write_text(json.dumps(status,indent=2))
    print(json.dumps(status,indent=2))


if __name__=="__main__":
    main()
