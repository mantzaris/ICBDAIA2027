"""Archive the runnable protocol before any evaluated-model test predictions."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--compatibility-evidence',required=True,type=Path)
    args=ap.parse_args()
    output=Path('configs/final_protocol_freeze.json')
    if output.exists():raise RuntimeError('Final protocol already frozen; preserve it and document any correction separately')
    for phase in ['main','development','ablation_inference','budget2','budget4','robustness','sensitivity','localization/predictions']:
        if any((Path('results')/phase).glob('*/*/episode.json')):
            raise RuntimeError(f'Evaluated-model outputs already exist in {phase}; cannot claim pre-test freeze')
    if not args.compatibility_evidence.is_file():raise ValueError('Actual completed development compatibility evidence is required')
    if not Path('configs/ai_reference_freeze.json').is_file():raise ValueError('AI-reference eligibility selection is not frozen')
    paths=sorted(set(list(Path('src/sonar').glob('*.py'))+list(Path('scripts').glob('*.py'))+list(Path('tests').glob('*.py'))+
        list(Path('configs').glob('*.json'))+list(Path('data/manifests').glob('*.jsonl'))+
        [Path('requirements.gpu.txt'),Path('reports/model_revision.json')]))
    directory=Path('provenance/final_protocol');directory.mkdir(parents=True,exist_ok=False)
    hashes={}
    for source in paths:
        target=directory/source;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        hashes[str(source)]=sha(target)
    # Presentation code may evolve during final layout; its own artifacts carry
    # source hashes and a separately frozen case-selection policy. Scientific
    # inference/training/scoring and the already frozen annotation protocol do
    # not change when a cursor, caption or page layout changes.
    drivers={f'scripts/{name}.py' for name in ['run_benchmark','train_perception','train_localization',
        'evaluate_benchmark','evaluate_sensitivity','compose_ablations','run_study_phases',
        'freeze_manifests','freeze_mechanism_tests','build_diagnostic_references']}
    enforced={path:digest for path,digest in hashes.items() if path.startswith(('src/sonar/','configs/','data/manifests/'))
              or path in drivers or path in ['requirements.gpu.txt','reports/model_revision.json']}
    record=dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),sha256=enforced,archived_sha256=hashes,
        presentation_policy='Case-selection rules frozen separately against final reference/scene manifests; final presentation source hashes accompany artifacts.',
        archived_sources=str(directory),compatibility_evidence=str(args.compatibility_evidence),
        compatibility_evidence_sha256=sha(args.compatibility_evidence),
        development_model_runs='Preserved pilots and compatibility fixtures only; no evaluated-model main predictions',
        evaluation_interpretation='AI-reference agreement on real recordings; exact scoped simulator correctness reported separately',
        confirmatory_comparator='Select G3 or G4 from validation AI-reference agreement. If eligible real references are unavailable, confirmatory real comparison is unavailable; no synthetic fallback is relabelled confirmatory.',
        author_authorization='User confirmed existing two authors and affiliations',
        annotation_authorization='User explicitly replaced unavailable human annotators with AI annotations')
    output.write_text(json.dumps(record,indent=2)+'\n')
    (directory/'freeze.json').write_text(output.read_text())
    print(json.dumps({'protocol':str(output),'sha256':sha(output),'archived_files':len(hashes)}))


if __name__=='__main__':main()
