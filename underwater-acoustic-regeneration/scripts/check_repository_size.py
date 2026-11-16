"""Check tracked working files before a commit; never delete or modify artifacts."""
import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIB = 1024 * 1024
PAPER_FIGURES = {
    'acoustic_case_studies.pdf', 'gpu_agent_comparison.pdf', 'velocity_correction.pdf'
}


def excluded(path):
    name = path.as_posix()
    if any(part in {'.cache', '.local-history', 'checkpoints', '.venv'}
           or part.startswith('.venv-') for part in path.parts):
        return True
    if name.startswith(('data/raw/', 'data/processed/', 'provenance/uniqueship_metadata/')):
        return True
    if name in {'manifests/sources.jsonl.gz', 'manifests/source_metadata.jsonl.gz'}:
        return True
    if name.startswith('figures/') and path.suffix in {'.pdf', '.png', '.svg'}:
        return True
    if name.startswith('manuscript/') and path.suffix == '.pdf':
        return path.name not in PAPER_FIGURES
    return path.suffix in {'.pt', '.pth', '.ckpt', '.safetensors', '.onnx', '.gguf', '.bundle', '.zip'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-total-mib', type=float, default=40)
    parser.add_argument('--max-file-mib', type=float, default=2)
    args = parser.parse_args()
    paths = [Path(p.decode()) for p in subprocess.check_output(
        ['git', 'ls-files', '-z', '--', '.'], cwd=ROOT).split(b'\0') if p]
    files = []
    errors = []
    for path in paths:
        if not (ROOT / path).is_file():
            errors.append('Missing tracked file: ' + path.as_posix())
            continue
        size = (ROOT / path).stat().st_size
        files.append({'path': path.as_posix(), 'bytes': size})
        if excluded(path):
            errors.append('Rebuildable or bulk artifact tracked: ' + path.as_posix())
        if size > args.max_file_mib * MIB:
            errors.append('File exceeds size budget: ' + path.as_posix())
    total = sum(row['bytes'] for row in files)
    if total > args.max_total_mib * MIB:
        errors.append('Tracked files exceed total size budget')
    print(json.dumps({'status': 'failed' if errors else 'passed',
                      'scope': 'Tracked working files, not historical Git objects or ignored local backups',
                      'tracked_files': len(files), 'tracked_bytes': total,
                      'total_mib': round(total / MIB, 2),
                      'limits_mib': {'total': args.max_total_mib, 'file': args.max_file_mib},
                      'largest_files': sorted(files, key=lambda row: -row['bytes'])[:5],
                      'errors': errors}, indent=2))
    raise SystemExit(bool(errors))


if __name__ == '__main__':
    main()
