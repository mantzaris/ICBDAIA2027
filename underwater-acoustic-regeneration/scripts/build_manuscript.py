"""Compile the editable, single-file LaTeX manuscript without regenerating its text."""
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'manuscript'
BUILD = ROOT / '.cache/latex'
AUDIT = ROOT / 'reports/manuscript_audit'


def main():
    BUILD.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)
    author_info = json.loads((AUDIT / 'author_info.json').read_text())
    command = ['latexmk', '-pdf', '-interaction=nonstopmode', '-halt-on-error',
               '-file-line-error', '-pdflatex=pdflatex -no-shell-escape %O %S',
               '-outdir=' + str(BUILD), 'main.tex']
    log = BUILD / 'console.log'
    with log.open('w') as stream:
        result = subprocess.run(command, cwd=PAPER, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        print('\n'.join(log.read_text(errors='replace').splitlines()[-30:]))
        raise SystemExit(result.returncode)
    shutil.copyfile(BUILD / 'main.pdf', PAPER / 'main.pdf')
    figures = re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', (PAPER / 'main.tex').read_text())
    paths = ['main.tex', 'references.bib', 'llncs.cls', 'splncs04.bst', *figures, 'main.pdf']
    report = {'format': 'LaTeX', 'canonical_source': 'manuscript/main.tex',
              'bibliography': 'manuscript/references.bib',
              'author': author_info['document_metadata_author'],
              'submission_ready': False, 'main_study_complete': False,
              'command': command, 'source_text_regenerated': False,
              'artifact_sha256': {p: hashlib.sha256((PAPER / p).read_bytes()).hexdigest() for p in paths},
              'engines': {name: subprocess.check_output([name, '--version'], text=True).splitlines()[0]
                          for name in ('pdflatex', 'bibtex', 'latexmk')}}
    (AUDIT / 'build_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Built manuscript/main.pdf from main.tex and references.bib')


if __name__ == '__main__':
    main()
