"""Validate the single-file LaTeX paper, bibliography, colocated figures, and PDF."""
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'manuscript'
AUDIT = ROOT / 'reports/manuscript_audit'


def main():
    source = (PAPER / 'mainj.tex').read_text()
    assert [p.name for p in PAPER.glob('*.tex')] == ['mainj.tex']
    assert not re.search(r'\\(?:input|include)\s*\{', source), 'All manuscript text must be in mainj.tex'
    assert r'\bibliography{references}' in source and r'\bibliographystyle{splncs04}' in source
    assert r'\author{Anonymous submission}' in source
    assert not list(PAPER.rglob('*.docx'))
    template = json.loads((ROOT / 'provenance/latex_template.json').read_text())
    for name, expected in template['files'].items():
        assert hashlib.sha256((PAPER / name).read_bytes()).hexdigest() == expected, name
    figures = re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', source)
    assert set(figures) == {'operator_configuration.pdf', 'operator_decisions.pdf', 'acoustic_case_studies.pdf'}
    assert len(figures) == 3
    for name in figures:
        assert Path(name).name == name and (PAPER / name).is_file(), name
        assert (PAPER / name).read_bytes() == (ROOT / 'figures' / name).read_bytes(), name
    citations = set(key.strip() for group in re.findall(r'\\cite(?:\[[^]]*\])?\{([^}]+)\}', source) for key in group.split(','))
    entries = set(re.findall(r'@\w+\s*\{\s*([^,]+),', (PAPER / 'references.bib').read_text()))
    assert citations == entries and len(entries) == 14
    examples = json.loads((ROOT / 'figures/acoustic_example_selection.json').read_text())
    assert examples['script_sha256'] == hashlib.sha256((ROOT / 'scripts/build_acoustic_examples.py').read_bytes()).hexdigest()
    decisions = json.loads((ROOT / 'figures/operator_decisions.json').read_text())
    assert decisions['script_sha256'] == examples['script_sha256']
    assert decisions['cases'] == len(decisions['rows']) == 64
    assert sum(row['worse_than_baseline'] for row in decisions['rows']) == 52
    assert not any(row['checked_policy_uses_AI'] for row in decisions['rows'])
    assert decisions['calibrated_threshold'] is None
    for name, expected in decisions['source_files'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    configuration = json.loads((ROOT / 'figures/operator_configuration.json').read_text())
    assert configuration['script_sha256'] == examples['script_sha256']
    assert configuration['identification_evaluated'] is False
    assert configuration['hardware_configuration_evaluated'] is False
    assert configuration['annotation_is_model_prediction'] is False
    assert [setting['window_samples'] for setting in configuration['settings']] == [256, 1024]
    assert [case['tag'] for case in examples['cases']] == ['A', 'B', 'C']
    audio_files = []
    for case in examples['cases']:
        assert case['source_partition'] == 'V' and case['pilot_role'] == 'dev_test'
        assert case['policy_accepted'] is False
        assert len(case['audio_files']) == 4
        for artifact in case['audio_files'].values():
            path = ROOT / artifact['path']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact['sha256'], path
            audio_files.append(artifact['path'])
    log = (ROOT / '.cache/latex/mainj.log').read_text(errors='replace')
    assert not re.search(r'undefined|Overfull \\[hv]box|LaTeX Error', log, re.IGNORECASE)
    info = subprocess.check_output(['pdfinfo', str(PAPER / 'mainj.pdf')], text=True)
    pages = int(re.search(r'^Pages:\s+(\d+)', info, re.MULTILINE)[1])
    assert pages <= 8, pages
    assert re.search(r'^Author:\s+Anonymous\s*$', info, re.MULTILINE)
    text = subprocess.check_output(['pdftotext', str(PAPER / 'mainj.pdf'), '-'], text=True)
    assert 'Anonymous submission' in text and 'References' in text
    assert not any(value in text for value in ('LAYOUT PREVIEW', '[PENDING', '\ufffd'))
    assert len(re.findall(r'\\begin\{table\}', source)) == 3
    assert len(re.findall(r'\\begin\{equation\}', source)) == 2
    # Ensure all existing numerical table entries survived the format conversion.
    archived = json.loads((AUDIT / 'word_draft/render_plan.json').read_text())
    for page in archived['pages']:
        for block in page:
            if block['kind'] == 'table':
                for row in block['rows']:
                    for value in row:
                        if re.fullmatch(r'[\d,.]+', value):
                            assert value in source, value
    report = {'status': 'passed', 'format': 'LaTeX', 'pages': pages, 'author': 'Anonymous',
              'single_manuscript_tex': True, 'bibtex_entries': len(entries), 'resolved_citations': len(citations),
              'publisher_class_and_style_unmodified': True,
              'acoustic_case_studies': 3, 'verified_listening_clips': len(audio_files),
              'operator_plot_cases_verified': 64, 'analysis_window_configurations': 2,
              'identification_and_hardware_control_evaluated': False,
              'figures_in_same_directory': figures, 'tables': 3, 'equations': 2,
              'no_undefined_references_or_overfull_boxes': True,
              'pdf_sha256': hashlib.sha256((PAPER / 'mainj.pdf').read_bytes()).hexdigest(),
              'source_sha256': hashlib.sha256((PAPER / 'mainj.tex').read_bytes()).hexdigest(),
              'scientific_content_scope': 'Negative development pilot with post hoc acoustic examples; full study remains incomplete',
              'visual_review': 'Separate page-by-page inspection required'}
    (AUDIT / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
