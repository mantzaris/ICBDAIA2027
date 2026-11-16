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
    assert set(figures) == {'agent_episode.pdf', 'velocity_correction.pdf', 'acoustic_case_studies.pdf'}
    assert len(figures) == 3
    for name in figures:
        assert Path(name).name == name and (PAPER / name).is_file(), name
        assert (PAPER / name).read_bytes() == (ROOT / 'figures' / name).read_bytes(), name
    citations = set(key.strip() for group in re.findall(r'\\cite(?:\[[^]]*\])?\{([^}]+)\}', source) for key in group.split(','))
    entries = set(re.findall(r'@\w+\s*\{\s*([^,]+),', (PAPER / 'references.bib').read_text()))
    assert citations == entries and len(entries) == 16
    examples = json.loads((ROOT / 'figures/acoustic_example_selection.json').read_text())
    assert examples['script_sha256'] == hashlib.sha256((ROOT / 'scripts/build_acoustic_examples.py').read_bytes()).hexdigest()
    for name, expected in examples['source_files'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
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
    agent = json.loads((ROOT / 'reports/cpu_agent_validation.json').read_text())
    assert agent['status'] == 'passed' and agent['CPU_residency_verified'] is True
    assert agent['tool_returns_replayed_exactly'] == 7 and agent['model_turns'] == 4
    assert all(agent['continued_checks'].values())
    assert agent['unsupported_original_comparison_claim_retained'] is True
    for name, expected in agent['artifact_sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    card = json.loads((ROOT / 'reports/cpu_agent_operator_card.json').read_text())
    assert card['free_text_model_report_verified'] is False and card['identity_status'] == 'unconfirmed'
    correction = json.loads((ROOT / 'reports/velocity_followup_validation.json').read_text())
    assert correction['status'] == 'passed' and correction['development_groups'] == 192
    assert correction['verified_gap_artifacts'] == 192
    assert correction['new_velocity_draw_gap_metrics_recomputed'] == 1536
    assert correction['old_candidate_metrics_recomputed_and_matched'] is True
    assert correction['local_error_amplification_identity_verified'] is True
    assert correction['script_sha256'] == hashlib.sha256((ROOT / 'scripts/plot_velocity_followup.py').read_bytes()).hexdigest()
    assert correction['figure_sha256'] == hashlib.sha256((PAPER / 'velocity_correction.pdf').read_bytes()).hexdigest()
    assert correction['date_sensitivity_sha256'] == hashlib.sha256((ROOT / 'reports/velocity_followup_date_sensitivity.json').read_bytes()).hexdigest()
    for name, expected in correction['source_files'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    training_match = json.loads((ROOT / 'reports/velocity_training_match.json').read_text())
    assert training_match['status'] == 'passed' and training_match['updates_each'] == 5000
    assert training_match['ending_python_numpy_torch_cuda_RNG_states_match'] is True
    for label in ['eq:amplification', 'eq:velocity', 'eq:velocity_loss']:
        assert r'\label{' + label + '}' in source
    assert [case['tag'] for case in examples['cases']] == ['A', 'B', 'C']
    audio_files = []
    for case in examples['cases']:
        assert case['source_partition'] == 'V' and case['pilot_role'] == 'dev_test'
        assert case['policy_accepted'] is False
        assert len(case['audio_files']) == 5
        assert case['velocity_release_gate_fitted'] is False and case['velocity_candidate_index'] == 0
        for artifact in case['audio_files'].values():
            path = ROOT / artifact['path']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact['sha256'], path
            audio_files.append(artifact['path'])
    log = (ROOT / '.cache/latex/mainj.log').read_text(errors='replace')
    assert not re.search(r'undefined|Overfull \\[hv]box|LaTeX Error', log, re.IGNORECASE)
    info = subprocess.check_output(['pdfinfo', str(PAPER / 'mainj.pdf')], text=True)
    pages = int(re.search(r'^Pages:\s+(\d+)', info, re.MULTILINE)[1])
    assert pages <= 10, pages
    assert re.search(r'^Author:\s+Anonymous\s*$', info, re.MULTILINE)
    text = subprocess.check_output(['pdftotext', str(PAPER / 'mainj.pdf'), '-'], text=True)
    assert 'Anonymous submission' in text and 'References' in text
    assert not any(value in text for value in ('LAYOUT PREVIEW', '[PENDING', '\ufffd'))
    assert len(re.findall(r'\\begin\{table\}', source)) == 3
    assert len(re.findall(r'\\begin\{equation\}', source)) == 5
    cells = json.loads((AUDIT / 'table_cells.json').read_text())
    for cell in cells:
        if cell['table'] == 2 and cell['row'].startswith('Velocity:'):
            line = next(line for line in source.splitlines() if line.startswith(cell['row'] + ' &'))
            precision = 2 if cell['column'] == 'AI use' else 3
            formatted = str(cell['value']) if cell['column'] == 'Calls' else format(cell['value'], '.' + str(precision) + 'f')
            column = {'Error Q': 1, 'NMSE': 2, 'AI use': 3, 'Worse': 4, 'Calls': 5}[cell['column']]
            actual = line.split('&')[column].strip().rstrip('\\').strip()
            assert formatted == actual, (cell, line)
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
              'CPU_agent_demonstration_cases': 1, 'CPU_agent_model_turns': 4, 'CPU_agent_tool_calls': 7,
              'CPU_agent_action_fix_verified': True, 'CPU_agent_prose_limitations_disclosed': True,
              'identification_and_hardware_control_evaluated': False,
              'figures_in_same_directory': figures, 'tables': 3, 'equations': 5,
              'targeted_correction_development_groups': 192,
              'training_intervention_state_match_verified': True,
              'page_limit_basis': 'Ten total pages requested by user; upper end of the CFP wording at most 8 to 10 pages',
              'no_undefined_references_or_overfull_boxes': True,
              'pdf_sha256': hashlib.sha256((PAPER / 'mainj.pdf').read_bytes()).hexdigest(),
              'source_sha256': hashlib.sha256((PAPER / 'mainj.tex').read_bytes()).hexdigest(),
              'scientific_content_scope': 'Development pilot and targeted correction on reused V groups; full study remains incomplete',
              'visual_review': 'Separate page-by-page inspection required'}
    (AUDIT / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
