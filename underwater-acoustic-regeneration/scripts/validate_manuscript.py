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
    assert set(figures) == {'gpu_agent_comparison.pdf', 'velocity_correction.pdf', 'acoustic_case_studies.pdf'}
    assert len(figures) == 3
    for name in figures:
        assert Path(name).name == name and (PAPER / name).is_file(), name
        assert (PAPER / name).read_bytes() == (ROOT / 'figures' / name).read_bytes(), name
    citations = set(key.strip() for group in re.findall(r'\\cite(?:\[[^]]*\])?\{([^}]+)\}', source) for key in group.split(','))
    entries = set(re.findall(r'@\w+\s*\{\s*([^,]+),', (PAPER / 'references.bib').read_text()))
    assert citations == entries and len(entries) == 19
    bibliography = json.loads((AUDIT/'bibliography.json').read_text())
    bibliography_order = re.findall(r'\\bibitem\{([^}]+)\}', (ROOT/'.cache/latex/mainj.bbl').read_text())
    assert [item['bibtex_key'] for item in bibliography] == bibliography_order
    assert [item['number'] for item in bibliography] == list(range(1,20))
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
    agent = json.loads((ROOT / 'reports/gpu_agent_validation.json').read_text())
    assert agent['status'] == 'passed' and agent['CUDA_model_turns_verified'] > 128
    assert agent['source_groups'] == 32 and agent['source_dates'] == 32
    assert agent['all_scores_recomputed'] is True
    assert agent['hidden_references_or_dataset_classes_sent_to_models'] is False
    for name, expected in agent['artifact_sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    gpu_summary = json.loads((ROOT / 'results/gpu_agent_benchmark/summary.json').read_text())
    assert gpu_summary['cases'] == 32 and gpu_summary['episodes'] == 160 and gpu_summary['neural_episodes'] == 128
    assert hashlib.sha256((ROOT / 'results/gpu_agent_benchmark/summary.json').read_bytes()).hexdigest() == agent['summary_sha256']
    guarded = json.loads((ROOT / 'reports/gpu_agent_guard_validation.json').read_text())
    assert guarded['status'] == 'passed' and guarded['cases'] == 16 and guarded['episodes'] == 32
    assert guarded['group_and_date_disjoint_from_primary'] is True
    assert hashlib.sha256((ROOT / 'results/gpu_agent_guard/summary.json').read_bytes()).hexdigest() == guarded['summary_sha256']
    for name,expected in guarded['artifact_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected,name
    budgeted = json.loads((ROOT/'reports/gpu_agent_budget_validation.json').read_text())
    assert budgeted['status']=='passed' and budgeted['new_episodes']==16 and budgeted['paired_primary_controls']==16
    assert budgeted['summary_sha256']==hashlib.sha256((ROOT/'results/gpu_agent_budget/summary.json').read_bytes()).hexdigest()
    for name,expected in budgeted['artifact_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    for report_name, script_name in [('gpu_agent_failure_audit.json','audit_gpu_agent_failures.py'),
                                     ('gpu_agent_budget_audit.json','audit_gpu_agent_budgets.py')]:
        report = json.loads((ROOT/'reports'/report_name).read_text())
        key = 'script_sha256' if report_name=='gpu_agent_failure_audit.json' else 'analysis_sha256'
        assert report[key] == hashlib.sha256((ROOT/'scripts'/script_name).read_bytes()).hexdigest()
    gpu_figure = json.loads((ROOT / 'figures/gpu_agent_comparison.json').read_text())
    for name, expected in {**gpu_figure['source_sha256'], **gpu_figure['figure_sha256']}.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    assert gpu_figure['example_long_budget_applied_window_ms'] is not None
    first_case = next(case for case in json.loads((ROOT/'results/gpu_agent_benchmark/inputs/manifest.json').read_text())['cases']
                      if case['split'] == 'evaluation')
    assert gpu_figure['example_case_id'] == first_case['case_id']
    assert gpu_figure['example_cue_label'] == first_case['expected_report']['cue_label']
    assert gpu_figure['example_spectral_window_ms'] == first_case['expected_report']['window_ms']
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
    assert len(cells)==len({(c['table'],c['row'],c['column']) for c in cells})==110
    for cell in cells:
        if cell['table'] == 1:
            label='Test' if cell['row']=='test' else cell['row']
            line=next(line for line in source.splitlines() if line.startswith(label+' &'))
            column={'Sources':1,'Groups':2,'Pilot use':3}[cell['column']]
            actual=line.split('&')[column].strip().rstrip('\\').strip()
            expected=format(cell['value'],',d') if isinstance(cell['value'],int) else cell['value']
            if cell['column']=='Pilot use' and isinstance(cell['value'],int):
                expected+=' sources' if cell['row']=='G' else ' groups'
            assert expected==actual,(cell,line)
        if cell['table'] == 2:
            labels={'zero':'Zero fill','linear':'Linear interpolation','B':'Baseline fill B',
                    'C1':'One AI proposal (C1)','C2':'Central of four (C2)',
                    'C3':'Always use selected AI (C3)','full':'Checked output P'}
            label=labels.get(cell['row'],cell['row'])
            line = next(line for line in source.splitlines() if line.startswith(label + ' &'))
            precision = 2 if cell['column'] == 'AI use' else 3
            formatted = ('---' if cell['value']=='—' else str(cell['value']) if cell['column']=='Calls'
                         else format(cell['value'], '.'+str(precision)+'f'))
            column = {'Error Q': 1, 'NMSE': 2, 'AI use': 3, 'Worse': 4, 'Calls': 5}[cell['column']]
            actual = line.split('&')[column].strip().rstrip('\\').strip()
            assert formatted == actual, (cell, line)
        if cell['table'] == 3:
            line = next(line for line in source.splitlines() if line.startswith(cell['row'] + ' &'))
            column = {'Setting':1,'Cue':2,'Complete':3,'Tokens':4,'Seconds':5}[cell['column']]
            precision = 1 if cell['column']=='Seconds' else 0
            formatted = format(cell['value'], '.'+str(precision)+'f')
            actual = line.split('&')[column].strip().rstrip('\\').strip()
            assert formatted == actual, (cell,line)
    assert not re.search(r'% GPU_[A-Z_]+',source), 'Unfilled experiment result placeholder'
    report = {'status': 'passed', 'format': 'LaTeX', 'pages': pages, 'author': 'Anonymous',
              'single_manuscript_tex': True, 'bibtex_entries': len(entries), 'resolved_citations': len(citations),
              'publisher_class_and_style_unmodified': True,
              'acoustic_case_studies': 3, 'verified_listening_clips': len(audio_files),
              'operator_plot_cases_verified': 64, 'analysis_window_configurations': 2,
              'GPU_agent_cases': 32, 'GPU_agent_approaches': 5, 'GPU_agent_neural_episodes': 128,
              'GPU_agent_tool_returns_replayed': agent['tool_returns_replayed'],
              'GPU_guard_followup_new_groups':16,'GPU_guard_followup_episodes':32,
              'GPU_budget_followup_reused_tasks':16,'GPU_budget_followup_new_episodes':16,
              'GPU_agent_reasoning_budget_and_scope_disclosed': True,
              'identification_and_hardware_control_evaluated': False,
              'figures_in_same_directory': figures, 'tables': 3, 'equations': 5,
              'targeted_correction_development_groups': 192,
              'training_intervention_state_match_verified': True,
              'page_limit_basis': 'Ten total pages requested by user; upper end of the CFP wording at most 8 to 10 pages',
              'no_undefined_references_or_overfull_boxes': True,
              'pdf_sha256': hashlib.sha256((PAPER / 'mainj.pdf').read_bytes()).hexdigest(),
              'source_sha256': hashlib.sha256((PAPER / 'mainj.tex').read_bytes()).hexdigest(),
              'scientific_content_scope': 'Development repair correction and a frozen GPU comparison of bounded acoustic agent tasks on reused V recordings; original full restoration study remains incomplete',
              'visual_review': 'Separate page-by-page inspection required'}
    (AUDIT / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
