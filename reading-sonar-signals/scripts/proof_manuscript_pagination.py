#!/usr/bin/env python3
"""CPU-only page-budget proof with labeled figure bounds, never result graphics."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'manuscript/main.tex')
    parser.add_argument('--name', default='compact')
    parser.add_argument('--reserve-lines', type=int, default=0,
                        help='Extra labeled text-height allowance for pending secondary results.')
    args = parser.parse_args()
    source = args.source.resolve()
    out = ROOT / 'reports/pagination_layout_proof' / args.name
    out.mkdir(parents=True, exist_ok=True)
    # Development proofs measure layout only; their extra proof title makes
    # these conservative proxies. No saved experimental values are evaluated.
    dimensions = {}
    for number, measured in [(1, 1), (2, 2), (3, 2), (4, 4), (5, 5)]:
        path = ROOT / f'reports/presentation_layout_proof/development_figure{measured:02d}.pdf'
        info = subprocess.check_output(['pdfinfo', str(path)], text=True)
        match = re.search(r'Page size:\s+([\d.]+) x ([\d.]+) pts', info)
        width, height = map(float, match.groups())
        dimensions[number] = dict(width_pt=width, height_pt=height,
                                  source=str(path.relative_to(ROOT)), sha256=digest(path),
                                  kind='measured development layout' if number != 3 else 'same layout as figure02')
    # Final quantitative renders do not exist yet. Include a 0.2-inch height
    # allowance, while conservatively retaining the smaller 6.4-inch width.
    for number, height in [(6, 6.3), (7, 6.7)]:
        dimensions[number] = dict(width_pt=6.4*72, height_pt=height*72,
                                  source='declared 6.4-inch source width and height plus allowance',
                                  kind='proxy bound; actual render pending')
    tex = source.read_text()
    macro = r'''\newcommand{\layoutproxy}[4]{%
  \begin{figure}[tbp]\centering
  \setlength{\fboxsep}{0pt}%
  \fbox{\parbox[c][\dimexpr #4\linewidth-2\fboxrule\relax][c]{\dimexpr\linewidth-2\fboxrule\relax}{%
    \centering\textbf{LAYOUT PROXY ONLY}\\[6pt]
    #1\\[6pt]No scientific image or result is represented.\\
    Measured/proxy bounds; final recorded figure pending.}}%
  \caption{#2}\label{#3}\end{figure}}
'''
    proof_notice = r'''\usepackage{eso-pic}
\usepackage{xcolor}
\AddToShipoutPictureFG{\AtPageLowerLeft{\raisebox{14pt}{%
  \makebox[\paperwidth]{\color{red}\footnotesize LAYOUT PROOF ONLY: proxy bounds, no scientific figures}}}}
\hypersetup{pdftitle={LAYOUT PROOF ONLY: measured and proxy figure bounds}}
'''
    tex = tex.replace(r'\begin{document}', proof_notice + macro + '\n' + r'\begin{document}')
    tex = tex.replace(r'\titlerunning{Bounded Inspection of Acoustic Spectrograms}',
                      r'\titlerunning{LAYOUT PROOF ONLY: proxy figure bounds}')
    calls = 0
    for number, dimension in dimensions.items():
        name = f'figure{number:02d}.pdf'
        pattern = r'\\researchfigure\{' + re.escape(name) + r'\}\{([^\n]*)\}\{([^\n]*)\}'
        ratio = dimension['height_pt'] / dimension['width_pt']
        tex, count = re.subn(pattern, lambda m: r'\layoutproxy{' + name + '}{' + m[1] + '}{' + m[2] + '}{' + f'{ratio:.9f}' + '}', tex)
        calls += count
    assert calls == 7, calls
    if args.reserve_lines:
        marker_match = re.search(r'\\pending\{Add [^}]*results and review final figures\.\}', tex)
        assert marker_match, "Pending secondary-results insertion point is missing"
        marker = marker_match[0]
        # Breakable lines model normal prose; an unbreakable tall box would
        # force an artificial float-page overflow unrelated to word budget.
        reserve = '\n' + '\n'.join(
            r'\par\noindent\textit{LAYOUT-ONLY: reserved secondary-results line ' + str(n+1) + '.}'
            for n in range(args.reserve_lines)) + '\n'
        tex=tex.replace(marker,marker+reserve)
    tex = tex.replace(r'\requiredartifact{figure08.pdf}', r'% Actual standalone figure08.pdf remains required in canonical source.')
    tex = tex.replace(r'\requiredartifact{figure09.pdf}', r'% Actual standalone figure09.pdf remains required in canonical source.')
    (out/'layout.tex').write_text(tex)
    for name in ['llncs.cls','splncs04.bst','references.bib']:
        shutil.copyfile(ROOT/'manuscript'/name,out/name)
    shutil.copyfile(ROOT/'manuscript/main.bbl',out/'layout.bbl')
    for attempt in [1,2]:
        proc=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','layout.tex'],cwd=out,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (out/f'compile{attempt}.log').write_text(proc.stdout)
        if proc.returncode:
            raise RuntimeError(proc.stdout[-3000:])
    info=subprocess.check_output(['pdfinfo',str(out/'layout.pdf')],text=True)
    pages=int(re.search(r'Pages:\s+(\d+)',info)[1])
    log=(out/'compile2.log').read_text()
    receipt={'status':'temporary_layout_proof_only','source':str(source.relative_to(ROOT)),
             'source_sha256':digest(source),'proof_pdf':str((out/'layout.pdf').relative_to(ROOT)),
             'proof_sha256':digest(out/'layout.pdf'),'pages':pages,'figure_bounds':dimensions,
             'main_figure_calls':calls,'scientific_data_or_metrics_generated':False,
             'additional_reserved_text_lines':args.reserve_lines,
             'GPU_use':False,'class_font_margins_changed':False,
             'overfull_warning_lines':[line for line in log.splitlines() if 'Overfull' in line],
             'limitation':'Page estimate only: actual final figure extents and pending secondary-results prose may change pagination. Proxies never replace canonical figures.'}
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ['status','pages','proof_pdf','overfull_warning_lines']},indent=2))


if __name__=='__main__':
    main()
