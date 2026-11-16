"""Scientific figure from verified GPU comparison outputs and the first frozen recording."""
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np

from acoustic_regeneration.agents.benchmark_tools import BenchmarkTools, REGIONS

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/gpu_agent_benchmark'
NAMES={'rules':'Fixed rules','direct':'Direct agent','reasoning':'Reasoning agent',
       'self_review':'Reasoning + self-review','two_agents':'Planner + reviewer'}
COLORS={'rules':'#657482','direct':'#cf8035','reasoning':'#1f6c9f','self_review':'#3d8c8b','two_agents':'#6c5095'}


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    summary=json.loads((BASE/'summary.json').read_text())
    validation=json.loads((ROOT/'reports/gpu_agent_validation.json').read_text())
    assert validation['status']=='passed' and validation['summary_sha256']==sha(BASE/'summary.json')
    cases=[c for c in json.loads((BASE/'inputs/manifest.json').read_text())['cases'] if c['split']=='evaluation']
    # Case 1 is selected by frozen input order, not by model success or failure.
    case=cases[0]; cid=case['case_id']; window=case['expected_report']['window_ms']
    task=json.loads((BASE/'inputs'/f'{cid}.json').read_text())
    with np.load(BASE/'inputs'/f'{cid}.npz',allow_pickle=False) as d: tools=BenchmarkTools(d['audio'],d['observed'],task)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,
                         'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(7.1,4.6),layout='constrained')
    grid=fig.add_gridspec(2,2,height_ratios=[1.45,1],width_ratios=[1.05,1.35])
    ax=fig.add_subplot(grid[0,0]); n=window*8; fs=8000; hann=np.hanning(n+1)[:-1]
    frequencies=np.fft.rfftfreq(n,1/fs); band=(frequencies>=20)&(frequencies<=600)
    spectra=[]
    for region,(lo,hi) in REGIONS.items():
        starts=np.arange(round(lo*fs),round(hi*fs)-n+1,n//8)
        frames=np.stack([tools.audio[s:s+n] for s in starts if tools.observed[s:s+n].all()])
        power=np.mean(np.abs(np.fft.rfft(frames*hann,axis=1))**2,axis=0)
        spectra.append((region,power))
    reference=max(power[band].max() for _,power in spectra)
    for (region,power),color in zip(spectra,['#1f6c9f','#cf8035']):
        ax.plot(frequencies[band],10*np.log10(np.maximum(power[band]/reference,1e-8)),label=region.capitalize()+' gap',color=color,lw=1.2)
    ax.set(xlim=(20,600),ylim=(-55,3),xlabel='Frequency (Hz)',ylabel='Relative power (dB)',title='A  Real recorded acoustic evidence',)
    ax.legend(frameon=False,fontsize=7,loc='lower right'); ax.grid(alpha=.2)
    ax.text(.03,.96,f'{window} ms window; erased gap excluded\nCue: {case["expected_report"]["cue_label"]}; identity unknown',
            transform=ax.transAxes,va='top',fontsize=7,bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
    ax=fig.add_subplot(grid[0,1]); arms=list(NAMES)
    for i,arm in enumerate(arms):
        m=summary['metrics'][arm]; value=100*m['success_rate'];lo,hi=np.array(m['wilson_95'])*100
        ax.barh(i,value,color=COLORS[arm],height=.62,alpha=.9)
        if arm!='rules': ax.errorbar(value,i,xerr=[[value-lo],[hi-value]],color='#253340',capsize=2,lw=.9)
        ax.text(102,i,f'{m["successes"]}/32',va='center',fontsize=8)
    ax.set(yticks=range(len(arms)),yticklabels=[NAMES[a] for a in arms],xlim=(0,118),xticks=[0,25,50,75,100],
           xlabel='Complete, evidence-backed task success (%)',title='B  Same tasks; five approaches')
    ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    ax=fig.add_subplot(grid[1,:]); neural=arms[1:]
    families=['frequency_detail','timing_detail','incompatible_constraints','fragmented_observation']
    ordered=[c for family in families for c in cases if c['family']==family]
    matrix=np.zeros((4,32),dtype=int)
    for r,arm in enumerate(neural):
        for c,case in enumerate(ordered):
            item=json.loads((BASE/'evaluation'/f'{case["case_id"]}__{arm}.json').read_text())
            checks=item['checks'];matrix[r,c]=2 if checks['strict_success'] else 1 if checks['setting_correct'] else 0
    ax.imshow(matrix,aspect='auto',vmin=0,vmax=2,cmap=ListedColormap(['#e1e5e8','#edbc67','#287e84']),interpolation='nearest')
    ax.set(yticks=range(4),yticklabels=[NAMES[a] for a in neural],xticks=[3.5,11.5,19.5,27.5],
           xticklabels=['Frequency detail','Timing detail','Conflicting limits','Fragmented audio'],
           title='C  Each cell is one recording-based operator task')
    ax.tick_params(axis='both',length=0)
    for x in [7.5,15.5,23.5]:ax.axvline(x,color='white',lw=2)
    for y in [.5,1.5,2.5]:ax.axhline(y,color='white',lw=.6)
    ax.legend(handles=[Patch(facecolor='#287e84',label='Complete success'),Patch(facecolor='#edbc67',label='Setting correct; other error'),
                       Patch(facecolor='#e1e5e8',label='Setting wrong or report missing')],
              loc='upper center',bbox_to_anchor=(.5,-.19),ncol=3,frameon=False,fontsize=7)
    for axis in fig.axes:
        axis.spines['top'].set_visible(False);axis.spines['right'].set_visible(False)
    paths=[]
    for ext in ('pdf','png','svg'):
        p=ROOT/'figures'/('gpu_agent_comparison.'+ext)
        fig.savefig(p,dpi=450,bbox_inches='tight',metadata={'Creator':'Matplotlib; frozen experiment outputs'})
        paths.append(p)
    plt.close(fig)
    shutil.copyfile(ROOT/'figures/gpu_agent_comparison.pdf',ROOT/'manuscript/gpu_agent_comparison.pdf')
    report={'status':'passed','example_case_id':cid,'example_selection':'First frozen case, independent of agent outcomes',
            'cases':32,'arms':arms,'intervals':'Wilson 95% intervals; task-specific deterministic rule baseline shown without inferential error bar',
            'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [BASE/'summary.json',BASE/'inputs/manifest.json',Path(__file__)]},
            'figure_sha256':{str(p.relative_to(ROOT)):sha(p) for p in paths},'cell_order':[c['case_id'] for c in ordered]}
    (ROOT/'figures/gpu_agent_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Built GPU comparison figure from all 160 episodes and first frozen acoustic case.')


if __name__=='__main__': main()
