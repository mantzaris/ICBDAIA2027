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
NAMES={'rules':'Fixed rules','direct':'Direct','reasoning':'Reasoning',
       'self_review':'Self-review','two_agents':'Two agents'}
COLORS={'rules':'#657482','direct':'#cf8035','reasoning':'#1f6c9f','self_review':'#3d8c8b','two_agents':'#6c5095'}


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    summary=json.loads((BASE/'summary.json').read_text())
    validation=json.loads((ROOT/'reports/gpu_agent_validation.json').read_text())
    assert validation['status']=='passed' and validation['summary_sha256']==sha(BASE/'summary.json')
    for name,expected in validation['artifact_sha256'].items():assert sha(ROOT/name)==expected,name
    source_paths=[BASE/'summary.json',BASE/'inputs/manifest.json',ROOT/'reports/gpu_agent_validation.json',Path(__file__).resolve()]
    cases=[c for c in json.loads((BASE/'inputs/manifest.json').read_text())['cases'] if c['split']=='evaluation']
    # Case 1 is selected by frozen input order, not by model success or failure.
    example_case=cases[0]; cid=example_case['case_id']; window=example_case['expected_report']['window_ms']
    task=json.loads((BASE/'inputs'/f'{cid}.json').read_text())
    with np.load(BASE/'inputs'/f'{cid}.npz',allow_pickle=False) as d: tools=BenchmarkTools(d['audio'],d['observed'],task)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.5,'axes.titlesize':11,'axes.labelsize':10.5,
                         'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
                         'svg.hashsalt':'gpu-agent-comparison-v1'})
    # Fixed panel rectangles keep long labels from shrinking the acoustic plot.
    fig=plt.figure(figsize=(6.4,4.4))
    ax=fig.add_axes([.12,.49,.35,.43]); n=window*8; fs=8000; hann=np.hanning(n+1)[:-1]
    frequencies=np.fft.rfftfreq(n,1/fs); band=(frequencies>=20)&(frequencies<=600)
    spectra=[]
    for region,(lo,hi) in REGIONS.items():
        starts=np.arange(round(lo*fs),round(hi*fs)-n+1,n//8)
        frames=np.stack([tools.audio[s:s+n] for s in starts if tools.observed[s:s+n].all()])
        power=np.mean(np.abs(np.fft.rfft(frames*hann,axis=1))**2,axis=0)
        spectra.append((region,power))
    reference=max(power[band].max() for _,power in spectra)
    peaks={region:float(frequencies[band][np.argmax(power[band])]) for region,power in spectra}
    for region,peak in peaks.items():
        assert peak==example_case['expected_report'][f'peak_{region}_hz']
    for (region,power),color in zip(spectra,['#1f6c9f','#cf8035']):
        ax.plot(frequencies[band],10*np.log10(np.maximum(power[band]/reference,1e-8)),
                label=f'{region.capitalize()}: {peaks[region]:.1f} Hz',color=color,lw=1.2)
    ax.set(xlim=(20,160),ylim=(-55,3),xticks=[40,80,120,160],xlabel='Frequency (Hz)',ylabel='Relative power (dB)')
    ax.set_title('A  Recorded sound',loc='left')
    ax.legend(frameon=False,fontsize=9,loc='upper right'); ax.grid(alpha=.2)
    example_path=BASE/'evaluation'/f'{cid}__reasoning.json'
    example=json.loads(example_path.read_text());source_paths.append(example_path)
    applied=32
    for event in example['tool_trace']:
        if event['name']=='configure_display' and 'error' not in event['result']:
            applied=event['result']['current_window_ms']
    stated=example['final_report'].get('window_ms','missing') if example['final_report'] else 'missing'
    long_stated=long_applied=None
    callout=f'Reasoner stated {stated} ms\nActually applied {applied} ms\nSource identity: unknown'
    if (ROOT/'reports/gpu_agent_budget_validation.json').exists():
        budget_validation=json.loads((ROOT/'reports/gpu_agent_budget_validation.json').read_text())
        assert budget_validation['status']=='passed'
        long_path=ROOT/'results/gpu_agent_budget/evaluation'/f'{cid}__reasoning_long.json'
        assert sha(long_path)==budget_validation['artifact_sha256'][str(long_path.relative_to(ROOT))]
        long_example=json.loads(long_path.read_text());long_applied=32
        for event in long_example['tool_trace']:
            if event['name']=='configure_display' and 'error' not in event['result']:
                long_applied=event['result']['current_window_ms']
        long_stated=long_example['final_report'].get('window_ms','missing') if long_example['final_report'] else 'missing'
        label='none' if long_stated is None else long_stated
        callout=f'Reasoner: stated / applied ms\n512 tokens: {stated} / {applied}\n2,048 tokens: {label} / {long_applied}'
        source_paths.extend([long_path,ROOT/'reports/gpu_agent_budget_validation.json'])
    ax.text(.03,.04,callout,
            transform=ax.transAxes,va='bottom',fontsize=8.5,bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
    ax=fig.add_axes([.67,.49,.29,.43]); arms=list(NAMES)
    for i,arm in enumerate(arms):
        m=summary['metrics'][arm]; value=100*m['success_rate'];lo,hi=np.array(m['wilson_95'])*100
        ax.barh(i,value,color=COLORS[arm],height=.62,alpha=.9)
        if arm!='rules': ax.errorbar(value,i,xerr=[[value-lo],[hi-value]],color='#253340',capsize=2,lw=.9)
        ax.text(103,i,f'{m["successes"]}/32',va='center',fontsize=9)
    ax.set(yticks=range(len(arms)),yticklabels=[NAMES[a] for a in arms],xlim=(0,130),xticks=[0,50,100],
           xlabel='Complete tasks (%)')
    ax.set_title('B  Task completion',loc='left')
    ax.tick_params(axis='y',labelsize=9.5)
    ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    ax=fig.add_axes([.16,.16,.80,.18]); neural=arms[1:]
    families=['frequency_detail','timing_detail','incompatible_constraints','fragmented_observation']
    ordered=[c for family in families for c in cases if c['family']==family]
    matrix=np.zeros((4,32),dtype=int)
    for r,arm in enumerate(neural):
        for c,case in enumerate(ordered):
            item=json.loads((BASE/'evaluation'/f'{case["case_id"]}__{arm}.json').read_text())
            checks=item['checks'];matrix[r,c]=2 if checks['strict_success'] else 1 if checks['setting_correct'] else 0
    ax.imshow(matrix,aspect='auto',vmin=0,vmax=2,cmap=ListedColormap(['#e1e5e8','#edbc67','#287e84']),interpolation='nearest')
    ax.set(yticks=range(4),yticklabels=[NAMES[a] for a in neural],xticks=[3.5,11.5,19.5,27.5],
           xticklabels=['Frequency\ndetail','Timing\ndetail','Conflicting\nlimits','Fragmented\naudio'])
    ax.set_title('C  Outcomes for individual recordings',loc='left')
    ax.tick_params(axis='y',labelsize=9.5)
    ax.tick_params(axis='both',length=0)
    for x in [7.5,15.5,23.5]:ax.axvline(x,color='white',lw=2)
    for y in [.5,1.5,2.5]:ax.axhline(y,color='white',lw=.6)
    ax.legend(handles=[Patch(facecolor='#287e84',label='Complete'),Patch(facecolor='#edbc67',label='Setting only'),
                       Patch(facecolor='#e1e5e8',label='Wrong/missing setting')],
              loc='upper center',bbox_to_anchor=(.5,-.67),ncol=3,frameon=False,fontsize=9)
    for axis in fig.axes:
        axis.spines['top'].set_visible(False);axis.spines['right'].set_visible(False)
    paths=[]
    for ext in ('pdf','png','svg'):
        p=ROOT/'figures'/('gpu_agent_comparison.'+ext)
        metadata={'Creator':'Matplotlib; frozen experiment outputs'}
        if ext=='pdf': metadata.update({'CreationDate':None,'ModDate':None})
        elif ext=='svg': metadata['Date']=None
        fig.savefig(p,dpi=450,bbox_inches='tight',metadata=metadata)
        if ext=='svg': p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
        paths.append(p)
    plt.close(fig)
    shutil.copyfile(ROOT/'figures/gpu_agent_comparison.pdf',ROOT/'manuscript/gpu_agent_comparison.pdf')
    report={'status':'passed','example_case_id':cid,'example_selection':'First frozen case, independent of agent outcomes',
            'example_stated_window_ms':stated,'example_applied_window_ms':applied,
            'example_spectral_window_ms':window,'example_cue_label':example_case['expected_report']['cue_label'],
            'example_spectral_peaks_hz':peaks,'displayed_frequency_range_hz':[20,160],
            'measurement_band_hz':[20,600],'relative_power_reference':'Shared maximum over the full measurement band and both observed regions',
            'example_long_budget_stated_window_ms':long_stated,'example_long_budget_applied_window_ms':long_applied,
            'cases':32,'arms':arms,'intervals':'Wilson 95% intervals; task-specific deterministic rule baseline shown without inferential error bar',
            'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in source_paths},
            'figure_sha256':{str(p.relative_to(ROOT)):sha(p) for p in paths},'cell_order':[c['case_id'] for c in ordered]}
    (ROOT/'figures/gpu_agent_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Built GPU comparison figure from all 160 episodes and first frozen acoustic case.')


if __name__=='__main__': main()
