import torch
import pytest
from sonar.input_interventions import localized_attenuation,load_experimental_waveform,DisplayOverride


def test_stft_sham_identity_and_localized_selectivity():
    sr=8000;t=torch.arange(sr*30,device='cuda')/sr
    target=torch.sin(2*torch.pi*300*t);control=torch.sin(2*torch.pi*700*t)
    x=target+control
    identity=localized_attenuation(x,sr,[200,400],[10,20],1.)
    assert (x-identity).abs().max()<2e-6
    attenuated=localized_attenuation(x,sr,[200,400],[10,20],.1)
    active=(t>12)&(t<18);outside=(t>2)&(t<8)
    def amplitude(wave,basis,mask):return 2*(wave[mask]*basis[mask]).mean()
    assert amplitude(attenuated,target,active).item()==pytest.approx(.1,abs=.01)
    assert amplitude(attenuated,control,active).item()==pytest.approx(1.,abs=.01)
    assert amplitude(attenuated,target,outside).item()==pytest.approx(1.,abs=.01)


def test_duration_updates_exact_chirp_coordinates():
    base=dict(stratum='synthetic',seed=811200,split='train',kind='chirp',sample_rate=16000,duration_seconds=30,snr_db=10)
    x,sr,origin,truth=load_experimental_waveform(dict(sensitivity_base=base,sensitivity=dict(name='duration',duration_seconds=10)))
    assert x.numel()==10*sr
    assert truth['events']
    for event in truth['events']:
        assert origin<=event['time_s'][0]<event['time_s'][1]<=10
        if event.get('frequency_track'):
            assert event['frequency_track'][0][0]==pytest.approx(event['time_s'][0])
            assert event['frequency_track'][-1][0]==pytest.approx(event['time_s'][1])


def test_real_parent_removal_preserves_original_scale(monkeypatch):
    import sonar.input_interventions as module
    first=torch.ones(20,device='cuda');second=torch.full((20,),3.,device='cuda')
    a={'name':'a'};b={'name':'b'}
    base=dict(stratum='semi_synthetic',parents=[a,b],offset_seconds=1)
    mixture=torch.zeros(30,device='cuda');mixture[:20]+=first/4;mixture[10:]+=second*2/4
    def fake_load(row):
        if row is base:return mixture,10,0.,dict(gain_second=2.,global_scale=4.)
        return (first if row is a else second),10,0.,None
    monkeypatch.setattr(module,'load_waveform',fake_load)
    actual,_,_,_=load_experimental_waveform(dict(sensitivity_base=base,sensitivity=dict(name='remove_real_parent',removed_parent=1)))
    expected=torch.zeros(30,device='cuda');expected[:20]=first/4
    assert torch.equal(actual,expected)


def test_display_probe_replaces_only_initial_demon():
    class Capture:
        def generate(self,prompt,images,max_new_tokens,**kwargs):return {'images':images,'kwargs':kwargs}
    model=DisplayOverride(Capture(),'replacement.png',{'kind':'fixture'})
    result=model.generate('prompt',['x/global_lofar.png','x/global_demon.png','x/action_0_E1.png'],30,schema={'type':'object'})
    assert result['images']==['x/global_lofar.png','replacement.png','x/action_0_E1.png']
    assert result['display_intervention']['replaced_initial_DEMON']
    assert result['kwargs']['schema']=={'type':'object'}
