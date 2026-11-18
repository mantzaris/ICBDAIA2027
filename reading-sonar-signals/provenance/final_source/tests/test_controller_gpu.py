"""Scientific numeric verifier checks execute on CUDA only."""
import unittest
from copy import deepcopy
import torch
from sonar.controller import verify_report


@unittest.skipUnless(torch.cuda.is_available(),"CUDA is required; no CPU numeric fallback")
class TrackVerificationTests(unittest.TestCase):
    def test_supplied_tone_track_must_agree_with_cited_measurement(self):
        component=dict(id="C1",type="tone",frequency_domain="acoustic",time_s=[0.,10.],
            frequency_hz=[100.,100.],evidence_ids=["E0"],source_identity="unknown")
        candidate=dict(component,id="C0000",frequency_track=[[0.,100.],[10.,100.]],bin_spacing_hz=1.)
        state=dict(components=[candidate])
        report=dict(components=[component],relations=[],claims=[],unresolved_questions=[],abstained=False)
        verified,audit=verify_report(report,state)
        self.assertEqual(len(verified["components"]),1)
        report=deepcopy(report)
        report["components"][0]["frequency_track"]=[[0.,200.],[10.,200.]]
        verified,audit=verify_report(report,state)
        self.assertEqual(verified["components"],[])
        self.assertEqual(audit[0]["status"],"withheld")

    def test_modulation_track_and_carrier_checked_in_modulation_domain(self):
        component=dict(id="C1",type="modulation",frequency_domain="modulation",time_s=[0.,10.],
            frequency_hz=[12.,12.],frequency_track=[[0.,12.],[10.,12.]],carrier_band_hz=[500.,2000.],
            evidence_ids=["E0"],source_identity="unknown")
        candidate=dict(component,id="C0000",bin_spacing_hz=.25)
        report=dict(components=[component],relations=[],claims=[],unresolved_questions=[],abstained=False)
        state=dict(components=[candidate])
        self.assertEqual(len(verify_report(report,state)[0]["components"]),1)
        report=deepcopy(report);report["components"][0]["frequency_track"]=[[0.,20.],[10.,20.]]
        self.assertEqual(verify_report(report,state)[0]["components"],[])


if __name__=="__main__":unittest.main(verbosity=2)
