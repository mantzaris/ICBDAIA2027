"""Prompt projection is a serialization operation, not a new measurement."""
from copy import deepcopy
import unittest
from sonar.prompting import compact_measurements, prompt_numbers


class PromptProjectionTests(unittest.TestCase):
    def test_preserves_ids_geometry_endpoints_and_original_record(self):
        state=dict(scene_id="opaque",time_s=[100.,130.],sample_rate_hz=16000.,limitations=[],
            components=[dict(id="C0000",scene_id="opaque",type="chirp",frequency_domain="acoustic",
                time_s=[100.123456,120.876543],frequency_hz=[200.123456,300.987654],
                frequency_track=[[100.,200.],[105.,225.],[110.,250.],[115.,275.],[120.,300.]],
                evidence_ids=["E0000"],confidence_kind="uncalibrated_detector_contrast")],
            measurements=[dict(id="E0000",scene_id="opaque",tool="inspect_region",component_ids=["C0000"],
                band_power=dict(mean_square=1.23456789e-12,level_db=-119.0848,calibrated=False,
                                units="digital_amplitude_squared",time_s=[100.,120.]),
                configuration=dict(window_s=1.,hop_s=.25,window_samples=16000,device="cuda:0"))],
            relations=[])
        original=deepcopy(state)
        projected=compact_measurements(state)
        self.assertEqual(state,original)
        component=projected["components"][0]
        self.assertEqual(component["frequency_track"],[[100.,200.],[110.,250.],[120.,300.]])
        self.assertEqual(component["id"],"C0000")
        self.assertEqual(component["evidence_ids"],["E0000"])
        self.assertNotIn("scene_id",component)
        self.assertEqual(component["time_s"],[100.123,120.877])
        measurement=projected["measurement_index"][0]
        self.assertEqual(measurement["band_power"]["mean_square"],1.23457e-12)
        self.assertNotIn("device",measurement["configuration"])
        self.assertFalse(measurement["band_power"]["calibrated"])

    def test_no_quantitative_keeps_evidence_ids_and_flat_mode_hides_canonical_links(self):
        state=dict(components=[],relations=[dict(type="overlaps_time",component_ids=["C1","C2"])],
            measurements=[dict(id="E0000",component_ids=["C1"],input_component_ids=["C1"],relations=[])])
        image_only=compact_measurements(state,quantitative=False,graph=False)
        self.assertEqual(image_only["available_evidence_ids"],["E0000"])
        self.assertNotIn("components",image_only)
        self.assertNotIn("measurement_index",image_only)
        flat=compact_measurements(state,graph=False)
        self.assertNotIn("relations",flat)
        self.assertEqual(flat["measurement_index"],[dict(id="E0000")])

    def test_action_indices_and_labels_are_not_rounded_or_replaced(self):
        value=dict(action_index=7,name="inspect_region",time_s=[0.123456,29.987654])
        result=prompt_numbers(value)
        self.assertEqual(result["action_index"],7)
        self.assertIs(type(result["action_index"]),int)
        self.assertEqual(result["name"],value["name"])
        self.assertEqual(result["time_s"],[.123,29.988])

    def test_peak_summary_retains_fixed_detector_order_and_full_trace(self):
        peaks=[dict(frequency_hz=float(100+i),contrast_db=float(20-i),bin_spacing_hz=1.,psd=1.e-9) for i in range(9)]
        state=dict(components=[],relations=[],measurements=[dict(id="E0",tool="measure_peaks",peaks=peaks)])
        projected=compact_measurements(state)
        summarized=projected["measurement_index"][0]["peaks"]
        self.assertEqual([p["frequency_hz"] for p in summarized],[100.,101.,102.,103.,104.,105.])
        self.assertEqual(len(state["measurements"][0]["peaks"]),9)
        self.assertTrue(all("psd" not in p for p in summarized))


if __name__=="__main__":unittest.main(verbosity=2)
