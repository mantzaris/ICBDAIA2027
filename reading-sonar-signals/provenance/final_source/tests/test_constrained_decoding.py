"""Host-only grammar checks; scientific arithmetic and model inference are absent."""
from copy import deepcopy
import json
import unittest

from sonar.constrained_decoding import (action_json_schema, report_json_schema,
    make_parser, RowGrammarDispatcher, grammar_provenance)


def accepted(value, schema):
    parser = make_parser(schema)
    text = json.dumps(value, separators=(",", ":"))
    for character in text:
        if character not in parser.get_allowed_characters():
            return False
        parser = parser.add_character(character)
    return parser.can_end()


def sample_report():
    return dict(components=[dict(id="C1", type="tone", frequency_domain="acoustic",
                time_s=[10., 20.], frequency_hz=[150., 150.], evidence_ids=[])],
                relations=[], claims=[], unresolved_questions=[], abstained=False)


class GrammarTests(unittest.TestCase):
    def test_valid_tone_chirp_modulation_and_empty_reports(self):
        schema = report_json_schema()
        value = sample_report()
        self.assertTrue(accepted(value, schema))
        value["components"][0].update(type="chirp", frequency_track=[[10.,150.],[20.,250.]])
        self.assertTrue(accepted(value, schema))
        value["components"][0].update(type="modulation", frequency_domain="modulation", carrier_band_hz=[500.,2000.])
        self.assertTrue(accepted(value, schema))
        value["components"] = []; value["abstained"] = True
        self.assertTrue(accepted(value, schema))

    def test_cross_domain_missing_carrier_and_missing_chirp_track_rejected(self):
        value = sample_report(); schema = report_json_schema()
        value["components"][0]["frequency_domain"] = "modulation"
        self.assertFalse(accepted(value, schema))
        value["components"][0]["type"] = "modulation"
        self.assertFalse(accepted(value, schema))
        value = sample_report(); value["components"][0]["type"] = "chirp"
        self.assertFalse(accepted(value, schema))

    def test_component_track_claim_and_string_bounds(self):
        schema = report_json_schema(); value = sample_report()
        value["components"] *= 7
        self.assertFalse(accepted(value, schema))
        value = sample_report(); value["components"][0]["frequency_track"] = [[1,1],[2,2],[3,3],[4,4]]
        self.assertFalse(accepted(value, schema))
        value = sample_report(); value["claims"] = [dict(text="test",kind="observation",component_ids=[],evidence_ids=[])]*4
        self.assertFalse(accepted(value, schema))
        value = sample_report(); value["unresolved_questions"] = ["x"*201]
        self.assertFalse(accepted(value, schema))

    def test_unknown_properties_wrong_boolean_and_nested_pairs_rejected(self):
        schema = report_json_schema()
        for modify in [lambda r:r.update(extra=1),lambda r:r.update(abstained=0),
                       lambda r:r["components"][0].update(time_s=[[10],20])]:
            value = sample_report(); modify(value)
            self.assertFalse(accepted(value,schema))

    def test_only_available_integer_action_indices(self):
        schema = action_json_schema([0, 2, 10])
        for index in [0,2,10]:
            self.assertTrue(accepted(dict(action_index=index,reason="inspect"),schema))
        for index in [1,-1,True,2.,"2"]:
            self.assertFalse(accepted(dict(action_index=index,reason="inspect"),schema))
        for indices in [[],[0,0],[True],[1.]]:
            with self.assertRaises(ValueError): action_json_schema(indices)

    def test_syntactically_valid_unsupported_values_not_injected_or_repaired(self):
        value = sample_report()
        value["components"][0].update(time_s=[20,10],frequency_hz=[100000,100001],
                                       evidence_ids=["made_up"],source_identity="unsupported")
        self.assertTrue(accepted(value,report_json_schema()))
        # Ordering, evidence existence, and unsupported physical identity belong
        # to the separate validator/verifier, rather than a correct-answer grammar.

    def test_row_dispatch_keeps_schema_identity_and_caps_without_parsing_padding(self):
        calls = []
        def callback(which):
            def call(_, suffix):
                calls.append((which,list(suffix))); return [which+10]
            return call
        dispatch = RowGrammarDispatcher([callback(0),callback(1)],3,[2,8],[99])
        self.assertEqual(dispatch(0,[11,12,13]),[10])
        self.assertEqual(dispatch(1,[0,0,13,5]),[11])
        self.assertEqual(dispatch(0,[11,12,13,5,6]),[99])
        self.assertEqual(dispatch(1,[0,0,13,5,99]),[99])
        self.assertEqual(dispatch(1,[0,0,13,5,99,0]),[99])
        self.assertEqual(calls,[(0,[]),(1,[5])])

    def test_provenance_pins_protocol_and_package(self):
        provenance = grammar_provenance()
        self.assertEqual(provenance["package_version"],"0.11.3")
        self.assertEqual(provenance["logit_mask_device"],"cuda")
        self.assertEqual(len(provenance["report_schema_sha256"]),64)


if __name__ == "__main__": unittest.main(verbosity=2)
