"""Administrative JSON checks: truncated reports must never become successes."""
import pytest
from sonar.vlm import parse_json


def test_complete_report():
    assert parse_json('{"components": [{"id": "C1"}]}')['components'][0]['id']=='C1'


def test_truncated_outer_report_does_not_salvage_component():
    with pytest.raises(ValueError):
        parse_json('{"components": [{"id": "C1"}, {"id": "C2"}')


def test_fenced_json():
    assert parse_json('```json\n{"action_index": 2}\n```')['action_index']==2


def test_missing_json():
    with pytest.raises(ValueError):
        parse_json('No output.')
