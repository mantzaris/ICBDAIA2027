"""Stable strict JSON extraction shared by reference-label pipelines."""
from __future__ import annotations
import json
from typing import Any


def parse_json(text: str) -> dict[str, Any]:
    """Retain invalid generations; extract one JSON object without repairing values."""
    decoder = json.JSONDecoder()
    index=text.find('{')
    if index<0:
        raise ValueError('Model output contains no JSON object')
    try:
        value, _ = decoder.raw_decode(text[index:])
    except json.JSONDecodeError as error:
        # Never salvage an inner component from a truncated outer report.
        raise ValueError('The first JSON object is incomplete or invalid') from error
    if not isinstance(value,dict):
        raise ValueError('Expected one JSON object')
    return value

