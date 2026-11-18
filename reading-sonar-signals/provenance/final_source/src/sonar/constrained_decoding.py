"""Content-agnostic JSON grammar for bounded acoustic reports and action choices.

The grammar constrains serialization, types, and output size. It supplies no
event frequencies, times, evidence IDs, or expected answers. Numeric ordering,
unique IDs, and measurement consistency remain independent validation checks.
Grammar state is host symbolic work; Transformers applies its mask to CUDA
logits. Token-budget truncation is preserved as an incomplete raw generation.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.metadata
import json

LMFE_VERSION = "0.11.3"
PROTOCOL_VERSION = "bounded_json_grammar_v1"
PARSER_CONFIG = dict(max_consecutive_whitespaces=2, force_json_field_order=False,
                     max_json_array_length=6)


def _string(maximum=80, minimum=0):
    return dict(type="string", minLength=minimum, maxLength=maximum)


def _array(items, maximum, minimum=0):
    return dict(type="array", items=items, minItems=minimum, maxItems=maximum)


def _object(properties, required=None):
    return dict(type="object", properties=properties,
                required=list(properties) if required is None else required,
                additionalProperties=False)


def report_json_schema():
    pair = _array(dict(type="number"), 2, 2)
    ids = _array(_string(40, 1), 6)
    properties = dict(id=_string(40, 1), type=dict(type="string"),
                      frequency_domain=dict(type="string"),
                      time_s=pair, frequency_hz=pair,
                      frequency_track=_array(pair, 3, 2),
                      carrier_band_hz=dict(anyOf=[deepcopy(pair), dict(type="null")]),
                      source_identity=dict(anyOf=[_string(80), dict(type="null")]),
                      evidence_ids=ids)
    required = ["id", "type", "frequency_domain", "time_s", "frequency_hz", "evidence_ids"]
    variants = []
    for kinds, domain, extra in [(["tone", "transient", "broadband"], "acoustic", []),
                                  (["chirp"], "acoustic", ["frequency_track"]),
                                  (["modulation"], "modulation", ["carrier_band_hz"])]:
        fields = deepcopy(properties)
        fields["type"]["enum"] = kinds
        fields["frequency_domain"]["enum"] = [domain]
        if domain == "modulation":
            fields["carrier_band_hz"] = deepcopy(pair)
        variants.append(_object(fields, required + extra))
    relation = _object(dict(type=dict(type="string", enum=["overlaps_time", "harmonic_candidate", "unresolved_association"]),
                            component_ids=_array(_string(40, 1), 6, 2), evidence_ids=ids))
    claim = _object(dict(text=_string(200, 1), component_ids=ids, evidence_ids=ids,
                        kind=_string(40, 1)))
    return _object(dict(components=_array(dict(anyOf=variants), 6),
                        relations=_array(relation, 6), claims=_array(claim, 3),
                        unresolved_questions=_array(_string(200, 1), 3),
                        abstained=dict(type="boolean")))


def action_json_schema(indices):
    choices = list(indices)
    if not choices or any(type(i) is not int or i < 0 for i in choices) or len(set(choices)) != len(choices):
        raise ValueError("Action grammar requires distinct nonnegative integer choices")
    return _object(dict(action_index=dict(type="integer", enum=choices), reason=_string(120)))


def schema_sha256(schema):
    return hashlib.sha256(json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def grammar_provenance():
    installed = importlib.metadata.version("lm-format-enforcer")
    if installed != LMFE_VERSION:
        raise RuntimeError(f"Expected lm-format-enforcer=={LMFE_VERSION}; found {installed}")
    distribution = importlib.metadata.distribution("lm-format-enforcer")
    package_sources = {str(path):hashlib.sha256(distribution.locate_file(path).read_bytes()).hexdigest()
                       for path in (distribution.files or [])
                       if str(path).startswith("lmformatenforcer/") and str(path).endswith(".py")}
    dependencies = {}
    for name in ["pydantic", "pydantic-core", "interegular", "transformers"]:
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None  # Standalone host grammar tests need no model runtime.
    return dict(protocol=PROTOCOL_VERSION, package="lm-format-enforcer", package_version=installed,
                package_source_sha256=package_sources, dependency_versions=dependencies,
                report_schema_sha256=schema_sha256(report_json_schema()), parser_config=dict(PARSER_CONFIG),
                scope="serialization_types_size_and_available_action_indices_only",
                numeric_ordering_and_evidence_validation="post_generation",
                implementation="official_transformers_prefix_allowed_tokens_fn_per_row",
                grammar_device="host_symbolic", logit_mask_device="cuda",
                token_limit_policy="preserve_incomplete_raw_output_as_failure")


def make_parser(schema, alphabet=None):
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.characterlevelparser import CharacterLevelParserConfig
    settings = dict(PARSER_CONFIG)
    if alphabet is not None:
        settings["alphabet"] = alphabet
    return JsonSchemaParser(schema, config=CharacterLevelParserConfig(**settings))


class RowGrammarDispatcher:
    """Independent row grammar state; works with unequal prompts and token caps.

    Token IDs after the shared left-padded prompt width are symbolic strings, not
    signal processing or model logits. Already stopped/capped rows permit EOS so
    Transformers can keep padding them while other rows continue. No incomplete
    prefix is ever repaired or represented as schema-complete.
    """
    def __init__(self, callbacks, prompt_width, limits, eos_token_ids):
        if len(callbacks) != len(limits) or not eos_token_ids:
            raise ValueError("Invalid row grammar configuration")
        self.callbacks = callbacks
        self.prompt_width = int(prompt_width)
        self.limits = list(limits)
        self.eos_token_ids = [int(i) for i in eos_token_ids]
        self.finished = set()

    def __call__(self, batch_id, sent):
        if not 0 <= batch_id < len(self.callbacks):
            raise IndexError("Unexpected generation batch row")
        suffix = sent[self.prompt_width:]
        if batch_id in self.finished or len(suffix) >= self.limits[batch_id]:
            return self.eos_token_ids
        # Natural EOS is the only special token the grammar can emit. Finished
        # rows subsequently receive pad tokens from Transformers; skip parsing.
        if len(suffix) and int(suffix[-1]) in self.eos_token_ids:
            self.finished.add(batch_id)
            return self.eos_token_ids
        return self.callbacks[batch_id](0, suffix)


class GrammarFactory:
    """One tokenizer trie per backend; never shared mutable parser state."""
    def __init__(self, tokenizer):
        from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data
        self.provenance = grammar_provenance()
        self.tokenizer_data = build_token_enforcer_tokenizer_data(tokenizer)

    def for_batch(self, schemas, prompt_width, limits, eos_token_ids):
        from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
        from lmformatenforcer.characterlevelparser import CharacterLevelParserConfig
        callbacks = []
        for schema in schemas:
            if schema is None:
                raise ValueError("Cannot mix constrained and unconstrained requests in one batch")
            parser = make_parser(schema, self.tokenizer_data.tokenizer_alphabet)
            callback = build_transformers_prefix_allowed_tokens_fn(self.tokenizer_data, parser)
            # LMFE 0.11.3 replaces config in TokenEnforcer.__init__. Restore our
            # explicit settings before any prefix is consumed (upstream main
            # fixes this, but the published dependency is pinned deliberately).
            parser.config = CharacterLevelParserConfig(alphabet=self.tokenizer_data.tokenizer_alphabet,
                                                       **PARSER_CONFIG)
            callbacks.append(callback)
        return RowGrammarDispatcher(callbacks, prompt_width, limits, eos_token_ids)
