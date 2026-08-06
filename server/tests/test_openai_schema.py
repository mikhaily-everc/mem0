"""Tests for the OpenAI-compatible schema-enforcing LLM override.

The defect guarded here is a one-word difference with a huge behavioural gap:
`response_format={"type": "json_object"}` (what mem0 sends) is a polite request that fails to
parse on ~19% of real inputs and does so SILENTLY, while
`{"type": "json_schema", ..., "strict": true}` is grammar-constrained. Nothing raises when the
weak form is used, so these assert on the REQUEST handed to the parent class rather than on the
return value.

The *shape* of that schema is pinned separately, in tests/test_extraction_schema.py, against the
installed mem0 library — that is the one that goes red if the envelope key drifts away from what
mem0 reads.

stdlib `unittest`, no pytest — same constraint as the other server tests. The OpenAI client is
constructed for real (it makes no network call at init) and the parent's `generate_response`
is patched, so nothing here talks to a backend.
"""

from __future__ import annotations

import json
import unittest
from typing import Any
from unittest import mock

from mem0.llms.openai import OpenAILLM
from mem0.utils.factory import LlmFactory, load_class

import llms
from llms.openai_schema import SchemaEnforcingOpenAILLM
from llms.schema import MEMORY_SCHEMA, OPENAI_STRICT_RESPONSE_FORMAT, ParsedJson

WEAK = {"type": "json_object"}
MESSAGES = [
    {"role": "system", "content": "You extract facts."},
    {"role": "user", "content": "Input:\nThe gateway runs on port 8800."},
]


def _payload(*texts: str) -> str:
    """A schema-valid extraction response, in the shape the deployed prompt asks for."""
    return json.dumps(
        {
            "memory": [
                {
                    "id": str(i),
                    "text": text,
                    "attributed_to": "user",
                    "linked_memory_ids": [],
                }
                for i, text in enumerate(texts)
            ]
        }
    )


GOOD = _payload("The gateway runs on port 8800")

# The shape this override USED to force. mem0 reads `memory`, so a response like this stores
# nothing and reports 200 + `results: []`. Kept as a literal so the regression cannot come back
# by way of a helper that was updated in lockstep with the bug.
OLD_FACTS_SHAPE = '{"facts": ["The gateway runs on port 8800"]}'


def _llm(model: str = "gpt-5.5") -> SchemaEnforcingOpenAILLM:
    return SchemaEnforcingOpenAILLM(
        {"model": model, "api_key": "test-key", "openai_base_url": "http://bifrost.invalid/v1"}
    )


def _call(llm: SchemaEnforcingOpenAILLM, returns: str = GOOD, **kwargs: Any):
    """Invoke through the override with the parent patched; returns (result, parent_kwargs)."""
    with mock.patch.object(OpenAILLM, "generate_response", return_value=returns) as parent:
        result = llm.generate_response(MESSAGES, **kwargs)
    return result, (parent.call_args.kwargs if parent.call_args else {})


def _every_object(node: Any):
    """Yield every JSON-Schema object node, so assertions cover nested models too."""
    if isinstance(node, dict):
        if node.get("type") == "object":
            yield node
        for value in node.values():
            yield from _every_object(value)
    elif isinstance(node, list):
        for value in node:
            yield from _every_object(value)


class FactoryRegistrationTest(unittest.TestCase):
    def test_openai_is_rebound(self) -> None:
        llms.install_overrides()
        dotted, _ = LlmFactory.provider_to_class["openai"]
        self.assertIs(load_class(dotted), SchemaEnforcingOpenAILLM)

    def test_config_class_is_preserved(self) -> None:
        _, before = LlmFactory.provider_to_class["openai"]
        llms.install_overrides()
        _, after = LlmFactory.provider_to_class["openai"]
        self.assertIs(after, before)

    def test_both_providers_registered_together(self) -> None:
        llms.install_overrides()
        for provider in ("openai", "aws_bedrock"):
            with self.subTest(provider=provider):
                self.assertIsInstance(LlmFactory.provider_to_class[provider], tuple)


class WeakJsonDetectionTest(unittest.TestCase):
    """Deliberately narrow: only mem0's exact form is rewritten."""

    def test_matches_mem0s_form(self) -> None:
        self.assertTrue(SchemaEnforcingOpenAILLM._is_weak_json(WEAK))
        self.assertTrue(SchemaEnforcingOpenAILLM._is_weak_json("json_object"))

    def test_does_not_match_an_explicit_schema(self) -> None:
        """A caller who already supplied a schema made a considered choice — never silently
        replace it with the extraction schema."""
        self.assertFalse(SchemaEnforcingOpenAILLM._is_weak_json({"type": "json_schema", "json_schema": {}}))

    def test_does_not_match_text_or_absent(self) -> None:
        for value in (None, "text", {}, {"type": "text"}, 7):
            with self.subTest(value=value):
                self.assertFalse(SchemaEnforcingOpenAILLM._is_weak_json(value))


class UpgradeTest(unittest.TestCase):
    def test_weak_request_becomes_strict(self) -> None:
        _, kwargs = _call(_llm(), response_format=WEAK)
        sent = kwargs["response_format"]
        self.assertEqual(sent["type"], "json_schema")
        self.assertTrue(sent["json_schema"]["strict"])
        self.assertEqual(sent["json_schema"]["schema"], MEMORY_SCHEMA)

    def test_schema_meets_strict_mode_requirements(self) -> None:
        """`strict: true` is rejected unless EVERY object forbids extra properties and lists
        every property in `required` — the same two rules Bedrock enforces. Checked on every
        nested object, not just the envelope, which is where a new field would slip through."""
        objects = list(_every_object(MEMORY_SCHEMA))
        self.assertEqual(len(objects), 2, "expected the envelope and the memory item")
        for obj in objects:
            self.assertFalse(obj["additionalProperties"])
            self.assertEqual(sorted(obj["required"]), sorted(obj["properties"]))

    def test_no_refs_survive_into_the_wire_schema(self) -> None:
        """Pydantic emits nested models as `$ref` + `$defs`. OpenAI resolves those; Bedrock's
        documented subset does not promise to, and there an unresolvable schema is rejected at
        grammar-compile time — every extraction 400s, not just the odd one."""
        serialised = json.dumps(MEMORY_SCHEMA)
        self.assertNotIn("$ref", serialised)
        self.assertNotIn("$defs", serialised)

    def test_explicit_schema_is_passed_through_untouched(self) -> None:
        mine = {"type": "json_schema", "json_schema": {"name": "mine", "schema": {}}}
        _, kwargs = _call(_llm(), response_format=mine)
        self.assertEqual(kwargs["response_format"], mine)

    def test_absent_response_format_stays_absent(self) -> None:
        """The memory-UPDATE path sends none; it must not acquire an extraction schema."""
        _, kwargs = _call(_llm())
        self.assertIsNone(kwargs["response_format"])

    def test_tool_calls_are_not_upgraded(self) -> None:
        tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
        _, kwargs = _call(_llm(), response_format=WEAK, tools=tools)
        self.assertEqual(kwargs["response_format"], WEAK)

    def test_env_escape_hatch_disables_the_upgrade(self) -> None:
        """For an OpenAI-compatible backend that only supports the weak form, upgrading would
        turn flaky-but-working extraction into a hard 400."""
        with mock.patch.dict("os.environ", {"MEM0_LLM_STRICT_SCHEMA": "false"}):
            _, kwargs = _call(_llm(), response_format=WEAK)
        self.assertEqual(kwargs["response_format"], WEAK)


class ValidationTest(unittest.TestCase):
    def test_valid_output_is_returned_byte_for_byte(self) -> None:
        raw = '{"memory": [{"id": "0",   "text": "one", "attributed_to": "user", "linked_memory_ids": []}]}'
        result, _ = _call(_llm(), returns=raw, response_format=WEAK)
        self.assertEqual(result, raw)

    def test_the_old_facts_shape_is_now_rejected(self) -> None:
        """The regression this file exists for. `{"facts": [...]}` satisfied the previous schema
        and mem0 read `memory` off it, got nothing, and answered 200 with `results: []`."""
        with self.assertRaises(ValueError):
            _call(_llm(), returns=OLD_FACTS_SHAPE, response_format=WEAK)

    def test_schema_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            _call(_llm(), returns='{"wrong": []}', response_format=WEAK)

    def test_a_memory_missing_text_raises(self) -> None:
        """`text` is the one field mem0 cannot do without — it is the memory, the embedding
        input and the md5 dedup key. An element without it is dropped silently by mem0."""
        with self.assertRaises(ValueError):
            _call(
                _llm(),
                returns='{"memory": [{"id": "0", "attributed_to": "user", "linked_memory_ids": []}]}',
                response_format=WEAK,
            )

    def test_truncated_output_raises_and_says_why(self) -> None:
        """max_tokens mid-document is the realistic cause; the message must name it so the
        operator does not go hunting for a schema bug."""
        with self.assertRaises(ValueError) as caught:
            _call(_llm(), returns='{"memory": [{"id": "0", "text": "tw', response_format=WEAK)
        self.assertIn("max_tokens", str(caught.exception))

    def test_non_json_raises(self) -> None:
        with self.assertRaises(ValueError):
            _call(_llm(), returns="I cannot help with that.", response_format=WEAK)

    def test_unupgraded_calls_are_not_validated(self) -> None:
        """Only what we constrained is our responsibility — a tool call returns a dict, and
        validating it as extraction JSON would break the update path."""
        result, _ = _call(_llm(), returns="not json at all")
        self.assertEqual(result, "not json at all")


class ParsedJsonTest(unittest.TestCase):
    """The return value has to satisfy two callers at once: mem0, which treats it as text, and
    server-side code, which wants the model without parsing twice."""

    def test_the_result_is_still_a_str_for_mem0(self) -> None:
        """mem0 runs remove_code_blocks() then json.loads() on this. A BaseModel breaks both,
        and that library code ships in the PyPI wheel — it cannot be patched from server/."""
        result, _ = _call(_llm(), response_format=WEAK)
        self.assertIsInstance(result, str)
        self.assertEqual(json.loads(result)["memory"][0]["text"], "The gateway runs on port 8800")

    def test_the_result_carries_the_parsed_model(self) -> None:
        result, _ = _call(_llm(), returns=_payload("one", "two"), response_format=WEAK)
        self.assertIsInstance(result, ParsedJson)
        self.assertEqual([m.text for m in result.parsed.memory], ["one", "two"])
        self.assertEqual([m.id for m in result.parsed.memory], ["0", "1"])

    def test_an_unupgraded_result_is_a_plain_str(self) -> None:
        """Nothing was constrained, so there is nothing to have parsed."""
        result, _ = _call(_llm(), returns="anything")
        self.assertNotIsInstance(result, ParsedJson)


class SharedShapeTest(unittest.TestCase):
    def test_strict_constant_matches_what_is_sent(self) -> None:
        _, kwargs = _call(_llm(), response_format=WEAK)
        self.assertEqual(kwargs["response_format"], OPENAI_STRICT_RESPONSE_FORMAT)

    def test_bedrock_and_openai_share_one_schema(self) -> None:
        """Two providers, one declared shape — they must not drift."""
        from llms.aws_bedrock import MEMORY_SCHEMA as BEDROCK_SCHEMA

        self.assertIs(BEDROCK_SCHEMA, MEMORY_SCHEMA)

    def test_no_prose_rides_on_the_wire(self) -> None:
        """Pydantic puts each model DOCSTRING into `description` and field names into `title`,
        and the schema is sent with every extraction request. This was observed verbatim in
        bifrost's echoed raw_request — hundreds of characters of internal commentary per call.
        Neither key affects constrained decoding, so both are stripped."""

        def walk(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    self.assertNotIn(key, ("description", "title"))
                    walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)

        walk(MEMORY_SCHEMA)

    def test_stripping_left_the_load_bearing_keys(self) -> None:
        """The strip must not take the keys the providers actually require."""
        self.assertFalse(MEMORY_SCHEMA["additionalProperties"])
        self.assertEqual(MEMORY_SCHEMA["required"], ["memory"])
        self.assertEqual(MEMORY_SCHEMA["properties"]["memory"]["type"], "array")


if __name__ == "__main__":
    unittest.main(verbosity=2)
