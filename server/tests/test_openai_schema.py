"""Tests for the OpenAI-compatible schema-enforcing LLM override.

The defect guarded here is a one-word difference with a huge behavioural gap:
`response_format={"type": "json_object"}` (what mem0 sends) is a polite request that fails to
parse on ~19% of real inputs and does so SILENTLY, while
`{"type": "json_schema", ..., "strict": true}` is grammar-constrained. Nothing raises when the
weak form is used, so these assert on the REQUEST handed to the parent class rather than on the
return value.

stdlib `unittest`, no pytest — same constraint as the other server tests. The OpenAI client is
constructed for real (it makes no network call at init) and the parent's `generate_response`
is patched, so nothing here talks to a backend.
"""

from __future__ import annotations

import unittest
from typing import Any
from unittest import mock

from mem0.llms.openai import OpenAILLM
from mem0.utils.factory import LlmFactory, load_class

import llms
from llms.openai_schema import SchemaEnforcingOpenAILLM
from llms.schema import FACTS_SCHEMA, OPENAI_STRICT_RESPONSE_FORMAT

WEAK = {"type": "json_object"}
MESSAGES = [
    {"role": "system", "content": "You extract facts."},
    {"role": "user", "content": "Input:\nThe gateway runs on port 8800."},
]
GOOD = '{"facts": ["The gateway runs on port 8800"]}'


def _llm(model: str = "gpt-5.5") -> SchemaEnforcingOpenAILLM:
    return SchemaEnforcingOpenAILLM(
        {"model": model, "api_key": "test-key", "openai_base_url": "http://bifrost.invalid/v1"}
    )


def _call(llm: SchemaEnforcingOpenAILLM, returns: str = GOOD, **kwargs: Any):
    """Invoke through the override with the parent patched; returns (result, parent_kwargs)."""
    with mock.patch.object(OpenAILLM, "generate_response", return_value=returns) as parent:
        result = llm.generate_response(MESSAGES, **kwargs)
    return result, (parent.call_args.kwargs if parent.call_args else {})


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
        replace it with the facts schema."""
        self.assertFalse(
            SchemaEnforcingOpenAILLM._is_weak_json({"type": "json_schema", "json_schema": {}})
        )

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
        self.assertEqual(sent["json_schema"]["schema"], FACTS_SCHEMA)

    def test_schema_meets_strict_mode_requirements(self) -> None:
        """`strict: true` is rejected unless every object forbids extra properties and lists
        every property in `required` — the same two rules Bedrock enforces."""
        self.assertFalse(FACTS_SCHEMA["additionalProperties"])
        self.assertEqual(sorted(FACTS_SCHEMA["required"]), sorted(FACTS_SCHEMA["properties"]))

    def test_explicit_schema_is_passed_through_untouched(self) -> None:
        mine = {"type": "json_schema", "json_schema": {"name": "mine", "schema": {}}}
        _, kwargs = _call(_llm(), response_format=mine)
        self.assertEqual(kwargs["response_format"], mine)

    def test_absent_response_format_stays_absent(self) -> None:
        """The memory-UPDATE path sends none; it must not acquire a facts schema."""
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
        raw = '{"facts": ["one",   "two"]}'
        result, _ = _call(_llm(), returns=raw, response_format=WEAK)
        self.assertEqual(result, raw)

    def test_schema_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            _call(_llm(), returns='{"wrong": []}', response_format=WEAK)

    def test_truncated_output_raises_and_says_why(self) -> None:
        """max_tokens mid-document is the realistic cause; the message must name it so the
        operator does not go hunting for a schema bug."""
        with self.assertRaises(ValueError) as caught:
            _call(_llm(), returns='{"facts": ["one", "tw', response_format=WEAK)
        self.assertIn("max_tokens", str(caught.exception))

    def test_non_json_raises(self) -> None:
        with self.assertRaises(ValueError):
            _call(_llm(), returns="I cannot help with that.", response_format=WEAK)

    def test_unupgraded_calls_are_not_validated(self) -> None:
        """Only what we constrained is our responsibility — a tool call returns a dict, and
        validating it as facts JSON would break the update path."""
        result, _ = _call(_llm(), returns="not json at all")
        self.assertEqual(result, "not json at all")


class SharedShapeTest(unittest.TestCase):
    def test_strict_constant_matches_what_is_sent(self) -> None:
        _, kwargs = _call(_llm(), response_format=WEAK)
        self.assertEqual(kwargs["response_format"], OPENAI_STRICT_RESPONSE_FORMAT)

    def test_bedrock_and_openai_share_one_schema(self) -> None:
        """Two providers, one declared shape — they must not drift."""
        from llms.aws_bedrock import FACTS_SCHEMA as BEDROCK_SCHEMA

        self.assertIs(BEDROCK_SCHEMA, FACTS_SCHEMA)


if __name__ == "__main__":
    unittest.main(verbosity=2)
