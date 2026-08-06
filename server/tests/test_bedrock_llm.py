"""Tests for the server-owned Bedrock LLM and its factory registration.

The defect being guarded is a SILENT one: the library's `AWSBedrockLLM.generate_response`
accepts `response_format` and never references it, so requesting JSON is a no-op. Nothing
raises — you simply get unconstrained text, and mem0's caller reports 200 with an empty
results list when the parse fails. A test that only checked "does it return a string" would
pass against the broken version, so these assert on the REQUEST that reaches Bedrock.

Also pinned: the model-support gate. `output_config` works on Haiku 4.5 / Sonnet 4.6 /
Opus 4.6 and is rejected by Opus 4.8 / Opus 5 / Sonnet 5, so a newer model must DEGRADE to
prompt-only JSON rather than 400 on every extraction.

The *shape* of the schema is pinned separately, in tests/test_extraction_schema.py, against the
installed mem0 library.

stdlib `unittest`, no pytest — same constraint as the other server tests. boto3 is imported
for real but no client is built: `boto3.client` is patched before construction.
"""

from __future__ import annotations

import json
import unittest
from typing import Any
from unittest import mock

from mem0.utils.factory import LlmFactory, load_class

import llms
from llms.aws_bedrock import MEMORY_SCHEMA, ConstrainedAWSBedrockLLM
from llms.schema import ParsedJson

HAIKU = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
SONNET_46 = "us.anthropic.claude-sonnet-4-6"
OPUS_48 = "us.anthropic.claude-opus-4-8"

JSON_OBJECT = {"type": "json_object"}


def _payload(*texts: str) -> str:
    """A schema-valid extraction response, in the shape the deployed prompt asks for."""
    return json.dumps(
        {
            "memory": [
                {"id": str(i), "text": t, "attributed_to": "user", "linked_memory_ids": []} for i, t in enumerate(texts)
            ]
        }
    )


ONE_MEMORY = _payload("a fact")


class _FakeBedrockClient:
    """Records converse/invoke_model calls and replays a canned Converse response."""

    def __init__(self, text: str = ONE_MEMORY, stop_reason: str = "end_turn") -> None:
        self.text = text
        self.stop_reason = stop_reason
        self.converse_calls: list[dict] = []
        self.invoke_calls: list[dict] = []

    def converse(self, **params: Any) -> dict:
        self.converse_calls.append(params)
        return {
            "output": {"message": {"content": [{"text": self.text}]}},
            "stopReason": self.stop_reason,
        }

    def invoke_model(self, **params: Any) -> dict:  # pragma: no cover - fallback path only
        self.invoke_calls.append(params)
        raise AssertionError("invoke_model should not be reached for an Anthropic model")

    def list_foundation_models(self, **_params: Any) -> dict:
        """AWSBedrockLLM.__init__ calls this to "verify model availability".

        Stubbed so the real warning path stays visible: without it every test logs
        `Could not verify model availability`, which would mask a genuine failure. Note the
        production implication — constructing the LLM makes a live
        `bedrock:ListFoundationModels` call, so the pod's role needs that permission or every
        startup logs the warning (non-fatal; the check is wrapped).
        """
        return {"modelSummaries": []}

    @property
    def last_converse(self) -> dict:
        return self.converse_calls[-1]


def _llm(
    model: str = HAIKU,
    text: str = ONE_MEMORY,
    stop_reason: str = "end_turn",
):
    client = _FakeBedrockClient(text, stop_reason)
    with mock.patch("boto3.client", return_value=client):
        instance = ConstrainedAWSBedrockLLM({"model": model, "aws_region": "us-east-1"})
    # __init__ built the client under the patch; keep a handle for assertions.
    instance.client = client
    return instance, client


MESSAGES = [
    {"role": "system", "content": "You extract facts."},
    {"role": "user", "content": "Input:\nThe gateway runs on port 8800."},
]


class FactoryRegistrationTest(unittest.TestCase):
    def test_install_overrides_rebinds_aws_bedrock(self) -> None:
        llms.install_overrides()
        dotted, _config_cls = LlmFactory.provider_to_class["aws_bedrock"]
        self.assertIs(load_class(dotted), ConstrainedAWSBedrockLLM)

    def test_the_config_class_is_preserved(self) -> None:
        """LLM factory entries are (path, config_class) TUPLES, unlike the embedder factory's
        bare strings. Dropping the config class would break AWSBedrockConfig construction."""
        _, before = LlmFactory.provider_to_class["aws_bedrock"]
        llms.install_overrides()
        _, after = LlmFactory.provider_to_class["aws_bedrock"]
        self.assertIs(after, before)

    def test_install_is_idempotent(self) -> None:
        llms.install_overrides()
        first = LlmFactory.provider_to_class["aws_bedrock"]
        llms.install_overrides()
        self.assertEqual(LlmFactory.provider_to_class["aws_bedrock"], first)


class WantsJsonTest(unittest.TestCase):
    def test_recognises_what_mem0_actually_sends(self) -> None:
        self.assertTrue(ConstrainedAWSBedrockLLM._wants_json(JSON_OBJECT))

    def test_recognises_the_strict_form_and_bare_strings(self) -> None:
        self.assertTrue(ConstrainedAWSBedrockLLM._wants_json({"type": "json_schema"}))
        self.assertTrue(ConstrainedAWSBedrockLLM._wants_json("json_object"))

    def test_rejects_absent_and_text(self) -> None:
        for value in (None, "text", {}, {"type": "text"}, 5):
            with self.subTest(value=value):
                self.assertFalse(ConstrainedAWSBedrockLLM._wants_json(value))


class ConstrainedRequestTest(unittest.TestCase):
    """The core assertions: the schema must reach Bedrock, in Converse's shape."""

    def test_output_config_is_sent(self) -> None:
        llm, client = _llm()
        llm.generate_response(MESSAGES, response_format=JSON_OBJECT)
        text_format = client.last_converse["outputConfig"]["textFormat"]
        self.assertEqual(text_format["type"], "json_schema")

    def test_schema_is_a_json_STRING_not_an_object(self) -> None:
        """Converse nests differently from InvokeModel and wants a stringified schema.
        Passing the dict is a 400 that reads like a schema-validity problem."""
        llm, client = _llm()
        llm.generate_response(MESSAGES, response_format=JSON_OBJECT)
        schema = client.last_converse["outputConfig"]["textFormat"]["structure"]["jsonSchema"]["schema"]
        self.assertIsInstance(schema, str)
        self.assertEqual(json.loads(schema), MEMORY_SCHEMA)

    def test_schema_meets_bedrocks_hard_requirements(self) -> None:
        """Bedrock rejects a schema at compile time without these two, on EVERY object — so the
        nested memory item counts, not just the envelope."""
        objects = [
            MEMORY_SCHEMA,
            MEMORY_SCHEMA["properties"]["memory"]["items"],
        ]
        for obj in objects:
            self.assertFalse(obj["additionalProperties"])
            self.assertEqual(sorted(obj["required"]), sorted(obj["properties"]))

    def test_the_nested_item_is_inlined_not_a_ref(self) -> None:
        """Pydantic emits nested models as `$ref` + `$defs`. Bedrock's accepted JSON Schema
        subset does not promise to resolve those, and there the failure is the whole schema
        rejected at grammar-compile time — every extraction 400s, not just the odd one."""
        serialised = json.dumps(MEMORY_SCHEMA)
        self.assertNotIn("$ref", serialised)
        self.assertNotIn("$defs", serialised)

    def test_system_message_is_hoisted_out_of_messages(self) -> None:
        """Converse takes the system prompt top-level; leaving it in `messages` is a 400."""
        llm, client = _llm()
        llm.generate_response(MESSAGES, response_format=JSON_OBJECT)
        call = client.last_converse
        self.assertEqual(call["system"], [{"text": "You extract facts."}])
        self.assertNotIn("system", [m.get("role") for m in call["messages"]])

    def test_returns_the_model_text(self) -> None:
        llm, _ = _llm(text=_payload("x", "y"))
        out = llm.generate_response(MESSAGES, response_format=JSON_OBJECT)
        self.assertEqual([m["text"] for m in json.loads(out)["memory"]], ["x", "y"])

    def test_the_result_is_a_str_that_carries_the_parsed_model(self) -> None:
        """mem0 runs remove_code_blocks() then json.loads() on this value, so it must stay a
        `str`; server-side callers get `.parsed` without parsing it a second time."""
        llm, _ = _llm(text=_payload("x", "y"))
        out = llm.generate_response(MESSAGES, response_format=JSON_OBJECT)
        self.assertIsInstance(out, str)
        self.assertIsInstance(out, ParsedJson)
        self.assertEqual([m.text for m in out.parsed.memory], ["x", "y"])

    def test_sonnet_4_6_is_also_schema_capable(self) -> None:
        llm, client = _llm(model=SONNET_46)
        llm.generate_response(MESSAGES, response_format=JSON_OBJECT)
        self.assertIn("outputConfig", client.last_converse)


class ResponseValidationTest(unittest.TestCase):
    """A grammar guarantees the shape, not that generation finished or that the constraint
    even applied. These are the two ways a "successful" call still yields bad data."""

    def test_truncation_raises_instead_of_storing_a_partial_result(self) -> None:
        """maxTokens mid-document leaves a valid JSON PREFIX — it parses, so nothing
        downstream notices that memories are missing."""
        llm, _ = _llm(text=ONE_MEMORY, stop_reason="max_tokens")
        with self.assertRaises(ValueError) as caught:
            llm.generate_response(MESSAGES, response_format=JSON_OBJECT)
        self.assertIn("MEM0_LLM_MAX_TOKENS", str(caught.exception))

    def test_schema_mismatch_raises_rather_than_returning_it(self) -> None:
        """Unreachable while the grammar holds, which is why it must be loud if it happens:
        the alternative is mem0 turning it into a silent 200 + empty results."""
        llm, _ = _llm(text='{"wrong_key": []}')
        with self.assertRaises(ValueError):
            llm.generate_response(MESSAGES, response_format=JSON_OBJECT)

    def test_the_old_facts_shape_is_now_rejected(self) -> None:
        """The regression this rewrite exists for: `{"facts": [...]}` passed the previous schema,
        and mem0 reads `memory` — so every extraction stored nothing and reported 200 with
        `results: []`, indistinguishable from "no new facts found"."""
        llm, _ = _llm(text='{"facts": ["one", "two"]}')
        with self.assertRaises(ValueError):
            llm.generate_response(MESSAGES, response_format=JSON_OBJECT)

    def test_non_json_raises(self) -> None:
        llm, _ = _llm(text="I cannot help with that.")
        with self.assertRaises(ValueError):
            llm.generate_response(MESSAGES, response_format=JSON_OBJECT)

    def test_valid_output_is_returned_byte_for_byte(self) -> None:
        """mem0 runs its own extract_json/json.loads on this string; re-serialising through
        Pydantic would churn key order and whitespace for no gain."""
        raw = '{"memory": [{"id": "0",   "text": "one", "attributed_to": "user", "linked_memory_ids": []}]}'
        llm, _ = _llm(text=raw)
        self.assertEqual(llm.generate_response(MESSAGES, response_format=JSON_OBJECT), raw)

    def test_extra_keys_are_rejected(self) -> None:
        """extra="forbid" is what emits additionalProperties:false for Bedrock; assert the
        model actually enforces it so the two cannot drift apart."""
        llm, _ = _llm(text='{"memory": [], "sneaky": 1}')
        with self.assertRaises(ValueError):
            llm.generate_response(MESSAGES, response_format=JSON_OBJECT)

    def test_extra_keys_on_a_memory_item_are_rejected(self) -> None:
        """The nested model needs its own extra="forbid" — the envelope's does not reach it."""
        llm, _ = _llm(
            text='{"memory": [{"id": "0", "text": "one", "attributed_to": "user", '
            '"linked_memory_ids": [], "sneaky": 1}]}'
        )
        with self.assertRaises(ValueError):
            llm.generate_response(MESSAGES, response_format=JSON_OBJECT)


class DelegationTest(unittest.TestCase):
    """Everything the override must NOT touch."""

    def test_no_response_format_delegates_unconstrained(self) -> None:
        llm, client = _llm()
        llm.generate_response(MESSAGES)
        self.assertNotIn("outputConfig", client.last_converse)

    def test_tool_calls_delegate(self) -> None:
        """Tools have their own structured surface (`strict`), and mem0's memory-UPDATE path
        passes no response_format at all — neither should acquire a facts schema."""
        llm, client = _llm()
        tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
        with mock.patch.object(ConstrainedAWSBedrockLLM, "_generate_with_tools", return_value="ok") as gen:
            llm.generate_response(MESSAGES, response_format=JSON_OBJECT, tools=tools)
        gen.assert_called_once()
        self.assertEqual(client.converse_calls, [])

    def test_unsupported_model_degrades_instead_of_400ing(self) -> None:
        """Opus 4.8 rejects output_config on every AWS surface. Extraction must still run."""
        llm, client = _llm(model=OPUS_48)
        self.assertFalse(llm.supports_schema())
        with self.assertLogs("llms.aws_bedrock", level="WARNING") as logged:
            llm.generate_response(MESSAGES, response_format=JSON_OBJECT)
        self.assertNotIn("outputConfig", client.last_converse)
        self.assertTrue(any("cannot enforce a JSON schema" in m for m in logged.output))


if __name__ == "__main__":
    unittest.main(verbosity=2)
