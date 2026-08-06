"""Bedrock LLM that actually honours `response_format`.

`mem0.llms.aws_bedrock.AWSBedrockLLM.generate_response` accepts a `response_format`
argument and then never references it — it falls straight through to `_generate_standard`.
So asking mem0 for JSON via the Bedrock provider is a silent no-op: you get whatever the
model felt like emitting, and the caller's `json.loads` fails some fraction of the time.
That is the same defect shape as the Bedrock *embedder* accepting `memory_action` and
discarding it.

Measured on this corpus, mem0's prompt-only JSON mode returns unparseable output on ~19% of
real inputs, and each failure is silent: `POST /memories` still answers 200 with
`{"results": []}`, which is byte-identical to "no new facts found". Grammar-constrained
decoding removes that failure class outright — 0 invalid responses in 24 measured calls.

Only the Anthropic-on-Bedrock path is reimplemented, because that is the one where a schema
can be enforced, and only when the caller actually asked for JSON. Everything else — other
providers, tool calls, unconstrained calls — delegates to the library.

IMPORTANT, model support is narrow and NOT newest-first. Verified against live Bedrock in
us-east-1 on 2026-08-05: `output_config` works on Haiku 4.5, Sonnet 4.6 and Opus 4.6, and is
rejected with `ValidationException: output_config.format: Extra inputs are not permitted` on
Opus 4.8, Opus 5 and Sonnet 5. The boundary is 4.6-and-older. Picking a "better" model here
is therefore a downgrade to prompt-only JSON, so `_SCHEMA_CAPABLE_MODELS` gates it and an
unsupported model falls back rather than 400-ing every extraction.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union

from llms.schema import FACTS_SCHEMA, FactExtraction
from mem0.llms.aws_bedrock import AWSBedrockLLM
from pydantic import ValidationError

logger = logging.getLogger(__name__)

__all__ = ["ConstrainedAWSBedrockLLM", "FACTS_SCHEMA", "FactExtraction"]

# Substrings, not exact ids, because the deployed id carries an inference-profile prefix and a
# version suffix (e.g. `us.anthropic.claude-haiku-4-5-20251001-v1:0`).
_SCHEMA_CAPABLE_MODELS = ("claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-6")


class ConstrainedAWSBedrockLLM(AWSBedrockLLM):
    """`AWSBedrockLLM` that turns a `response_format` request into a real grammar constraint."""

    def supports_schema(self) -> bool:
        model = self.config.model or ""
        return self.provider == "anthropic" and any(m in model for m in _SCHEMA_CAPABLE_MODELS)

    @staticmethod
    def _wants_json(response_format: Any) -> bool:
        """mem0 asks with `{"type": "json_object"}`; tolerate `json_schema` and a bare string."""
        if isinstance(response_format, str):
            return "json" in response_format
        if isinstance(response_format, dict):
            return "json" in str(response_format.get("type", ""))
        return False

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Any] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        stream: bool = False,
        **kwargs,
    ) -> Union[str, Dict[str, Any]]:
        # Tool calls have their own structured surface (`strict` on the tool), and an
        # unconstrained call has nothing to constrain — leave both to the library.
        if tools or not self._wants_json(response_format):
            return super().generate_response(
                messages, response_format=response_format, tools=tools,
                tool_choice=tool_choice, stream=stream, **kwargs,
            )

        if not self.supports_schema():
            # Deliberately a warning and a fallback, not an exception: extraction degrades to
            # prompt-only JSON (what upstream always did) instead of every write failing.
            logger.warning(
                "model %r cannot enforce a JSON schema on Bedrock; falling back to prompt-only "
                "JSON. Structured outputs need Haiku 4.5 / Sonnet 4.6 / Opus 4.6 — 4.7+ reject "
                "output_config.",
                self.config.model,
            )
            return super().generate_response(
                messages, response_format=response_format, tools=tools,
                tool_choice=tool_choice, stream=stream, **kwargs,
            )

        return self._generate_constrained(messages)

    def _generate_constrained(self, messages: List[Dict[str, str]]) -> str:
        """Converse with `outputConfig.textFormat` set, so decoding is grammar-constrained.

        Converse rather than InvokeModel because that is what the library's Anthropic path
        already uses, so message formatting and inference config stay shared. Note Converse
        nests the schema differently from InvokeModel: `outputConfig.textFormat.structure.
        jsonSchema.schema`, and that `schema` is a STRING, not an object.
        """
        formatted_messages, system_message = self._format_messages_anthropic(messages)

        params: Dict[str, Any] = {
            "modelId": self.config.model,
            "messages": formatted_messages,
            "inferenceConfig": self._build_inference_config(),
            "outputConfig": {
                "textFormat": {
                    "type": "json_schema",
                    "structure": {
                        "jsonSchema": {
                            "name": "fact_extraction",
                            "description": "Facts extracted from the input.",
                            "schema": json.dumps(FACTS_SCHEMA),
                        }
                    },
                }
            },
        }
        if system_message:
            params["system"] = [{"text": system_message}]

        response = self.client.converse(**params)

        # A grammar guarantees the SHAPE, not that generation finished. If maxTokens is hit
        # mid-document the output is a valid *prefix* and nothing upstream notices: mem0's
        # caller would parse a short fact list and store it as if complete. Check before
        # parsing, per the documented structured-outputs failure mode.
        stop_reason = response.get("stopReason")
        if stop_reason == "max_tokens":
            raise ValueError(
                "Bedrock truncated the extraction at maxTokens — the JSON is a valid prefix, "
                "not a complete result. Raise MEM0_LLM_MAX_TOKENS."
            )

        text = response["output"]["message"]["content"][0]["text"]

        # Validate rather than trust. The grammar should make this unreachable, which is
        # precisely why a failure here is worth surfacing loudly: it means the constraint did
        # not apply (an unexpected fallback, a refusal, a provider change) and we would
        # otherwise hand mem0 something it will silently turn into "no new facts found".
        # mem0's caller re-raises LLM errors as LLMError, so this reaches the operator instead
        # of becoming a 200 with an empty results list.
        try:
            FactExtraction.model_validate_json(text)
        except ValidationError as exc:
            raise ValueError(
                f"Bedrock returned output that does not match the facts schema despite "
                f"grammar constraint (stopReason={stop_reason!r}): {exc}"
            ) from exc

        # Return the original text, not a re-serialisation: mem0 does its own
        # extract_json/json.loads on this string, and round-tripping through Pydantic would
        # change key order and whitespace for no benefit.
        return text
