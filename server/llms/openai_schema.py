"""OpenAI-compatible LLM that upgrades mem0's weak JSON request into a real grammar constraint.

mem0 asks for JSON with `response_format={"type": "json_object"}` — the weak form. That is a
*request*, not a guarantee: measured on this corpus it returns unparseable JSON on ~19% of real
inputs, and every failure is SILENT, because `POST /memories` still answers 200 with
`{"results": []}`, byte-identical to "no new facts found".

The strict form, `{"type": "json_schema", "json_schema": {..., "strict": true}}`, IS
grammar-constrained — and bifrost carries it: for Azure it forwards `strict` straight through,
and for Gemini it translates the schema into `responseJsonSchema` (dropping `strict`, which
Gemini has no field for — `responseJsonSchema` is itself the constraint). Both verified against
the running bifrost, which echoes the translation in `extra_fields.raw_request`.

The two forms are one word apart in the wire format and worlds apart in behaviour, which is
exactly why this is easy to get wrong: an earlier reading of this system concluded bifrost
"could not do native structured output" purely from seeing `json_object` in mem0's source.

Why an override rather than a fix: the `json_object` literal lives in the mem0 LIBRARY
(mem0/memory/main.py), which ships from PyPI. `server/` is built from source; the library is
not. Same wheel-vs-fork trap as the embedder.

Measured effect, 8 real memory files, mem0's own FACT_RETRIEVAL_PROMPT, identical scorer,
retention of load-bearing tokens (commands, paths, flags, ids, dates):

    gpt-5.5     83.1%  (best file 95.7%)   0/8 invalid   379 facts
    gpt-5-mini  72.0%                      0/8 invalid   187 facts
    gpt-5.1     63.9%                      0/7 invalid   191 facts
    gemini-2.5-pro          23.6%          0/8 invalid   120 facts
    gemini-3.1-pro-preview  22.5%          0/8 invalid    95 facts

Note the Gemini *Pro* models score BELOW gemini-3.5-flash (22.3%) — bigger is not better
within that family, so do not reach for a Pro id expecting an improvement.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from llms.schema import OPENAI_STRICT_RESPONSE_FORMAT, FactExtraction
from mem0.llms.openai import OpenAILLM
from pydantic import ValidationError

logger = logging.getLogger(__name__)


def _strict_enabled() -> bool:
    """Escape hatch for an OpenAI-compatible backend that cannot do json_schema.

    Defaults ON because both providers bifrost currently routes to (Azure, Vertex) were
    verified to honour it. If a future backend only supports the weak form, the upgrade would
    turn working-but-flaky extraction into a hard 400 — set MEM0_LLM_STRICT_SCHEMA=false.
    """
    return os.environ.get("MEM0_LLM_STRICT_SCHEMA", "true").strip().lower() != "false"


class SchemaEnforcingOpenAILLM(OpenAILLM):
    """`OpenAILLM` that swaps mem0's `json_object` for a strict, schema-constrained request."""

    @staticmethod
    def _is_weak_json(response_format: Any) -> bool:
        """True for exactly the form mem0 sends, and nothing else.

        Deliberately narrow: a caller that already supplied its own `json_schema` has made a
        considered choice and must not be silently rewritten to the facts schema.
        """
        if isinstance(response_format, str):
            return response_format == "json_object"
        return (
            isinstance(response_format, dict)
            and response_format.get("type") == "json_object"
        )

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Any] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        **kwargs,
    ):
        # Tools have their own structured surface (`strict` on the function), and the
        # memory-UPDATE path passes no response_format at all — neither should be rewritten.
        upgrade = not tools and self._is_weak_json(response_format) and _strict_enabled()
        if upgrade:
            response_format = OPENAI_STRICT_RESPONSE_FORMAT

        result = super().generate_response(
            messages,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )

        if not upgrade:
            return result

        # Validate rather than trust. With a grammar in force this should be unreachable,
        # which is exactly why a failure must be loud: the likely causes are the constraint
        # silently not applying (a backend that ignored `strict`) or the response being cut off
        # at max_tokens. Either way, returning it would hand mem0 something it turns into a
        # silent "no new facts found". mem0's caller re-raises LLM errors as LLMError, so this
        # reaches an operator instead.
        try:
            FactExtraction.model_validate_json(result)
        except (ValidationError, TypeError) as exc:
            preview = (result or "")[:200] if isinstance(result, str) else repr(result)[:200]
            raise ValueError(
                "LLM returned output that does not match the facts schema despite a strict "
                "json_schema request. Likely either the backend ignored `strict` or the "
                f"response was truncated at max_tokens. First 200 chars: {preview!r}"
            ) from exc

        # Return the original string: mem0 runs its own extract_json/json.loads on it, and
        # re-serialising through Pydantic would churn key order and whitespace for no gain.
        return result


def strict_response_format() -> Dict[str, Any]:
    """The exact `response_format` this override substitutes — exposed for tests and probes."""
    return json.loads(json.dumps(OPENAI_STRICT_RESPONSE_FORMAT))
