"""The one schema mem0's extraction path ever asks for.

Shared by both provider overrides so the shape is declared exactly once. Both call sites that
pass `response_format` (mem0/memory/main.py, sync fact extraction at :917 and its async twin
at :2556) expect `{"facts": [str]}`. The memory-UPDATE path passes no `response_format` at
all, so it never reaches either override — there is no second shape to select between.
"""

from typing import Any, Dict

from pydantic import BaseModel, ConfigDict


class FactExtraction(BaseModel):
    """Extraction result.

    `extra="forbid"` is NOT stylistic: it is what makes `model_json_schema()` emit
    `additionalProperties: false`, which BOTH backends require on every object — Bedrock
    rejects the schema at grammar-compile time without it, and OpenAI/Azure reject
    `strict: true` without it. `facts` being non-optional is what puts it in `required`,
    the other shared hard requirement.
    """

    model_config = ConfigDict(extra="forbid")

    facts: list[str]


# Resolved once at import. The grammar is compiled and cached per-schema on the provider side,
# so these bytes must be stable across requests or every call pays first-compile latency.
FACTS_SCHEMA: Dict[str, Any] = FactExtraction.model_json_schema()

# OpenAI-compatible strict form. `strict: true` is what turns Azure's json_schema mode into
# constrained decoding; bifrost forwards it, and for Gemini it drops the flag and translates
# the schema to `responseJsonSchema` (which is inherently constrained). Verified on both.
OPENAI_STRICT_RESPONSE_FORMAT: Dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "fact_extraction",
        "strict": True,
        "schema": FACTS_SCHEMA,
    },
}
