"""The one schema mem0's extraction path ever asks for.

Shared by both provider overrides so the shape is declared exactly once.

## The key is `memory`, and getting that wrong is total silent data loss

An earlier revision of this file constrained extraction to `{"facts": [str]}`. mem0 does not
read `facts`. Both call sites that pass `response_format` — `mem0/memory/main.py`, sync
extraction at :931 and its async twin at :2568, verified against the DEPLOYED wheel
(mem0ai 2.0.14, not this repo's top-level `mem0/`, which is 2.0.12) — do:

    extracted_memories = json.loads(response, strict=False).get("memory", [])

and then read exactly two fields off each element (:986/:1006 and :2618/:2638):

    text            required; the memory itself, and the embedding + md5 dedup key
    attributed_to   optional; copied into payload metadata under `if mem.get(...)`

So a grammar that forces `facts` makes mem0's `.get("memory", [])` return `[]`, and
`POST /memories` answers **200 with `{"results": []}`** — byte-identical to "no new facts
found". Nothing raises, nothing logs. `tests/test_extraction_schema.py` pins the key against
the installed library rather than against this docstring, because a docstring cannot go red.

## Why four fields, when mem0 reads two

The system prompt on that path is mem0's own `ADDITIVE_EXTRACTION_PROMPT`
(`mem0/configs/prompts.py`), and its OUTPUT FORMAT section mandates four:

    id                 (string, required)  sequential, "0", "1", ...
    text               (string, required)
    attributed_to      (string, required)  "user" | "assistant"
    linked_memory_ids  (array,  optional)  "Omit or pass [] if no existing memories"

mem0 ignores `id` and `linked_memory_ids` entirely. They are in the schema anyway so that the
grammar and the prompt AGREE: under `additionalProperties: false` a two-field schema would make
two of the four keys the prompt demands unrepresentable, clipping the model's token
distribution against its own instructions on every object. The wasted bytes are a rounding
error next to that; prompt/grammar disagreement is a quality risk that is invisible in tests.

Both are non-nullable and required, which is also what lets `linked_memory_ids` stay honest:
strict mode cannot express "optional" by omission (see below), and the prompt itself offers
`[]` as the empty case, so `[]` is the encoding — no nullable union needed for it.

## Two rules both backends enforce, and one they do not share

`extra="forbid"` is NOT stylistic: it is what makes `model_json_schema()` emit
`additionalProperties: false` on every object. Bedrock rejects the schema at grammar-compile
time without it; OpenAI/Azure reject `strict: true` without it. The second shared rule is that
EVERY property must appear in `required` — optionality has to be expressed as a nullable type,
never by omission. Nothing here needs a nullable type, which is the cheapest way to satisfy
that rule.

`$ref`/`$defs` is the unshared part. OpenAI strict mode resolves them; Anthropic-on-Bedrock's
accepted JSON Schema subset is not documented to, and a schema rejected at compile time fails
every extraction. Pydantic emits a `$ref` for any nested model, so `_inline_refs` flattens it
back out — see there.
"""

from typing import Any, Dict, FrozenSet

from pydantic import BaseModel, ConfigDict


class ExtractedMemory(BaseModel):
    """One extracted memory, in the shape ADDITIVE_EXTRACTION_PROMPT asks for."""

    model_config = ConfigDict(extra="forbid")

    id: str
    text: str
    # NOT `Literal["user", "assistant"]`, however much it looks like it should be. A Literal
    # becomes an `enum` in the wire schema, i.e. a hard grammar constraint, and the prompt asks
    # for a THIRD case: "in multi-speaker conversations the 'assistant' role may represent a real
    # person sharing their own life ... attributed to the speaker by name". Narrowing this makes
    # a speaker's name unrepresentable — the model would be instructed to emit it and forbidden
    # from doing so, and the fallback is mis-attributing that memory to "assistant".
    attributed_to: str
    linked_memory_ids: list[str]


class MemoryExtraction(BaseModel):
    """The envelope. `memory` is the key mem0 reads; nothing else is looked at."""

    model_config = ConfigDict(extra="forbid")

    memory: list[ExtractedMemory]


class ParsedJson(str):
    """A JSON string that also carries its validated model.

    Subclasses `str` deliberately. mem0's caller runs `remove_code_blocks(response)` and
    `json.loads(response, strict=False)` on whatever `generate_response` returns, so returning a
    `BaseModel` — the obvious way to "return a typed object" — breaks both. That code is in the
    PyPI wheel and cannot be patched from `server/`.

    This way the library keeps working byte-for-byte while server-side callers can reach
    `.parsed` without parsing the string a second time. Note the model does not survive mem0's
    `remove_code_blocks`, which returns a plain `str` — that is fine, the library only ever
    wanted the text.
    """

    parsed: MemoryExtraction

    def __new__(cls, raw: str, parsed: MemoryExtraction) -> "ParsedJson":
        obj = super().__new__(cls, raw)
        obj.parsed = parsed
        return obj


def _strip_prose(node: Any) -> Any:
    """Remove `description`/`title` from a generated schema, recursively.

    Pydantic puts each model's DOCSTRING into the schema as `description` and every field name
    into `title`. Both then ride on the wire with every single extraction request — the
    docstrings above are hundreds of characters of internal commentary that the provider neither
    needs nor benefits from, and it was observed verbatim in bifrost's echoed `raw_request`.

    Stripped rather than avoided by shortening the docstrings, so the explanation can stay where
    a reader will find it. Neither key carries meaning for constrained decoding: the grammar is
    built from types, `required` and `additionalProperties`.
    """
    if isinstance(node, dict):
        return {k: _strip_prose(v) for k, v in node.items() if k not in ("description", "title")}
    if isinstance(node, list):
        return [_strip_prose(v) for v in node]
    return node


def _inline_refs(node: Any, defs: Dict[str, Any], seen: FrozenSet[str] = frozenset()) -> Any:
    """Replace every `$ref` with the definition it points at, recursively.

    Pydantic emits nested models as `{"$ref": "#/$defs/ExtractedMemory"}` plus a `$defs` block.
    OpenAI strict mode resolves that; Bedrock's documented subset does not promise to, and there
    the failure mode is the whole schema rejected at grammar-compile time — i.e. every
    extraction 400s. Inlining is unambiguously portable and this schema is three keys deep, so
    it costs nothing.

    `seen` makes a cyclic model a loud error instead of a hang at import time. Nothing here is
    recursive today; a future field that is would otherwise take the pod down on startup with a
    RecursionError from the module body.
    """
    if isinstance(node, dict):
        ref = node.get("$ref")
        if ref is not None:
            name = ref.rsplit("/", 1)[-1]
            if name in seen:
                raise ValueError(f"cyclic $ref: {name!r} cannot be inlined into a flat schema")
            if name not in defs:
                raise ValueError(f"dangling $ref {ref!r}: {sorted(defs)} are defined")
            # Sibling keys of a $ref (e.g. a per-use override) win over the definition.
            merged = {**defs[name], **{k: v for k, v in node.items() if k != "$ref"}}
            return _inline_refs(merged, defs, seen | {name})
        return {k: _inline_refs(v, defs, seen) for k, v in node.items() if k != "$defs"}
    if isinstance(node, list):
        return [_inline_refs(v, defs, seen) for v in node]
    return node


def _wire_schema(model: type[BaseModel]) -> Dict[str, Any]:
    """`model_json_schema()` reduced to what a grammar compiler will accept from every backend."""
    raw = model.model_json_schema()
    return _strip_prose(_inline_refs(raw, raw.get("$defs", {})))


# Resolved once at import. The grammar is compiled and cached per-schema on the provider side,
# so these bytes must be stable across requests or every call pays first-compile latency.
MEMORY_SCHEMA: Dict[str, Any] = _wire_schema(MemoryExtraction)

# OpenAI-compatible strict form. `strict: true` is what turns Azure's json_schema mode into
# constrained decoding; bifrost forwards it, and for Gemini it drops the flag and translates
# the schema to `responseJsonSchema` (which is inherently constrained). Verified on both.
OPENAI_STRICT_RESPONSE_FORMAT: Dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "memory_extraction",
        "strict": True,
        "schema": MEMORY_SCHEMA,
    },
}
