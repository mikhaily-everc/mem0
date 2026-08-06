"""Pins the extraction schema against the mem0 library that is actually installed.

This is the test the shipped bug needed and did not have. `server/llms/schema.py` constrained
extraction to `{"facts": [str]}`; mem0 reads `.get("memory", [])`. So the grammar forced `facts`,
mem0 asked for `memory`, got `[]`, and `POST /memories` answered **200 with `{"results": []}`** —
byte-identical to "no new facts found". Nothing raised. Nothing logged. Every other test in this
directory passed, because they all asserted our schema against our own constant.

The fix is not "assert the key is `memory`" — that is another constant, and it would have been
written wrong at the same moment for the same reason. These tests read the INSTALLED library and
the DEPLOYED prompt and compare our schema to those.

Consequence worth stating: a mem0 wheel bump that renames the envelope key or adds a field to the
prompt turns this file red. That is the point. The failure is loud here or silent in production;
there is no third option. When it goes red, re-read `mem0/memory/main.py` around the
`response_format` call sites and `ADDITIVE_EXTRACTION_PROMPT` in `mem0/configs/prompts.py`, then
update `server/llms/schema.py` to match.

stdlib `unittest`, no pytest — same constraint as the other server tests. No network, no client.
"""

from __future__ import annotations

import inspect
import json
import re
import unittest

import mem0.memory.main as mem0_main
from mem0.configs.prompts import ADDITIVE_EXTRACTION_PROMPT

from llms.schema import MEMORY_SCHEMA, ExtractedMemory, MemoryExtraction

_SOURCE = inspect.getsource(mem0_main)

# `extracted_memories = json.loads(response, strict=False).get("memory", [])` — the envelope key.
# Matched loosely on the `.get(...)` that follows a json.loads of the LLM response, so a rename of
# the local variable does not silently stop matching.
_ENVELOPE_READ_RE = re.compile(r"json\.loads\(\s*\w+\s*,\s*strict=False\s*\)\.get\(\s*[\"'](\w+)[\"']")

# Reads off ONE extracted element: `mem.get("text")`, `mem["attributed_to"]`, `m.get("text", "")`.
_ELEMENT_READ_RE = re.compile(r"\b(?:mem|m)(?:\.get\(\s*[\"'](\w+)[\"']|\[\s*[\"'](\w+)[\"']\s*\])")

# Method boundary, used to bound the scan to the extraction method that just read the envelope.
# Without it the scan picks up `mem.get('id')` / `mem.get('score')` from the SEARCH path, which
# iterates vector-store rows rather than extracted memories.
_METHOD_DEF_RE = re.compile(r"^\s{4}(?:async\s+)?def\s", re.M)


def _envelope_keys() -> list[str]:
    return _ENVELOPE_READ_RE.findall(_SOURCE)


def _element_fields() -> set[str]:
    """Every field mem0 reads off an extracted memory, taken from the library source.

    Scans forward from each envelope read to the end of its method, which is exactly the region
    that handles extracted memories.
    """
    fields: set[str] = set()
    for match in _ENVELOPE_READ_RE.finditer(_SOURCE):
        end = _METHOD_DEF_RE.search(_SOURCE, match.end())
        window = _SOURCE[match.end() : end.start() if end else len(_SOURCE)]
        for get_name, index_name in _ELEMENT_READ_RE.findall(window):
            fields.add(get_name or index_name)
    return fields


def _item_schema() -> dict:
    """Our per-memory schema, reached through the key mem0 READS rather than a literal.

    Raises rather than KeyError-ing so a wrong envelope key fails as one legible assertion
    instead of an exception in a class body, which would abort module import and take every
    other test in this file down with it — hiding the diagnosis behind a stack trace.
    """
    key = _envelope_keys()[0]
    properties = MEMORY_SCHEMA["properties"]
    if key not in properties:
        raise AssertionError(
            "the extraction schema has no {!r} property, so mem0's `.get({!r}, [])` returns "
            "[] and every extraction stores NOTHING while `POST /memories` answers 200 with "
            "an empty results list. Declared instead: {}".format(key, key, sorted(properties))
        )
    return properties[key]["items"]


def _prompt_fields() -> list[str]:
    """The field names ADDITIVE_EXTRACTION_PROMPT's OUTPUT FORMAT section documents."""
    start = ADDITIVE_EXTRACTION_PROMPT.index("## Fields")
    end = ADDITIVE_EXTRACTION_PROMPT.index("## Rules", start)
    section = ADDITIVE_EXTRACTION_PROMPT[start:end]
    return re.findall(r"^- \*\*(\w+)\*\* \(", section, re.M)


class LibraryIsStillShapedTheWayWeThinkTest(unittest.TestCase):
    """Guards the scans themselves. A regex that quietly stops matching would make every
    assertion below vacuously true — the exact failure mode that shipped."""

    def test_the_envelope_read_was_found(self) -> None:
        keys = _envelope_keys()
        self.assertTrue(
            keys,
            "could not find `json.loads(..., strict=False).get(...)` in mem0.memory.main. The "
            "library changed shape; re-read the response_format call sites before trusting any "
            "other assertion in this file.",
        )

    def test_both_the_sync_and_async_paths_were_found(self) -> None:
        """mem0 has two extraction implementations and they drift independently."""
        self.assertGreaterEqual(len(_envelope_keys()), 2, _envelope_keys())

    def test_element_reads_were_found(self) -> None:
        self.assertTrue(_element_fields(), "no per-memory field reads found near the envelope read")


class EnvelopeKeyTest(unittest.TestCase):
    def test_we_emit_the_key_mem0_reads(self) -> None:
        for key in _envelope_keys():
            with self.subTest(key=key):
                self.assertIn(key, MEMORY_SCHEMA["properties"])

    def test_the_envelope_has_no_other_key(self) -> None:
        """A second top-level key would be a field the model spends tokens on and mem0 discards."""
        self.assertEqual(set(MEMORY_SCHEMA["properties"]), set(_envelope_keys()))

    def test_both_paths_read_the_same_key(self) -> None:
        self.assertEqual(len(set(_envelope_keys())), 1, _envelope_keys())


class ElementFieldsTest(unittest.TestCase):
    def test_every_field_mem0_reads_is_in_the_schema(self) -> None:
        """The direction that loses data. A field mem0 reads and the grammar cannot produce is
        silently absent from every stored memory."""
        item = _item_schema()
        for field in _element_fields():
            with self.subTest(field=field):
                self.assertIn(field, item["properties"])

    def test_text_is_required_not_merely_allowed(self) -> None:
        """`text` IS the memory: it is the embedding input and the md5 dedup key. mem0 skips any
        element without it, so an optional `text` is a silent partial loss."""
        item = _item_schema()
        self.assertIn("text", item["required"])
        self.assertEqual(item["properties"]["text"]["type"], "string")

    def test_the_fields_mem0_ignores_are_exactly_the_ones_the_prompt_demands(self) -> None:
        """The other direction. Extra fields are not free, so each one needs a reason; the reason
        here is that ADDITIVE_EXTRACTION_PROMPT mandates them, and a grammar that forbids what the
        prompt demands clips the model against its own instructions. If mem0's prompt stops asking
        for one, drop it from the schema rather than widening this assertion."""
        ignored = set(_item_schema()["properties"]) - _element_fields()
        self.assertEqual(ignored, {"id", "linked_memory_ids"})
        for field in ignored:
            with self.subTest(field=field):
                self.assertIn(field, _prompt_fields())


class PromptAgreementTest(unittest.TestCase):
    def test_the_prompt_fields_were_found(self) -> None:
        self.assertTrue(
            _prompt_fields(),
            "could not parse the OUTPUT FORMAT field list out of ADDITIVE_EXTRACTION_PROMPT",
        )

    def test_the_schema_and_the_prompt_ask_for_the_same_fields(self) -> None:
        """Under `additionalProperties: false` the grammar is the harder constraint, so any field
        the prompt documents and the schema omits is unrepresentable — the model is told to emit
        it and forbidden from doing so, on every object."""
        self.assertEqual(set(_prompt_fields()), set(_item_schema()["properties"]))

    def test_the_prompt_asks_for_the_same_envelope_key(self) -> None:
        self.assertIn('"memory"', ADDITIVE_EXTRACTION_PROMPT)


class SilentDropTest(unittest.TestCase):
    """Why `log_extraction` warns on blank text at all.

    `llms/schema.py` claims mem0 discards a memory with empty `text` in COMPLETE silence, and the
    provider overrides emit a WARNING on the strength of that claim. If it were wrong the warning
    would be noise. So the claim is checked against the installed library rather than asserted in
    a docstring.
    """

    # `if not text or text not in embed_map:` — the falsy-text guard, in the extraction loop.
    _GUARD_RE = re.compile(r"if\s+not\s+text\s+or\s+text\s+not\s+in\s+\w+\s*:")

    def _extraction_windows(self) -> list[str]:
        windows = []
        for match in _ENVELOPE_READ_RE.finditer(_SOURCE):
            end = _METHOD_DEF_RE.search(_SOURCE, match.end())
            windows.append(_SOURCE[match.end() : end.start() if end else len(_SOURCE)])
        return windows

    def test_the_falsy_text_guard_is_still_there(self) -> None:
        hits = [w for w in self._extraction_windows() if self._GUARD_RE.search(w)]
        self.assertTrue(
            hits,
            "mem0 no longer guards on falsy `text` the way llms/schema.py's blank_text_ids "
            "docstring says. Re-read the extraction loop: either the drop is now logged (in "
            "which case our WARNING is redundant) or it changed shape.",
        )

    def test_the_drop_is_unlogged(self) -> None:
        """The `continue` must NOT be preceded by a log call — an unlogged drop is the whole
        justification for warning about it on our side. Note the sibling duplicate-hash branch
        immediately below it DOES log, at debug, which is why only blank text gets a warning."""
        for window in self._extraction_windows():
            match = self._GUARD_RE.search(window)
            if not match:
                continue
            # The guard body: up to and including its `continue`.
            body = window[match.end() : match.end() + 200]
            body = body[: body.index("continue") + len("continue")] if "continue" in body else body
            with self.subTest(body=body.strip()[:80]):
                self.assertNotIn("logger", body)

    def test_the_duplicate_branch_does_log_so_we_only_count_it(self) -> None:
        for window in self._extraction_windows():
            if "Skipping duplicate memory" in window:
                self.assertIn("logger.debug", window)
                return
        self.fail("mem0's duplicate-hash branch not found; re-check duplicate_text_count()")


class SalvagePathIsUnreachableTest(unittest.TestCase):
    """Why `extract_json` is deliberately NOT reimplemented on the Pydantic side.

    mem0's extraction parse has a salvage branch:

        try:    json.loads(response, strict=False).get("memory", [])
        except json.JSONDecodeError:
                json.loads(extract_json(response), strict=False).get("memory", [])

    `extract_json` (mem0/memory/utils.py) strips ``` fences, or failing that takes the span from
    the first `{` to the last `}`. It exists for prose-wrapped or truncated output.

    On the CONSTRAINED path it cannot run. The override validates with
    `MemoryExtraction.model_validate_json` and raises on failure, so every value it returns has
    already parsed as JSON; mem0 then parses with `strict=False`, which is strictly MORE permissive
    than JSON (it tolerates raw control characters inside strings). Validated therefore implies
    parseable, and the `except` is dead.

    That is the argument for leaving it alone: a Pydantic reimplementation would have no caller.
    These tests keep the argument honest — if pydantic-core and `json.loads` ever disagree on a
    payload we would hand over, or if upstream restructures the fallback, this goes red.

    Note the salvage branch is still LIVE on the unconstrained paths — MEM0_LLM_STRICT_SCHEMA=false
    and Bedrock models that cannot enforce a schema — where we deliberately degrade to upstream
    behaviour. `extract_json` must keep working; it just never sees our output.
    """

    # Valid under the schema, and awkward in the ways that historically broke JSON here: quoted
    # JSON/YAML inside a memory (the documented cause of the ~19% invalid-JSON rate), backslashes,
    # newlines, unicode, and a lone `}` that would fool a first-{-to-last-} salvage.
    AWKWARD = [
        'the gateway config is `{"port": 8800, "host": "0.0.0.0"}`',
        "a windows path C:\\\\Users\\\\test and a regex \\\\d{7,40}",
        "line one\nline two\ttabbed",
        "em-dash — typographic “quotes” and an emoji 🚀",
        "a closing brace } and an opening { out of order",
        'nested "double \\"escaped\\"" quotes',
        "",  # blank text is schema-valid; mem0 drops it, log_extraction warns
    ]

    def _payload(self, text: str) -> str:
        return MemoryExtraction(
            memory=[ExtractedMemory(id="0", text=text, attributed_to="user", linked_memory_ids=[])]
        ).model_dump_json()

    def test_pydantic_validated_output_always_parses_for_mem0(self) -> None:
        """The invariant the whole argument rests on: validated => mem0's json.loads succeeds,
        so the salvage branch never runs on our output."""
        for text in self.AWKWARD:
            with self.subTest(text=text[:40]):
                raw = self._payload(text)
                # Ours.
                MemoryExtraction.model_validate_json(raw)
                # Theirs, verbatim from main.py — must not raise.
                recovered = json.loads(raw, strict=False).get("memory", [])
                self.assertEqual(recovered[0]["text"], text)

    def test_the_salvage_branch_is_still_shaped_as_assumed(self) -> None:
        """If upstream stops routing the JSONDecodeError through extract_json, the reasoning above
        needs re-checking rather than silently continuing to hold."""
        self.assertIn("extract_json", _SOURCE)
        self.assertRegex(_SOURCE, r"except json\.JSONDecodeError:")

    def test_mem0_parses_with_strict_false(self) -> None:
        """`strict=False` is what makes "validated implies parseable" true rather than merely
        likely — it is more permissive than JSON, not less. If it ever becomes a strict parse,
        pydantic-core and json.loads could disagree and the salvage branch becomes reachable."""
        self.assertRegex(_SOURCE, r"json\.loads\(\s*\w+\s*,\s*strict=False\s*\)")


class RoundTripTest(unittest.TestCase):
    """End to end through mem0's own parse, rather than through our model."""

    def _mem0_parse(self, raw: str) -> list[dict]:
        """The library's exact expression, from main.py."""
        return json.loads(raw, strict=False).get(_envelope_keys()[0], [])

    def test_a_schema_valid_response_survives_mem0s_parse(self) -> None:
        raw = MemoryExtraction(
            memory=[
                ExtractedMemory(
                    id="0",
                    text="The gateway runs on port 8800.",
                    attributed_to="user",
                    linked_memory_ids=[],
                )
            ]
        ).model_dump_json()

        extracted = self._mem0_parse(raw)
        self.assertEqual(len(extracted), 1)
        # The two reads that decide whether anything is stored at all.
        self.assertEqual(extracted[0].get("text"), "The gateway runs on port 8800.")
        self.assertEqual(extracted[0].get("attributed_to"), "user")

    def test_the_old_facts_shape_parses_to_nothing(self) -> None:
        """The shipped bug, reproduced. Valid JSON, schema-conformant under the old model, and
        mem0 extracts zero memories from it — which it reports as HTTP 200 + `results: []`."""
        self.assertEqual(self._mem0_parse('{"facts": ["a fact", "another"]}'), [])

    def test_an_empty_extraction_is_representable(self) -> None:
        """ "Nothing worth extracting" must be expressible, or the grammar forces invention."""
        raw = MemoryExtraction(memory=[]).model_dump_json()
        self.assertEqual(self._mem0_parse(raw), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
