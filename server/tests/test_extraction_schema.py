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
