"""Which rows a memory read is allowed to enumerate, in both directions.

Two properties, and both fail INVISIBLY: the endpoint answers 200 either way and
only the contents are wrong.

NARROWING — `include_agent_scoped`. The vector store's `list()` is
`SELECT id, payload FROM <col> WHERE <filters> LIMIT %s` — no ORDER BY, no
OFFSET. So a listing is an arbitrary heap-order slice and nothing can page past
the cap.

That was harmless while a whole scope fit under the cap. It stopped fitting on
2026-08-07, when a derived fact layer (one agent-scoped row per extracted
statement, ~13 per source document) grew to roughly 13x the number of source
documents. Measured the next day: `top_k=1000` returned 593 documents + 407
derived rows, `max(created_at) = 2026-08-07T21:53:58Z`, and memories written
after that were absent from the LIVE response — not merely from a client cache.
Nothing errored; LIMIT is not an error. Enumeration had silently stopped.

`include_agent_scoped=false` restores it by asking only for rows carrying no
`agent_id`. These tests pin the filter that expresses it, because the failure it
prevents is invisible: drop the filter and the endpoint still answers 200, just
with a truncated mix again.

WIDENING — `include_shared`. The opposite direction, and the opposite risk. Every
memory is pinned to its author's Auth0 sub, so a repo-wide note whose URL is
committed to version control 404s for every colleague who follows the link.
`include_shared` adds a second pass over the reserved shared scopes to fix that,
and it must stay OPT-IN: `.claude/hooks/mem0/mem0ctl.py` recall runs at
thresholds (0.350 vector, 0.30 rerank) fitted against exactly today's corpus, and
a default-on widening would move the separation they were derived from with
nothing to report it. The read/write boundary the shared scope opens is asserted
in test_auth0.py, next to `MemoryScope`; what lives here is what gets ENUMERATED.

Tests target `listing.py` rather than the route: main.py builds a real `Memory`
at import scope, so importing it needs credentials and a database. The two things
that cannot be reached that way — the wire default of `include_shared`, which is
the contract every existing caller actually gets, and WHICH GUARD each route
calls, which is the contract the store gets — are asserted against the routing
modules' ASTs instead. stdlib `unittest` and `ast` for the reason at the top of
test_auth0.py: nothing may be added to the resolved dependency graph, because
that graph is what the image ships.

Deliberately importing nothing but `listing` and the stdlib, so this file runs
against a bare interpreter (`PYTHONPATH=server python3 tests/test_list_scoping.py`)
with no wheels resolved. The scope PREDICATES, and the mem0-library behaviour that
makes one of the guards necessary, live in test_auth0.py, which needs the venv.
"""

from __future__ import annotations

import ast
import pathlib
import unittest

import listing
from listing import SHARED_SCOPES, listing_filters, read_scopes

USER = "auth0|someone"


def _module_tree(*parts: str) -> ast.Module:
    """Parse a server module by path, located through `listing.__file__`.

    Through `listing` rather than a path relative to this test, so it does not
    depend on the runfiles layout: under bazel the test and the library it is
    testing do not necessarily share a directory.
    """
    return ast.parse(pathlib.Path(listing.__file__).parent.joinpath(*parts).read_text())


def _default_of(call: ast.AST) -> object:
    """The default a `Query(...)` / `Field(...)` declares, positional or keyword."""
    if not isinstance(call, ast.Call):
        raise AssertionError(f"expected a Query()/Field() call, got {ast.dump(call)}")
    if call.args:
        return ast.literal_eval(call.args[0])
    for keyword in call.keywords:
        if keyword.arg == "default":
            return ast.literal_eval(keyword.value)
    raise AssertionError("Query()/Field() declared no default at all")


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} is gone")


def _argument_defaults(func: ast.FunctionDef) -> dict[str, ast.expr]:
    return dict(
        zip(
            [arg.arg for arg in func.args.args][-len(func.args.defaults) :],
            func.args.defaults,
            # The slice is derived from `defaults` itself, so the two are the same
            # length by construction — `strict` asserts that rather than silently
            # truncating if it ever stops holding.
            strict=True,
        )
    )


def _names_called_in(func: ast.FunctionDef) -> set[str]:
    """Every function and method name called in this function's OWN body.

    Scoped to the one `FunctionDef`, which is the whole point: `_fetch_writable_memory`
    calls `_fetch_owned_memory` internally, so a module-wide walk would make
    "this route does not call `_fetch_owned_memory`" impossible to state.
    """
    called: set[str] = set()
    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            called.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            called.add(node.func.attr)
    return called


class ListingFiltersTest(unittest.TestCase):
    def test_excluding_the_agent_layer_asks_for_rows_with_no_agent_id(self) -> None:
        """`$not` over the `"*"` key-exists test is the only way to say "no agent_id".

        The filter language has no IS NULL, and `{"agent_id": None}` would compile
        to `payload->>'agent_id' = 'None'` — a filter matching nothing, returning an
        empty list, exiting 200.
        """
        filters = listing_filters(USER, None, None, include_agent_scoped=False)
        self.assertEqual(filters["$not"], [{"agent_id": "*"}])
        self.assertEqual(filters["user_id"], USER)

    def test_the_default_is_unchanged(self) -> None:
        """Every existing caller must keep seeing exactly what it saw before."""
        self.assertEqual(listing_filters(USER, None, None), {"user_id": USER})

    def test_an_explicit_agent_id_is_never_negated(self) -> None:
        """Asking for one agent's rows is a different question from excluding them all.

        Negating here would turn a legitimate `agent_id=facts` listing into a query
        that cannot match anything, which is worse than ignoring the flag.
        """
        filters = listing_filters(USER, None, "facts", include_agent_scoped=False)
        self.assertEqual(filters["agent_id"], "facts")
        self.assertNotIn("$not", filters)

    def test_the_user_scope_is_never_dropped(self) -> None:
        """The exclusion narrows a listing; it must never widen one past the caller."""
        for include in (True, False):
            with self.subTest(include_agent_scoped=include):
                self.assertEqual(listing_filters(USER, None, None, include)["user_id"], USER)

    def test_run_id_still_travels(self) -> None:
        """The run scope is how the fact layer addresses one parent; it must survive."""
        filters = listing_filters(USER, "run-1", None)
        self.assertEqual(filters["run_id"], "run-1")

    def test_empty_identifiers_are_dropped_not_sent_as_empty_strings(self) -> None:
        """An empty string is a filter that matches nothing, not an absent filter."""
        self.assertEqual(listing_filters(USER, "", ""), {"user_id": USER})


class SharedReadScopeTest(unittest.TestCase):
    """`include_shared` widens a read to the repo-wide scopes, and only on request.

    WHY THIS EXISTS. Pinning a memory to its author's Auth0 sub also made its URL
    useless to everyone else: a link committed to a shared file 404s for every
    colleague who follows it, because `MemoryScope.owns` matched the stored
    `user_id` against the caller's own and nothing else. The shared scope is the
    fix, and these pin the read half of it.

    The DEFAULT is the load-bearing half. `.claude/hooks/mem0/mem0ctl.py` recall
    runs at thresholds (0.350 vector, 0.30 rerank) fitted by measurement against
    exactly today's corpus; a widening that arrived by default would move the
    separation those numbers were derived from, and nothing would report it —
    recall would simply start returning slightly different rows.
    """

    MINE = "auth0:alice"
    SHARED = "repo:ai-utils"

    def test_the_default_is_the_callers_own_scope_and_nothing_else(self) -> None:
        """Every caller that does not opt in must see exactly what it saw before."""
        self.assertEqual(read_scopes(self.MINE, False), (self.MINE,))

    def test_opting_in_appends_the_shared_scope(self) -> None:
        self.assertEqual(read_scopes(self.MINE, True), (self.MINE, self.SHARED))

    def test_the_callers_own_scope_is_queried_first(self) -> None:
        """A listing is LIMIT-only per scope; the caller's own rows must not be the
        ones that fall off the end of a truncated merge."""
        self.assertEqual(read_scopes(self.MINE, True)[0], self.MINE)

    def test_naming_the_shared_scope_outright_does_not_double_it(self) -> None:
        """An unpinned caller may pass `user_id=repo:ai-utils` directly. Without the
        set subtraction that read would run twice and return every row twice."""
        self.assertEqual(read_scopes(self.SHARED, True), (self.SHARED,))

    def test_an_absent_user_id_is_not_given_the_shared_scope(self) -> None:
        """The no-identifier listing is the admin "everything" branch. Appending a
        scope to it would turn one raw scan into a scan plus a scoped listing."""
        self.assertEqual(read_scopes(None, True), (None,))

    def test_a_shared_scope_can_never_collide_with_an_auth0_one(self) -> None:
        """`Auth0Claims.memory_scope` mints `auth0:<sub>`. If a shared value could
        be spelled that way, a tenant that let someone control their own `sub`
        could mint a scope every caller may read."""
        for value in SHARED_SCOPES:
            with self.subTest(scope=value):
                self.assertFalse(value.startswith("auth0:"))

    def test_the_set_is_the_single_place_a_scope_is_added(self) -> None:
        """Guards the shape of the extension point, not the current membership: the
        design is one named set so a second shared scope is a one-line change."""
        self.assertIsInstance(SHARED_SCOPES, frozenset)
        self.assertIn(self.SHARED, SHARED_SCOPES)


class SharedIsOffOnTheWireTest(unittest.TestCase):
    """`include_shared` must default to False in the ROUTE SIGNATURES too.

    `read_scopes(..., False)` being correct proves nothing if `GET /memories`
    declares `Query(True)`: the wire default is the contract every existing caller
    actually gets, and flipping it is a one-character edit no other test here can
    see.

    Asserted against the routing modules' ASTs rather than by importing them,
    because main.py builds a real `Memory` at import scope — embedder client and all
    — so importing it needs credentials and a database, and routers/entities.py
    imports main.py's dependencies transitively. They are located through `listing`'s
    own `__file__` so this does not depend on the runfiles layout.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.tree = _module_tree("main.py")
        cls.entities_tree = _module_tree("routers", "entities.py")

    def test_the_listing_route_defaults_to_the_callers_own_scope(self) -> None:
        defaults = _argument_defaults(_function(self.tree, "get_all_memories"))
        self.assertIn("include_shared", defaults, "GET /memories lost its include_shared param")
        self.assertIs(_default_of(defaults["include_shared"]), False)

    def test_the_search_route_defaults_to_the_callers_own_scope(self) -> None:
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef) and node.name == "SearchRequest":
                for statement in node.body:
                    if not isinstance(statement, ast.AnnAssign):
                        continue
                    if getattr(statement.target, "id", None) == "include_shared":
                        # `AnnAssign.value` is Optional — a bare `include_shared: bool`
                        # with no default parses fine and would mean the field lost its
                        # default entirely, which is exactly the regression this guards.
                        if statement.value is None:
                            raise AssertionError("include_shared declares no default at all")
                        self.assertIs(_default_of(statement.value), False)
                        return
                raise AssertionError("SearchRequest lost its include_shared field")
        raise AssertionError("SearchRequest is gone from main.py")

    def test_the_entities_route_defaults_to_the_callers_own_scope(self) -> None:
        """`GET /entities` was the one route whose results changed for an existing
        caller with no opt-in when the shared scope arrived: it authorised with
        `MemoryScope.owns`, which answers True for a shared row. It is in
        `assemble-spec.ts`'s KEEP set, so an LLM reaches it."""
        defaults = _argument_defaults(_function(self.entities_tree, "list_entities"))
        self.assertIn("include_shared", defaults, "GET /entities lost its include_shared param")
        self.assertIs(_default_of(defaults["include_shared"]), False)

    def test_the_entities_route_does_not_authorise_with_the_read_predicate(self) -> None:
        """`owns` is the READ predicate and is True for every shared row, so using it
        to decide what to BUCKET silently folds shared rows into the default listing.
        The route matches the scopes exactly instead, through `read_scopes`."""
        called = _names_called_in(_function(self.entities_tree, "list_entities"))
        self.assertNotIn("owns", called, "GET /entities is back to authorising with `owns`")
        self.assertIn("read_scopes", called, "GET /entities no longer derives its scopes from read_scopes")


class RoutesCallTheRightGuardTest(unittest.TestCase):
    """Which guard each memory route actually calls, asserted on main.py's AST.

    WHY A TEST ABOUT CALL NAMES. Every guard in `MemoryScope` is unit-tested next to
    it in test_auth0.py, and all of those tests stay green if a route stops calling
    the guard. That is not hypothetical: swapping `_fetch_writable_memory` back to
    `_fetch_owned_memory` in `update_memory` and `delete_memory` is a one-token edit
    that reopens the entire shared-row hole — `owns` answers True for any pinned
    caller on a shared row, so anyone could overwrite or delete any repo-wide note —
    and the whole suite would have passed.

    So these assert the WIRING, in both directions. A route naming the read predicate
    where the write one belongs is the regression; a route naming neither is the same
    regression with an extra step.

    Names, not behaviour, because the behaviour cannot be reached from here: main.py
    builds a real `Memory` at import scope. An AST assertion is weaker than a request
    through the route and it is what is available; it is worth having because the
    edit it catches is small, plausible, and invisible in every other test.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.tree = _module_tree("main.py")

    def _called_in(self, route: str) -> set[str]:
        return _names_called_in(_function(self.tree, route))

    def test_the_mutating_routes_use_the_write_predicate(self) -> None:
        """`owns` is True for a shared row for EVERY pinned caller. `may_modify` —
        reached only through `_fetch_writable_memory` — is what narrows that back to
        the author or an admin."""
        for route in ("update_memory", "delete_memory"):
            with self.subTest(route=route):
                called = self._called_in(route)
                self.assertIn("_fetch_writable_memory", called)
                self.assertNotIn(
                    "_fetch_owned_memory",
                    called,
                    f"{route} authorises with the READ predicate; any pinned caller could "
                    "then overwrite or delete any shared memory",
                )

    def test_the_read_routes_still_use_the_read_predicate(self) -> None:
        """The other direction. Tightening these to the write predicate would 403 a
        colleague following a committed memory URL, which is the failure the shared
        scope exists to fix."""
        for route in ("get_memory", "memory_history"):
            with self.subTest(route=route):
                self.assertIn("_fetch_owned_memory", self._called_in(route))

    def test_search_refuses_scope_escaping_filters_before_pinning(self) -> None:
        """`filters` is the only free-form dict a caller puts in front of the vector
        store, and `{**filters, "user_id": pinned}` does not settle it — mem0 flattens
        an `AND` group back over the top level afterwards."""
        self.assertIn("reject_scope_escaping_filters", self._called_in("search_memories"))

    def test_a_shared_write_cannot_choose_its_own_grouping_ids(self) -> None:
        """A shared row carrying `run_id: <someone else's memory id>` renders as a
        fact of that memory in the console, and the fact layer is what recall
        retrieves — so on this path the grouping ids are an injection primitive."""
        self.assertIn("reject_shared_grouping_ids", self._called_in("add_memory"))

    def test_update_runs_both_halves_of_the_put_guard(self) -> None:
        """Two calls and not one, because they can run at different times: the
        `user_id` refusal needs nothing but the body and so precedes the `get`, while
        the derived-key comparison and the shared-row grouping refusal need the
        stored row. Dropping either leaves the other looking sufficient."""
        update = self._called_in("update_memory")
        for guard in ("reject_ownership_metadata", "reject_row_metadata_conflicts"):
            with self.subTest(guard=guard):
                self.assertIn(guard, update)


if __name__ == "__main__":
    unittest.main(verbosity=2)
