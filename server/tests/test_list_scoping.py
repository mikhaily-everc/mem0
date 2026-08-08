"""`GET /memories` must be able to exclude the agent-scoped layer.

WHY THIS EXISTS. The vector store's `list()` is `SELECT id, payload FROM <col>
WHERE <filters> LIMIT %s` — no ORDER BY, no OFFSET. So a listing is an arbitrary
heap-order slice and nothing can page past the cap.

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

Tests target `listing.py` rather than the route: main.py builds a real `Memory`
at import scope, so importing it needs credentials and a database. stdlib
`unittest` for the reason at the top of test_auth0.py — nothing may be added to
the resolved dependency graph, because that graph is what the image ships.
"""

from __future__ import annotations

import unittest

from listing import listing_filters

USER = "auth0|someone"


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
