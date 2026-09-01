"""Scope and filter construction for the memory read routes.

Its own module, rather than inline in main.py, because main.py builds a real
`Memory` at import scope — embedder client and all — so anything living there can
only be tested with credentials and a database. These are pure functions over
strings; they need neither.

It is also the leaf of the import graph, which is why `SHARED_SCOPES` lives here
rather than next to `MemoryScope` in auth.py: auth.py imports it, main.py imports
it, and nothing it imports can import them back.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

# The filter language's key-exists test. `$not` negates the group containing it,
# so `{"$not": [{"agent_id": AGENT_KEY_EXISTS}]}` reads as "rows with no agent_id".
# There is no IS NULL: `{"agent_id": None}` compiles to `payload->>'agent_id' =
# 'None'` and matches nothing, which is a listing that returns zero rows and still
# exits 200.
AGENT_KEY_EXISTS = "*"

# Reserved `user_id` values that are NOT a person. A row stored under one of these
# is a repo-wide note — the kind whose URL gets committed to version control — and
# `MemoryScope.owns` lets any pinned caller read it.
#
# Why the values live here, in the one module with no imports: `auth.py` needs them
# to answer "may this caller read this row", `main.py` needs them to build the
# second listing, and both of those can only be imported with credentials and a
# database behind them. Keeping the set in a leaf module is what makes the scope
# rules assertable at all.
#
# The `<origin>:<name>` shape is deliberate and matches `Auth0Claims.memory_scope`,
# which mints `auth0:<sub>`. A shared scope therefore cannot collide with any
# Auth0-derived one no matter what a tenant puts in a `sub`.
SHARED_SCOPES: frozenset[str] = frozenset({"repo:ai-utils"})


def listing_filters(
    user_id: Optional[str],
    run_id: Optional[str],
    agent_id: Optional[str],
    include_agent_scoped: bool = True,
) -> Dict[str, Any]:
    """Build the filter dict for a scoped `GET /memories`.

    `include_agent_scoped=False` restricts the listing to rows carrying no
    `agent_id`. That exists because the vector store's `list()` is
    `SELECT id, payload FROM <col> WHERE <filters> LIMIT %s` — no ORDER BY, no
    OFFSET — so a listing is an arbitrary heap-order slice that nothing can page
    past.

    Harmless while a whole scope fits under the cap; not harmless once a derived
    agent-scoped layer outgrows the documents it was derived from. Measured here
    2026-08-08: `top_k=1000` returned 593 documents plus 407 derived rows, and
    every memory written after the layer was built was absent from the response.
    No error — LIMIT is not an error — so enumeration had silently stopped.

    An explicit `agent_id` is a different question ("that agent's rows") and is
    never negated; asking for both is a contradiction, and the explicit value wins.
    """
    filters: Dict[str, Any] = {
        key: value for key, value in (("user_id", user_id), ("run_id", run_id), ("agent_id", agent_id)) if value
    }
    if not include_agent_scoped and not agent_id:
        filters["$not"] = [{"agent_id": AGENT_KEY_EXISTS}]
    return filters


def read_scopes(user_id: Optional[str], include_shared: bool) -> Tuple[Optional[str], ...]:
    """The `user_id` values one logical read has to be run against, in order.

    `include_shared=False` — the default everywhere — returns the caller's own
    scope alone, so an existing caller's result set is bit-for-bit what it was.
    That default is load-bearing rather than cautious: `.claude/hooks/mem0/`
    recall runs against thresholds (0.350 vector, 0.30 rerank) derived by
    measurement over exactly today's corpus, and quietly enlarging the corpus
    would move the separation those numbers were fitted to.

    Two calls, not one filter. There is no way to express "mine OR shared" in a
    single `get_all`/`search`: both require a top-level `user_id`/`agent_id`/`run_id`
    and both push that value through `_validate_and_trim_entity_id`, which does
    `str(value)` on anything non-string — so `{"user_id": ["a", "b"]}` becomes the
    literal `"['a', 'b']"`, matches no row, and returns 200 with an empty list. A
    top-level `$or` beside `user_id` is ANDed with it and collapses back to "mine".
    Two calls, each carrying one plain string, also depend on nothing but the
    documented contract — `requirements.txt` pins `mem0ai>=0.1.48`, a floor, so the
    filter-compiler internals that a `$or` scheme would rest on are not a fixed
    target.

    The shared scope is subtracted from itself so a caller who names a shared
    `user_id` outright does not get every one of its rows twice.
    """
    if not include_shared or not user_id:
        return (user_id,)
    return (user_id, *sorted(SHARED_SCOPES - {user_id}))
