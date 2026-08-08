"""Filter construction for `GET /memories`.

Its own module, rather than inline in main.py, because main.py builds a real
`Memory` at import scope — embedder client and all — so anything living there can
only be tested with credentials and a database. This is a pure function over
strings; it needs neither.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# The filter language's key-exists test. `$not` negates the group containing it,
# so `{"$not": [{"agent_id": AGENT_KEY_EXISTS}]}` reads as "rows with no agent_id".
# There is no IS NULL: `{"agent_id": None}` compiles to `payload->>'agent_id' =
# 'None'` and matches nothing, which is a listing that returns zero rows and still
# exits 200.
AGENT_KEY_EXISTS = "*"


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
        key: value
        for key, value in (("user_id", user_id), ("run_id", run_id), ("agent_id", agent_id))
        if value
    }
    if not include_agent_scoped and not agent_id:
        filters["$not"] = [{"agent_id": AGENT_KEY_EXISTS}]
    return filters
