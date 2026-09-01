from collections import defaultdict
from datetime import datetime
from typing import Any, Literal, Optional

from auth import MemoryScope, require_admin, resolve_scope
from errors import upstream_error
from fastapi import APIRouter, Depends, HTTPException, Query
from listing import read_scopes
from pydantic import BaseModel
from schemas import MessageResponse
from server_state import get_memory_instance

router = APIRouter(prefix="/entities", tags=["entities"])

SCAN_LIMIT = 10_000

EntityType = Literal["user", "agent", "run"]
TYPE_TO_FIELD: dict[EntityType, str] = {"user": "user_id", "agent": "agent_id", "run": "run_id"}


class Entity(BaseModel):
    id: str
    type: EntityType
    total_memories: int
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


def _iter_payloads() -> list[dict[str, Any]]:
    results = get_memory_instance().vector_store.list(top_k=SCAN_LIMIT)
    rows = results[0] if results and isinstance(results, list) and isinstance(results[0], list) else results or []
    return [getattr(row, "payload", None) or {} for row in rows]


def _parse_timestamp(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


@router.get("", response_model=list[Entity])
def list_entities(
    include_shared: bool = Query(
        False,
        description=(
            "Also count the reserved repo-wide scopes. Off by default so an existing "
            "caller's result set is unchanged, and so the entity names a shared row "
            "carries are not disclosed to someone who did not ask for them."
        ),
    ),
    scope: MemoryScope = Depends(resolve_scope),
):
    """List the entities present in the caller's own memories.

    Unscoped, this route was the widest read in the API: an unfiltered 10k-row
    vector-store scan that hands any authenticated caller every `user_id`,
    `agent_id` and `run_id` in the store with per-entity counts. Unlike
    `GET /memories` it had no admin branch at all, so a plain user saw everything.

    The scopes are matched exactly rather than through `MemoryScope.owns`, which is
    the READ predicate and answers True for a shared row. Leaving `owns` here made
    this the ONE route whose results changed for an existing caller with no opt-in
    when the shared scope arrived — and it is in `assemble-spec.ts`'s KEEP set, so an
    LLM reaches it. `include_shared` restores the choice and matches `GET /memories`
    and `POST /search`, which take the same flag with the same default for the same
    reason.

    What the opt-in actually discloses, since the default's guarantee is stronger
    than it looks: a shared row's `agent_id` and `run_id`. Those are grouping ids
    from a repo-wide row every pinned caller may already read in full through
    `GET /memories?include_shared=true`, so surfacing them under the identical flag
    is consistent rather than a second, quieter widening. What stays out at BOTH
    settings is any entity from another person's PRIVATE row.
    """
    buckets: dict[tuple[EntityType, str], dict[str, Any]] = defaultdict(
        lambda: {"total_memories": 0, "created_at": None, "updated_at": None}
    )
    # `read_scopes` rather than a hand-rolled pair, so "which scopes does one logical
    # read cover" has exactly one definition and this route cannot drift from the two
    # that already use it. None for an unpinned caller: they see the whole store here
    # as they always did.
    scopes = read_scopes(scope.pinned_user_id, include_shared) if scope.is_pinned else None

    for payload in _iter_payloads():
        # Drop rows outside the caller's scopes BEFORE bucketing, so no foreign
        # agent_id or run_id can be inferred from a memory the caller did not ask
        # to see.
        if scopes is not None and payload.get("user_id") not in scopes:
            continue
        created = _parse_timestamp(payload.get("created_at"))
        updated = _parse_timestamp(payload.get("updated_at")) or created

        for entity_type, field in TYPE_TO_FIELD.items():
            value = payload.get(field)
            if not value:
                continue
            bucket = buckets[(entity_type, str(value))]
            bucket["total_memories"] += 1
            if created and (bucket["created_at"] is None or created < bucket["created_at"]):
                bucket["created_at"] = created
            if updated and (bucket["updated_at"] is None or updated > bucket["updated_at"]):
                bucket["updated_at"] = updated

    return [
        Entity(id=entity_id, type=entity_type, **data)
        for (entity_type, entity_id), data in sorted(buckets.items(), key=lambda item: (item[0][0], item[0][1]))
    ]


@router.delete("/{entity_type}/{entity_id}", response_model=MessageResponse)
def delete_entity(
    entity_type: EntityType,
    entity_id: str,
    _auth=Depends(require_admin),
    scope: MemoryScope = Depends(resolve_scope),
):
    # A pinned caller may only delete their own `user` entity. Without this an
    # `mcp-admin` holder could delete a colleague's entire memory set by id, and
    # for agent/run entities there is no owner field to check against at all.
    if scope.is_pinned and not (entity_type == "user" and entity_id == scope.pinned_user_id):
        raise HTTPException(status_code=404, detail="Entity not found.")
    try:
        get_memory_instance().delete_all(**{TYPE_TO_FIELD[entity_type]: entity_id})
    except Exception as exc:
        raise upstream_error() from exc
    return MessageResponse(message="Entity deleted")
