import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from auth import generate_api_key, require_auth
from db import get_db
from models import APIKey, User
from schemas import MessageResponse

router = APIRouter(prefix="/api-keys", tags=["api-keys"])


class CreateKeyRequest(BaseModel):
    label: str


class CreateKeyResponse(BaseModel):
    id: str
    key: str
    label: str
    key_prefix: str
    created_at: datetime


class KeyListItem(BaseModel):
    id: str
    label: str
    key_prefix: str
    created_at: datetime
    last_used_at: datetime | None

    model_config = {"from_attributes": True}


@router.get("", response_model=list[KeyListItem])
def list_keys(user: User = Depends(require_auth), db: Session = Depends(get_db)):
    keys = (
        db.execute(
            select(APIKey)
            .where(APIKey.created_by == user.id, APIKey.revoked_at.is_(None))
            .order_by(APIKey.created_at.desc())
        )
        .scalars()
        .all()
    )
    return [
        KeyListItem(
            id=str(k.id),
            label=k.label,
            key_prefix=k.key_prefix,
            created_at=k.created_at,
            last_used_at=k.last_used_at,
        )
        for k in keys
    ]


@router.post("", response_model=CreateKeyResponse, status_code=201)
def create_key(
    request: Request, body: CreateKeyRequest, user: User = Depends(require_auth), db: Session = Depends(get_db)
):
    # An SSO caller may NOT mint a long-lived key.
    #
    # `X-API-Key` survives on purpose, for agents and CI that Auth0 has never
    # heard of. The problem is the other direction: a human who reached this
    # route with an Auth0 token walking away with a credential that Auth0 cannot
    # revoke. `_resolve_user_from_api_key` consults neither Auth0 nor JumpCloud,
    # so such a key keeps working after the person is disabled — the precise
    # failure this server's SSO work exists to remove, reintroduced through the
    # dashboard's "create API key" button.
    #
    # Gate on HOW THIS REQUEST AUTHENTICATED, not on `user.auth0_sub`. For an
    # `ADMIN_API_KEY` or `AUTH_DISABLED` caller `require_auth` substitutes the
    # OLDEST user row, which here is an adopted SSO account — so keying on the
    # resolved user would break the break-glass path that exists to issue agent
    # keys in the first place.
    if getattr(request.state, "auth_type", "none") == "auth0":
        raise HTTPException(
            status_code=403,
            detail=(
                "API keys cannot be created with an SSO session, because such a key would outlive "
                "SSO revocation. Ask an operator to issue one for a non-human caller."
            ),
        )

    full_key, prefix, key_hash = generate_api_key()
    api_key = APIKey(key_prefix=prefix, key_hash=key_hash, label=body.label, created_by=user.id)
    db.add(api_key)
    db.commit()
    db.refresh(api_key)

    return CreateKeyResponse(
        id=str(api_key.id),
        key=full_key,
        label=api_key.label,
        key_prefix=prefix,
        created_at=api_key.created_at,
    )


@router.delete("/{key_id}", response_model=MessageResponse)
def revoke_key(key_id: str, user: User = Depends(require_auth), db: Session = Depends(get_db)):
    try:
        key_uuid = uuid.UUID(key_id)
    except (TypeError, ValueError):
        # `from None`: an unparseable id is input validation, and the 404 is meant to
        # read identically to "exists but is not yours" — see the ownership check
        # below. The parse error adds nothing.
        raise HTTPException(status_code=404, detail="API key not found.") from None
    api_key = db.get(APIKey, key_uuid)
    if api_key is None or api_key.created_by != user.id:
        raise HTTPException(status_code=404, detail="API key not found.")
    if api_key.revoked_at is not None:
        raise HTTPException(status_code=400, detail="API key is already revoked.")

    api_key.revoked_at = datetime.now(timezone.utc)
    db.commit()
    return MessageResponse(message="API key revoked.")
