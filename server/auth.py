import logging
import os
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from auth0 import Auth0Claims, Auth0VerificationError, looks_like_auth0_token, verify_auth0_token
from db import get_db
from fastapi import Depends, HTTPException, Request
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from models import APIKey, RefreshTokenJti, User
from passlib.context import CryptContext
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

JWT_SECRET = os.environ.get("JWT_SECRET", "")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 30
ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "")
AUTH_DISABLED = os.environ.get("AUTH_DISABLED", "").lower() in {"1", "true", "yes", "on"}

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    # An SSO-provisioned row has `password_hash = ""`. passlib raises on an empty
    # hash rather than returning False, and an exception escaping here would be a
    # 500 instead of a clean 401 — so short-circuit it. Without this, the password
    # login path is the one place an SSO account could be reached without Auth0.
    if not hashed:
        return False
    return pwd_context.verify(plain, hashed)


def dummy_verify_password() -> None:
    """Burn the same bcrypt cycles as a real verify so login timing doesn't leak whether an email exists."""
    pwd_context.dummy_verify()


def generate_api_key() -> tuple[str, str, str]:
    """Returns (full_key, prefix, hash)."""
    raw = secrets.token_urlsafe(32)
    full_key = f"m0sk_{raw}"
    prefix = full_key[:12]
    key_hash = pwd_context.hash(full_key)
    return full_key, prefix, key_hash


def verify_api_key_hash(plain_key: str, hashed: str) -> bool:
    return pwd_context.verify(plain_key, hashed)


def _get_secret() -> str:
    if not JWT_SECRET:
        raise HTTPException(status_code=500, detail="JWT_SECRET is not configured.")
    return JWT_SECRET


def create_access_token(user_id: str, role: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": user_id, "role": role, "exp": expire, "type": "access"}
    return jwt.encode(payload, _get_secret(), algorithm=JWT_ALGORITHM)


def create_refresh_token(user_id: str, db: Session) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    jti = uuid.uuid4()
    db.add(RefreshTokenJti(jti=jti, user_id=uuid.UUID(user_id), expires_at=expire))
    db.commit()
    payload = {"sub": user_id, "exp": expire, "jti": str(jti), "type": "refresh"}
    return jwt.encode(payload, _get_secret(), algorithm=JWT_ALGORITHM)


def consume_refresh_jti(jti: str, db: Session) -> None:
    """Atomically mark a refresh token's jti as used. Raises 401 if missing, already used, or expired.

    The conditional UPDATE closes the read-check-write race: concurrent replays of the same
    token race on a single row, so at most one update affects a row and the rest see rowcount 0.
    """
    try:
        jti_uuid = uuid.UUID(jti)
    except (TypeError, ValueError):
        raise HTTPException(status_code=401, detail="Refresh token is no longer valid.")
    now = datetime.now(timezone.utc)
    result = db.execute(
        update(RefreshTokenJti)
        .where(
            RefreshTokenJti.jti == jti_uuid,
            RefreshTokenJti.used_at.is_(None),
            RefreshTokenJti.expires_at > now,
        )
        .values(used_at=now)
    )
    if result.rowcount == 0:
        raise HTTPException(status_code=401, detail="Refresh token is no longer valid.")
    db.commit()


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, _get_secret(), algorithms=[JWT_ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")


bearer_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _mark_auth_type(request: Request, auth_type: str) -> None:
    request.state.auth_type = auth_type


def _get_default_user(db: Session) -> User | None:
    return db.scalar(select(User).order_by(User.created_at.asc()))


def _resolve_user_from_jwt(token: str, db: Session) -> User:
    payload = decode_token(token)
    if payload.get("type") != "access":
        raise HTTPException(status_code=401, detail="Invalid token type.")
    user = db.get(User, payload.get("sub"))
    if user is None:
        raise HTTPException(status_code=401, detail="User not found.")
    return user


def _resolve_user_from_api_key(key: str, db: Session) -> User:
    prefix = key[:12] if len(key) >= 12 else key
    candidates = (
        db.execute(select(APIKey).where(APIKey.key_prefix == prefix, APIKey.revoked_at.is_(None))).scalars().all()
    )

    for candidate in candidates:
        if verify_api_key_hash(key, candidate.key_hash):
            candidate.last_used_at = datetime.now(timezone.utc)
            db.commit()
            user = db.get(User, candidate.created_by)
            if user is None:
                raise HTTPException(status_code=401, detail="API key owner not found.")
            return user

    raise HTTPException(status_code=401, detail="Invalid API key.")


def _resolve_user_from_auth0(claims: Auth0Claims, db: Session) -> User:
    """Map a verified Auth0 identity onto a `users` row, provisioning on first use.

    The row is the durable anchor for everything already keyed on `users.id`
    (API keys, request logs, `/auth/me`); the Auth0 token remains the only way to
    authenticate as it, since `password_hash` stays empty.

    Three cases, in order:

    1. A row already carries this `auth0_sub` — use it.
    2. A row carries the same `email` but no `auth0_sub` — ADOPT it. This is the
       normal upgrade path for anyone who had a password account before SSO (and
       is not hypothetical: the POC admin account exists with a real address).
       Inserting instead would violate the unique `email` index and strand the
       user behind a 500, and even if `email` were not unique it would split one
       person across two ids.
    3. Nothing matches — insert.

    `users.role` is NEVER set from the roles claim. Revision 004 created
    `ix_users_only_one_admin`, a partial unique index over `role` where
    `role = 'admin'`, so the table permits AT MOST ONE admin row — writing `admin`
    for a second SSO caller is an IntegrityError, not a policy decision. Admin-ness
    for Auth0 callers therefore lives in the claim and is read by `require_admin`
    and `resolve_scope`, which is where it belongs: the claim is the source of
    truth, and caching it in a uniqueness-constrained column could not represent
    two admins even if it wanted to.

    An adopted row keeps whatever `role` it already had — downgrading someone's
    existing account as a side effect of their first SSO login would be a
    surprising write, and `require_admin` no longer depends on it for this path.
    """
    user = db.scalar(select(User).where(User.auth0_sub == claims.sub))

    if user is None and claims.email:
        existing = db.scalar(select(User).where(User.email == claims.email))
        if existing is not None:
            if existing.auth0_sub not in (None, claims.sub):
                # Same address, different SSO subject. Adopting would hand this
                # caller another identity's memories, so fail closed instead.
                raise HTTPException(
                    status_code=409,
                    detail="This email is already linked to a different SSO identity.",
                )
            existing.auth0_sub = claims.sub
            user = existing

    if user is None:
        user = User(
            auth0_sub=claims.sub,
            # A token with no `email` claim still needs a value for the NOT NULL
            # unique column; key it on the sub so it cannot collide.
            email=claims.email or f"{claims.sub}@sso.invalid",
            name=claims.email or claims.sub,
            password_hash="",
            # Always "user" — see the note above about ix_users_only_one_admin.
            role="user",
        )
        db.add(user)

    try:
        db.commit()
    except IntegrityError:
        # Two concurrent first-time logins raced on the unique index. The loser
        # rolls back and re-reads the winner's row — same pattern as the register
        # handler in routers/auth.py.
        db.rollback()
        user = db.scalar(select(User).where(User.auth0_sub == claims.sub))
        if user is None:
            raise HTTPException(status_code=500, detail="Could not provision SSO account.")
        return user

    db.refresh(user)
    return user


async def verify_auth(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    x_api_key: str | None = Depends(api_key_header),
    db: Session = Depends(get_db),
) -> User | None:
    """Authenticate via Auth0 RS256, native HS256 JWT, X-API-Key, or legacy ADMIN_API_KEY.

    Returns a User, or None for the credential-less admin paths (ADMIN_API_KEY and
    AUTH_DISABLED), which callers distinguish via `request.state.auth_type`.
    """
    if credentials is not None:
        token = credentials.credentials
        # Route by key type, not by trying one and falling back: an RS256 token
        # with a `kid` can only be Auth0's, and the native path is HS256 with no
        # `kid`. Falling back would let a rejected Auth0 token be re-tried against
        # the shared HS256 secret.
        if looks_like_auth0_token(token):
            try:
                claims = verify_auth0_token(token)
            except Auth0VerificationError as exc:
                # Logged server-side, never echoed to the client: the operator
                # needs to tell "the tenant is misconfigured" from "that token is
                # genuinely bad", but telling the CALLER whether it failed on
                # signature, audience, issuer or expiry is an oracle. Without this
                # log a misconfiguration is indistinguishable from a bad token
                # from the outside, which is a bad place to debug from.
                logging.warning(
                    "auth0: rejected a bearer token: %s (cause: %r)", exc, exc.__cause__
                )
                raise HTTPException(
                    status_code=401,
                    detail="Invalid or expired token.",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            _mark_auth_type(request, "auth0")
            request.state.auth0_claims = claims
            return _resolve_user_from_auth0(claims, db)

        _mark_auth_type(request, "bearer")
        return _resolve_user_from_jwt(token, db)

    if x_api_key is not None:
        if ADMIN_API_KEY and secrets.compare_digest(x_api_key, ADMIN_API_KEY):
            _mark_auth_type(request, "admin_api_key")
            return None
        _mark_auth_type(request, "api_key")
        return _resolve_user_from_api_key(x_api_key, db)

    if AUTH_DISABLED:
        _mark_auth_type(request, "disabled")
        return None

    raise HTTPException(
        status_code=401,
        detail="Authentication required. Provide a Bearer token or X-API-Key header.",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def require_auth(
    request: Request,
    user: User | None = Depends(verify_auth),
    db: Session = Depends(get_db),
) -> User:
    """Like verify_auth but guarantees a non-None User. Use for endpoints that require auth."""
    if user is None:
        if getattr(request.state, "auth_type", "none") in {"admin_api_key", "disabled"}:
            default_user = _get_default_user(db)
            if default_user is not None:
                return default_user
        raise HTTPException(status_code=401, detail="Authentication required.")
    return user


_BOOTSTRAP_ADMIN = User(
    id=uuid.UUID(int=0), name="admin_api_key", email="", password_hash="", role="admin", created_at=datetime.min.replace(tzinfo=timezone.utc),
)


async def require_admin(
    request: Request,
    user: User | None = Depends(verify_auth),
    db: Session = Depends(get_db),
) -> User:
    """Like require_auth but also enforces admin role.

    ADMIN_API_KEY and AUTH_DISABLED callers are treated as admin even when
    the users table is empty (fresh-deploy bootstrap).
    """
    auth_type = getattr(request.state, "auth_type", "none")
    if user is None:
        if auth_type in {"admin_api_key", "disabled"}:
            default_user = _get_default_user(db)
            if default_user is not None:
                if default_user.role != "admin":
                    raise HTTPException(status_code=403, detail="Admin role required.")
                return default_user
            return _BOOTSTRAP_ADMIN
        raise HTTPException(status_code=401, detail="Authentication required.")
    if auth_type == "auth0":
        # Admin-ness comes from the roles claim, not `user.role`, which is always
        # "user" for SSO rows — `ix_users_only_one_admin` (revision 004) permits at
        # most one admin row, so the column cannot represent a second SSO admin.
        # Checking the claim also means revoking `mcp-admin` in Auth0 takes effect
        # on the caller's next request with no DB write.
        claims: Auth0Claims = request.state.auth0_claims
        if not claims.is_admin:
            raise HTTPException(status_code=403, detail="Admin role required.")
        return user
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin role required.")
    return user


# ---------------------------------------------------------------------------
# Memory scope
#
# The bug this exists to close: every memory route resolved an identity and then
# discarded it, taking `user_id` from the request body instead. So authentication
# never implied isolation — any caller could read, overwrite or delete any other
# caller's memories just by naming their `user_id`, or by omitting it and reading
# the shared pool.
#
# The fix is deliberately an EXPLICIT dependency rather than something stashed on
# `request.state`. A route that needs scoping and forgets to ask for it is then a
# visible omission in its signature, which is reviewable; an implicit lookup that
# silently returns "unscoped" is exactly how the original hole survived.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemoryScope:
    """Which memories the current caller may see and write.

    `pinned_user_id` is non-None only for Auth0 callers. The legacy credential
    paths (native HS256 JWT, `X-API-Key`, `ADMIN_API_KEY`, `AUTH_DISABLED`) keep
    the pre-existing caller-supplied behaviour so agents and CI do not break; see
    the `X-API-Key` note in the deployment docs.
    """

    pinned_user_id: str | None
    is_admin: bool

    @property
    def is_pinned(self) -> bool:
        return self.pinned_user_id is not None

    def pin(self, requested: str | None) -> str | None:
        """The `user_id` to actually use, ignoring whatever the caller asked for."""
        return self.pinned_user_id if self.is_pinned else requested

    def owns(self, stored_user_id: str | None) -> bool:
        """Whether a stored memory belongs to this caller.

        Unpinned callers retain full access. A pinned caller matches only their own
        scope — note a stored `None` never matches, so pre-existing unscoped
        memories become unreachable rather than shared.
        """
        if not self.is_pinned:
            return True
        return stored_user_id == self.pinned_user_id


async def resolve_scope(
    request: Request,
    user: User | None = Depends(verify_auth),
) -> MemoryScope:
    """Derive the caller's memory scope from their verified credential."""
    auth_type = getattr(request.state, "auth_type", "none")
    if auth_type == "auth0":
        claims: Auth0Claims = request.state.auth0_claims
        # Admin-ness governs the "list everything" branch only; it deliberately
        # does NOT unpin the scope, so an admin's own reads and writes still land
        # in their own scope rather than the shared pool.
        return MemoryScope(pinned_user_id=claims.memory_scope, is_admin=claims.is_admin)
    return MemoryScope(pinned_user_id=None, is_admin=(user is None or user.role == "admin"))
