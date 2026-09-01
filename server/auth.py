import logging
import os
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from auth0 import Auth0Claims, Auth0VerificationError, looks_like_auth0_token, verify_auth0_token
from db import get_db
from fastapi import Depends, HTTPException, Request
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from listing import SHARED_SCOPES
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
        # `from None`, not `from exc`: a malformed jti is input validation, and the
        # 401 deliberately reads the same as a replayed or expired one — the parse
        # error carries nothing an operator wants and telling the two apart is an
        # oracle. Same reasoning as the auth0 branch below.
        raise HTTPException(status_code=401, detail="Refresh token is no longer valid.") from None
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
    except JWTError as exc:
        raise HTTPException(status_code=401, detail="Invalid or expired token.") from exc


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


def adopt_row_into_sso(existing: User, sub: str) -> None:
    """Link a pre-existing account to an Auth0 subject AND close its password door.

    Extracted from `_resolve_user_from_auth0` so the property can be asserted
    without standing up a database — the mutation is the security-relevant part,
    and it should not be reachable only through a Session.

    Clearing the hash is the half that is easy to miss. `verify_password` short-
    circuits only on an EMPTY hash: a row created by the insert path gets
    `password_hash=""` and is SSO-only, but an ADOPTED row keeps whatever hash it
    had. Without the clear, that person keeps a working password and reaches the
    dashboard without ever touching Auth0 — the revocation hole the SSO work
    exists to close, left open on the account most likely to already have a
    password. Not hypothetical: the POC admin is an adopted row with a real
    address.

    Safe to clear unconditionally. `ADMIN_API_KEY` is an independent break-glass
    that never consults `users.password_hash`, so an operator is not locked out by
    an Auth0 outage.
    """
    existing.auth0_sub = sub
    existing.password_hash = ""


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
            adopt_row_into_sso(existing, claims.sub)
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
    except IntegrityError as exc:
        # Two concurrent first-time logins raced on the unique index. The loser
        # rolls back and re-reads the winner's row — same pattern as the register
        # handler in routers/auth.py.
        db.rollback()
        user = db.scalar(select(User).where(User.auth0_sub == claims.sub))
        if user is None:
            # Chained: reaching here means the row is missing after a constraint
            # violation said it exists, which is not the documented race and the
            # constraint name is the only clue to what actually happened.
            raise HTTPException(status_code=500, detail="Could not provision SSO account.") from exc
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
                logging.warning("auth0: rejected a bearer token: %s (cause: %r)", exc, exc.__cause__)
                # `from None`: the cause has just been logged above, deliberately
                # server-side only. Chaining it here would put the same detail into
                # the traceback of a response the comment above says must stay
                # indistinguishable from any other bad token.
                raise HTTPException(
                    status_code=401,
                    detail="Invalid or expired token.",
                    headers={"WWW-Authenticate": "Bearer"},
                ) from None
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
    id=uuid.UUID(int=0),
    name="admin_api_key",
    email="",
    password_hash="",
    role="admin",
    created_at=datetime.min.replace(tzinfo=timezone.utc),
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
#
# The one crack deliberately left in that wall is the SHARED scope, below: pinning
# also made a memory's URL useless to anyone but its author, so a repo-wide code
# note committed to version control 404s for every colleague who follows the link.
# The crack is read-only and its width is one enumerated set of `user_id` values —
# read shared, write attributed, never a blanket unpin.
# ---------------------------------------------------------------------------

#: Metadata key a writer sets to opt a row into a shared scope, and the key the
#: server stamps their verified subject onto in return. Both are ordinary payload
#: keys once stored — mem0 flattens `metadata` into the row's payload — so they are
#: readable back through `Memory.get()`'s `metadata` dict and filterable in a query.
SHARED_SCOPE_KEY = "scope"
AUTHOR_SUB_KEY = "author_sub"

#: Metadata keys a pinned caller may never write through `PUT /memories/{id}`,
#: whatever the row says.
#: mem0's `_update_memory` does `new_metadata = deepcopy(existing.payload)` then
#: `new_metadata.update(metadata)`, so a metadata dict naming `user_id` OVERWRITES
#: the row's identity in place: it re-keys the row into another caller's scope, or
#: into the shared one, bypassing the `POST /memories` opt-in entirely. There is no
#: legitimate spelling of that, from any caller, on any row — hence a flat refusal
#: rather than a comparison against what is stored.
OWNERSHIP_METADATA_KEYS = ("user_id",)

#: Metadata keys the SERVER derives, which a `PUT` may ECHO BACK but never CHANGE.
#:
#: These two used to sit in `OWNERSHIP_METADATA_KEYS`, i.e. naming either in a `PUT`
#: was a flat 400 — and that broke the documented round trip. mem0's `Memory.get()`
#: returns BOTH inside `metadata` (it promotes only the entity ids and buckets every
#: other payload key there), and the recorded workflow is "search first and update
#: the existing row rather than adding a near-duplicate": GET, edit, PUT what you
#: read. So a shared row's own author could not update it without knowing to strip
#: two keys the error message never mentioned, and the 400 read like a permission
#: problem rather than a payload one.
#:
#: `reject_row_metadata_conflicts` compares them against the stored row instead —
#: equal is a no-op and passes, different is still a 400. Silently dropping them was
#: the other option, and a request that quietly means something other than what it
#: says is the failure shape this file keeps getting bitten by.
DERIVED_METADATA_KEYS = (SHARED_SCOPE_KEY, AUTHOR_SUB_KEY)

#: Metadata keys that GROUP rows rather than own them.
#:
#: Free on a private row, and deliberately so: `.claude/hooks/mem0/build_fact_layer.py`
#: writes both on every fact it derives. They stop being mere grouping on a SHARED
#: row, where they become a forgery primitive aimed at somebody else's context — see
#: `reject_shared_grouping_ids`, which refuses them there for a pinned caller only,
#: leaving the unpinned fact-layer build untouched.
GROUPING_METADATA_KEYS = ("agent_id", "run_id")

#: The JSON scalars a pinned caller's `/search` filters may carry. `bool` is a
#: subclass of `int` and so redundant here; it is spelled out because the set is the
#: whole security argument of `reject_scope_escaping_filters` and a reader should not
#: have to know that to see that booleans are allowed.
FILTER_SCALAR_TYPES = (str, int, float, bool)


@dataclass(frozen=True)
class MemoryScope:
    """Which memories the current caller may see and write.

    `pinned_user_id` is non-None only for Auth0 callers. The legacy credential
    paths (native HS256 JWT, `X-API-Key`, `ADMIN_API_KEY`, `AUTH_DISABLED`) keep
    the pre-existing caller-supplied behaviour so agents and CI do not break; see
    the `X-API-Key` note in the deployment docs.

    Three predicates, and they are NOT interchangeable — picking the wrong one is
    how a read-only widening turns into a write-anywhere one:

    * `owns`     — may READ this row. True for the shared scopes.
    * `may_modify` — may OVERWRITE or DELETE it. Shared rows: author or admin only.
    * `shared_write` — resolves a write's `metadata.scope` opt-in, and stamps the
      attribution `may_modify` will later read back.

    Plus four refusals, which exist because the predicates above answer questions
    about a row that is ALREADY in the store, and every one of these is about a
    request that has not reached it yet:

    * `reject_scope_escaping_filters` — a `/search` filter that could outrun the pin.
    * `reject_ownership_metadata`     — a `PUT` re-keying the row's `user_id`.
    * `reject_row_metadata_conflicts` — a `PUT` changing what the server derived,
      or grouping a shared row into somebody else's fact layer.
    * `reject_shared_grouping_ids`    — the same forgery on the `POST` path.
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
        """Whether this caller may READ a stored memory. Not the write predicate.

        Unpinned callers retain full access. A pinned caller matches their own
        scope, plus the reserved shared scopes — note a stored `None` still never
        matches, so pre-existing unscoped memories stay unreachable rather than
        quietly becoming everyone's.

        Reading a shared row is the entire point of the shared scope: a memory URL
        in a committed file has to resolve for the colleague who follows it, and
        before this it 404'd for everyone but the author. It says nothing about
        writing — `may_modify` is the check for that, and every mutating route must
        use it instead of this one.
        """
        if not self.is_pinned:
            return True
        if stored_user_id in SHARED_SCOPES:
            return True
        return stored_user_id == self.pinned_user_id

    def may_modify(self, stored_user_id: str | None, author_sub: str | None) -> bool:
        """Whether this caller may OVERWRITE or DELETE a stored memory.

        Strictly narrower than `owns`. Everyone reads a shared row; only the
        identity that wrote it — recorded as `metadata.author_sub` at creation from
        the verified credential, never from the request body — may change or remove
        it. Without this split, widening `owns` alone would have handed every
        authenticated caller edit rights over every repo-wide note, which is a
        strictly worse hole than the one the shared scope exists to fix.

        An SSO admin IS allowed through here, and that is a deliberate departure
        from `delete_all_memories`, where admin-ness explicitly does not let you
        destroy a colleague's memories. The two are consistent on the principle: a
        shared row is repo-wide by construction and belongs to no colleague's
        private scope, so an admin acting on one is moderating common property, not
        reaching into someone's account. Nothing here lets an admin touch a
        PERSONAL row of another caller — `owns` fails first.

        A shared row whose `author_sub` is missing (written before this existed, or
        stripped by hand in the store) is admin-only, not everyone-writable. Failing
        closed on absent attribution is the whole reason the check reads the author
        rather than merely checking that one exists.
        """
        if not self.owns(stored_user_id):
            return False
        if stored_user_id not in SHARED_SCOPES:
            return True
        if not self.is_pinned:
            # Legacy credentials (X-API-Key, ADMIN_API_KEY, AUTH_DISABLED) keep the
            # pre-existing unrestricted behaviour deliberately; operational cleanup
            # of shared rows runs through the machine credential.
            return True
        return self.is_admin or author_sub == self.pinned_user_id

    def shared_write(self, metadata: dict[str, Any] | None) -> tuple[str, dict[str, Any]] | None:
        """Resolve a `metadata.scope` opt-in into the (user_id, metadata) to store.

        Returns None for an ordinary write, leaving the caller's own pinned scope
        in force.

        The opt-in travels in METADATA and not in a `user_id` argument, which looks
        like an odd choice until you read `tools/mcp/mem0/scripts/assemble-spec.ts`:
        it strips `user_id`/`agent_id`/`run_id` from every generated MCP tool schema
        AND from request-body properties, so an MCP caller has no way to name a
        `user_id` at all. Metadata passes through verbatim. Routing the opt-in
        through it keeps the OpenAPI spec and the gateway's `include_tools`
        allowlist untouched.

        Membership is tested against the enumerated set, never by prefix. A
        `startswith("repo:")` test would be the whole security check on this path,
        and it would accept `auth0:someone-else` the moment the shape changed; an
        exact set cannot be talked into writing into a colleague's scope.

        An unrecognised value is a 400 rather than ordinary metadata. A typo'd
        `repo:aiutils` that silently produced a PRIVATE row is the exact failure
        this whole change exists to fix, arriving as a 200 and discovered only when
        a colleague follows the URL.

        Unpinned callers are passed through untouched: they can already name any
        `user_id` outright, so honouring the opt-in would buy them nothing and would
        mint a shared row with no author — one nothing but the machine credential
        could ever edit again.

        The test is "is the key PRESENT", not "is its value non-null", and the
        difference is a whole failure mode of its own. `metadata.get(...) is None`
        short-circuited BEFORE the allowlist ran, so `{"scope": null}` stored a
        PRIVATE row carrying a useless `scope: None` key and answered 200 — the exact
        silent success the 400 two paragraphs up exists to prevent, reached through
        the one input shape a typo'd string cannot produce. `null` is what a client
        library emits for an unset optional field, so it arrives without anyone
        having decided to send it, and it is now a 400 like every other value that
        is not a known scope.
        """
        metadata = metadata or {}
        if not self.is_pinned or SHARED_SCOPE_KEY not in metadata:
            return None
        requested = metadata[SHARED_SCOPE_KEY]
        # `in` on a frozenset raises TypeError for an unhashable value, and a JSON
        # body can carry a list or an object here — that would be a 500 on a
        # malformed request.
        if not isinstance(requested, str) or requested not in SHARED_SCOPES:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown metadata.{SHARED_SCOPE_KEY}. Shared scopes are: " + ", ".join(sorted(SHARED_SCOPES)),
            )
        stored = dict(metadata)
        # Overwritten, never merged from the body: attribution is what gates every
        # later write to this row, so it may only ever come from the verified
        # credential.
        stored[AUTHOR_SUB_KEY] = self.pinned_user_id
        return requested, stored

    def reject_shared_grouping_ids(
        self,
        agent_id: str | None,
        run_id: str | None,
        metadata: dict[str, Any] | None,
    ) -> None:
        """Refuse a pinned caller's attempt to file a SHARED write into another person's group.

        `agent_id`/`run_id` are grouping and not ownership — true right up until the
        row is shared, at which point they are a fact-forgery primitive that lands
        inside a colleague's agent context:

        the console's memory page resolves a row's derived facts with
        `post_search filters={"run_id": <that memory's id>}` plus `include_shared`
        (`tools/web/console/src/lib/mem0/api.ts`, `factsFor`). So any SHARED row
        carrying `agent_id: "facts"` and `run_id: <Alice's memory id>` renders as a
        fact extracted from Alice's memory. Bob may write it — `metadata.scope` is an
        open opt-in and he authors the row, so `may_modify` keeps letting him edit
        it — and the fact layer is what recall retrieves, which makes this an
        injection path into someone else's context rather than merely a wrong-looking
        page.

        BOTH carriers are refused, and they are NOT equivalent. mem0's
        `_build_filters_and_metadata` does `base_metadata_template =
        deepcopy(input_metadata)` and only THEN `base_metadata_template["agent_id"] =
        agent_id`, under `if agent_id:` — so a grouping id smuggled through
        `metadata` lands in the payload untouched whenever the top-level parameter is
        absent. Guarding only the parameters would have looked complete and stopped
        nothing.

        A refusal rather than a cross-check against "a row this writer may modify",
        because nothing in this deployment writes a shared row with grouping ids at
        all: `.claude/hooks/mem0/build_fact_layer.py` writes both legitimately but
        goes through ADMIN_API_KEY, which is UNPINNED and so never reaches this
        check, and an MCP caller cannot name them at all (`assemble-spec.ts` strips
        all three entity ids from every generated tool schema). So this costs a real
        caller nothing today, and a 400 naming the key is recoverable where a
        silently re-homed row is not.

        The `POST` half. `PUT` is `reject_row_metadata_conflicts`, and BOTH are
        required: this one alone leaves "publish an empty shared row, then PUT the
        grouping ids into it".
        """
        if not self.is_pinned:
            return
        # Keyed BY NAME rather than zipped against the constant: a zip would make the
        # mapping depend on `GROUPING_METADATA_KEYS`' order, and reordering a tuple of
        # two strings is exactly the kind of edit nothing would flag.
        supplied = {"agent_id": agent_id, "run_id": run_id}
        named = [key for key in GROUPING_METADATA_KEYS if key in (metadata or {}) or supplied.get(key)]
        if named:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"A write opting into metadata.{SHARED_SCOPE_KEY} may not set "
                    + ", ".join(named)
                    + ". Grouping ids on a shared row are server-controlled, because they decide "
                    "whose memory the row is presented as belonging to."
                ),
            )

    def reject_ownership_metadata(self, metadata: dict[str, Any] | None) -> None:
        """Refuse a `PUT` whose metadata would rewrite who owns the row.

        Loud rather than silent: quietly dropping the key would let a caller believe
        they had published a note repo-wide when they had not, which is the same
        class of plausible-looking success as the typo'd scope above.

        Only pinned callers are held to it. The legacy credential paths are
        deliberately unscoped everywhere else in this class and re-keying a row is
        within what they could already do by deleting and re-adding it.

        Deliberately row-INDEPENDENT, so it can run before the `get` that fetches the
        row is spent. `user_id` is the one key with no legitimate spelling from any
        pinned caller on any row; everything that needs to know what the row already
        says is in `reject_row_metadata_conflicts`.
        """
        if not self.is_pinned or not metadata:
            return
        offending = [key for key in OWNERSHIP_METADATA_KEYS if key in metadata]
        if offending:
            raise HTTPException(
                status_code=400,
                detail="metadata may not set " + ", ".join(offending) + " on an existing memory.",
            )

    def reject_row_metadata_conflicts(
        self,
        metadata: dict[str, Any] | None,
        stored_user_id: str | None,
        stored_metadata: dict[str, Any] | None,
    ) -> None:
        """The half of the `PUT` guard that needs the stored row in hand.

        Runs AFTER `_fetch_writable_memory` rather than beside
        `reject_ownership_metadata`, because both of its properties are comparisons
        against what the row already says:

        1. `scope` and `author_sub` may be ECHOED BACK, never CHANGED. mem0's
           `Memory.get()` returns both inside `metadata`, and the recorded workflow
           is GET, edit, PUT what you read — so a flat refusal locked a shared row's
           own author out of updating it. Equal to the stored value is a no-op and
           passes; anything else is still a 400. Note that changing `scope` alone
           could not by itself re-home a row — `owns` and `read_scopes` key on the
           payload's `user_id`, which `reject_ownership_metadata` still refuses
           outright — but a row whose metadata CLAIMS a shared scope it does not have
           is precisely the plausible-looking lie the opt-in's 400 exists to prevent,
           and it is the second door on the forgery in (2).

        2. A pinned caller may not add or change a grouping id on a SHARED row, for
           the reason spelled out in `reject_shared_grouping_ids`: `run_id` is what
           decides whose memory a fact is presented as belonging to. Without this,
           the `POST`-side refusal is bypassed by publishing an empty shared row and
           then PUTting the ids into it — one extra request.

        Private rows are untouched by (2). The fact layer's `agent_id`/`run_id` are
        ordinary grouping there, and `build_fact_layer.py` rewrites them.
        """
        if not self.is_pinned or not metadata:
            return
        stored_metadata = stored_metadata or {}
        changed = sorted(
            key for key in DERIVED_METADATA_KEYS if key in metadata and metadata[key] != stored_metadata.get(key)
        )
        if changed:
            raise HTTPException(
                status_code=400,
                detail=(
                    "metadata may echo "
                    + ", ".join(DERIVED_METADATA_KEYS)
                    + " back unchanged but may not alter them. These do not match the stored memory: "
                    + ", ".join(changed)
                    + "."
                ),
            )
        if stored_user_id not in SHARED_SCOPES:
            return
        regrouped = sorted(key for key in GROUPING_METADATA_KEYS if key in metadata)
        if regrouped:
            raise HTTPException(
                status_code=400,
                detail=(
                    "metadata may not set "
                    + ", ".join(regrouped)
                    + " on a shared memory. Grouping ids there are server-controlled, because they "
                    "decide whose memory the row is presented as belonging to."
                ),
            )

    def reject_scope_escaping_filters(self, filters: dict[str, Any] | None) -> None:
        """Refuse a pinned caller's `/search` filters that could outrun the `user_id` pin.

        THE HOLE THIS CLOSES, reproduced by executing mem0's own code rather than by
        reading it:

            caller filters: {"user_id": "anything", "AND": [{"user_id": "<victim>"}]}
            server pins:    {**caller_filters, "user_id": "<caller>"}
            -> effective user_id == "<victim>", i.e. a full read of another person's
               private memories through a route that looks pinned.

        Three library behaviours line up to do it. `{**d, k: v}` keeps the ORIGINAL
        position of a key already present in `d`, so a caller who writes `user_id`
        before `AND` in their JSON controls the iteration order of the merged dict.
        `Memory._process_metadata_filters` then FLATTENS `AND` to the top level
        through `merge_filters`, whose scalar branch is a plain `target[key] = value`
        — an overwrite, not a conjunction. Finally `search()` does
        `effective_filters.update(processed_filters)`, copying the attacker's
        `user_id` back over the pin. Measured both ways: with `user_id` first it
        escapes, without it, it does not — so no fix that depends on key order is a
        fix.

        Which is why this is an ALLOWLIST over VALUES and not a list of the operator
        keys that happen to be dangerous today. `OR`/`NOT` nest into `$or`/`$not`,
        which pgvector joins with AND, so those two do not escape — currently.
        `requirements.txt` pins `mem0ai>=0.1.48`, a FLOOR: the filter compiler is not
        a fixed target, and a denylist over it stops covering the moment a release
        adds a fourth logical key, silently, with nothing able to notice. One rule —
        every value must be a plain scalar — rejects every dict, every list and every
        logical operator, present and future. It is the same argument `shared_write`
        makes for testing scope membership against an enumerated set rather than a
        prefix.

        It costs the real callers nothing: every filter this deployment sends is a
        single scalar — `{"agent_id": "facts"}` from the console and from
        `.claude/hooks/mem0/`, `{"run_id": <parent id>}` from the console's fact
        panel. It DOES newly 400 an operator-shaped `mem0ctl.py search --filters`
        payload, which accepts arbitrary JSON through a pinned credential; that is
        the intended behaviour change and not an oversight.

        `user_id` is exempt because the route overwrites it with the pin one line
        later, and that overwrite is `pin`'s documented, tested behaviour. There is
        no gap between what is checked here and what is sent: the key this skips is
        the one key the server itself supplies.

        `None` is refused along with the containers rather than waved through as a
        harmless scalar. `{"k": null}` compiles to `payload->>'k' = 'None'` — a
        filter matching no row, returning 200 with an empty list, which is the
        silent-wrong-answer shape the rest of this module exists to keep out.

        `/search` is the whole surface. `GET /memories` accepts no `filters` at all:
        it builds its own dict from three scalar query params through
        `listing.listing_filters`, and FastAPI binds an `Optional[str]` query param
        to the last value, never to a list. Unpinned callers are not held to any of
        it — they can already name any `user_id` outright, so there is no pin to
        escape.
        """
        if not self.is_pinned or not filters:
            return
        offending = sorted(
            key for key, value in filters.items() if key != "user_id" and not isinstance(value, FILTER_SCALAR_TYPES)
        )
        if offending:
            raise HTTPException(
                status_code=400,
                detail=(
                    "filters may only carry plain string, number or boolean values for an SSO caller. "
                    "Nested filter operators are not available on this path. Offending keys: "
                    + ", ".join(offending)
                    + "."
                ),
            )


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
