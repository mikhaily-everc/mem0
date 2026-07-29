"""Auth0 RS256 access-token verification.

The mem0 server's own auth (`auth.py`) issues HS256 tokens signed with a shared
`JWT_SECRET` and accepts long-lived `X-API-Key` values. Neither is tied to the
company IdP, so disabling someone in JumpCloud does not revoke their mem0 access.

This module adds the third credential path: an Auth0 access token minted for this
server's own resource-server audience, carrying the caller's JumpCloud identity as
`sub`. Because the token is short-lived and can only be re-minted from a refresh
token that Auth0 revokes when the user is blocked, revocation finally propagates.

Deliberately separate from `auth.py`: the HS256 path there uses a hardcoded
`JWT_ALGORITHM = "HS256"` module constant shared by token *creation* and
*verification*. Widening it to accept RS256 would also widen what
`create_access_token` can be coerced into emitting, and would make an
algorithm-confusion bug a one-line mistake. Two modules, two key types, no
overlap.

Mirrors `tools/mcp/mysql/server/src/auth/index.ts:85-121` in the ai-utils repo —
same issuer/audience/algorithm pinning, same rejection of a `sub`-less token, same
roles-claim → admin mapping.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass

import httpx
from jose import jwt
from jose.exceptions import JWTError

logger = logging.getLogger(__name__)

AUTH0_DOMAIN = os.environ.get("AUTH0_DOMAIN", "")
MEM0_AUDIENCE = os.environ.get("MEM0_AUDIENCE", "")
#: Namespaced custom claim carrying the user's roles. Same claim the mysql MCP
#: server reads, so one Auth0 role assignment governs both.
AUTH0_ROLES_CLAIM = os.environ.get("AUTH0_ROLES_CLAIM", "https://everc.io/roles")
#: Role that maps onto mem0's `admin` role.
MEM0_ADMIN_ROLE = os.environ.get("MEM0_ADMIN_ROLE", "mcp-admin")

#: Only RS256. Never widen this — and in particular never let it come from the
#: token itself, which is the classic algorithm-confusion hole.
ALLOWED_ALGORITHMS = ["RS256"]

_JWKS_TTL_SECONDS = 10 * 60
_JWKS_TIMEOUT_SECONDS = 5.0


def auth0_enabled() -> bool:
    """Whether this deployment is configured to accept Auth0 tokens at all."""
    return bool(AUTH0_DOMAIN and MEM0_AUDIENCE)


def issuer() -> str:
    """The expected `iss`. Auth0 always emits the trailing slash."""
    return f"https://{AUTH0_DOMAIN}/"


@dataclass(frozen=True)
class Auth0Claims:
    """The subset of a verified Auth0 token this server acts on."""

    sub: str
    email: str
    roles: tuple[str, ...]

    @property
    def is_admin(self) -> bool:
        return MEM0_ADMIN_ROLE in self.roles

    @property
    def memory_scope(self) -> str:
        """The `user_id` every memory of this caller is stored under.

        Prefixed so an Auth0-scoped memory can never collide with a legacy
        caller-supplied `user_id`, and so the origin of a scope is readable
        straight out of the vector store.
        """
        return f"auth0:{self.sub}"


class _JwksCache:
    """Fetch-and-cache the tenant JWKS, refetching on an unknown `kid`.

    Key rotation publishes a new `kid` before retiring the old one, so a cache
    miss is the rotation signal. `_last_miss_fetch` bounds how often an unknown
    kid can force an outbound fetch, so a stream of bogus kids cannot be turned
    into a request amplifier against Auth0.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._keys: dict[str, dict] = {}
        self._fetched_at = 0.0
        self._last_miss_fetch = 0.0

    def _fetch(self) -> dict[str, dict]:
        url = f"https://{AUTH0_DOMAIN}/.well-known/jwks.json"
        response = httpx.get(url, timeout=_JWKS_TIMEOUT_SECONDS)
        response.raise_for_status()
        keys = {k["kid"]: k for k in response.json().get("keys", []) if k.get("kid")}
        if not keys:
            raise ValueError("JWKS contained no usable keys")
        return keys

    def get(self, kid: str) -> dict | None:
        now = time.monotonic()
        with self._lock:
            fresh = now - self._fetched_at < _JWKS_TTL_SECONDS
            if kid in self._keys and fresh:
                return self._keys[kid]
            # Unknown kid, or the cache aged out: refetch, but never more than
            # once per TTL window for a kid we have never seen.
            if not fresh or (now - self._last_miss_fetch) >= _JWKS_TTL_SECONDS:
                try:
                    self._keys = self._fetch()
                    self._fetched_at = now
                except Exception:
                    # exc_info so the cause is recoverable: degrading to the cached
                    # keys is correct, but a silent degrade makes "the tenant is
                    # unreachable" look identical to "that token is bad" from the
                    # outside, which is a very bad place to debug from.
                    logger.warning("auth0: JWKS fetch failed", exc_info=True)
                    # Fall through to whatever is still cached.
                finally:
                    if kid not in self._keys:
                        self._last_miss_fetch = now
            return self._keys.get(kid)

    def clear(self) -> None:
        with self._lock:
            self._keys = {}
            self._fetched_at = 0.0
            self._last_miss_fetch = 0.0


_jwks = _JwksCache()


def looks_like_auth0_token(token: str) -> bool:
    """Cheap, unverified triage: is this an RS256 JWT with a `kid`?

    Used only to decide WHICH verifier to route a bearer token to. The mem0-native
    path is HS256 and carries no `kid`, so the two are unambiguous. This function
    trusts nothing — the answer is re-established by real verification.
    """
    if not auth0_enabled():
        return False
    try:
        header = jwt.get_unverified_header(token)
    except JWTError:
        return False
    return header.get("alg") in ALLOWED_ALGORITHMS and bool(header.get("kid"))


class Auth0VerificationError(Exception):
    """The token is not a valid Auth0 token for this server."""


def verify_auth0_token(token: str) -> Auth0Claims:
    """Verify an Auth0 access token, or raise `Auth0VerificationError`.

    Pins the signing algorithm, the issuer and the audience, and requires a `sub`.
    The raised message is deliberately coarse — the caller turns it into a 401
    without echoing it, so nothing about key state or claim contents leaks.
    """
    if not auth0_enabled():
        raise Auth0VerificationError("Auth0 is not configured")

    try:
        header = jwt.get_unverified_header(token)
    except JWTError as exc:
        raise Auth0VerificationError("malformed token header") from exc

    alg = header.get("alg")
    if alg not in ALLOWED_ALGORITHMS:
        # Covers `alg: none` and every HMAC variant, which would otherwise let a
        # caller sign with a value derived from the public key.
        raise Auth0VerificationError(f"unsupported algorithm: {alg!r}")

    kid = header.get("kid")
    if not kid:
        raise Auth0VerificationError("token has no kid")

    key = _jwks.get(kid)
    if key is None:
        raise Auth0VerificationError("unknown signing key")

    try:
        claims = jwt.decode(
            token,
            key,
            algorithms=ALLOWED_ALGORITHMS,
            audience=MEM0_AUDIENCE,
            issuer=issuer(),
        )
    except JWTError as exc:
        # Signature, `exp`, `aud` and `iss` failures all land here.
        raise Auth0VerificationError("token verification failed") from exc

    sub = claims.get("sub")
    if not isinstance(sub, str) or not sub:
        # A token with no `sub` has no identity to scope memories to. Rejecting is
        # the only safe option: the alternative is scoping every such caller to the
        # literal string "None" and silently pooling their memories together.
        raise Auth0VerificationError("token has no sub")

    raw_roles = claims.get(AUTH0_ROLES_CLAIM)
    roles = tuple(r for r in raw_roles if isinstance(r, str)) if isinstance(raw_roles, list) else ()

    email = claims.get("email")
    return Auth0Claims(
        sub=sub,
        email=email if isinstance(email, str) else "",
        roles=roles,
    )
