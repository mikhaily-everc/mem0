"""Tests for Auth0 token verification and memory-scope pinning.

These cover the rejection cases that are awkward to produce against a live tenant
(`alg: none`, a token signed by the wrong key, a wrong audience or issuer) by
signing tokens locally with a throwaway RSA key and seeding the JWKS cache with
its public half. No network, no database.

stdlib `unittest` rather than pytest, deliberately. rules_py's `uv.project`
derives the wheel hub's dep-group config settings from the resolved graph, and
adding a PEP 735 `[dependency-groups]` block to pyproject.toml made the hub emit
ONLY that group — the `mem0_server` group disappeared and every
`@pypi_mem0//<pkg>` label failed at analysis. `unittest` needs nothing added to
the lock, so the shipped image's dependency graph stays untouched.
"""

from __future__ import annotations

import json
import time
import unittest
from types import SimpleNamespace

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException
from jose import jwt
from jose.utils import base64url_encode
from starlette.requests import Request

import auth as auth_module
import auth0
import rate_limit
from auth import MemoryScope
from routers import api_keys as api_keys_router
from routers import auth as auth_router

DOMAIN = "test-tenant.us.auth0.com"
AUDIENCE = "https://mem0.test.invalid"
ISSUER = f"https://{DOMAIN}/"
ROLES_CLAIM = "https://everc.io/roles"
KID = "test-key-1"
SUB = "samlp|SAML-JumpCloud-Dev|someone@everc.com"


def _int_to_b64(value: int) -> str:
    return base64url_encode(value.to_bytes((value.bit_length() + 7) // 8, "big")).decode()


def _new_key() -> tuple[str, dict]:
    """A throwaway RSA key as (PEM private key, public JWK)."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    numbers = key.public_key().public_numbers()
    return pem, {
        "kty": "RSA",
        "kid": KID,
        "use": "sig",
        "alg": "RS256",
        "n": _int_to_b64(numbers.n),
        "e": _int_to_b64(numbers.e),
    }


class Auth0TokenTest(unittest.TestCase):
    """Verification of a real RS256 token against a seeded JWKS."""

    @classmethod
    def setUpClass(cls) -> None:
        # 2048-bit keygen is slow enough to be worth doing once for the class.
        cls.pem, cls.jwk = _new_key()

    def setUp(self) -> None:
        self._restore = {
            "AUTH0_DOMAIN": auth0.AUTH0_DOMAIN,
            "MEM0_AUDIENCE": auth0.MEM0_AUDIENCE,
            "AUTH0_ROLES_CLAIM": auth0.AUTH0_ROLES_CLAIM,
            "MEM0_ADMIN_ROLE": auth0.MEM0_ADMIN_ROLE,
        }
        auth0.AUTH0_DOMAIN = DOMAIN
        auth0.MEM0_AUDIENCE = AUDIENCE
        auth0.AUTH0_ROLES_CLAIM = ROLES_CLAIM
        auth0.MEM0_ADMIN_ROLE = "mcp-admin"
        # Seed the cache and stub the fetch to serve the same key offline.
        #
        # Counting rather than failing on fetch: `_JwksCache.get` swallows every
        # exception from `_fetch` (so a JWKS outage degrades to the cached keys
        # instead of 500ing), which means an `assert`/`fail` raised in here would
        # be silently absorbed and the guard would never fire. A counter checked
        # from the test body cannot be swallowed.
        self.fetch_count = 0
        auth0._jwks._keys = {KID: self.jwk}
        auth0._jwks._fetched_at = time.monotonic()
        auth0._jwks._last_miss_fetch = 0.0
        # Rebinding an instance attribute over a method is how the fetch gets counted. ty
        # is right that a bound method is not assignable to the declared function type;
        # that mismatch is the technique, not a mistake.
        auth0._jwks._fetch = self._counting_fetch  # ty: ignore[invalid-assignment]

    def _counting_fetch(self) -> dict[str, dict]:
        self.fetch_count += 1
        return {KID: self.jwk}

    def tearDown(self) -> None:
        for name, value in self._restore.items():
            setattr(auth0, name, value)
        auth0._jwks.clear()

    # -- helpers ----------------------------------------------------------- #

    def _claims(self, **overrides) -> dict:
        claims = {
            "sub": SUB,
            "aud": AUDIENCE,
            "iss": ISSUER,
            "exp": int(time.time()) + 300,
            "iat": int(time.time()),
            "email": "someone@everc.com",
            ROLES_CLAIM: [],
        }
        claims.update(overrides)
        return claims

    def _sign(self, claims: dict, *, kid: str = KID, pem: str | None = None) -> str:
        return jwt.encode(claims, pem or self.pem, algorithm="RS256", headers={"kid": kid})

    def _assert_rejected(self, token: str, because: str) -> None:
        with self.assertRaises(auth0.Auth0VerificationError, msg=because):
            auth0.verify_auth0_token(token)

    # -- acceptance -------------------------------------------------------- #

    def test_accepts_a_valid_token(self) -> None:
        result = auth0.verify_auth0_token(self._sign(self._claims()))
        self.assertEqual(result.sub, SUB)
        self.assertEqual(result.email, "someone@everc.com")
        self.assertFalse(result.is_admin)
        # A warm cache must not be re-fetched on every request.
        self.assertEqual(self.fetch_count, 0)

    def test_memory_scope_is_derived_from_sub(self) -> None:
        result = auth0.verify_auth0_token(self._sign(self._claims()))
        self.assertEqual(result.memory_scope, f"auth0:{SUB}")

    def test_admin_role_is_read_from_the_namespaced_claim(self) -> None:
        token = self._sign(self._claims(**{ROLES_CLAIM: ["other", "mcp-admin"]}))
        self.assertTrue(auth0.verify_auth0_token(token).is_admin)

    def test_a_non_list_roles_claim_does_not_crash(self) -> None:
        result = auth0.verify_auth0_token(self._sign(self._claims(**{ROLES_CLAIM: "mcp-admin"})))
        self.assertEqual(result.roles, ())
        self.assertFalse(result.is_admin)

    def test_missing_email_yields_empty_string(self) -> None:
        claims = self._claims()
        del claims["email"]
        self.assertEqual(auth0.verify_auth0_token(self._sign(claims)).email, "")

    # -- rejection: each of these was a way in before the change ----------- #

    def test_rejects_wrong_audience(self) -> None:
        self._assert_rejected(
            self._sign(self._claims(aud="https://someone-elses-api.invalid")),
            "a token minted for another API must not be accepted here",
        )

    def test_rejects_wrong_issuer(self) -> None:
        self._assert_rejected(
            self._sign(self._claims(iss="https://evil.us.auth0.com/")),
            "another tenant must not be able to mint tokens for us",
        )

    def test_rejects_expired_token(self) -> None:
        self._assert_rejected(
            self._sign(self._claims(exp=int(time.time()) - 60)),
            "expiry is what makes Auth0 revocation propagate",
        )

    def test_rejects_alg_none(self) -> None:
        header = base64url_encode(json.dumps({"alg": "none", "typ": "JWT", "kid": KID}).encode()).decode()
        payload = base64url_encode(json.dumps(self._claims()).encode()).decode()
        self._assert_rejected(f"{header}.{payload}.", "an unsigned token is not a credential")

    def test_rejects_hs256_signed_with_the_public_key(self) -> None:
        """Algorithm confusion: sign with the public modulus as an HMAC secret."""
        token = jwt.encode(self._claims(), self.jwk["n"], algorithm="HS256", headers={"kid": KID})
        self._assert_rejected(token, "the algorithm must come from our allowlist, not the token")

    def test_rejects_unknown_kid(self) -> None:
        self._assert_rejected(
            self._sign(self._claims(), kid="not-a-key-we-know"),
            "an unknown kid must not fall back to some other key",
        )
        # An unknown kid is the key-rotation signal, so ONE refetch is correct —
        # but it must not then accept the token under a different key.
        self.assertEqual(self.fetch_count, 1)

    def test_repeated_unknown_kids_do_not_amplify_fetches(self) -> None:
        """A stream of bogus kids must not become a request amplifier at Auth0."""
        for i in range(5):
            self._assert_rejected(
                self._sign(self._claims(), kid=f"bogus-{i}"),
                "a bogus kid must be rejected",
            )
        self.assertEqual(self.fetch_count, 1)

    def test_rejects_signature_from_another_key(self) -> None:
        other_pem, _ = _new_key()
        self._assert_rejected(
            self._sign(self._claims(), pem=other_pem),
            "the kid must actually match the verifying key",
        )

    def test_rejects_token_with_no_sub(self) -> None:
        claims = self._claims()
        del claims["sub"]
        self._assert_rejected(self._sign(claims), "no sub means no scope to pin memories to")

    def test_rejects_token_with_empty_sub(self) -> None:
        self._assert_rejected(self._sign(self._claims(sub="")), "an empty sub is not an identity")

    def test_rejects_garbage(self) -> None:
        self._assert_rejected("not-a-jwt-at-all", "malformed input must not raise past the caller")

    # -- routing triage ---------------------------------------------------- #

    def test_looks_like_auth0_token_accepts_rs256_with_kid(self) -> None:
        self.assertTrue(auth0.looks_like_auth0_token(self._sign(self._claims())))

    def test_looks_like_auth0_token_rejects_the_native_hs256_token(self) -> None:
        """The mem0-native path is HS256 with no kid, so the two are unambiguous."""
        native = jwt.encode({"sub": "x", "type": "access"}, "shared-secret", algorithm="HS256")
        self.assertFalse(auth0.looks_like_auth0_token(native))

    def test_looks_like_auth0_token_is_false_when_unconfigured(self) -> None:
        token = self._sign(self._claims())
        auth0.MEM0_AUDIENCE = ""
        self.assertFalse(auth0.looks_like_auth0_token(token))

    def test_verify_refuses_when_unconfigured(self) -> None:
        token = self._sign(self._claims())
        auth0.AUTH0_DOMAIN = ""
        self._assert_rejected(token, "an unconfigured deployment must accept no Auth0 token")


class RegistrationClosedUnderSsoTest(unittest.TestCase):
    """`POST /auth/register` is unauthenticated and mints an `admin`.

    Under SSO the handler refuses outright, but that guard lives in the function
    body — it is invisible in `/openapi.json`, which lists the route with no
    `security` entry. An audit reading the spec alone concludes the route is open,
    and an empty-body probe returns 422 (FastAPI validating the body before the
    handler runs), which reads as further confirmation. That false alarm has now
    been raised twice.

    So pin the guard where it actually has to be: FIRST in the body, ahead of the
    password-length check and any database access. Every argument below is a
    sentinel that would raise `AttributeError` if it were touched, so a future
    refactor that moves the guard down fails here rather than in production.
    """

    def setUp(self) -> None:
        self._restore = (auth0.AUTH0_DOMAIN, auth0.MEM0_AUDIENCE)
        # Rate limiting is orthogonal, and slowapi's decorator reaches for
        # `request.app` when enabled — which a bare Request does not carry.
        self._limiter_enabled = rate_limit.limiter.enabled
        rate_limit.limiter.enabled = False

    def tearDown(self) -> None:
        auth0.AUTH0_DOMAIN, auth0.MEM0_AUDIENCE = self._restore
        rate_limit.limiter.enabled = self._limiter_enabled

    @staticmethod
    def _request() -> Request:
        return Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/auth/register",
                "headers": [],
                "client": ("203.0.113.1", 1234),
            }
        )

    def test_register_is_refused_while_auth0_is_configured(self) -> None:
        auth0.AUTH0_DOMAIN = DOMAIN
        auth0.MEM0_AUDIENCE = AUDIENCE
        self.assertTrue(auth0.auth0_enabled(), "precondition: this deployment accepts Auth0 tokens")

        with self.assertRaises(HTTPException) as caught:
            auth_router.register(request=self._request(), body=object(), db=object())

        self.assertEqual(caught.exception.status_code, 403)

    def test_the_guard_runs_before_the_body_is_read(self) -> None:
        """A caller-supplied body must never reach validation on an SSO deployment."""
        auth0.AUTH0_DOMAIN = DOMAIN
        auth0.MEM0_AUDIENCE = AUDIENCE

        class Exploding:
            def __getattr__(self, name: str):
                raise AssertionError(f"register() read body.{name} before refusing")

        with self.assertRaises(HTTPException) as caught:
            auth_router.register(request=self._request(), body=Exploding(), db=Exploding())

        self.assertEqual(caught.exception.status_code, 403)


class MemoryScopeTest(unittest.TestCase):
    """The pinning contract every memory route relies on."""

    PINNED = MemoryScope(pinned_user_id="auth0:alice", is_admin=False)
    UNPINNED = MemoryScope(pinned_user_id=None, is_admin=False)

    def test_pin_overwrites_a_caller_supplied_user_id(self) -> None:
        self.assertEqual(self.PINNED.pin("bob"), "auth0:alice")

    def test_pin_fills_in_a_missing_user_id(self) -> None:
        self.assertEqual(self.PINNED.pin(None), "auth0:alice")

    def test_pin_is_a_passthrough_when_unpinned(self) -> None:
        self.assertEqual(self.UNPINNED.pin("bob"), "bob")
        self.assertIsNone(self.UNPINNED.pin(None))

    def test_owns_only_the_callers_own_scope(self) -> None:
        self.assertTrue(self.PINNED.owns("auth0:alice"))
        self.assertFalse(self.PINNED.owns("auth0:bob"))

    def test_owns_rejects_unscoped_rows_for_a_pinned_caller(self) -> None:
        """Memories written before pinning become unreachable, not shared."""
        self.assertFalse(self.PINNED.owns(None))

    def test_unpinned_callers_retain_full_access(self) -> None:
        self.assertTrue(self.UNPINNED.owns("anything"))
        self.assertTrue(self.UNPINNED.owns(None))

    def test_admin_does_not_unpin_the_scope(self) -> None:
        """An SSO admin may list broadly, but their reads and writes stay in scope."""
        admin = MemoryScope(pinned_user_id="auth0:alice", is_admin=True)
        self.assertEqual(admin.pin("bob"), "auth0:alice")
        self.assertFalse(admin.owns("auth0:bob"))

    def test_is_pinned_reflects_the_presence_of_a_scope(self) -> None:
        self.assertTrue(self.PINNED.is_pinned)
        self.assertFalse(self.UNPINNED.is_pinned)


class AdoptionClosesThePasswordDoorTest(unittest.TestCase):
    """Adopting a pre-existing account into SSO must clear its password hash.

    `verify_password` short-circuits only on an EMPTY hash. A row provisioned by
    the insert path gets `password_hash=""` and is SSO-only; an ADOPTED row keeps
    whatever hash it had. Without the clear, the person keeps a working password
    and reaches the dashboard without ever touching Auth0 — the revocation hole
    phases D-H exist to close, left open on the one account most likely to have a
    password already: the POC admin.

    Asserted through `verify_password` rather than by eyeballing the field, so the
    test states the property that matters (this credential no longer authenticates)
    rather than one particular way of achieving it.
    """

    def test_an_adopted_row_can_no_longer_be_reached_by_password(self) -> None:
        hashed = auth_module.hash_password("the-old-password")
        self.assertTrue(
            auth_module.verify_password("the-old-password", hashed),
            "precondition: the hash must be valid before adoption, or the test proves nothing",
        )

        adopted = SimpleNamespace(auth0_sub=None, email="poc-admin@everc.com", password_hash=hashed)
        # A real `User` would need a Session and a live table to prove a two-attribute
        # mutation. adopt_row_into_sso touches only `auth0_sub` and `password_hash`.
        auth_module.adopt_row_into_sso(adopted, "auth0|123")  # ty: ignore[invalid-argument-type]

        self.assertFalse(
            auth_module.verify_password("the-old-password", adopted.password_hash),
            "an adopted account must not remain reachable with its old password",
        )

    def test_adoption_still_links_the_sso_subject(self) -> None:
        """Closing the door must not cost the linkage the adoption exists to create."""
        adopted = SimpleNamespace(auth0_sub=None, email="poc-admin@everc.com", password_hash="x")
        # Same duck-typed row as the test above.
        auth_module.adopt_row_into_sso(adopted, "auth0|123")  # ty: ignore[invalid-argument-type]
        self.assertEqual(adopted.auth0_sub, "auth0|123")


class SsoCallersCannotMintApiKeysTest(unittest.TestCase):
    """`POST /api-keys` must refuse an Auth0-authenticated caller.

    `X-API-Key` survives deliberately for agents and CI. The hazard is the other
    direction: a human with an SSO session minting a key that outlives their own
    revocation, since the API-key path consults neither Auth0 nor JumpCloud.

    The guard keys on `request.state.auth_type`, NOT on `user.auth0_sub`, and that
    distinction is load-bearing: for an `ADMIN_API_KEY` caller `require_auth`
    substitutes the oldest user row, which is an adopted SSO account — so keying on
    the resolved user would break the very break-glass path used to issue agent
    keys. Both directions are pinned below.

    Every argument other than the request is a sentinel that would raise
    `AttributeError` on touch, so a future refactor that moves the guard below the
    key generation fails here rather than in production.
    """

    @staticmethod
    def _request(auth_type: str):
        return SimpleNamespace(state=SimpleNamespace(auth_type=auth_type))

    def test_an_sso_session_is_refused(self) -> None:
        with self.assertRaises(HTTPException) as caught:
            api_keys_router.create_key(self._request("auth0"), object(), user=object(), db=object())
        self.assertEqual(caught.exception.status_code, 403)

    def test_the_admin_break_glass_path_is_not_refused(self) -> None:
        """Must fail PAST the guard, not at it — an operator has to be able to issue agent keys."""
        with self.assertRaises(Exception) as caught:
            api_keys_router.create_key(self._request("admin_api_key"), object(), user=object(), db=object())
        self.assertNotIsInstance(
            caught.exception, HTTPException, "the break-glass path must reach key generation, not the 403"
        )

    def test_an_api_key_caller_is_not_refused(self) -> None:
        """CI renewing its own credential is the Phase H case and must keep working."""
        with self.assertRaises(Exception) as caught:
            api_keys_router.create_key(self._request("api_key"), object(), user=object(), db=object())
        self.assertNotIsInstance(caught.exception, HTTPException)


if __name__ == "__main__":
    unittest.main(verbosity=2)
