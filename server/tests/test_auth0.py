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

import mem0.memory.main as mem0_main
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


class SharedScopeIsReadableByEveryoneTest(unittest.TestCase):
    """The read half of the shared scope: a committed memory URL must resolve.

    WHY THIS EXISTS. Every memory is pinned to its author's Auth0 sub, and
    `_fetch_owned_memory` turns a scope miss into a 404 so the routes are not an id
    oracle. Correct for personal memories, and fatal for the other kind: a
    repo-wide code note whose URL is committed to version control 404'd for every
    colleague who followed the link, including the ones the note was written for.
    The author saw it working, which is why it survived.

    A reserved `user_id` — not an unpinned admin, not a per-row ACL — is the whole
    mechanism, so these assert on the exact set membership.
    """

    SHARED = "repo:ai-utils"
    ALICE = MemoryScope(pinned_user_id="auth0:alice", is_admin=False)
    BOB = MemoryScope(pinned_user_id="auth0:bob", is_admin=False)

    def test_a_second_identity_can_read_a_shared_row(self) -> None:
        """The property the whole change exists for."""
        self.assertTrue(self.BOB.owns(self.SHARED))

    def test_the_author_still_reads_their_own_shared_row(self) -> None:
        self.assertTrue(self.ALICE.owns(self.SHARED))

    def test_sharing_does_not_open_anyone_elses_personal_scope(self) -> None:
        """The widening is one enumerated value, not "any scope that is not mine"."""
        self.assertFalse(self.BOB.owns("auth0:alice"))

    def test_a_lookalike_scope_is_not_shared(self) -> None:
        """Membership is exact, never a `repo:` prefix test. A prefix rule would be
        the entire check on this path and would accept anything shaped like it."""
        for lookalike in ("repo:ai-utils-2", "repo:", "repo:ai_utils", "ai-utils"):
            with self.subTest(user_id=lookalike):
                self.assertFalse(self.BOB.owns(lookalike))

    def test_unscoped_rows_are_still_unreachable(self) -> None:
        """Pre-pinning rows carry `user_id = None`. Sharing must not sweep them in —
        they were written before anyone chose to publish anything."""
        self.assertFalse(self.BOB.owns(None))


class SharedRowsAreWriteAttributedTest(unittest.TestCase):
    """The write half: read shared, write attributed. Never a blanket unpin.

    Widening `owns` alone would have been a strictly worse hole than the one being
    fixed — every authenticated caller would have gained edit and delete rights
    over every repo-wide note. `may_modify` is the second predicate that stops it,
    and every mutating route must call it instead of `owns`.
    """

    SHARED = "repo:ai-utils"
    ALICE = MemoryScope(pinned_user_id="auth0:alice", is_admin=False)
    BOB = MemoryScope(pinned_user_id="auth0:bob", is_admin=False)
    ADMIN = MemoryScope(pinned_user_id="auth0:carol", is_admin=True)
    UNPINNED = MemoryScope(pinned_user_id=None, is_admin=False)

    def test_a_second_identity_cannot_overwrite_a_row_it_did_not_author(self) -> None:
        """Reading it is the point; rewriting it is not."""
        self.assertTrue(self.BOB.owns(self.SHARED), "precondition: bob must be able to READ it")
        self.assertFalse(self.BOB.may_modify(self.SHARED, "auth0:alice"))

    def test_the_author_may_still_change_their_own_shared_row(self) -> None:
        self.assertTrue(self.ALICE.may_modify(self.SHARED, "auth0:alice"))

    def test_an_sso_admin_may_moderate_a_shared_row(self) -> None:
        """Deliberately different from `delete_all_memories`, where admin-ness does
        NOT let you destroy a colleague's memories. Consistent on the principle: a
        shared row is repo-wide by construction and sits in nobody's private scope."""
        self.assertTrue(self.ADMIN.may_modify(self.SHARED, "auth0:alice"))

    def test_an_admin_still_cannot_touch_a_colleagues_personal_row(self) -> None:
        """The admin exception is scoped to shared rows and nothing else. If this
        goes green-to-red the admin branch has escaped past `owns`."""
        self.assertFalse(self.ADMIN.may_modify("auth0:alice", None))
        self.assertFalse(self.ADMIN.may_modify("auth0:alice", "auth0:carol"))

    def test_a_shared_row_with_no_author_fails_closed(self) -> None:
        """Rows predating attribution, or ones whose `author_sub` was stripped in the
        store, are admin-only — not everyone-writable. Reading the author rather
        than merely checking one exists is exactly what buys this."""
        self.assertFalse(self.BOB.may_modify(self.SHARED, None))
        self.assertTrue(self.ADMIN.may_modify(self.SHARED, None))

    def test_a_forged_author_on_a_personal_row_buys_nothing(self) -> None:
        """`author_sub` is consulted only after `owns` has already passed, so it can
        never be the thing that grants access to someone else's private scope."""
        self.assertFalse(self.BOB.may_modify("auth0:alice", "auth0:bob"))

    def test_a_pinned_caller_still_owns_their_own_personal_rows(self) -> None:
        """The narrowing applies to shared rows only; nothing about a private row
        changed, and it has no `author_sub` to check."""
        self.assertTrue(self.ALICE.may_modify("auth0:alice", None))

    def test_legacy_credentials_keep_their_pre_existing_freedom(self) -> None:
        """X-API-Key / ADMIN_API_KEY / AUTH_DISABLED are unpinned by design, and
        operational cleanup of shared rows runs through the machine credential."""
        self.assertTrue(self.UNPINNED.may_modify(self.SHARED, "auth0:alice"))
        self.assertTrue(self.UNPINNED.may_modify("auth0:alice", None))


class SharedWriteOptInTest(unittest.TestCase):
    """`metadata.scope` is how a write opts into a shared scope, and the only way.

    It travels in METADATA rather than as a `user_id` argument because
    `tools/mcp/mem0/scripts/assemble-spec.ts` strips `user_id`/`agent_id`/`run_id`
    from every generated MCP tool schema AND from request-body properties — an MCP
    caller cannot name a `user_id` at all. Metadata passes through verbatim, so the
    OpenAPI spec and the gateway's `include_tools` allowlist stay untouched.

    Which makes this the one place a caller-supplied string decides where a row
    lands. These pin the checks on it.
    """

    SHARED = "repo:ai-utils"
    ALICE = MemoryScope(pinned_user_id="auth0:alice", is_admin=False)
    UNPINNED = MemoryScope(pinned_user_id=None, is_admin=False)

    def _opt_in(self, metadata: dict[str, object]) -> tuple[str, dict[str, object]]:
        """`shared_write` returns None for an ordinary write, so every test that
        expects an opt-in has to prove it got one before unpacking. Doing that here
        rather than in each test keeps the None case a real assertion instead of a
        TypeError that happens to fail."""
        result = self.ALICE.shared_write(metadata)
        self.assertIsNotNone(result, "expected the opt-in to be honoured")
        assert result is not None  # narrows for the type checker
        return result

    def test_an_ordinary_write_is_left_alone(self) -> None:
        self.assertIsNone(self.ALICE.shared_write(None))
        self.assertIsNone(self.ALICE.shared_write({"name": "a-note"}))

    def test_opting_in_redirects_the_row_and_stamps_the_author(self) -> None:
        user_id, metadata = self._opt_in({"scope": self.SHARED, "name": "a-note"})
        self.assertEqual(user_id, self.SHARED)
        self.assertEqual(metadata["author_sub"], "auth0:alice")
        self.assertEqual(metadata["name"], "a-note", "the caller's own metadata must survive")

    def test_the_body_can_never_supply_its_own_author(self) -> None:
        """`author_sub` gates every later write to the row. Merging it from the
        request body instead of overwriting would let a caller publish a note
        already attributed to somebody else, or one only they could ever edit."""
        _, metadata = self._opt_in({"scope": self.SHARED, "author_sub": "auth0:bob"})
        self.assertEqual(metadata["author_sub"], "auth0:alice")

    def test_the_request_body_is_not_mutated(self) -> None:
        """Same reason `add_memory` assigns into `params` and not into the model:
        the body as sent is what the request log records."""
        body = {"scope": self.SHARED, "name": "a-note"}
        self.ALICE.shared_write(body)
        self.assertEqual(body, {"scope": self.SHARED, "name": "a-note"})

    def test_an_unknown_scope_is_refused_not_quietly_stored_privately(self) -> None:
        """A typo'd `repo:aiutils` that returned 200 and produced a PRIVATE row is
        the exact failure this change exists to fix — discovered weeks later, by a
        colleague following a link."""
        for typo in ("repo:aiutils", "repo:ai-utils ", "ai-utils", ""):
            with self.subTest(scope=typo):
                with self.assertRaises(HTTPException) as caught:
                    self.ALICE.shared_write({"scope": typo})
                self.assertEqual(caught.exception.status_code, 400)

    def test_a_scope_naming_another_caller_is_refused(self) -> None:
        """The exact-set test is what stops the opt-in being a write-anywhere
        primitive: nothing here may ever land a row in a colleague's scope."""
        with self.assertRaises(HTTPException) as caught:
            self.ALICE.shared_write({"scope": "auth0:bob"})
        self.assertEqual(caught.exception.status_code, 400)

    def test_a_non_string_scope_is_a_400_and_not_a_500(self) -> None:
        """A JSON body can carry a list or an object here, and `in` on a frozenset
        raises TypeError for an unhashable value — which would surface as a 500 on a
        merely malformed request."""
        for malformed in ([self.SHARED], {"name": self.SHARED}, 7, True):
            with self.subTest(scope=malformed):
                with self.assertRaises(HTTPException) as caught:
                    self.ALICE.shared_write({"scope": malformed})
                self.assertEqual(caught.exception.status_code, 400)

    def test_a_null_scope_is_a_400_and_not_a_silent_private_write(self) -> None:
        """`null` is the one malformed value the old check could not see. It tested
        `metadata.get(SHARED_SCOPE_KEY) is None`, which short-circuited BEFORE the
        allowlist, so `{"scope": null}` stored a PRIVATE row carrying a useless
        `scope: None` key and answered 200 — the same silent success as a typo'd
        scope, reached through the shape a typo cannot produce. It is not exotic:
        `null` is what a client library emits for an unset optional field, so it
        arrives without anyone deciding to send it."""
        with self.assertRaises(HTTPException) as caught:
            self.ALICE.shared_write({"scope": None})
        self.assertEqual(caught.exception.status_code, 400)

    def test_the_key_being_absent_is_still_an_ordinary_write(self) -> None:
        """The distinction the fix rests on is PRESENT-and-null versus ABSENT, and
        collapsing it the other way would turn every private write into a 400."""
        self.assertIsNone(self.ALICE.shared_write({"name": "a-note", "type": "reference"}))
        self.assertIsNone(self.ALICE.shared_write({}))
        self.assertIsNone(self.ALICE.shared_write(None))

    def test_an_unpinned_caller_is_still_passed_through_on_a_null_scope(self) -> None:
        """The unpinned check comes first for a reason: a machine credential sending
        `{"scope": null}` must keep behaving exactly as it did, since it never opted
        into any of this."""
        self.assertIsNone(self.UNPINNED.shared_write({"scope": None}))

    def test_an_unpinned_caller_is_passed_through_untouched(self) -> None:
        """They can already name any `user_id` outright, so honouring the opt-in
        would buy them nothing — and would mint a shared row with no author, one
        nothing but the machine credential could ever edit again."""
        self.assertIsNone(self.UNPINNED.shared_write({"scope": self.SHARED}))


class UpdateCannotRewriteOwnershipTest(unittest.TestCase):
    """`PUT /memories/{id}` must not be a second, unguarded way to re-key a row.

    mem0's `_update_memory` does `new_metadata = deepcopy(existing.payload)` then
    `new_metadata.update(metadata)`, and the payload is where `user_id` lives. So a
    metadata dict naming `user_id` REWRITES the row's owner in place — which would
    have made every check above decorative: publish by PUT with no scope validation
    and no attribution, or forge the `author_sub` that gates the write path.

    Loud rather than silent. Dropping the key quietly would let a caller believe
    they had published a note repo-wide when they had not.

    This half of the guard is ROW-INDEPENDENT and so runs before the `get` that
    fetches the row is spent; `PutMetadataRoundTripTest` below covers the half that
    needs the row in hand.
    """

    ALICE = MemoryScope(pinned_user_id="auth0:alice", is_admin=False)
    UNPINNED = MemoryScope(pinned_user_id=None, is_admin=False)

    def _assert_refused(self, metadata: dict) -> None:
        with self.assertRaises(HTTPException) as caught:
            self.ALICE.reject_ownership_metadata(metadata)
        self.assertEqual(caught.exception.status_code, 400)

    def test_an_update_may_not_move_a_row_into_another_scope(self) -> None:
        self._assert_refused({"user_id": "auth0:bob"})

    def test_an_update_may_not_publish_a_row_by_naming_the_shared_user_id(self) -> None:
        """`user_id` is what `owns` and `read_scopes` actually key on, so this is the
        spelling that would really re-home the row — skipping `shared_write`'s scope
        allowlist AND its attribution stamp, leaving an authorless shared row."""
        self._assert_refused({"user_id": "repo:ai-utils"})

    def test_ordinary_metadata_still_updates(self) -> None:
        """The refusal is ONE key wide, not "metadata is now read-only"."""
        self.ALICE.reject_ownership_metadata({"name": "a-note", "type": "reference"})
        self.ALICE.reject_ownership_metadata(None)
        self.ALICE.reject_ownership_metadata({})

    def test_the_grouping_ids_are_not_refused_by_this_half(self) -> None:
        """`agent_id`/`run_id` group rows, they do not own them, and the derived fact
        layer writes them on every private row it creates. Whether they are allowed
        depends on the row, which this half cannot see — `reject_row_metadata_conflicts`
        is where a SHARED row refuses them."""
        self.ALICE.reject_ownership_metadata({"agent_id": "facts", "run_id": "parent-id"})

    def test_the_server_derived_keys_are_not_refused_by_this_half_either(self) -> None:
        """They used to be, and it broke the documented round trip — see
        `PutMetadataRoundTripTest`. Whether they are acceptable depends on what the
        row already stores, so the answer moved to the row-aware guard."""
        self.ALICE.reject_ownership_metadata({"scope": "repo:ai-utils", "author_sub": "auth0:alice"})

    def test_legacy_credentials_are_not_held_to_it(self) -> None:
        """They are unscoped everywhere else in this class, and re-keying a row is
        inside what they could already do by deleting and re-adding it."""
        self.UNPINNED.reject_ownership_metadata({"user_id": "auth0:bob"})


class PutMetadataRoundTripTest(unittest.TestCase):
    """A `PUT` may ECHO the server-derived keys back; it may not CHANGE them.

    WHY THIS EXISTS. `scope` and `author_sub` were a flat 400 on any `PUT`, and that
    broke the workflow the deployment documents. mem0's `Memory.get()` promotes only
    the entity ids and buckets every other payload key into `metadata`, so BOTH come
    back inside `metadata` on a read — and the recorded instruction is "search first
    and update the existing row rather than adding a near-duplicate", i.e. GET, edit,
    PUT what you read. A shared row's own author therefore could not update their own
    note without knowing to strip two keys the error never mentioned, and the 400 read
    like a permission problem rather than a payload one.

    Comparing against the stored row is what makes the round trip lossless while
    keeping a real ownership change loud. Re-deriving them silently was the other
    option and is the same class of plausible-looking success as a typo'd scope.
    """

    SHARED = "repo:ai-utils"
    ALICE = MemoryScope(pinned_user_id="auth0:alice", is_admin=False)
    UNPINNED = MemoryScope(pinned_user_id=None, is_admin=False)
    #: What `Memory.get()` hands back for a shared row Alice wrote.
    STORED = {"scope": SHARED, "author_sub": "auth0:alice", "name": "a-note"}

    def _assert_refused(self, metadata: dict, stored_user_id: str | None, stored: dict | None) -> None:
        with self.assertRaises(HTTPException) as caught:
            self.ALICE.reject_row_metadata_conflicts(metadata, stored_user_id, stored)
        self.assertEqual(caught.exception.status_code, 400)

    def test_putting_back_exactly_what_get_returned_is_allowed(self) -> None:
        """The property the whole change exists for: GET, edit the prose, PUT."""
        self.ALICE.reject_row_metadata_conflicts(dict(self.STORED), self.SHARED, self.STORED)

    def test_editing_ordinary_metadata_in_the_round_trip_is_allowed(self) -> None:
        edited = {**self.STORED, "name": "a-renamed-note"}
        self.ALICE.reject_row_metadata_conflicts(edited, self.SHARED, self.STORED)

    def test_omitting_the_derived_keys_entirely_is_allowed(self) -> None:
        """mem0 deep-copies the existing payload before updating, so keys the body
        does not mention survive. Nothing has to be echoed for the row to stay
        correct — echoing is merely permitted."""
        self.ALICE.reject_row_metadata_conflicts({"name": "a-note"}, self.SHARED, self.STORED)

    def test_changing_the_scope_is_still_refused(self) -> None:
        """Changing `scope` alone cannot re-home a row on its own — `owns` and
        `read_scopes` read the payload's `user_id`, which `reject_ownership_metadata`
        refuses outright. But a row whose metadata CLAIMS a scope it does not have is
        exactly the plausible-looking lie the opt-in's 400 exists to prevent, and it
        is the second door on the shared-grouping forgery."""
        self._assert_refused({"scope": self.SHARED}, "auth0:alice", {})
        self._assert_refused({"scope": "repo:something-else"}, self.SHARED, self.STORED)
        self._assert_refused({"scope": None}, self.SHARED, self.STORED)

    def test_changing_the_attribution_is_still_refused(self) -> None:
        """`author_sub` is what `may_modify` reads to decide who may change a shared
        row. A caller who can rewrite it can hand their own note to somebody else, or
        keep one nobody but the machine credential could ever edit."""
        self._assert_refused({"author_sub": "auth0:bob"}, self.SHARED, self.STORED)
        self._assert_refused({"author_sub": "auth0:alice"}, "auth0:alice", {})

    def test_the_refusal_says_which_key_did_not_match(self) -> None:
        """The 400 this replaces said only that the keys were forbidden, which sent
        the author looking for a permission they did not lack. Naming the mismatch is
        the whole difference between "you may not do this" and "your body disagrees
        with the row"."""
        with self.assertRaises(HTTPException) as caught:
            self.ALICE.reject_row_metadata_conflicts({"author_sub": "auth0:bob"}, self.SHARED, self.STORED)
        self.assertIn("do not match the stored memory: author_sub", caught.exception.detail)

    def test_legacy_credentials_are_not_held_to_it(self) -> None:
        """Unscoped everywhere else in this class; `build_fact_layer.py` rewrites
        whatever it likes through the machine credential."""
        self.UNPINNED.reject_row_metadata_conflicts({"author_sub": "auth0:bob"}, self.SHARED, self.STORED)


class SharedRowsCannotBeGroupedIntoAnothersFactsTest(unittest.TestCase):
    """A pinned caller may not choose the grouping ids on a SHARED row.

    THE FORGERY. `agent_id`/`run_id` are grouping and not ownership, which holds
    right up until the row is shared. The console's memory page resolves a row's
    derived facts with `post_search filters={"run_id": <that memory's id>}` and
    `include_shared` (`tools/web/console/src/lib/mem0/api.ts`, `factsFor`), so a
    SHARED row carrying `agent_id: "facts"` and `run_id: <Alice's memory id>` renders
    as a fact extracted from Alice's memory. Bob may write exactly that: the
    `metadata.scope` opt-in is open to everyone and he authors the row, so
    `may_modify` keeps letting him edit it afterwards. Because the fact layer is what
    recall retrieves, that is an injection path into a colleague's agent context, not
    merely a wrong-looking page.

    TWO ROUTES, two guards, and both are needed: refusing it only on `POST` leaves
    "publish an empty shared row, then PUT the ids into it", one extra request.
    """

    SHARED = "repo:ai-utils"
    ALICE = MemoryScope(pinned_user_id="auth0:alice", is_admin=False)
    ADMIN = MemoryScope(pinned_user_id="auth0:carol", is_admin=True)
    UNPINNED = MemoryScope(pinned_user_id=None, is_admin=False)

    def _assert_post_refused(self, agent_id=None, run_id=None, metadata=None) -> HTTPException:
        with self.assertRaises(HTTPException) as caught:
            self.ALICE.reject_shared_grouping_ids(agent_id, run_id, metadata)
        self.assertEqual(caught.exception.status_code, 400)
        return caught.exception

    def test_a_shared_post_may_not_name_a_run_id(self) -> None:
        self._assert_post_refused(run_id="alices-memory-id")

    def test_a_shared_post_may_not_name_an_agent_id(self) -> None:
        self._assert_post_refused(agent_id="facts")

    def test_the_metadata_carrier_is_refused_too_and_it_is_not_the_same_one(self) -> None:
        """mem0's `_build_filters_and_metadata` does `deepcopy(input_metadata)` and
        only THEN `base_metadata_template["agent_id"] = agent_id`, under `if agent_id:`.
        So a grouping id smuggled through `metadata` reaches the payload untouched
        whenever the top-level parameter is absent — guarding the parameters alone
        would have looked complete and stopped nothing."""
        self._assert_post_refused(metadata={"scope": self.SHARED, "run_id": "alices-memory-id"})
        self._assert_post_refused(metadata={"scope": self.SHARED, "agent_id": "facts"})

    def test_the_refusal_names_the_key(self) -> None:
        self.assertIn("run_id", self._assert_post_refused(run_id="alices-memory-id").detail)

    def test_an_ordinary_shared_write_is_untouched(self) -> None:
        """The refusal is two keys wide. A repo-wide note with a name and a type is
        the whole point of the shared scope."""
        self.ALICE.reject_shared_grouping_ids(None, None, {"scope": self.SHARED, "name": "a-note"})
        self.ALICE.reject_shared_grouping_ids(None, None, None)

    def test_the_put_route_refuses_the_same_thing_on_a_shared_row(self) -> None:
        """Otherwise the POST refusal is bypassed by publishing an empty shared row
        and then PUTting the grouping ids into it."""
        with self.assertRaises(HTTPException) as caught:
            self.ALICE.reject_row_metadata_conflicts(
                {"agent_id": "facts", "run_id": "alices-memory-id"}, self.SHARED, {"author_sub": "auth0:alice"}
            )
        self.assertEqual(caught.exception.status_code, 400)

    def test_an_sso_admin_is_not_exempt(self) -> None:
        """`may_modify` lets an SSO admin moderate a shared row, which is about
        removing or correcting common property. Re-pointing one at a colleague's
        memory is not moderation, and the fact layer is built through the unpinned
        machine credential anyway — so the admin carve-out stops here."""
        with self.assertRaises(HTTPException):
            self.ADMIN.reject_shared_grouping_ids("facts", "alices-memory-id", None)

    def test_a_private_row_keeps_its_grouping_ids(self) -> None:
        """The derived fact layer writes both on every private row it creates.
        Refusing them there would break `build_fact_layer.py` for a pinned caller and
        buy nothing: a private row is not reachable through anyone else's read."""
        self.ALICE.reject_row_metadata_conflicts({"agent_id": "facts", "run_id": "parent-id"}, "auth0:alice", {})

    def test_the_unpinned_fact_layer_build_is_untouched(self) -> None:
        """`.claude/hooks/mem0/build_fact_layer.py` writes `agent_id` AND `run_id` on
        every fact, through ADMIN_API_KEY. That credential is unpinned, so it must
        never reach either guard — this is the check the fix was warned not to
        break."""
        self.UNPINNED.reject_shared_grouping_ids("facts", "parent-id", {"scope": self.SHARED})
        self.UNPINNED.reject_row_metadata_conflicts(
            {"agent_id": "facts", "run_id": "parent-id"}, self.SHARED, {"author_sub": "auth0:alice"}
        )


class SearchFiltersCannotOutrunThePinTest(unittest.TestCase):
    """`POST /search` filters may only carry scalars for a pinned caller.

    THE HOLE. `search_memories` pins with `{**caller_filters, "user_id": <caller>}`,
    which looks final and is not. `{**d, k: v}` keeps the ORIGINAL position of a key
    already in `d`, so a caller who writes `user_id` before `AND` controls the
    iteration order of the merged dict; `Memory._process_metadata_filters` then
    FLATTENS the `AND` group back to the top level through `merge_filters`, whose
    scalar branch is a plain `target[key] = value`; and `search()` finishes with
    `effective_filters.update(processed_filters)`. Net effect: the caller's
    `user_id` lands back over the pin and the search runs against any other person's
    private scope. `Mem0FlattensAndOverThePinTest` below EXECUTES that against the
    installed wheel rather than asserting it from a reading of the source.

    AN ALLOWLIST, NOT A LIST OF DANGEROUS OPERATORS. `OR`/`NOT` nest into
    `$or`/`$not`, which pgvector joins with AND, so they do not escape TODAY.
    `requirements.txt` pins `mem0ai>=0.1.48` — a floor — so the filter compiler is
    not a fixed target, and a denylist over it would stop covering the moment a
    release adds a fourth logical key, with nothing able to notice. The rule is
    therefore about values, not keys: a container of any kind is refused.
    """

    ALICE = MemoryScope(pinned_user_id="auth0:alice", is_admin=False)
    UNPINNED = MemoryScope(pinned_user_id=None, is_admin=False)
    VICTIM = "auth0:bob"

    def _assert_refused(self, filters: dict) -> HTTPException:
        with self.assertRaises(HTTPException) as caught:
            self.ALICE.reject_scope_escaping_filters(filters)
        self.assertEqual(caught.exception.status_code, 400)
        return caught.exception

    def test_the_and_escape_is_refused_in_the_ordering_that_works(self) -> None:
        """`user_id` first is the ordering that actually escapes."""
        self._assert_refused({"user_id": "anything", "AND": [{"user_id": self.VICTIM}]})

    def test_the_and_escape_is_refused_in_the_ordering_that_does_not(self) -> None:
        """The other ordering is harmless with today's wheel, and the fix must not
        depend on that — dict ordering is the attacker's to choose, and it is the one
        input a caller controls that no filter-shape check can see."""
        self._assert_refused({"AND": [{"user_id": self.VICTIM}], "user_id": "anything"})

    def test_the_operators_that_are_currently_safe_are_refused_anyway(self) -> None:
        """`OR`/`NOT`/`$or`/`$not` do not escape against the pinned floor of the
        library. They are refused because "does not escape" is a property of a
        version this server does not pin, not of the filter language."""
        for key in ("OR", "NOT", "$and", "$or", "$not"):
            with self.subTest(operator=key):
                self._assert_refused({key: [{"user_id": self.VICTIM}]})

    def test_a_comparison_operator_dict_is_refused(self) -> None:
        """`{"k": {"eq": ...}}` is what flips mem0 into advanced-filter processing at
        all, which is the code path the escape lives on."""
        self._assert_refused({"metadata_key": {"eq": "x"}})

    def test_a_null_value_is_refused_rather_than_waved_through(self) -> None:
        """`{"k": null}` compiles to `payload->>'k' = 'None'` — a filter matching no
        row, returning 200 with an empty list. Not an escape; the other failure this
        module keeps closing, where the answer is wrong and nothing says so."""
        self._assert_refused({"name": None})

    def test_the_refusal_names_the_offending_key(self) -> None:
        detail = self._assert_refused({"AND": [{"user_id": self.VICTIM}]}).detail
        self.assertIn("AND", detail)

    def test_the_filters_every_real_caller_sends_still_work(self) -> None:
        """The console's fact panel (`filters={"run_id": <parent id>}`), its layer
        scoping and `.claude/hooks/mem0/` recall (`{"agent_id": "facts"}`). If this
        goes red the guard has been widened past what it is for."""
        for filters in (
            {"agent_id": "facts"},
            {"run_id": "parent-id"},
            {"agent_id": "facts", "run_id": "parent-id"},
            {"variant": "fact", "source_profile": "max"},
            {"agent_id": "*"},
            None,
            {},
        ):
            with self.subTest(filters=filters):
                self.ALICE.reject_scope_escaping_filters(filters)

    def test_a_caller_supplied_user_id_is_exempt_because_the_pin_overwrites_it(self) -> None:
        """`pin` overwriting a caller-supplied `user_id` is documented and tested
        behaviour, and the route applies it one line later. Rejecting it here would
        break the deprecated top-level `user_id` field, which `search_memories`
        copies into `filters` before pinning."""
        self.ALICE.reject_scope_escaping_filters({"user_id": self.VICTIM})
        self.ALICE.reject_scope_escaping_filters({"user_id": {"eq": self.VICTIM}})

    def test_unpinned_callers_keep_the_full_filter_language(self) -> None:
        """They can already name any `user_id` outright, so there is no pin to
        escape — and CI, `mem0ctl.py` under the machine credential and the console's
        admin paths all run through them."""
        self.UNPINNED.reject_scope_escaping_filters({"AND": [{"user_id": self.VICTIM}]})


class Mem0FlattensAndOverThePinTest(unittest.TestCase):
    """The library behaviour that makes the guard above necessary, EXECUTED.

    Every other test in this file compares this server against its own intent. This
    one runs the INSTALLED `mem0ai` wheel — the copy that actually serves requests;
    the image installs from PyPI, so `mem0/` in this fork is an upstream contribution
    that reaches production no sooner than the next release — and reproduces the
    escape end to end. It is the difference between "the source reads as though an
    `AND` group overwrites the pin" and "it does".

    Still no network and no database: `_process_metadata_filters` touches no `self`,
    so it is called unbound. Constructing a `Memory` is what would need credentials,
    and nothing here does.

    IF THIS GOES RED, read it before touching anything. A failure means the wheel's
    filter semantics moved, which is not itself bad news — upstream may have fixed
    the overwrite. It is NOT on its own a reason to delete
    `reject_scope_escaping_filters`: the guard is an allowlist precisely because
    `requirements.txt` pins a floor rather than a version, so an older wheel with the
    old behaviour remains installable. Update the test to describe what the wheel now
    does, and leave the guard.
    """

    PINNED = "auth0:alice"
    VICTIM = "auth0:bob"

    @classmethod
    def _effective(cls, caller_filters: dict) -> dict:
        """main.py's own pin, then `Memory.search`'s advanced-operator branch, verbatim.

        Transcribed rather than called, because main.py builds a real `Memory` at
        import scope and `Memory.search` would run an embedder and a query. What is
        reproduced is `search_memories`'s `{**filters, "user_id": sid}` followed by
        the six lines of `Memory.search` that sit between `_has_advanced_operators`
        and `_search_vector_store` — including the pops, so that the dict this
        returns is the one the vector store would receive.
        `RoutesCallTheRightGuardTest` in test_list_scoping.py is what pins that the
        route still contains the first of those lines.
        """
        pinned = {**caller_filters, "user_id": cls.PINNED}
        processed = mem0_main.Memory._process_metadata_filters(None, pinned)
        for logical_key in ("AND", "OR", "NOT"):
            pinned.pop(logical_key, None)
        for key in list(pinned):
            if key not in ("AND", "OR", "NOT", "user_id", "agent_id", "run_id") and isinstance(pinned.get(key), dict):
                pinned.pop(key, None)
        pinned.update(processed)
        return pinned

    def test_the_wheel_still_has_the_method_this_rests_on(self) -> None:
        """A rename would make every assertion below an AttributeError with no
        indication of which side changed."""
        self.assertTrue(hasattr(mem0_main.Memory, "_process_metadata_filters"))

    def test_an_and_group_overwrites_the_pin_when_user_id_comes_first(self) -> None:
        """The escape. `merge_filters`' scalar branch is `target[key] = value`, so
        flattening `AND` back to the top level is an overwrite, not a conjunction."""
        escaped = self._effective({"user_id": "anything", "AND": [{"user_id": self.VICTIM}]})
        self.assertEqual(
            escaped["user_id"],
            self.VICTIM,
            "the pin held — the wheel's filter semantics have changed; read this class's docstring",
        )

    def test_the_same_payload_with_the_keys_reversed_does_not_escape(self) -> None:
        """Which is why no fix may depend on key order: `{**d, k: v}` keeps the
        ORIGINAL position of a key already in `d`, and that position is the
        attacker's to choose."""
        held = self._effective({"AND": [{"user_id": self.VICTIM}], "user_id": "anything"})
        self.assertEqual(held["user_id"], self.PINNED)

    def test_or_and_not_nest_instead_of_flattening(self) -> None:
        """The reason the guard is an allowlist rather than a list of operators: these
        two are safe by an implementation detail of a version this server does not
        pin (they become `$or`/`$not`, which pgvector joins with AND), not by
        anything in the filter language's contract."""
        for operator, compiled in (("OR", "$or"), ("NOT", "$not")):
            with self.subTest(operator=operator):
                nested = self._effective({operator: [{"user_id": self.VICTIM}]})
                self.assertEqual(nested["user_id"], self.PINNED)
                self.assertIn(compiled, nested)

    def test_the_guard_refuses_every_payload_above(self) -> None:
        """The two halves joined up: what the library does, and what now never
        reaches it."""
        for payload in (
            {"user_id": "anything", "AND": [{"user_id": self.VICTIM}]},
            {"AND": [{"user_id": self.VICTIM}], "user_id": "anything"},
            {"OR": [{"user_id": self.VICTIM}]},
            {"NOT": [{"user_id": self.VICTIM}]},
        ):
            with self.subTest(filters=payload):
                with self.assertRaises(HTTPException) as caught:
                    MemoryScope(pinned_user_id=self.PINNED, is_admin=False).reject_scope_escaping_filters(payload)
                self.assertEqual(caught.exception.status_code, 400)


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
