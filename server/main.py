import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional

import telemetry
from auth import (
    ADMIN_API_KEY,
    AUTH_DISABLED,
    JWT_SECRET,
    MemoryScope,
    require_admin,
    resolve_scope,
    verify_auth,
)
from db import SessionLocal
from dotenv import load_dotenv
from embedders import install_overrides as install_embedder_overrides
from errors import (
    UpstreamError,
    install_request_id_logging,
    new_request_id,
    request_id_var,
    upstream_error,
    upstream_error_handler,
)
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from listing import listing_filters
from llms import install_overrides as install_llm_overrides
from mem0.exceptions import ValidationError as Mem0ValidationError
from models import RequestLog, User
from pydantic import BaseModel, Field
from rate_limit import limiter
from routers import api_keys as api_keys_router
from routers import auth as auth_router
from routers import entities as entities_router
from routers import requests as requests_router
from schemas import MessageResponse
from server_state import (
    get_current_config,
    get_memory_instance,
    initialize_state,
    set_session_factory,
    update_config,
)
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from sqlalchemy import func, select

load_dotenv()

install_request_id_logging()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - [%(request_id)s] %(message)s")

MIN_KEY_LENGTH = 16
SENSITIVE_CONFIG_KEYS = {
    "admin_api_key",
    "api_key",
    "authorization",
    "jwt_secret",
    "password",
    "password_hash",
    "secret",
    "token",
}
SKIPPED_REQUEST_LOG_PATHS = {"/api/health", "/docs", "/redoc", "/openapi.json"}
SKIPPED_REQUEST_LOG_PREFIXES = ("/requests",)

# evercompliant POC patch: ONE bundled provider, because there is one gateway.
#
# Every model this deployment reaches — Gemini and Anthropic included — is served
# by bifrost over its OpenAI-compatible API, so the provider is always "openai"
# with OPENAI_BASE_URL pointed at it (see MEM0_DEFAULT_LLM_MODEL and
# OPENAI_BASE_URL in k8s/manifests/mem0-server-configmap.yaml). The vendor SDKs
# behind "gemini" and "anthropic" were therefore never imported at runtime while
# costing ~140MB of a 632MB image — google-generativeai drags in
# google-api-python-client (99MB) on its own.
#
# Re-adding a provider means restoring its package in server/requirements.txt AND
# in ai-utils' tools/mcp/mem0/server/pyproject.toml, which transcribes that file,
# and re-running `aspect run //tools/mcp/mem0/server:lock`. Prefer pointing
# bifrost at the model instead — that is the whole reason it is in front.
# aws_bedrock is bundled for the LLM too (boto3 is already in the image for the embedder). It
# is the only way to get grammar-constrained extraction: bifrost CAN carry a strict schema
# (`response_format: {type: json_schema}` -> Gemini `responseJsonSchema`), but mem0 hardcodes
# the weak `{"type": "json_object"}` form in the LIBRARY, which ships from PyPI and cannot be
# patched from this fork. See MEM0_LLM_PROVIDER below.
BUNDLED_LLM_PROVIDERS = ("openai", "aws_bedrock")
# aws_bedrock is bundled for the EMBEDDER only (boto3 in server/requirements.txt). It is the
# one path bifrost cannot serve: its virtual key is not authorised for cohere, and its
# OpenAI-compatible surface carries no input_type, so an asymmetric embedder is reduced to a
# symmetric one. Requires AWS credentials in the pod — see MEM0_EMBEDDER_PROVIDER below.
BUNDLED_EMBEDDER_PROVIDERS = ("openai", "aws_bedrock")


def _warn_if_unconfigured() -> None:
    """Pre-auth deployments upgrading into this build will 401 everywhere until
    an admin key or admin user exists. Surface the fix before the support tickets."""
    try:
        with SessionLocal() as session:
            if session.scalar(select(func.count(User.id))) > 0:
                return
    except Exception:
        return

    logging.warning(
        "\n%s\n"
        "  Auth is enabled by default and this server has no admin configured.\n"
        "  Protected endpoints will return 401 until you either:\n"
        "    1. Set ADMIN_API_KEY=<long-random-value>  (fastest, no client changes)\n"
        "    2. Register an admin at http://<host>:3000/setup\n"
        "    3. Set AUTH_DISABLED=true                 (local development only)\n"
        "  Docs: https://docs.mem0.ai/open-source/features/rest-api#authentication\n"
        "%s",
        "=" * 72,
        "=" * 72,
    )


if not AUTH_DISABLED and not JWT_SECRET:
    raise RuntimeError(
        "JWT_SECRET is required. Set it in .env (generate with `openssl rand -base64 48`) "
        "or set AUTH_DISABLED=true for local development only."
    )

if AUTH_DISABLED:
    logging.warning("AUTH_DISABLED is enabled. Protected endpoints are open for local development only.")
elif ADMIN_API_KEY and len(ADMIN_API_KEY) < MIN_KEY_LENGTH:
    logging.warning(
        "ADMIN_API_KEY is shorter than %d characters - consider using a longer key for production.",
        MIN_KEY_LENGTH,
    )
elif not ADMIN_API_KEY:
    _warn_if_unconfigured()

telemetry.log_status()

POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "postgres")
POSTGRES_PORT = os.environ.get("POSTGRES_PORT", "5432")
POSTGRES_DB = os.environ.get("POSTGRES_DB", "postgres")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "postgres")
POSTGRES_COLLECTION_NAME = os.environ.get("POSTGRES_COLLECTION_NAME", "memories")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
HISTORY_DB_PATH = os.environ.get("HISTORY_DB_PATH", "/app/history/history.db")
DEFAULT_LLM_MODEL = os.environ.get("MEM0_DEFAULT_LLM_MODEL", "gpt-4.1-nano-2025-04-14")
DEFAULT_EMBEDDER_MODEL = os.environ.get("MEM0_DEFAULT_EMBEDDER_MODEL", "text-embedding-3-small")
# evercompliant POC patch: upstream has no env for embedding dims, but without it the
# openai embedder never sends `dimensions` and the pgvector store defaults the memories
# table to VECTOR(1536) — both wrong for Nova embeddings pinned to 1024 via bifrost.
_EMBEDDING_DIMS_ENV = os.environ.get("MEM0_EMBEDDING_DIMS")
EMBEDDING_DIMS = int(_EMBEDDING_DIMS_ENV) if _EMBEDDING_DIMS_ENV else None
# evercompliant POC patch: the pgvector store's psycopg pool defaults to min_size=1 — a
# permanently-idle connection that blocks Aurora Serverless v2 auto-pause. minconn=0 is
# valid (max_size stays at the maxconn default 5); idle connections then drain after the
# pool's max_idle (600s), letting the cluster pause.
PG_MINCONN = int(os.environ.get("MEM0_PG_MINCONN", "1"))
# evercompliant POC patch: the LIBRARY already accepts `reasoning_effort` and
# `is_reasoning_model` (mem0/configs/llms/base.py), but the server never passed either, so a
# deployment had no way to set them. That gap is load-bearing for any thinking-capable
# extraction model, because `LLMBase._is_reasoning_model` classifies by NAME against a set of
# OpenAI ids only ({o1, o3, gpt-5, ...}): `gemini-3.5-flash` misses it, is treated as a regular
# model, and so is sent `max_tokens` — which its reasoning tokens consume before it emits the
# fact JSON (measured through bifrost: 866 reasoning vs 202 text tokens on a 2KB input). The
# JSON arrives truncated, extraction logs "Error parsing extraction response: Unterminated
# string", and POST /memories STILL returns 200 with an empty results list — so total
# extraction failure is indistinguishable from "no new facts found" at the API.
#
# MEM0_LLM_IS_REASONING_MODEL=true takes precedence over that heuristic and drops max_tokens
# (and temperature) entirely; MEM0_LLM_REASONING_EFFORT=none additionally stops the thinking
# tokens being spent at all (verified via bifrost -> vertex: reasoning tokens go to 0,
# finish_reason=stop). Both default to None, i.e. unchanged upstream behaviour.
LLM_REASONING_EFFORT = os.environ.get("MEM0_LLM_REASONING_EFFORT") or None
_IS_REASONING_ENV = os.environ.get("MEM0_LLM_IS_REASONING_MODEL")
LLM_IS_REASONING_MODEL = _IS_REASONING_ENV.strip().lower() == "true" if _IS_REASONING_ENV else None

# evercompliant POC patch: `temperature` must be OMITTABLE, not merely settable. Newer models
# reject the parameter outright rather than ignoring it — claude-sonnet-5 on Bedrock answers
# HTTP 400 `The model returned the following errors: `temperature` is deprecated for this
# model.` So a hardcoded temperature does not degrade on such a model, it makes the model
# entirely unreachable: every extraction call fails, and mem0 surfaces that as 200 with an
# empty results list, i.e. silently storing nothing.
#
# Set MEM0_LLM_TEMPERATURE to a float to pin it, or to the empty string to omit the key
# altogether. Default "0.2" preserves the previous behaviour exactly.
_TEMPERATURE_ENV = os.environ.get("MEM0_LLM_TEMPERATURE", "0.2").strip()
LLM_TEMPERATURE = float(_TEMPERATURE_ENV) if _TEMPERATURE_ENV else None

# evercompliant POC patch: the embedder provider is pinned to "openai" upstream, which here
# means bifrost via OPENAI_BASE_URL. That is the right default — bifrost holds the credentials
# and this pod has none — but it cannot reach models bifrost's virtual key is not authorised
# for (cohere returns 502 provider_auth_failed), and it flattens the query/document asymmetry
# that asymmetric embedders depend on.
#
# MEM0_EMBEDDER_PROVIDER=aws_bedrock switches to a direct Bedrock client. That requires AWS
# credentials IN THIS POD, which arrive via Otterize: an aws: target on the workload's
# ClientIntents makes the credentials-operator mint an IAM role and annotate the
# ServiceAccount. Without that intent applied first, this setting makes every embed call fail
# — and since embedding happens on add AND on search, that is a total outage of the store,
# not a degradation.
EMBEDDER_PROVIDER = os.environ.get("MEM0_EMBEDDER_PROVIDER", "openai").strip() or "openai"
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

# MEM0_LLM_PROVIDER=aws_bedrock switches fact extraction from bifrost to a direct Bedrock
# client. The reason to want that is GRAMMAR-CONSTRAINED output: mem0 asks for JSON with
# `response_format={"type": "json_object"}`, which is a request, not a guarantee — measured on
# this corpus it returns unparseable JSON on ~19% of real inputs, and every failure is SILENT
# (`POST /memories` answers 200 with `{"results": []}`, byte-identical to "no new facts").
# Bedrock's `output_config.format` compiles the schema into a grammar, which removes that
# failure class outright.
#
# Model support is narrow and NOT newest-first — see server/llms/aws_bedrock.py. Only
# 4.6-and-older Claude accepts it; Opus 4.8 / Opus 5 / Sonnet 5 reject it on every AWS surface,
# so choosing a "better" model silently degrades to prompt-only JSON.
#
# Needs the same pod AWS credentials as the embedder (Otterize aws: ClientIntents). Unlike the
# embedder, a failure here is a degradation rather than an outage: extraction only runs on
# `infer:true` writes, and this corpus is imported with `infer:false`.
LLM_PROVIDER = os.environ.get("MEM0_LLM_PROVIDER", "openai").strip() or "openai"

_LLM_MAX_TOKENS_ENV = os.environ.get("MEM0_LLM_MAX_TOKENS", "").strip()
LLM_MAX_TOKENS = int(_LLM_MAX_TOKENS_ENV) if _LLM_MAX_TOKENS_ENV else None

# Point "aws_bedrock" at embedders/aws_bedrock.py instead of the wheel's own version.
#
# This is NOT belt-and-braces over the fix in mem0/embeddings/aws_bedrock.py — it is the
# only copy that runs. The image installs mem0ai from PyPI (upstream's own production
# Dockerfile does the same; only dev.Dockerfile pip-installs the local library), so this
# repo's mem0/ is an upstream contribution that reaches the server no sooner than the next
# PyPI release. Stock Bedrock+Cohere cannot serve embed-v4 at all: it hardcodes
# input_type=search_document, never sends output_dimension (1536 dims into a fixed 1024
# pgvector column), and indexes [0] into v4's dict-shaped response.
#
# Unconditional, not gated on EMBEDDER_PROVIDER: it rebinds one entry in a lookup table
# and imports nothing until that provider is selected, so gating would only add a way for
# the two to disagree.
install_embedder_overrides()

# Point "aws_bedrock" at server/llms/aws_bedrock.py. The library's AWSBedrockLLM accepts a
# `response_format` argument and then never references it, so asking mem0's Bedrock provider for
# JSON is a silent no-op — the override is what makes MEM0_LLM_PROVIDER=aws_bedrock actually
# enforce a schema. Unconditional for the same reason as the embedder: it rebinds one entry in a
# lookup table and imports nothing until that provider is selected.
install_llm_overrides()

DEFAULT_CONFIG = {
    "version": "v1.1",
    "vector_store": {
        "provider": "pgvector",
        "config": {
            "host": POSTGRES_HOST,
            "port": int(POSTGRES_PORT),
            "dbname": POSTGRES_DB,
            "user": POSTGRES_USER,
            "password": POSTGRES_PASSWORD,
            "collection_name": POSTGRES_COLLECTION_NAME,
            "minconn": PG_MINCONN,
            **({"embedding_model_dims": EMBEDDING_DIMS} if EMBEDDING_DIMS else {}),
        },
    },
    "llm": {
        "provider": LLM_PROVIDER,
        "config": {
            # api_key is bifrost's virtual key; a direct Bedrock client authenticates with the
            # pod's IAM role instead, and AWSBedrockConfig has no api_key field at all.
            **({"api_key": OPENAI_API_KEY} if LLM_PROVIDER == "openai" else {}),
            **({"aws_region": AWS_REGION} if LLM_PROVIDER == "aws_bedrock" else {}),
            **({"temperature": LLM_TEMPERATURE} if LLM_TEMPERATURE is not None else {}),
            "model": DEFAULT_LLM_MODEL,
            # Bedrock's Converse path defaults maxTokens to 2000, which truncates extraction on
            # a large input (measured: up to 21 facts from one 5KB file). Pass it explicitly.
            **({"max_tokens": LLM_MAX_TOKENS} if LLM_MAX_TOKENS else {}),
            # Reasoning knobs are OpenAI-shaped and exist for the bifrost/gemini path only.
            # AWSBedrockConfig has no such fields, and the factory calls config_class(**config)
            # directly for a dict config, so passing them would raise TypeError at startup.
            **(
                {
                    **({"reasoning_effort": LLM_REASONING_EFFORT} if LLM_REASONING_EFFORT else {}),
                    **({"is_reasoning_model": LLM_IS_REASONING_MODEL} if LLM_IS_REASONING_MODEL is not None else {}),
                }
                if LLM_PROVIDER == "openai"
                else {}
            ),
        },
    },
    "embedder": {
        "provider": EMBEDDER_PROVIDER,
        "config": {
            # api_key is bifrost's virtual key and is meaningless to a direct Bedrock client,
            # which authenticates with the pod's IAM role instead.
            **({"api_key": OPENAI_API_KEY} if EMBEDDER_PROVIDER == "openai" else {}),
            **({"aws_region": AWS_REGION} if EMBEDDER_PROVIDER == "aws_bedrock" else {}),
            "model": DEFAULT_EMBEDDER_MODEL,
            **({"embedding_dims": EMBEDDING_DIMS} if EMBEDDING_DIMS else {}),
        },
    },
    "history_db_path": HISTORY_DB_PATH,
}


set_session_factory(SessionLocal)
initialize_state(DEFAULT_CONFIG)


app = FastAPI(
    title="Mem0 REST APIs",
    description=(
        "A REST API for managing and searching memories for your AI Agents and Apps.\n\n"
        "## Authentication\n"
        "Supports Bearer JWT tokens, per-user API keys via `X-API-Key` header, "
        "or the legacy `ADMIN_API_KEY` environment variable. Set `AUTH_DISABLED=true` for local development only."
    ),
    version="1.0.0",
    redirect_slashes=False,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_exception_handler(UpstreamError, upstream_error_handler)
DASHBOARD_URL = os.environ.get("DASHBOARD_URL", "http://localhost:3000")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[DASHBOARD_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router.router)
app.include_router(api_keys_router.router)
app.include_router(entities_router.router)
app.include_router(requests_router.router)


class Message(BaseModel):
    role: str = Field(..., description="Role of the message (user or assistant).")
    content: str = Field(..., description="Message content.")


class MemoryCreate(BaseModel):
    messages: List[Message] = Field(..., description="List of messages to store.")
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    expiration_date: Optional[str] = Field(None, description="Expiration date in YYYY-MM-DD format.")
    infer: Optional[bool] = Field(None, description="Whether to extract facts from messages. Defaults to True.")
    memory_type: Optional[str] = Field(None, description="Type of memory to store (e.g. 'core').")
    prompt: Optional[str] = Field(None, description="Custom prompt to use for fact extraction.")


class MemoryUpdate(BaseModel):
    text: Optional[str] = Field(None, description="New content to update the memory with.")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata to update.")
    expiration_date: Optional[str] = Field(None, description="Expiration date in YYYY-MM-DD format, or null to clear.")


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query.")
    user_id: Optional[str] = Field(None, description="Deprecated: pass inside `filters` instead.", deprecated=True)
    run_id: Optional[str] = Field(None, description="Deprecated: pass inside `filters` instead.", deprecated=True)
    agent_id: Optional[str] = Field(None, description="Deprecated: pass inside `filters` instead.", deprecated=True)
    filters: Optional[Dict[str, Any]] = None
    top_k: Optional[int] = Field(None, description="Maximum number of results to return.")
    threshold: Optional[float] = Field(None, description="Minimum similarity score for results.")
    explain: Optional[bool] = Field(None, description="Include score details for each search result.")
    show_expired: Optional[bool] = Field(None, description="Include expired memories.")


class GenerateInstructionsRequest(BaseModel):
    use_case: str = Field(..., description="Description of what the user will use Mem0 for.")


def _client_error(exc: Exception) -> HTTPException:
    """Map core validation / not-found errors to 4xx so clients can tell a bad
    request from an upstream outage. 'not found' is a 404, everything else a 400."""
    detail = str(exc)
    status_code = 404 if isinstance(exc, ValueError) and "not found" in detail.lower() else 400
    return HTTPException(status_code=status_code, detail=detail)


def _redact_config(value: Any, key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {item_key: _redact_config(item_value, item_key) for item_key, item_value in value.items()}
    if isinstance(value, list):
        return [_redact_config(item_value, key) for item_value in value]
    if key is not None and key.lower() in SENSITIVE_CONFIG_KEYS:
        return "[redacted]" if value else value
    return value


def _validate_bundled_providers(config: Dict[str, Any]) -> None:
    llm = config.get("llm")
    if isinstance(llm, dict) and (provider := llm.get("provider")) and provider not in BUNDLED_LLM_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"LLM provider '{provider}' is not bundled in this image. "
                f"Bundled providers: {', '.join(BUNDLED_LLM_PROVIDERS)}. "
                "To use another provider, install its Python package, rebuild the container, "
                "and extend BUNDLED_LLM_PROVIDERS in server/main.py."
            ),
        )

    embedder = config.get("embedder")
    if (
        isinstance(embedder, dict)
        and (provider := embedder.get("provider"))
        and provider not in BUNDLED_EMBEDDER_PROVIDERS
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Embedder provider '{provider}' is not bundled in this image. "
                f"Bundled providers: {', '.join(BUNDLED_EMBEDDER_PROVIDERS)}. "
                "To use another provider, install its Python package, rebuild the container, "
                "and extend BUNDLED_EMBEDDER_PROVIDERS in server/main.py."
            ),
        )


def _should_log_request(request: Request) -> bool:
    if request.method == "OPTIONS":
        return False
    path = request.url.path
    if path in SKIPPED_REQUEST_LOG_PATHS:
        return False
    return not path.startswith(SKIPPED_REQUEST_LOG_PREFIXES)


def _persist_request_log(method: str, path: str, status_code: int, latency_ms: float, auth_type: str) -> None:
    session = SessionLocal()

    try:
        session.add(
            RequestLog(
                method=method,
                path=path,
                status_code=status_code,
                latency_ms=latency_ms,
                auth_type=auth_type,
            )
        )
        session.commit()
    except Exception:
        session.rollback()
        logging.exception("Failed to persist request log")
    finally:
        session.close()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request.state.auth_type = getattr(request.state, "auth_type", "none")
    rid = new_request_id()
    token = request_id_var.set(rid)
    start = time.perf_counter()
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = rid
        return response
    except Exception:
        status_code = 500
        raise
    finally:
        request_id_var.reset(token)
        if _should_log_request(request):
            asyncio.get_running_loop().run_in_executor(
                None,
                _persist_request_log,
                request.method,
                request.url.path,
                status_code,
                round((time.perf_counter() - start) * 1000, 2),
                getattr(request.state, "auth_type", "none"),
            )


@app.get("/configure", summary="Get current Mem0 configuration")
def get_config(_auth=Depends(verify_auth)):
    return _redact_config(get_current_config())


@app.get("/configure/providers", summary="List bundled LLM and embedder providers")
def list_bundled_providers(_auth=Depends(verify_auth)):
    return {"llm": list(BUNDLED_LLM_PROVIDERS), "embedder": list(BUNDLED_EMBEDDER_PROVIDERS)}


@app.post("/configure", summary="Configure Mem0")
def set_config(config: Dict[str, Any], _auth=Depends(require_admin)):
    """Set memory configuration. Requires admin role."""
    _validate_bundled_providers(config)
    update_config(config)
    return {"message": "Configuration set successfully"}


@app.post("/generate-instructions", summary="Generate custom instructions from a use case")
def generate_instructions(req: GenerateInstructionsRequest, _auth=Depends(verify_auth)):
    """Generate custom instructions and a contextual test message tailored to a use case."""
    try:
        llm = get_memory_instance().llm
        prompt = (
            "You are configuring a memory system. Given the use case below, produce two things:\n"
            "1. INSTRUCTIONS: A short paragraph of custom instructions telling the memory extraction system "
            "what kinds of facts, preferences, and context to prioritize. Be specific to the use case.\n"
            "2. TEST_MESSAGE: A single realistic sentence a user in this use case would say, suitable for "
            "testing that the memory system works.\n\n"
            "Respond in exactly this format (no markdown, no extra text):\n"
            "INSTRUCTIONS: <your instructions>\n"
            f"TEST_MESSAGE: <your test message>\n\nUse case: {req.use_case}"
        )
        response = llm.generate_response([{"role": "user", "content": prompt}])
        instructions = response
        test_message = "I like to hike on weekends."
        if "INSTRUCTIONS:" in response and "TEST_MESSAGE:" in response:
            parts = response.split("TEST_MESSAGE:")
            instructions = parts[0].replace("INSTRUCTIONS:", "").strip()
            test_message = parts[1].strip()
        return {"custom_instructions": instructions, "test_message": test_message}
    except Exception as exc:
        raise upstream_error() from exc


@app.post("/memories", summary="Create memories")
def add_memory(memory_create: MemoryCreate, scope: MemoryScope = Depends(resolve_scope)):
    """Store new memories."""
    params = {k: v for k, v in memory_create.model_dump().items() if v is not None and k != "messages"}

    # Overwrite before validating, not after: a pinned caller supplies no
    # `user_id` at all (the tool schema no longer offers one), so the
    # "at least one identifier" check below would reject every request if it ran
    # first. Assigning into `params` rather than the model keeps the request body
    # untouched for logging.
    pinned = scope.pin(memory_create.user_id)
    if pinned is not None:
        params["user_id"] = pinned
    elif "user_id" in params:
        del params["user_id"]

    if not any([params.get("user_id"), params.get("agent_id"), params.get("run_id")]):
        raise HTTPException(status_code=400, detail="At least one identifier (user_id, agent_id, run_id) is required.")

    try:
        response = get_memory_instance().add(messages=[m.model_dump() for m in memory_create.messages], **params)
        if response.get("results"):
            telemetry.log_dashboard_nudge_once(DASHBOARD_URL)
        return JSONResponse(content=response)
    except (ValueError, Mem0ValidationError) as e:
        raise _client_error(e) from e
    except Exception as exc:
        raise upstream_error() from exc


ALL_MEMORIES_LIMIT = 1000
_RESERVED_PAYLOAD_KEYS = {
    "data",
    "user_id",
    "agent_id",
    "run_id",
    "hash",
    "created_at",
    "updated_at",
    "expiration_date",
}


def _serialize_memory(row: Any) -> Dict[str, Any]:
    payload = getattr(row, "payload", None) or {}
    return {
        "id": getattr(row, "id", None),
        "memory": payload.get("data"),
        "user_id": payload.get("user_id"),
        "agent_id": payload.get("agent_id"),
        "run_id": payload.get("run_id"),
        "hash": payload.get("hash"),
        "expiration_date": payload.get("expiration_date"),
        "metadata": {k: v for k, v in payload.items() if k not in _RESERVED_PAYLOAD_KEYS},
        "created_at": payload.get("created_at"),
        "updated_at": payload.get("updated_at"),
    }


def _list_all_memories(limit: int = ALL_MEMORIES_LIMIT) -> Dict[str, Any]:
    results = get_memory_instance().vector_store.list(top_k=limit)
    rows = results[0] if results and isinstance(results, list) and isinstance(results[0], list) else results or []
    return {"results": [_serialize_memory(row) for row in rows]}


@app.get("/memories", summary="Get memories")
def get_all_memories(
    request: Request,
    user_id: Optional[str] = None,
    run_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    include_agent_scoped: bool = Query(
        True,
        description=(
            "Set false to return only memories with no agent_id. Needed because this route "
            "is LIMIT-only with no ORDER BY or OFFSET: once agent-scoped rows outnumber the "
            "cap, unscoped ones become unreachable."
        ),
    ),
    top_k: Optional[int] = Query(None, ge=0, le=ALL_MEMORIES_LIMIT),
    show_expired: bool = Query(False),
    _auth=Depends(verify_auth),
    scope: MemoryScope = Depends(resolve_scope),
):
    """Retrieve stored memories. Lists all memories when no identifier is provided (admin only)."""
    try:
        user_id = scope.pin(user_id)
        if not any([user_id, run_id, agent_id]):
            auth_type = getattr(request.state, "auth_type", "none")
            if _auth is not None and _auth.role != "admin" and auth_type not in {"admin_api_key", "disabled"}:
                raise HTTPException(status_code=403, detail="Admin role required to list all memories.")
            # Admin all-memory listing is intentionally raw; scoped get_all below applies expiry visibility.
            # Unreachable for a pinned caller: `scope.pin` always yields a user_id,
            # so even an admin SSO caller takes the scoped branch and cannot read
            # the whole store through this route.
            return _list_all_memories(limit=top_k if top_k is not None else ALL_MEMORIES_LIMIT)
        # Both `$not` and the `"*"` key-exists test are handled by the vector store's
        # _build_filter_conditions, and they survive get_all untouched — it copies the
        # filter dict and validates only the three entity ids.
        filters = listing_filters(user_id, run_id, agent_id, include_agent_scoped)
        params = {"filters": filters}
        if top_k is not None:
            params["top_k"] = top_k
        params["show_expired"] = show_expired
        return get_memory_instance().get_all(**params)
    except HTTPException:
        raise
    except Exception as exc:
        raise upstream_error() from exc


def _fetch_owned_memory(memory_id: str, scope: MemoryScope) -> Dict[str, Any]:
    """Fetch a memory by id, or 404 if it does not exist OR is not the caller's.

    404 rather than 403 for the not-yours case, deliberately: a 403 would confirm
    that a given id exists, turning these routes into an id oracle. The two cases
    are indistinguishable from outside.
    """
    try:
        memory = get_memory_instance().get(memory_id)
    except Exception as exc:
        raise upstream_error() from exc

    if memory is None:
        raise HTTPException(status_code=404, detail="Memory not found.")
    owner = memory.get("user_id") if isinstance(memory, dict) else None
    if not scope.owns(owner):
        raise HTTPException(status_code=404, detail="Memory not found.")
    return memory


@app.get("/memories/{memory_id}", summary="Get a memory")
def get_memory(memory_id: str, scope: MemoryScope = Depends(resolve_scope)):
    """Retrieve a specific memory by ID."""
    return _fetch_owned_memory(memory_id, scope)


@app.post("/search", summary="Search memories")
def search_memories(search_req: SearchRequest, scope: MemoryScope = Depends(resolve_scope)):
    """Search for memories based on a query."""
    try:
        filters = dict(search_req.filters or {})
        deprecated_keys = []
        for entity_key in ("user_id", "agent_id", "run_id"):
            entity_val = getattr(search_req, entity_key, None)
            if entity_val:
                filters[entity_key] = entity_val
                deprecated_keys.append(entity_key)
        if deprecated_keys:
            logging.warning(
                "Top-level %s in /search is deprecated. Use filters={%s} instead.",
                ", ".join(deprecated_keys),
                ", ".join(f'"{k}": "..."' for k in deprecated_keys),
            )
        # Pin `filters["user_id"]`, NOT just the deprecated top-level field.
        # `filters` is a free-form passthrough dict and upstream's own deprecation
        # notice tells callers to put `user_id` in it, so pinning only the
        # top-level field would leave the documented route wide open.
        if scope.is_pinned:
            filters["user_id"] = scope.pinned_user_id
        params = {}
        if search_req.top_k is not None:
            params["top_k"] = search_req.top_k
        if search_req.threshold is not None:
            params["threshold"] = search_req.threshold
        if search_req.explain is not None:
            params["explain"] = search_req.explain
        if search_req.show_expired is not None:
            params["show_expired"] = search_req.show_expired
        return get_memory_instance().search(query=search_req.query, filters=filters, **params)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as exc:
        raise upstream_error() from exc


@app.put("/memories/{memory_id}", summary="Update a memory")
def update_memory(
    memory_id: str,
    updated_memory: MemoryUpdate,
    scope: MemoryScope = Depends(resolve_scope),
):
    """Update an existing memory."""
    _fetch_owned_memory(memory_id, scope)
    try:
        fields_set = getattr(updated_memory, "model_fields_set", getattr(updated_memory, "__fields_set__", set()))
        params = {"memory_id": memory_id}
        if "text" in fields_set:
            params["data"] = updated_memory.text
        if "metadata" in fields_set:
            params["metadata"] = updated_memory.metadata
        if "expiration_date" in fields_set:
            params["expiration_date"] = updated_memory.expiration_date
        return get_memory_instance().update(**params)
    except (ValueError, Mem0ValidationError) as e:
        raise _client_error(e) from e
    except Exception as exc:
        raise upstream_error() from exc


@app.get("/memories/{memory_id}/history", summary="Get memory history")
def memory_history(memory_id: str, scope: MemoryScope = Depends(resolve_scope)):
    """Retrieve memory history."""
    _fetch_owned_memory(memory_id, scope)
    try:
        return get_memory_instance().history(memory_id=memory_id)
    except Exception as exc:
        raise upstream_error() from exc


@app.delete("/memories/{memory_id}", summary="Delete a memory", response_model=MessageResponse)
def delete_memory(memory_id: str, scope: MemoryScope = Depends(resolve_scope)):
    """Delete a specific memory by ID."""
    _fetch_owned_memory(memory_id, scope)
    try:
        get_memory_instance().delete(memory_id=memory_id)
        return MessageResponse(message="Memory deleted successfully")
    except (ValueError, Mem0ValidationError) as e:
        raise _client_error(e) from e
    except Exception as exc:
        raise upstream_error() from exc


@app.delete("/memories", summary="Delete all memories", response_model=MessageResponse)
def delete_all_memories(
    user_id: Optional[str] = None,
    run_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    _auth=Depends(require_admin),
    scope: MemoryScope = Depends(resolve_scope),
):
    """Delete all memories for a given identifier. Requires admin role."""
    # An SSO admin is still pinned to their own scope here. Admin-ness governs
    # what you can LIST, not whose memories you can destroy — otherwise every
    # `mcp-admin` holder could wipe a colleague's memories by naming their
    # `user_id`. Operational cross-scope cleanup remains available through the
    # machine credential (ADMIN_API_KEY), which is unpinned by design.
    user_id = scope.pin(user_id)
    if not any([user_id, run_id, agent_id]):
        raise HTTPException(status_code=400, detail="At least one identifier is required.")
    try:
        params = {k: v for k, v in {"user_id": user_id, "run_id": run_id, "agent_id": agent_id}.items() if v}
        get_memory_instance().delete_all(**params)
        return MessageResponse(message="All relevant memories deleted")
    except Exception as exc:
        raise upstream_error() from exc


@app.post("/reset", summary="Reset all memories")
def reset_memory(_auth=Depends(require_admin), scope: MemoryScope = Depends(resolve_scope)):
    """Completely reset stored memories. Requires admin role."""
    # Inherently cross-scope: there is no way to express "reset only mine" here.
    # Refuse it for a pinned (SSO) caller rather than silently wiping everyone.
    if scope.is_pinned:
        raise HTTPException(
            status_code=403,
            detail="Reset affects every user's memories and is not available to SSO callers. "
            "Use DELETE /memories to clear your own.",
        )
    try:
        get_memory_instance().reset()
        return {"message": "All memories reset"}
    except Exception as exc:
        raise upstream_error() from exc


@app.get("/", summary="Redirect to the OpenAPI documentation", include_in_schema=False)
def home():
    """Redirect to the OpenAPI documentation."""
    return RedirectResponse(url="/docs")
