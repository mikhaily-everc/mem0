"""Tests for the server-owned Bedrock embedder and its factory registration.

Every assertion here is about a failure that is INVISIBLE until production: a wrong
`input_type` silently degrades recall with no error at all, a missing `output_dimension`
returns 1536 dims that pgvector rejects on insert, and reading `[0]` off v4's dict response
yields the string "float" which pgvector accepts as a value and rejects as a vector. None of
it shows up in a green rollout.

stdlib `unittest` rather than pytest, deliberately — same reason as test_auth0.py: adding a
`[dependency-groups]` block to tools/mcp/mem0/server/pyproject.toml made rules_py's uv hub
emit only that group, and every `@pypi_mem0//<pkg>` label failed at analysis.

boto3 is imported for real (it is a declared dep of `:server`), but no client is ever built:
`boto3.client` is patched before the embedder is constructed, so nothing here needs AWS
credentials, a region that exists, or a network.
"""

from __future__ import annotations

import io
import json
import unittest
from typing import Any, Optional
from unittest import mock

from mem0.configs.embeddings.base import BaseEmbedderConfig
from mem0.utils.factory import EmbedderFactory, load_class

import embedders
from embedders.aws_bedrock import AsymmetricAWSBedrockEmbedding

V4_MODEL = "cohere.embed-v4:0"
V3_MODEL = "cohere.embed-english-v3"
TITAN_MODEL = "amazon.titan-embed-text-v2:0"
DIMS = 1024


class _FakeBedrockClient:
    """Records the request body and replays a canned response.

    `invoke_model` returns `{"body": <file-like>}` because that is what botocore's
    StreamingBody looks like to the embedder — it calls `.get("body").read()`.
    """

    def __init__(self, response: Any) -> None:
        self._response = response
        self.calls: list[dict] = []

    def invoke_model(self, *, body: str, modelId: str, accept: str, contentType: str) -> dict:  # noqa: N803
        self.calls.append({"body": json.loads(body), "modelId": modelId})
        return {"body": io.BytesIO(json.dumps(self._response).encode())}

    @property
    def last_body(self) -> dict:
        return self.calls[-1]["body"]


def _embedder(
    model: str,
    response: Any,
    embedding_dims: Optional[int] = DIMS,
) -> tuple[AsymmetricAWSBedrockEmbedding, _FakeBedrockClient]:
    client = _FakeBedrockClient(response)
    with mock.patch("boto3.client", return_value=client):
        instance = AsymmetricAWSBedrockEmbedding(
            BaseEmbedderConfig(model=model, embedding_dims=embedding_dims, aws_region="us-east-1")
        )
    return instance, client


class FactoryRegistrationTest(unittest.TestCase):
    """The registration is the load-bearing half — the class is dead code without it."""

    def test_install_overrides_rebinds_aws_bedrock(self) -> None:
        embedders.install_overrides()
        self.assertIs(
            load_class(EmbedderFactory.provider_to_class["aws_bedrock"]),
            AsymmetricAWSBedrockEmbedding,
            "the factory must resolve aws_bedrock to the server's class, not the wheel's",
        )

    def test_registered_path_is_importable(self) -> None:
        """A rename would break `load_class` only when a memory is added or searched.

        By then the pod is Ready and the store is down, so resolve the dotted path here
        instead — this is the cheapest place to notice.
        """
        embedders.install_overrides()
        dotted = EmbedderFactory.provider_to_class["aws_bedrock"]
        self.assertEqual(dotted, "embedders.aws_bedrock.AsymmetricAWSBedrockEmbedding")
        self.assertTrue(callable(load_class(dotted)))

    def test_the_factory_builds_it_end_to_end(self) -> None:
        """What server/main.py's DEFAULT_CONFIG actually does, minus the network."""
        embedders.install_overrides()
        with mock.patch("boto3.client", return_value=_FakeBedrockClient({"embeddings": [[0.0] * DIMS]})):
            instance = EmbedderFactory.create(
                "aws_bedrock",
                {"model": V4_MODEL, "embedding_dims": DIMS, "aws_region": "us-east-1"},
                vector_config=None,
            )
        self.assertIsInstance(instance, AsymmetricAWSBedrockEmbedding)


class CohereInputTypeTest(unittest.TestCase):
    """Defect 1: memory_action was accepted and dropped, input_type hardcoded."""

    def test_search_asks_for_a_query_vector(self) -> None:
        instance, client = _embedder(V4_MODEL, {"embeddings": {"float": [[0.1] * DIMS]}})
        instance.embed("where does the gateway run", "search")
        self.assertEqual(client.last_body["input_type"], "search_query")

    def test_add_and_update_ask_for_a_document_vector(self) -> None:
        for action in ("add", "update"):
            with self.subTest(action=action):
                instance, client = _embedder(V4_MODEL, {"embeddings": {"float": [[0.1] * DIMS]}})
                instance.embed("the gateway runs in mlrc-gradle-cache", action)
                self.assertEqual(client.last_body["input_type"], "search_document")

    def test_an_unlabelled_call_is_treated_as_a_document(self) -> None:
        """Matches upstream's assumption, so an unlabelled caller is no worse off."""
        instance, client = _embedder(V4_MODEL, {"embeddings": {"float": [[0.1] * DIMS]}})
        instance.embed("some text")
        self.assertEqual(client.last_body["input_type"], "search_document")

    def test_batch_embedding_inherits_the_action(self) -> None:
        """EmbeddingBase.embed_batch routes through embed(), so this must not regress."""
        instance, client = _embedder(V4_MODEL, {"embeddings": {"float": [[0.1] * DIMS]}})
        instance.embed_batch(["a", "b"], memory_action="search")
        self.assertEqual([call["body"]["input_type"] for call in client.calls], ["search_query"] * 2)


class OutputDimensionTest(unittest.TestCase):
    """Defect 2: v4 defaults to 1536 dims against a pgvector column fixed at 1024."""

    def test_v4_is_sent_the_configured_dimension(self) -> None:
        instance, client = _embedder(V4_MODEL, {"embeddings": {"float": [[0.1] * DIMS]}})
        instance.embed("text", "add")
        self.assertEqual(client.last_body["output_dimension"], DIMS)
        self.assertEqual(
            client.last_body["embedding_types"],
            ["float"],
            "v4 keys its response by the requested types, so this is what makes the float branch apply",
        )

    def test_v3_is_not(self) -> None:
        """v3 has no such parameter and REJECTS it — sending it breaks the older model."""
        instance, client = _embedder(V3_MODEL, {"embeddings": [[0.1] * DIMS]})
        instance.embed("text", "add")
        self.assertNotIn("output_dimension", client.last_body)
        self.assertNotIn("embedding_types", client.last_body)

    def test_no_configured_dimension_means_no_parameter(self) -> None:
        instance, client = _embedder(V4_MODEL, {"embeddings": {"float": [[0.1] * DIMS]}}, embedding_dims=None)
        instance.embed("text", "add")
        self.assertNotIn("output_dimension", client.last_body)


class ResponseShapeTest(unittest.TestCase):
    """Defect 3: `[0]` on v4's dict response reads a KEY, and pgvector accepts a string."""

    def test_v4_dict_response_yields_the_vector(self) -> None:
        vector = [0.5] * DIMS
        instance, _ = _embedder(V4_MODEL, {"embeddings": {"float": [vector]}})
        self.assertEqual(instance.embed("text", "add"), vector)

    def test_v3_list_response_yields_the_vector(self) -> None:
        vector = [0.25] * DIMS
        instance, _ = _embedder(V3_MODEL, {"embeddings": [vector]})
        self.assertEqual(instance.embed("text", "add"), vector)

    def test_an_unexpected_embedding_type_key_still_resolves(self) -> None:
        """Asking for float and being answered in another type is better than a crash."""
        vector = [1.0] * DIMS
        instance, _ = _embedder(V4_MODEL, {"embeddings": {"int8": [vector]}})
        self.assertEqual(instance.embed("text", "add"), vector)

    def test_an_empty_dict_raises_instead_of_returning_a_key(self) -> None:
        instance, _ = _embedder(V4_MODEL, {"embeddings": {}})
        with self.assertRaises(ValueError):
            instance.embed("text", "add")

    def test_a_missing_embeddings_field_raises(self) -> None:
        instance, _ = _embedder(V4_MODEL, {"message": "throttled"})
        with self.assertRaises(ValueError):
            instance.embed("text", "add")

    def test_a_flat_vector_raises_instead_of_returning_one_float(self) -> None:
        """The same defect as returning a dict key, one level down. `[0]` on a list OF
        vectors is the vector; `[0]` on a flat vector is a float, and pgvector accepts
        that too — it stores a 1-dimensional embedding and every later similarity score
        is quietly meaningless. Surfaced by `ty`, which pointed out that the value being
        subscripted was never typed at all."""
        instance, _ = _embedder(V4_MODEL, {"embeddings": {"float": [0.1] * DIMS}})
        with self.assertRaises(ValueError) as caught:
            instance.embed("text", "add")
        self.assertIn("flat embedding", str(caught.exception))

    def test_a_non_list_embeddings_field_raises(self) -> None:
        """A provider error body where the vectors should be, e.g. a bare string."""
        instance, _ = _embedder(V4_MODEL, {"embeddings": "throttled"})
        with self.assertRaises(ValueError):
            instance.embed("text", "add")


class NonCohereProvidersUnchangedTest(unittest.TestCase):
    """Titan is what is deployed today; this override must not touch it."""

    def test_titan_delegates_to_the_library(self) -> None:
        instance, client = _embedder(TITAN_MODEL, {"embedding": [0.3] * DIMS})
        result = instance.embed("text", "search")
        self.assertEqual(result, [0.3] * DIMS)
        self.assertIn("inputText", client.last_body)
        self.assertNotIn(
            "input_type",
            client.last_body,
            "Titan is symmetric and takes no input_type — sending one would be an API error",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
