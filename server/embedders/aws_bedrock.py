"""Bedrock embedder that honours Cohere's query/document asymmetry.

This is the server-owned twin of the fix committed to `mem0/embeddings/aws_bedrock.py`.
Both are needed and neither is redundant: the library copy is the upstream contribution,
this one is what the deployed image actually executes (see `embedders/__init__.py` for
why the image runs the PyPI wheel rather than this repo's `mem0/`).

Three defects in the stock Bedrock embedder make `cohere.embed-v4:0` unusable:

1. `embed()` accepts `memory_action` ("add"/"search"/"update") and DISCARDS it, while
   `_get_embedding` hardcodes `input_type="search_document"`. Cohere is an ASYMMETRIC
   embedder — it builds a different vector for a short question than for the long
   document that answers it — so every query was embedded as though it were a document,
   discarding the single advantage Cohere has over a symmetric model like Titan.
   Measured on a 572-row corpus, honouring the asymmetry widens the margin between a
   relevant row and an irrelevant one by ~27% (+0.382 -> +0.486). The gain is almost
   entirely in pushing irrelevant content AWAY (+0.015 mean on the correct document,
   occasionally negative), which is the failure mode that actually matters for recall.

2. `embed-v4` is never sent `output_dimension`, so it returns its 1536-dim default. A
   pgvector column's width is fixed at table creation, making that a hard insert failure
   rather than a degradation. v3 has no such parameter (fixed 1024) and REJECTS it, so
   this is gated on the model id, not sent unconditionally.

3. `embed-v4` answers `{"embeddings": {"float": [[...]]}}` where v3 answers
   `{"embeddings": [[...]]}`. Indexing `[0]` into the v4 shape reads a dict KEY — a
   string — and hands it to pgvector as a vector.

Only the Cohere path is reimplemented. Titan and every other Bedrock provider delegate to
the wheel's own `_get_embedding`, so the currently deployed
`amazon.titan-embed-text-v2:0` behaviour is untouched by this file.
"""

import json
from typing import Literal, Optional

from mem0.embeddings.aws_bedrock import AWSBedrockEmbedding

# Cohere's embed-v4 accepts 256/512/1024/1536 and defaults to 1536; v3 is fixed at 1024
# and errors on the parameter. Keyed on the model id because there is no capability
# endpoint to ask.
_OUTPUT_DIMENSION_MODELS = ("v4",)


class AsymmetricAWSBedrockEmbedding(AWSBedrockEmbedding):
    """`AWSBedrockEmbedding` with the Cohere request/response shape corrected."""

    def _supports_output_dimension(self) -> bool:
        return any(marker in self.config.model for marker in _OUTPUT_DIMENSION_MODELS)

    def _get_embedding(self, text, memory_action: Optional[str] = None):
        provider = self.config.model.split(".")[0]
        if provider != "cohere":
            # Titan and friends are symmetric and take no input_type — leave them to the
            # library so this override cannot regress the model in production today.
            return super()._get_embedding(text)

        input_body = {
            # The whole point of the override: "add" and "update" store documents,
            # "search" issues a query. `None` means an unlabelled call from some other
            # code path, and treating that as a document matches upstream's assumption.
            "input_type": "search_query" if memory_action == "search" else "search_document",
            "texts": [text],
        }
        if self.config.embedding_dims is not None and self._supports_output_dimension():
            input_body["output_dimension"] = self.config.embedding_dims
            # v4 keys its response by the requested types, so asking for float is what
            # makes the {"float": [...]} branch below the shape we get.
            input_body["embedding_types"] = ["float"]

        try:
            response = self.client.invoke_model(
                body=json.dumps(input_body),
                modelId=self.config.model,
                accept="application/json",
                contentType="application/json",
            )
            response_body = json.loads(response.get("body").read())
            return self._extract_cohere_vector(response_body)
        except Exception as e:
            raise ValueError(f"Error getting embedding from AWS Bedrock: {e}")

    @staticmethod
    def _extract_cohere_vector(response_body: dict):
        """Read one vector out of either Cohere response shape.

        v3: `{"embeddings": [[...]]}`.  v4: `{"embeddings": {"float": [[...]]}}`.
        Raises rather than returning a dict key, because pgvector accepts a string and
        fails much later with an error naming neither the model nor this function.
        """
        embeddings = response_body.get("embeddings")
        if isinstance(embeddings, dict):
            vectors = embeddings.get("float") or next(iter(embeddings.values()), None)
            if not vectors:
                raise ValueError(f"no embedding vectors in response keys {list(embeddings)}")
            embeddings = vectors
        if not embeddings:
            raise ValueError("empty embeddings in Bedrock response")
        return embeddings[0]

    def embed(self, text, memory_action: Optional[Literal["add", "search", "update"]] = None):
        """Forward `memory_action` instead of dropping it.

        `EmbeddingBase.embed_batch` routes through here too, so batch calls inherit the
        same query/document labelling.
        """
        return self._get_embedding(text, memory_action=memory_action)
