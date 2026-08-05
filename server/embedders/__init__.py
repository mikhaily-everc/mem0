"""Embedder providers owned by the server rather than by the mem0 library.

Why this package exists at all: **the deployed image installs `mem0ai` from PyPI, so
nothing under this repo's top-level `mem0/` runs in production.** That is upstream's own
design, not a packaging mistake here — `server/Dockerfile` builds with `server/` as its
context, `pip install -r requirements.txt` (which pins `mem0ai`), then `COPY . .`. Only
`server/dev.Dockerfile`, whose context is the repo root, `pip install -e .`s the local
library. The Bazel build reproduces the production Dockerfile: `@pypi_mem0//mem0ai` is a
wheel, `@mem0//server` is source.

So a fix committed to `mem0/embeddings/*.py` is an upstream *contribution*; it reaches
this server only when it is released to PyPI. Anything the server needs NOW has to live
here, where it is built from source.

`install_overrides()` registers these classes into `EmbedderFactory.provider_to_class`,
which is a plain dict of provider name -> dotted path resolved lazily by
`mem0.utils.factory.load_class`. Rebinding an entry is the factory's own extension
mechanism; it is not a monkeypatch of library internals, and it survives a wheel bump as
long as that dict and `AWSBedrockEmbedding.__init__` keep their shape.
"""

from embedders.aws_bedrock import AsymmetricAWSBedrockEmbedding

__all__ = ["AsymmetricAWSBedrockEmbedding", "install_overrides"]


def _dotted_path(cls: type) -> str:
    """`load_class`-compatible path for cls, derived rather than written out.

    A literal string here would silently stop matching the day the module or class is
    renamed — `load_class` raises only when the provider is actually selected, which for
    an embedder is the first add or search, i.e. after the pod is Ready.
    """
    return f"{cls.__module__}.{cls.__qualname__}"


def install_overrides() -> None:
    """Point mem0's embedder factory at the server-owned providers.

    Idempotent, and safe to call before knowing which provider the deployment selected —
    it rebinds a name in a lookup table and imports nothing until that name is used.
    """
    from mem0.utils.factory import EmbedderFactory

    EmbedderFactory.provider_to_class["aws_bedrock"] = _dotted_path(AsymmetricAWSBedrockEmbedding)
