"""LLM providers owned by the server rather than by the mem0 library.

Same reason as `server/embedders/`: the deployed image installs `mem0ai` from PyPI, so
nothing under this repo's top-level `mem0/` runs in production. A fix there is an upstream
contribution; anything the server needs now lives here, where it is built from source.

`install_overrides()` rebinds entries in `LlmFactory.provider_to_class`, which is a plain
dict of provider name -> dotted path resolved lazily by `mem0.utils.factory.load_class`.
"""

from llms.aws_bedrock import ConstrainedAWSBedrockLLM

__all__ = ["ConstrainedAWSBedrockLLM", "install_overrides"]


def _dotted_path(cls: type) -> str:
    """`load_class`-compatible path for cls, derived rather than written out."""
    return f"{cls.__module__}.{cls.__qualname__}"


def install_overrides() -> None:
    """Point mem0's LLM factory at the server-owned providers.

    Idempotent, and imports nothing until the provider is actually selected.
    """
    from mem0.utils.factory import LlmFactory

    # provider_to_class values are (dotted_path, config_class) pairs for LLMs — unlike the
    # embedder factory, whose values are bare strings. Preserve the existing config class so
    # AWSBedrockConfig keeps being constructed.
    _, config_cls = LlmFactory.provider_to_class["aws_bedrock"]
    LlmFactory.provider_to_class["aws_bedrock"] = (
        _dotted_path(ConstrainedAWSBedrockLLM),
        config_cls,
    )
