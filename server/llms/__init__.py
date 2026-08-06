"""LLM providers owned by the server rather than by the mem0 library.

Same reason as `server/embedders/`: the deployed image installs `mem0ai` from PyPI, so nothing
under this repo's top-level `mem0/` runs in production. A fix there is an upstream
contribution; anything the server needs now lives here, where it is built from source.

Both overrides exist to solve ONE problem — mem0 asks for JSON with
`response_format={"type": "json_object"}`, which is a request rather than a guarantee
(~19% unparseable on this corpus, and silently so). Each provider needs a different mechanism
to turn that into a real grammar constraint:

    aws_bedrock   Converse `outputConfig.textFormat`; 4.6-and-older Claude only
    openai        the strict `json_schema` response_format, forwarded by bifrost to Azure
                  and translated to `responseJsonSchema` for Gemini

`install_overrides()` rebinds entries in `LlmFactory.provider_to_class`, whose values are
`(dotted_path, config_class)` TUPLES — unlike the embedder factory's bare strings. The config
class is preserved so provider-specific config construction keeps working.
"""

from llms.aws_bedrock import ConstrainedAWSBedrockLLM
from llms.openai_schema import SchemaEnforcingOpenAILLM

__all__ = [
    "ConstrainedAWSBedrockLLM",
    "SchemaEnforcingOpenAILLM",
    "install_overrides",
]

# provider name -> replacement class. Both are registered unconditionally: registration only
# rebinds a lookup-table entry and imports nothing until that provider is actually selected,
# so gating on MEM0_LLM_PROVIDER would add a way for the two to disagree.
_OVERRIDES = {
    "aws_bedrock": ConstrainedAWSBedrockLLM,
    "openai": SchemaEnforcingOpenAILLM,
}


def _dotted_path(cls: type) -> str:
    """`load_class`-compatible path for cls, derived rather than written out.

    A literal string would silently stop matching the day the module or class is renamed, and
    `load_class` raises only when the provider is selected — i.e. on the first add or search,
    after the pod is already Ready.
    """
    return f"{cls.__module__}.{cls.__qualname__}"


def install_overrides() -> None:
    """Point mem0's LLM factory at the server-owned providers. Idempotent."""
    from mem0.utils.factory import LlmFactory

    for provider, cls in _OVERRIDES.items():
        _, config_cls = LlmFactory.provider_to_class[provider]
        LlmFactory.provider_to_class[provider] = (_dotted_path(cls), config_cls)
