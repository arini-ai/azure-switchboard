from loguru import logger as _logger
from openai.types.chat import ParsedChatCompletion

from .anthropic_deployment import AnthropicDeployment
from .exceptions import SwitchboardError
from .foundry import Foundry
from .openai_deployment import OpenAIDeployment
from .switchboard import Switchboard

# As a library, do not configure sinks or emit logs by default.
# Applications can opt in explicitly.
_logger.disable("azure_switchboard")


__all__ = [
    "AnthropicDeployment",
    "Foundry",
    "OpenAIDeployment",
    "ParsedChatCompletion",
    "SwitchboardError",
    "Switchboard",
]
