from loguru import logger as _logger
from openai.types.chat import ParsedChatCompletion

from .openai_api import OpenAIConfig
from .exceptions import SwitchboardError
from .anthropic_api import AnthropicConfig
from .model import Model
from .switchboard import Switchboard

# As a library, do not configure sinks or emit logs by default.
# Applications can opt in explicitly.
_logger.disable("azure_switchboard")


__all__ = [
    "AnthropicConfig",
    "Model",
    "OpenAIConfig",
    "ParsedChatCompletion",
    "SwitchboardError",
    "Switchboard",
]
