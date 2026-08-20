import logging as _logging

from openai.types.chat import ParsedChatCompletion

from .anthropic_deployment import AnthropicDeployment
from .exceptions import SwitchboardError
from .resource import Foundry, Resource
from .openai_deployment import OpenAIDeployment
from .switchboard import Switchboard

# As a library, do not configure handlers or emit logs by default. An
# application opts in by configuring the "azure_switchboard" logger.
_logging.getLogger(__name__).addHandler(_logging.NullHandler())


__all__ = [
    "AnthropicDeployment",
    "Foundry",
    "OpenAIDeployment",
    "Resource",
    "ParsedChatCompletion",
    "SwitchboardError",
    "Switchboard",
]
