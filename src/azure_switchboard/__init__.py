from loguru import logger as _logger

from .deployment import Deployment
from .exceptions import SwitchboardError
from .model import Model
from .switchboard import Switchboard

# As a library, do not configure sinks or emit logs by default.
# Applications can opt in explicitly.
_logger.disable("azure_switchboard")


__all__ = [
    "Deployment",
    "Model",
    "SwitchboardError",
    "Switchboard",
]
