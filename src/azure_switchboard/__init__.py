from loguru import logger as _logger

from .deployment import Deployment
from .exceptions import SwitchboardError
from .model import Model
from .switchboard import Switchboard

# As a library, do not configure sinks or emit logs by default.
# Applications can opt in explicitly.
_LOG_NAMESPACE = "azure_switchboard"
_logger.disable(_LOG_NAMESPACE)


def enable_logging() -> None:
    _logger.enable(_LOG_NAMESPACE)


def disable_logging() -> None:
    _logger.disable(_LOG_NAMESPACE)


__all__ = [
    "Deployment",
    "Model",
    "enable_logging",
    "disable_logging",
    "SwitchboardError",
    "Switchboard",
]
