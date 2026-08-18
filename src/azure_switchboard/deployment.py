from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from loguru import logger
from opentelemetry import trace

from .exceptions import SwitchboardError

if TYPE_CHECKING:
    from .resource import Resource


@dataclass(frozen=True)
class UtilStats:
    util: float
    tpm: str
    rpm: str


class Cooldown:
    """Timed removal from selection.

    Both a deployment and the resource hosting it can be taken out, and the
    mechanics are the same either way. What differs is what a cooldown
    implicates: one deployment's quota, or every deployment on an unreachable
    host.
    """

    def __init__(self, default_cooldown: float = 10.0) -> None:
        self.default_cooldown = default_cooldown
        self.cooldown_until: float = 0

    def mark_down(self, seconds: float = 0.0) -> None:
        self.cooldown_until = time.time() + (seconds or self.default_cooldown)

    def mark_up(self) -> None:
        self.cooldown_until = 0

    def is_cooling(self) -> bool:
        return time.time() < self.cooldown_until


class ModelDeployment(Cooldown):
    """A model deployment: its quota, its live utilization, and its cooldown.

    Subclasses bind an API spec to it, supplying the calls that speak to an
    upstream. A model is declared standalone and bound to a Foundry when it is
    registered; it is not callable before then.
    """

    def __init__(
        self,
        name: str,
        tpm: int = 0,
        rpm: int = 0,
        *,
        endpoint: str | None = None,
        default_cooldown: float = 10.0,
    ):
        super().__init__(default_cooldown)
        self.name = name
        self.tpm_limit = tpm
        self.rpm_limit = rpm
        self.endpoint = endpoint

        self.tpm_usage: int = 0
        self.rpm_usage: int = 0

        self._resource: Resource | None = None

    def bind(self, resource: Resource) -> None:
        if self._resource is not None and self._resource is not resource:
            raise SwitchboardError(
                f"{self.name} is already bound to {self._resource.name}"
            )
        self._resource = resource

    @property
    def resource(self) -> Resource:
        if self._resource is None:
            raise SwitchboardError(
                f"{self.name} is not bound to a resource; pass it to Foundry(models=[...])"
            )
        return self._resource

    @property
    def url(self) -> str | None:
        """Where this deployment is served.

        An explicit endpoint wins. Otherwise it is the resource's base with
        this API's path appended — and None when the resource has no base, so
        the SDK falls back to its vendor default.
        """
        raise NotImplementedError

    # Each API's SDK raises its own classes for these, so subclasses name
    # theirs and _handle_error is written once.
    ratelimit_error: ClassVar[type[Exception]]
    timeout_error: ClassVar[type[Exception]]
    connection_error: ClassVar[type[Exception]]

    def _handle_error(self, exc: Exception, op: str, log=logger) -> None:
        """Scope a cooldown to what the error actually implicates.

        A 429 is one deployment's quota; a connection error is the whole
        resource. Timeouts during upstream-wide slowdowns are uncorrelated with
        which deployment was chosen, so they cool nothing.

        The timeout branch has to come before the connection one: in both SDKs
        the timeout error subclasses the connection error, so testing
        connection first would cool the whole resource on every timeout.
        """
        if isinstance(exc, self.ratelimit_error):
            log.exception(f"Marking down model for rate limit on {op}")
            self.mark_down()
        elif isinstance(exc, self.timeout_error):
            log.warning(f"Upstream timeout on {op}; not marking down")
        elif isinstance(exc, self.connection_error):
            log.exception(f"Marking down resource for connection error on {op}")
            self.resource.mark_down()

    def is_healthy(self) -> bool:
        return self.util < 1

    @property
    def util(self) -> float:
        """
        Calculate the load weight of this model as a value between 0 and 1.
        Lower weight means this model is a better choice for new requests.
        """
        # full utilization while cooling down keeps us out of selection. A
        # cooling resource takes every model on it out with it: a connection
        # error means the host is unreachable, not that one model is busy.
        if self.is_cooling() or self.resource.is_cooling():
            return 1

        # Azure buckets tokens on a non-sliding 60 second window
        token_util = self.tpm_usage / self.tpm_limit if self.tpm_limit > 0 else 0

        # Azure allocates RPM at a ratio of 6:1000 to TPM
        # Limits are enforced proportionally to the 60s limit in 1-10s sliding windows
        request_util = self.rpm_usage / self.rpm_limit if self.rpm_limit > 0 else 0

        # Add a small random factor to prevent oscillation
        return round(max(token_util, request_util) + random.uniform(0, 0.01), 3)

    def reset_usage(self) -> None:
        """Call periodically to reset usage counters"""

        self.tpm_usage = 0
        self.rpm_usage = 0

    def stats(self) -> UtilStats:
        return UtilStats(
            util=self.util,
            tpm=f"{self.tpm_usage}/{self.tpm_limit}",
            rpm=f"{self.rpm_usage}/{self.rpm_limit}",
        )

    def spend_request(self, n: int = 1) -> None:
        self.rpm_usage += n

    def spend_tokens(self, n: int) -> None:
        self.tpm_usage += n

    def _record_token_details(
        self, *, cached: int | None = None, reasoning: int | None = None
    ) -> None:
        """Record token details on the active span.

        Both providers report cached and reasoning tokens, under different
        field names; subclasses extract, this normalizes onto gen_ai.* attrs.
        """
        span = trace.get_current_span()
        if cached:
            span.set_attribute("gen_ai.usage.cached_tokens", cached)
        if reasoning:
            span.set_attribute("gen_ai.usage.reasoning_tokens", reasoning)

    def __repr__(self) -> str:
        # util reads the resource's cooldown, which an unregistered deployment
        # has no way to reach. repr has to describe one anyway: raising here
        # would break debuggers and swallow whatever error was being formatted.
        if self._resource is None:
            return f"{type(self).__name__}<{self.name}>(unbound)"
        stats = self.stats()
        return (
            f"{type(self).__name__}<{self.name}>"
            f"(util={stats.util} tpm='{stats.tpm}' rpm='{stats.rpm}')"
        )
