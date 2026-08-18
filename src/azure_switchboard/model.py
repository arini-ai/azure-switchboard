from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from opentelemetry import trace

from .exceptions import SwitchboardError

if TYPE_CHECKING:
    from .foundry import Foundry


@dataclass(frozen=True)
class UtilStats:
    util: float
    tpm: str
    rpm: str


class ModelBase:
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
        self.name = name
        self.tpm_limit = tpm
        self.rpm_limit = rpm
        self.endpoint = endpoint
        self.default_cooldown = default_cooldown

        self.tpm_usage: int = 0
        self.rpm_usage: int = 0
        self.cooldown_until: float = 0
        self.last_reset: float = 0

        self._foundry: Foundry | None = None

    def bind(self, foundry: Foundry) -> None:
        if self._foundry is not None and self._foundry is not foundry:
            raise SwitchboardError(
                f"{self.name} is already bound to {self._foundry.name}"
            )
        self._foundry = foundry

    @property
    def foundry(self) -> Foundry:
        if self._foundry is None:
            raise SwitchboardError(
                f"{self.name} is not bound to a foundry; pass it to Foundry(models=[...])"
            )
        return self._foundry

    @property
    def url(self) -> str | None:
        """Where this deployment is served.

        An explicit endpoint wins. Otherwise it is the resource's base with
        this API's path appended — and None when the resource has no base, so
        the SDK falls back to its vendor default.
        """
        raise NotImplementedError

    def mark_down(self, seconds: float = 0.0) -> None:
        self.cooldown_until = time.time() + (seconds or self.default_cooldown)

    def mark_up(self) -> None:
        self.cooldown_until = 0

    def is_healthy(self) -> bool:
        return self.util < 1

    def is_cooling(self) -> bool:
        return time.time() < self.cooldown_until

    @property
    def util(self) -> float:
        """
        Calculate the load weight of this model as a value between 0 and 1.
        Lower weight means this model is a better choice for new requests.
        """
        # full utilization while cooling down keeps us out of selection. A
        # cooling foundry takes every model on it out with it: a connection
        # error means the host is unreachable, not that one model is busy.
        if self.is_cooling() or (
            self._foundry is not None and self._foundry.is_cooling()
        ):
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
        self.last_reset = time.time()

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
        stats = self.stats()
        return (
            f"{type(self).__name__}<{self.name}>"
            f"(util={stats.util} tpm='{stats.tpm}' rpm='{stats.rpm}')"
        )
