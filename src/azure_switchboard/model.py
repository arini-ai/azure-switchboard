from dataclasses import dataclass
import random
import time


@dataclass(frozen=True)
class UtilStats:
    util: float
    tpm: str
    rpm: str


class Model:
    """Runtime state of a model in a deployment"""

    def __init__(
        self,
        name: str,
        tpm: int = 0,
        rpm: int = 0,
        default_cooldown: float = 10.0,
    ):
        self.name = name
        self.tpm_limit = tpm
        self.rpm_limit = rpm
        self.default_cooldown = default_cooldown

        self.tpm_usage: int = 0
        self.rpm_usage: int = 0
        self.cooldown_until: float = 0
        self.last_reset: float = 0

    def mark_down(self, seconds: float = 0.0) -> None:
        self.cooldown_until = time.time() + (seconds or self.default_cooldown)

    def mark_up(self) -> None:
        self.cooldown_until = 0

    def is_healthy(self) -> bool:
        """
        Check if the model is healthy based on utilization.
        """
        return self.util < 1

    def is_cooling(self) -> bool:
        return time.time() < self.cooldown_until

    @property
    def util(self) -> float:
        """
        Calculate the load weight of this client as a value between 0 and 1.
        Lower weight means this client is a better choice for new requests.
        """
        # return full utilization if we're cooling down to avoid selection
        if self.is_cooling():
            return 1

        # Calculate token utilization (as a percentage of max)
        # Azure buckets tokens on a non-sliding 60 second window
        token_util = self.tpm_usage / self.tpm_limit if self.tpm_limit > 0 else 0

        # Azure allocates RPM at a ratio of 6:1000 to TPM
        # Limits are enforced proportionally to the 60s limit in 1-10s sliding windows
        request_util = self.rpm_usage / self.rpm_limit if self.rpm_limit > 0 else 0

        # Use the higher of the two utilizations as the weight
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

    def __repr__(self) -> str:
        stats = self.stats()
        return (
            f"Model<{self.name}>("
            f"util={stats.util} tpm='{stats.tpm}' rpm='{stats.rpm}'"
            ")"
        )
