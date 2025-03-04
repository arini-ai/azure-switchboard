import asyncio
import logging
import random
from functools import lru_cache
from typing import AsyncGenerator, Callable, Dict

from openai.types.chat import ChatCompletion, ChatCompletionChunk
from tenacity import AsyncRetrying, stop_after_attempt

from .client import Client, Deployment, default_client_factory

logger = logging.getLogger(__name__)


class SwitchboardError(Exception):
    pass


class Switchboard:
    def __init__(
        self,
        deployments: list[Deployment],
        client_factory: Callable[[Deployment], Client] = default_client_factory,
        healthcheck_interval: int = 10,
        ratelimit_window: int = 60,  # Reset usage counters every minute
    ) -> None:
        self.deployments: Dict[str, Client] = {
            deployment.name: client_factory(deployment) for deployment in deployments
        }

        self.healthcheck_interval = healthcheck_interval
        self.ratelimit_window = ratelimit_window

        # Start background tasks if intervals are positive
        self.healthcheck_task = (
            asyncio.create_task(self.periodically_check_health())
            if self.healthcheck_interval > 0
            else None
        )

        self.ratelimit_reset_task = (
            asyncio.create_task(self.periodically_reset_usage())
            if self.ratelimit_window > 0
            else None
        )

        self.fallback_policy = AsyncRetrying(
            stop=stop_after_attempt(2),
        )

    async def periodically_check_health(self):
        """Periodically check the health of all deployments"""

        async def _check_health(client: Client):
            # splay outbound requests by a little bit
            await asyncio.sleep(random.uniform(0, 1))
            await client.check_health()

        while True:
            await asyncio.sleep(self.healthcheck_interval)
            await asyncio.gather(
                *[_check_health(client) for client in self.deployments.values()]
            )

    async def periodically_reset_usage(self):
        """Periodically reset usage counters on all clients.

        This is pretty naive but it will suffice for now."""
        while True:
            await asyncio.sleep(self.ratelimit_window)
            self.reset_usage()

    def select_deployment(self, session_id: str | None = None) -> Client:
        """
        Select a deployment using the power of two random choices algorithm.
        If session_id is provided, try to use that specific deployment first.
        """
        # Handle session-based routing first
        if session_id and session_id in self.deployments:
            client = self.deployments[session_id]
            if client.healthy:
                logger.debug(f"Using session-specific deployment: {client}")
                return client
            # If the requested client isn't healthy, fall back to load balancing
            logger.warning(
                f"Session-specific deployment {client} is unhealthy, falling back to load balancing"
            )

        # Get healthy deployments
        healthy_deployments = [c for c in self.deployments.values() if c.healthy]
        if not healthy_deployments:
            raise RuntimeError("No healthy deployments available")

        if len(healthy_deployments) == 1:
            return healthy_deployments[0]

        # Power of two random choices
        choices = random.sample(healthy_deployments, min(2, len(healthy_deployments)))

        # Select the client with the lower weight (lower weight = better choice)
        selected = min(choices, key=lambda c: c.util)
        logger.debug(f"Selected deployment {selected} with weight {selected.util}")
        return selected

    async def completion(
        self, session_id: str | None = None, **kwargs
    ) -> ChatCompletion | None:
        """
        Send a chat completion request to the selected deployment, with automatic fallback.
        """

        async for attempt in self.fallback_policy:
            with attempt:
                client = self.select_deployment(session_id)
                logger.debug(f"Sending completion request to {client}")
                return await client.completion(**kwargs)

    async def stream(
        self, session_id: str | None = None, **kwargs
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """
        Send a chat completion request to the selected deployment, with automatic fallback.
        """
        async for attempt in self.fallback_policy:
            with attempt:
                client = self.select_deployment(session_id)
                logger.debug(f"Sending streaming request to {client}")
                async for chunk in client.stream(**kwargs):
                    yield chunk

    def reset_usage(self) -> None:
        for client in self.deployments.values():
            client.reset_counters()

    def get_usage(self) -> dict[str, dict]:
        return {
            name: client.get_counters() for name, client in self.deployments.items()
        }

    def __repr__(self) -> str:
        return f"Switchboard({self.get_usage()})"

    async def close(self):
        for task in [self.healthcheck_task, self.ratelimit_reset_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
