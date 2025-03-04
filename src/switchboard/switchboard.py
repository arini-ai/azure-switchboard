import asyncio
import logging
import random
from typing import AsyncGenerator, Callable, Dict

from openai.types.chat import ChatCompletion, ChatCompletionChunk
from tenacity import AsyncRetrying, stop_after_attempt

from .client import Client, Deployment, default_client_factory

logger = logging.getLogger(__name__)


class Switchboard:
    def __init__(
        self,
        deployments: list[Deployment],
        client_factory: Callable[[Deployment], Client] = default_client_factory,
        healthcheck_interval: int = 10,
        usage_reset_interval: int = 60,  # Reset usage counters every minute
    ) -> None:
        self.deployments: Dict[str, Client] = {
            deployment.name: client_factory(deployment) for deployment in deployments
        }

        self.healthcheck_interval = healthcheck_interval
        self.usage_reset_interval = usage_reset_interval

        # Start background tasks if intervals are positive
        self.healthcheck_task = (
            asyncio.create_task(self.periodically_check_health())
            if self.healthcheck_interval > 0
            else None
        )

        self.usage_reset_task = (
            asyncio.create_task(self.periodically_reset_counters())
            if self.usage_reset_interval > 0
            else None
        )

        self.fallback_policy = AsyncRetrying(
            stop=stop_after_attempt(2),
        )

    async def periodically_check_health(self):
        """Periodically check the health of all deployments"""

        async def check_health(client: Client):
            try:
                # splay outbound requests by a little bit
                await asyncio.sleep(random.uniform(0, 1))
                await client.check_liveness()
            except Exception:
                logger.exception(f"{client}: healthcheck failed")

        while True:
            await asyncio.sleep(self.healthcheck_interval)
            await asyncio.gather(
                *[check_health(client) for client in self.deployments.values()]
            )

    async def periodically_reset_counters(self):
        """Periodically reset usage counters on all clients"""
        while True:
            await asyncio.sleep(self.usage_reset_interval)
            for client in self.deployments.values():
                await client.reset_counters()

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
        selected = min(choices, key=lambda c: c.utilization)
        logger.debug(
            f"Selected deployment {selected} with weight {selected.utilization}"
        )
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
    ) -> AsyncGenerator[ChatCompletionChunk, None] | None:
        """
        Send a chat completion request to the selected deployment, with automatic fallback.
        """

        async for attempt in self.fallback_policy:
            with attempt:
                client = self.select_deployment(session_id)
                logger.debug(f"Sending streaming request to {client}")
                return client.stream(**kwargs)

    def __getattr__(self, name: str):
        client = self.select_deployment()
        return getattr(client, name)

    def usage(self):
        return {
            name: {
                "tokens": deployment.ratelimit_tokens,
                "requests": deployment.ratelimit_requests,
            }
            for name, deployment in self.deployments.items()
        }

    def __str__(self):
        return f"Switchboard(deployments={self.deployments})"

    async def close(self):
        """Clean up resources when shutting down"""
        if self.healthcheck_task:
            self.healthcheck_task.cancel()
            try:
                await self.healthcheck_task
            except asyncio.CancelledError:
                pass

        if self.usage_reset_task:
            self.usage_reset_task.cancel()
            try:
                await self.usage_reset_task
            except asyncio.CancelledError:
                pass
