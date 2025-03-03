import asyncio
import logging
import random
from typing import Annotated, AsyncGenerator, Callable, List

from openai import AsyncAzureOpenAI, AsyncStream
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from pydantic import BaseModel, Field
from tenacity import AsyncRetrying, RetryError, stop_after_attempt

logger = logging.getLogger("switchboard")


class Deployment(BaseModel):
    """Metadata about the Azure deployment"""

    name: str
    api_base: str
    api_key: str
    api_version: str = "2024-10-21"
    timeout: float = 30.0
    max_tpm: Annotated[int, Field(description="TPM Ratelimit")] = 0
    max_rpm: Annotated[int, Field(description="RPM Ratelimit")] = 0
    healthcheck_interval: int = 30


class Client:
    """Runtime state of a deployment"""

    def __init__(self, config: Deployment, client: AsyncAzureOpenAI):
        self.name = config.name
        self.config = config
        self.client = client
        self.healthy = True

        self.token_usage = 0
        self.request_usage = 0

    def __str__(self):
        return f"Client(name={self.name}, healthy={self.healthy})"

    async def check_liveness(self):
        try:
            logger.debug(f"{self}: checking liveness")
            await self.client.models.list()
            self.healthy = True
        except Exception:
            logger.exception(f"{self}: liveness check failed")
            self.healthy = False

    async def chat_completion(
        self, **kwargs
    ) -> ChatCompletion | AsyncStream[ChatCompletionChunk]:
        response = await self.client.chat.completions.create(
            **kwargs, timeout=self.config.timeout
        )
        return response


class Switchboard:
    def __init__(
        self,
        deployments: List[Deployment],
        client_factory: Callable[[Deployment], Client] | None = None,
        healthcheck_interval: int = 10,
    ):
        self.client_factory = client_factory or self._default_client_factory
        self.deployments = {
            deployment.name: self.client_factory(deployment)
            for deployment in deployments
        }

        self.healthcheck_interval = healthcheck_interval
        self.healthcheck_task = (
            asyncio.create_task(self.run_healthchecks())
            if self.healthcheck_interval > 0
            else None
        )

        self.fallback_policy = AsyncRetrying(
            stop=stop_after_attempt(2),
        )

    def _default_client_factory(self, deployment: Deployment) -> Client:
        return Client(
            config=deployment,
            client=AsyncAzureOpenAI(
                azure_endpoint=deployment.api_base,
                api_key=deployment.api_key,
                api_version=deployment.api_version,
            ),
        )

    async def run_healthchecks(self):
        async def check_health(client: Client):
            try:
                await asyncio.sleep(
                    random.uniform(0, 1)
                )  # splay outbound requests by a little bit
                await client.check_liveness()
            except Exception:
                logger.exception(f"{client}: healthcheck failed")

        while True:
            await asyncio.gather(
                *[check_health(client) for client in self.deployments.values()]
            )
            await asyncio.sleep(self.healthcheck_interval)

    def select_deployment(self, session_id: str | None = None) -> Client:
        healthy_deployments = []
        # simple random for now, use session_id to select a specific deployment
        for name, client in self.deployments.items():
            if session_id == name:
                return client

            if client.healthy:
                healthy_deployments.append(client)

        if not healthy_deployments:
            raise RuntimeError("No healthy deployments")

        return random.choice(healthy_deployments)

    async def completion(
        self, session_id: str | None = None, stream: bool = False, **kwargs
    ) -> ChatCompletion | AsyncStream[ChatCompletionChunk] | None:
        try:
            async for attempt in self.fallback_policy:
                with attempt:
                    client = self.select_deployment(session_id)
                    if stream:
                        return await client.chat_completion(stream=True, **kwargs)
                    else:
                        return await client.chat_completion(**kwargs)
        except RetryError as e:
            logger.exception("All chat_completion attempts failed.")
            raise e
