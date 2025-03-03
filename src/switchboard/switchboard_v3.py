import asyncio
import logging
import random
from typing import Annotated, Callable, List

from openai import AsyncAzureOpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger("switchboard")


class AzureDeployment(BaseModel):
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

    def __init__(self, config: AzureDeployment, client: AsyncAzureOpenAI):
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

    async def chat_completion(self, **kwargs):
        try:
            response = await self.client.chat.completions.create(
                **kwargs, timeout=self.config.timeout
            )
            return response
        except Exception:
            logger.exception(f"{self}: chat completion failed")
            self.healthy = False
            raise


class Switchboard:
    def __init__(
        self,
        deployments: List[AzureDeployment],
        client_factory: Callable[[AzureDeployment], AsyncAzureOpenAI] | None = None,
        healthcheck_interval: int = 10,
    ):
        self.client_factory = client_factory or self._default_client_factory
        self.healthcheck_interval = healthcheck_interval

        self.deployments = {
            deployment.name: Client(deployment, self.client_factory(deployment))
            for deployment in deployments
        }

        self.healthcheck_task = (
            asyncio.create_task(self.run_healthchecks())
            if self.healthcheck_interval > 0
            else None
        )

    def _default_client_factory(self, deployment: AzureDeployment) -> AsyncAzureOpenAI:
        return AsyncAzureOpenAI(
            azure_endpoint=deployment.api_base,
            api_key=deployment.api_key,
            api_version=deployment.api_version,
        )

    async def run_healthchecks(self):
        while True:
            await self.check_liveness()
            await asyncio.sleep(self.healthcheck_interval)

    async def check_liveness(self):
        tasks = [client.check_liveness() for client in self.deployments.values()]
        await asyncio.gather(*tasks)

    def _select_deployment(self) -> Client:
        # simple random for now
        return random.choice(list(self.deployments.values()))

    async def chat_completion(self, **kwargs):
        client = self._select_deployment()
        return await client.chat_completion(**kwargs)
