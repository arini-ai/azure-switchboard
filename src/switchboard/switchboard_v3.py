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
        await self.client.models.list()

    async def chat_completion(self, **kwargs):
        try:
            response = await self.client.chat.completions.create(
                **kwargs, timeout=self.config.timeout
            )
            return response
        except Exception:
            logger.exception(f"Chat completion failed for {self}")
            self.healthy = False
            raise


class Switchboard:
    def __init__(
        self,
        deployments: List[AzureDeployment],
        client_factory: Callable[[AzureDeployment], AsyncAzureOpenAI] | None = None,
    ):
        self.client_factory = client_factory or self._default_client_factory

        self.deployments = {
            deployment.name: Client(deployment, self.client_factory(deployment))
            for deployment in deployments
        }

    def _default_client_factory(self, deployment: AzureDeployment) -> AsyncAzureOpenAI:
        return AsyncAzureOpenAI(
            azure_endpoint=deployment.api_base,
            api_key=deployment.api_key,
            api_version=deployment.api_version,
        )

    async def check_liveness(self):
        for name, deployment in self.deployments.items():
            try:
                await deployment.check_liveness()
            except Exception:
                logger.exception(f"Liveness check failed for {name}")
                deployment.healthy = False

    def _select_deployment(self) -> Client:
        # simple random for now
        return random.choice(list(self.deployments.values()))

    async def chat_completion(self, **kwargs):
        client = self._select_deployment()
        return await client.chat_completion(**kwargs)
