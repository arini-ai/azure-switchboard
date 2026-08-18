from __future__ import annotations

from collections.abc import Iterable

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from .anthropic_deployment import AnthropicDeployment
from .exceptions import SwitchboardError
from .deployment import Cooldown, UtilStats
from .openai_deployment import OpenAIDeployment

# Every deployment speaks one of the APIs switchboard serves. ModelDeployment
# carries the quota machinery they share and bounds the generic selection code,
# but nothing is ever only a ModelDeployment, so a resource says so.
Deployment = OpenAIDeployment | AnthropicDeployment


class Resource(Cooldown):
    """Somewhere deployments live: a credential, a timeout, and the clients
    they share.

    `base` is the origin a deployment appends its API path to, with no trailing
    slash, so a deployment writes f"{base}/openai/v1/". Without one,
    each SDK falls back to its vendor default — which is how the vendors' own
    APIs are reached.
    """

    def __init__(
        self,
        name: str,
        *,
        base: str | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
        models: Iterable[Deployment] = (),
        default_cooldown: float = 10.0,
    ) -> None:
        super().__init__(default_cooldown)
        self.name = name
        self.base = base
        self.api_key = api_key
        self.timeout = timeout

        # Keyed by URL: within one API, that is what distinguishes a client.
        # AsyncAnthropicFoundry subclasses AsyncAnthropic, and AsyncAzureOpenAI
        # subclasses AsyncOpenAI, so the base types cover every variant.
        self._openai_clients: dict[str | None, AsyncOpenAI] = {}
        self._anthropic_clients: dict[str | None, AsyncAnthropic] = {}

        self.models: dict[str, Deployment] = {}
        for model in models:
            self.add(model)

    def add(self, model: Deployment) -> None:
        """Register a deployment and hand it the client it will use.

        Deployments served from the same URL share a client, so two models of
        one API on this resource share a connection pool while one that
        overrides its endpoint gets its own.
        """
        if model.name in self.models:
            raise SwitchboardError(f"{self.name}: duplicate model {model.name}")
        # checked before anything is mutated, and here rather than in Switchboard
        # so a Resource used on its own cannot hold the wrong client either
        if not isinstance(model, (OpenAIDeployment, AnthropicDeployment)):
            raise SwitchboardError(
                f"{self.name}: {model.name} is not an OpenAIDeployment "
                "or AnthropicDeployment"
            )

        # url reads the resource's base, so binding comes first
        model.bind(self)
        self.models[model.name] = model

        url = model.url
        if isinstance(model, OpenAIDeployment):
            if url not in self._openai_clients:
                self._openai_clients[url] = model.new_client()
            model.client = self._openai_clients[url]
        else:
            if url not in self._anthropic_clients:
                self._anthropic_clients[url] = model.new_client()
            model.client = self._anthropic_clients[url]

    def reset_usage(self) -> None:
        for model in self.models.values():
            model.reset_usage()

    def stats(self) -> dict[str, UtilStats]:
        return {name: model.stats() for name, model in self.models.items()}

    def __repr__(self) -> str:
        elems = ", ".join(map(str, self.models.values()))
        return f"{type(self).__name__}<{self.name}>([{elems}])"


class Foundry(Resource):
    """An Azure AI Foundry resource: an endpoint, a credential, and the model
    deployments it hosts.

    `name` is the Azure resource name and resolves to
    `https://{name}.services.ai.azure.com`, onto which each model appends the
    path for the API it speaks. A model can override the whole URL with
    `endpoint=`, which covers legacy `<resource>.openai.azure.com` resources
    and non-Azure hosts.
    """

    def __init__(
        self,
        name: str,
        *,
        api_key: str | None = None,
        timeout: float = 30.0,
        models: Iterable[Deployment] = (),
        default_cooldown: float = 10.0,
    ) -> None:
        # base is passed down rather than assigned after, so it is in place
        # before any deployment is added and builds its client against it
        super().__init__(
            name,
            base=f"https://{name}.services.ai.azure.com",
            api_key=api_key,
            timeout=timeout,
            models=models,
            default_cooldown=default_cooldown,
        )
