from __future__ import annotations

import time
from collections.abc import Callable, Hashable, Iterable
from typing import Any, TypeVar

from .anthropic_model import AnthropicModel
from .exceptions import SwitchboardError
from .model import UtilStats
from .openai_model import OpenAIModel

_C = TypeVar("_C")

# Every deployment speaks one of the APIs switchboard serves. ModelBase carries
# the quota and cooldown machinery they share and bounds the generic selection
# code, but nothing is ever only a ModelBase, so a resource says so.
Deployment = OpenAIModel | AnthropicModel


class Foundry:
    """An Azure AI Foundry resource: an endpoint, a credential, and the model
    deployments it hosts.

    `name` is the Azure resource name and resolves to
    `https://{name}.services.ai.azure.com/`, onto which each model appends the
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
        self.name = name
        self.api_key = api_key
        self.timeout = timeout
        self.default_cooldown = default_cooldown
        self.cooldown_until: float = 0

        # The prefix each deployment appends its API path to.
        self.base: str | None = f"https://{name}.services.ai.azure.com/"

        self._clients: dict[Hashable, Any] = {}

        self.models: dict[str, Deployment] = {}
        for model in models:
            self.add(model)

    def add(self, model: Deployment) -> None:
        if model.name in self.models:
            raise SwitchboardError(f"{self.name}: duplicate model {model.name}")
        model.bind(self)
        self.models[model.name] = model

    def cached_client(self, key: Hashable, build: Callable[[], _C]) -> _C:
        """Share one SDK client across the deployments that can use it.

        Deployments call this with a key of their own choosing, normally their
        API and URL, so two models of the same API on this resource share a
        connection pool while one that overrides its endpoint gets its own.
        """
        if key not in self._clients:
            self._clients[key] = build()
        return self._clients[key]

    def mark_down(self, seconds: float = 0.0) -> None:
        """Take the whole resource out of selection.

        Reserved for errors that are properties of the host rather than of one
        model's quota — see AnthropicModel/OpenAIModel error handling.
        """
        self.cooldown_until = time.time() + (seconds or self.default_cooldown)

    def mark_up(self) -> None:
        self.cooldown_until = 0

    def is_cooling(self) -> bool:
        return time.time() < self.cooldown_until

    def reset_usage(self) -> None:
        for model in self.models.values():
            model.reset_usage()

    def stats(self) -> dict[str, UtilStats]:
        return {name: model.stats() for name, model in self.models.items()}

    def __repr__(self) -> str:
        elems = ", ".join(map(str, self.models.values()))
        return f"{type(self).__name__}<{self.name}>([{elems}])"


class FirstParty(Foundry):
    """The vendors' own APIs, used as a last-resort fallback.

    Not an Azure resource, so it derives no endpoint: each SDK is left to its
    own default base URL and its own environment variable for credentials.
    """

    def __init__(
        self, api: str, *, timeout: float = 30.0, models: Iterable[Deployment] = ()
    ):
        super().__init__(name=f"first-party-{api}", timeout=timeout, models=models)
        # no resource to derive from; each SDK falls back to its own default
        self.base = None
