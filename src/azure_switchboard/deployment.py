from __future__ import annotations

from typing import ClassVar, Literal

from opentelemetry import trace

from .exceptions import SwitchboardError
from .model import Model, UtilStats

Api = Literal["chat", "messages"]


class DeploymentBase:
    """Provider-agnostic runtime state for a deployment.

    Holds the per-model utilization counters that drive selection. Everything
    here is protocol-independent: subclasses add the calls that actually speak
    to an upstream API.
    """

    # Which wire protocol this deployment speaks. Selection filters on it so a
    # model name registered against one provider can never be routed to the other.
    api: ClassVar[Api]

    def __init__(self, name: str, models: list[Model]) -> None:
        self._name = name
        self.models = {m.name: m for m in models}

    def __repr__(self) -> str:
        elems = ", ".join(map(str, self.models.values()))
        return f"{type(self).__name__}<{self._name}>([{elems}])"

    @property
    def name(self) -> str:
        return self._name

    def reset_usage(self) -> None:
        for model in self.models.values():
            model.reset_usage()

    def stats(self) -> dict[str, UtilStats]:
        return {name: model.stats() for name, model in self.models.items()}

    def is_healthy(self, model: str) -> bool:
        return self.model(model).is_healthy() if model in self.models else False

    def util(self, model: str) -> float:
        return self.model(model).util if model in self.models else 0.0

    def model(self, name: str) -> Model:
        return self.models[name]

    def _check_model(self, model: str) -> Model:
        if model not in self.models:
            raise SwitchboardError(f"{model} not configured for deployment")
        return self.models[model]

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
