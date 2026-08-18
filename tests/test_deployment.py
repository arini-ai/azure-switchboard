import asyncio

import pytest

from azure_switchboard import Foundry, OpenAIDeployment, SwitchboardError


class TestBinding:
    """A deployment is declared standalone and bound when it is registered."""

    def test_unbound_deployment_has_no_resource(self, model: OpenAIDeployment):
        with pytest.raises(SwitchboardError, match="not bound to a resource"):
            _ = model.resource

    def test_unbound_deployment_has_no_client(self, model: OpenAIDeployment):
        """The client is assigned at registration, so there is nothing to read
        until a Foundry has taken the deployment."""
        with pytest.raises(AttributeError):
            _ = model.client

    def test_rebinding_to_another_resource_is_rejected(self, model: OpenAIDeployment):
        Foundry(name="first", api_key="k", models=[model])
        with pytest.raises(SwitchboardError, match="already bound to first"):
            Foundry(name="second", api_key="k", models=[model])

    def test_rebinding_to_the_same_resource_is_a_no_op(self, model: OpenAIDeployment):
        resource = Foundry(name="only", api_key="k")
        resource.add(model)
        model.bind(resource)
        assert model.resource is resource

    def test_resource_cooldown_takes_its_deployments_out(self, model: OpenAIDeployment):
        """A connection error means the host is unreachable, so every
        deployment on it is unreachable too."""
        resource = Foundry(name="east", api_key="k", models=[model])
        assert model.is_healthy()

        resource.mark_down()
        assert not model.is_healthy()
        assert model.util == 1
        assert not model.is_cooling()  # the deployment itself is fine

        resource.mark_up()
        assert model.is_healthy()


@pytest.fixture
def bound(model: OpenAIDeployment) -> OpenAIDeployment:
    """Utilization reads the resource's cooldown, so it needs one."""
    Foundry(name="east", api_key="k", models=[model])
    return model


class TestModel:
    """Model functionality tests."""

    async def test_init(self, bound: OpenAIDeployment):
        assert str(bound).startswith("OpenAIDeployment<gpt-4o-mini>(util=0.0")

    async def test_util(self, bound: OpenAIDeployment):
        assert bound.is_healthy()

        bound.tpm_usage = 2000
        assert not bound.is_healthy()

        bound.tpm_usage = 500
        assert bound.is_healthy()

        bound.rpm_usage = 10
        assert not bound.is_healthy()

        bound.rpm_usage = 5
        assert bound.is_healthy()

    async def test_markdown(self, bound: OpenAIDeployment):
        assert bound.is_healthy()

        bound.mark_down(1)
        assert not bound.is_healthy()

        await asyncio.sleep(0.5)
        assert not bound.is_healthy()

        await asyncio.sleep(0.5)
        assert bound.is_healthy()

        bound.mark_down(10)
        assert not bound.is_healthy()

        bound.mark_up()
        assert bound.is_healthy()

    async def test_repr_survives_an_unbound_deployment(self, model: OpenAIDeployment):
        """util needs a resource, but repr has to work regardless — a raising
        repr breaks debuggers and hides the error being formatted."""
        assert repr(model) == "OpenAIDeployment<gpt-4o-mini>(unbound)"
