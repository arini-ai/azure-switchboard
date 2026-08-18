import asyncio

import pytest

from azure_switchboard import Foundry, OpenAIModel, SwitchboardError


class TestBinding:
    """A deployment is declared standalone and bound when it is registered."""

    def test_unbound_deployment_has_no_foundry(self, model: OpenAIModel):
        with pytest.raises(SwitchboardError, match="not bound to a foundry"):
            _ = model.foundry

    def test_unbound_deployment_has_no_client(self, model: OpenAIModel):
        with pytest.raises(SwitchboardError, match="not bound to a foundry"):
            _ = model.client

    def test_rebinding_to_another_foundry_is_rejected(self, model: OpenAIModel):
        Foundry(name="first", api_key="k", models=[model])
        with pytest.raises(SwitchboardError, match="already bound to first"):
            Foundry(name="second", api_key="k", models=[model])

    def test_rebinding_to_the_same_foundry_is_a_no_op(self, model: OpenAIModel):
        resource = Foundry(name="only", api_key="k")
        resource.add(model)
        model.bind(resource)
        assert model.foundry is resource

    def test_foundry_cooldown_takes_its_deployments_out(self, model: OpenAIModel):
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


class TestModel:
    """Model functionality tests."""

    async def test_init(self, model: OpenAIModel):
        assert str(model).startswith("OpenAIModel<gpt-4o-mini>(util=0.0")

    async def test_util(self, model: OpenAIModel):
        assert model.is_healthy()

        model.tpm_usage = 2000
        assert not model.is_healthy()

        model.tpm_usage = 500
        assert model.is_healthy()

        model.rpm_usage = 10
        assert not model.is_healthy()

        model.rpm_usage = 5
        assert model.is_healthy()

    async def test_markdown(self, model: OpenAIModel):
        assert model.is_healthy()

        model.mark_down(1)
        assert not model.is_healthy()

        await asyncio.sleep(0.5)
        assert not model.is_healthy()

        await asyncio.sleep(0.5)
        assert model.is_healthy()

        model.mark_down(10)
        assert not model.is_healthy()

        model.mark_up()
        assert model.is_healthy()
