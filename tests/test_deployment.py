import asyncio

import pytest

from azure_switchboard import Foundry, OpenAIDeployment, SwitchboardError


class TestBinding:
    """A deployment is declared standalone and bound when it is registered."""

    def test_rebinding_to_another_resource_is_rejected(self, model: OpenAIDeployment):
        Foundry(name="first", api_key="k", models=[model])
        with pytest.raises(SwitchboardError, match="already bound to first"):
            Foundry(name="second", api_key="k", models=[model])

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

    def test_load_is_the_unjittered_util(self, bound: OpenAIDeployment):
        """util jitters to break ties between idle deployments; load is what
        gets reported, so it has to be the number that is actually true."""
        bound.tpm_usage = 500
        assert bound.load == 0.5
        assert 0.5 <= bound.util < 0.51

    def test_stats_report_both_quotas_numerically(self, bound: OpenAIDeployment):
        bound.tpm_usage = 250
        bound.rpm_usage = 3

        stats = bound.stats()
        assert stats.tpm == (250, 1000)
        assert stats.rpm.used == 3
        assert stats.util == 0.5  # rpm is the tighter of the two
