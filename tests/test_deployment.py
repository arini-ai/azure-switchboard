import asyncio

import pytest

from azure_switchboard import Foundry, OpenAIDeployment, SwitchboardError

from .conftest import openai_foundry


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


class TestUtilization:
    """The quota arithmetic every deployment shares, regardless of the API it
    speaks. An OpenAIDeployment stands in for any of them."""

    def test_the_tighter_of_the_two_quotas_wins(self, bound: OpenAIDeployment):
        bound.spend_tokens(500)  # 50% of 1000 TPM
        assert bound.load == 0.5

        bound.reset_usage()
        bound.spend_request(3)  # 50% of 6 RPM
        assert bound.load == 0.5

        bound.reset_usage()
        bound.spend_tokens(600)  # 60% of TPM
        bound.spend_request(3)  # 50% of RPM
        assert bound.load == 0.6

    def test_a_deployment_over_either_quota_is_unhealthy(self, bound: OpenAIDeployment):
        assert bound.is_healthy()

        bound.tpm_usage = 2000
        assert not bound.is_healthy()

        bound.tpm_usage = 500
        assert bound.is_healthy()

        bound.rpm_usage = 10
        assert not bound.is_healthy()

        bound.rpm_usage = 5
        assert bound.is_healthy()

    def test_load_is_the_unjittered_util(self, bound: OpenAIDeployment):
        """util jitters to break ties between idle deployments; load is what
        gets reported, so it has to be the number that is actually true."""
        bound.spend_tokens(500)
        assert bound.load == 0.5
        # the jitter is at most 0.01, and rounding can land exactly on it
        assert 0.5 <= bound.util <= 0.51

    def test_an_idle_deployment_still_jitters(self, bound: OpenAIDeployment):
        """Without it every idle deployment ties at zero and the same one keeps
        winning the comparison."""
        assert bound.load == 0
        assert 0 <= bound.util <= 0.01

    def test_a_deployment_with_no_limits_is_never_loaded(self):
        unlimited = OpenAIDeployment(name="unlimited")
        Foundry(name="east", api_key="k", models=[unlimited])

        unlimited.spend_tokens(10_000)
        unlimited.spend_request(500)
        assert unlimited.load == 0
        assert unlimited.is_healthy()


class TestCooldown:
    async def test_a_cooldown_expires_on_its_own(self, bound: OpenAIDeployment):
        assert bound.is_healthy()

        bound.mark_down(1)
        assert not bound.is_healthy()

        await asyncio.sleep(0.5)
        assert not bound.is_healthy()

        await asyncio.sleep(0.5)
        assert bound.is_healthy()

    def test_a_cooldown_can_be_lifted_early(self, bound: OpenAIDeployment):
        bound.mark_down(10)
        assert not bound.is_healthy()

        bound.mark_up()
        assert bound.is_healthy()

    def test_cooling_pins_load_to_one(self, bound: OpenAIDeployment):
        bound.mark_down()
        assert bound.load == 1
        assert bound.util == 1


class TestStats:
    def test_stats_report_both_quotas_numerically(self, bound: OpenAIDeployment):
        assert bound.stats().tpm == (0, 1000)
        assert bound.stats().rpm == (0, 6)

        bound.spend_tokens(250)
        bound.spend_request(3)

        stats = bound.stats()
        assert stats.tpm == (250, 1000)
        assert stats.rpm.used == 3
        assert stats.util == 0.5  # rpm is the tighter of the two

    def test_repr_shows_both_quotas(self, bound: OpenAIDeployment):
        bound.spend_tokens(250)
        assert repr(bound) == "OpenAIDeployment<gpt-4o-mini>(tpm=250/1000 rpm=0/6)"

    def test_repr_works_before_a_deployment_is_registered(self):
        """It reports no load, so it needs no resource -- and a __repr__ that
        raised would break debuggers and hide the error being formatted."""
        assert (
            repr(OpenAIDeployment(name="gpt-4o", tpm=1000))
            == "OpenAIDeployment<gpt-4o>(tpm=0/1000 rpm=0/0)"
        )

    def test_resetting_a_resource_resets_every_deployment_on_it(self):
        resource = openai_foundry("east")
        first, second = resource.models["gpt-4o-mini"], resource.models["gpt-4o"]

        first.spend_tokens(100)
        first.spend_request(5)
        assert second.stats().tpm.used == 0, "counters are per deployment"

        resource.reset_usage()
        assert first.stats().tpm == (0, first.tpm_limit)
        assert first.stats().rpm == (0, first.rpm_limit)
