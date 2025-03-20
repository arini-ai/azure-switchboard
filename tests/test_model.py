import asyncio


from azure_switchboard import Model


class TestModel:
    """Model functionality tests."""

    async def test_init(self, model: Model):
        assert str(model).startswith("Model<gpt-4o-mini>(util=0.0")

    async def test_util(self, model: Model):
        assert model.is_healthy()

        model.tpm_usage = 2000
        assert not model.is_healthy()

        model.tpm_usage = 500
        assert model.is_healthy()

        model.rpm_usage = 10
        assert not model.is_healthy()

        model.rpm_usage = 5
        assert model.is_healthy()

    async def test_markdown(self, model: Model):
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
