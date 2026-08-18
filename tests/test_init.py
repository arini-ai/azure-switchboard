import azure_switchboard
from loguru import logger as _logger

from azure_switchboard import Foundry, OpenAIModel, Switchboard


class TestInit:
    def test_public_exports_do_not_include_logging_helpers(self):
        assert "enable_logging" not in azure_switchboard.__all__
        assert "disable_logging" not in azure_switchboard.__all__

    def test_logging_activation_controls_switchboard_logs(self):
        records: list[dict] = []
        sink_id = _logger.add(lambda m: records.append(m.record))
        switchboard = Switchboard(
            foundries=[
                Foundry(
                    name="mini-only",
                    api_key="mini-only",
                    models=[OpenAIModel(name="gpt-4o-mini", tpm=1000, rpm=6)],
                ),
                Foundry(
                    name="full-only",
                    api_key="full-only",
                    models=[OpenAIModel(name="gpt-4o", tpm=1000, rpm=6)],
                ),
            ],
            ratelimit_window=0,
        )
        try:
            _logger.disable("azure_switchboard")
            switchboard.sessions["test"] = switchboard.foundries["mini-only"]
            _ = switchboard.select_deployment(session_id="test", model="gpt-4o")
            assert not records

            _logger.enable("azure_switchboard")
            switchboard.sessions["test"] = switchboard.foundries["mini-only"]
            _ = switchboard.select_deployment(session_id="test", model="gpt-4o")
            assert any(
                "is unhealthy on mini-only, reselecting" in r["message"]
                for r in records
            )
        finally:
            _logger.remove(sink_id)
            _logger.disable("azure_switchboard")

    def test_public_export_surface(self):
        assert set(azure_switchboard.__all__) == {
            "AnthropicModel",
            "Foundry",
            "OpenAIModel",
            "ParsedChatCompletion",
            "SwitchboardError",
            "Switchboard",
        }

    def test_config_classes_were_replaced(self):
        """Resources are Foundry, deployments are *Model; no back-compat aliases."""
        for gone in ("DeploymentConfig", "OpenAIConfig", "AnthropicConfig", "Model"):
            assert not hasattr(azure_switchboard, gone)
