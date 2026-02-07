import azure_switchboard
from loguru import logger as _logger

from azure_switchboard import Deployment, Model, Switchboard


class TestInit:
    def test_enable_logging(self):
        enabled: list[str] = []
        original_enable = azure_switchboard._logger.enable
        try:
            azure_switchboard._logger.enable = lambda name: enabled.append(name)
            azure_switchboard.enable_logging()
        finally:
            azure_switchboard._logger.enable = original_enable
        assert enabled == [azure_switchboard._LOG_NAMESPACE]

    def test_disable_logging(self):
        disabled: list[str] = []
        original_disable = azure_switchboard._logger.disable
        try:
            azure_switchboard._logger.disable = lambda name: disabled.append(name)
            azure_switchboard.disable_logging()
        finally:
            azure_switchboard._logger.disable = original_disable
        assert disabled == [azure_switchboard._LOG_NAMESPACE]

    def test_logging_activation_controls_switchboard_logs(self):
        records: list[dict] = []
        sink_id = _logger.add(lambda m: records.append(m.record))
        switchboard = Switchboard(
            deployments=[
                Deployment(
                    name="mini-only",
                    base_url="https://mini-only.openai.azure.com/openai/v1/",
                    api_key="mini-only",
                    models=[Model(name="gpt-4o-mini", tpm=1000, rpm=6)],
                ),
                Deployment(
                    name="full-only",
                    base_url="https://full-only.openai.azure.com/openai/v1/",
                    api_key="full-only",
                    models=[Model(name="gpt-4o", tpm=1000, rpm=6)],
                ),
            ],
            ratelimit_window=0,
        )
        try:
            azure_switchboard.disable_logging()
            switchboard.sessions["test"] = switchboard.deployments["mini-only"]
            _ = switchboard.select_deployment(session_id="test", model="gpt-4o")
            assert not records

            azure_switchboard.enable_logging()
            switchboard.sessions["test"] = switchboard.deployments["mini-only"]
            _ = switchboard.select_deployment(session_id="test", model="gpt-4o")
            assert any(
                "is unhealthy on mini-only, falling back to selection" in r["message"]
                for r in records
            )
        finally:
            _logger.remove(sink_id)
            azure_switchboard.disable_logging()
