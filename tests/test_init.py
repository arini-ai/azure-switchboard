import azure_switchboard
from loguru import logger as _logger

from azure_switchboard import Deployment, Model, Switchboard


class TestInit:
    def test_public_exports_do_not_include_logging_helpers(self):
        assert "enable_logging" not in azure_switchboard.__all__
        assert "disable_logging" not in azure_switchboard.__all__

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
            _logger.disable("azure_switchboard")
            switchboard.sessions["test"] = switchboard.deployments["mini-only"]
            _ = switchboard.select_deployment(session_id="test", model="gpt-4o")
            assert not records

            _logger.enable("azure_switchboard")
            switchboard.sessions["test"] = switchboard.deployments["mini-only"]
            _ = switchboard.select_deployment(session_id="test", model="gpt-4o")
            assert any(
                "is unhealthy on mini-only, falling back to selection" in r["message"]
                for r in records
            )
        finally:
            _logger.remove(sink_id)
            _logger.disable("azure_switchboard")
