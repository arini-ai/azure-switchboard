from unittest.mock import patch

import azure_switchboard


class TestInit:
    def test_enable_logging(self):
        with patch.object(azure_switchboard._logger, "enable") as mock_enable:
            azure_switchboard.enable_logging()
            mock_enable.assert_called_once_with("azure_switchboard")

    def test_disable_logging(self):
        with patch.object(azure_switchboard._logger, "disable") as mock_disable:
            azure_switchboard.disable_logging()
            mock_disable.assert_called_once_with("azure_switchboard")
