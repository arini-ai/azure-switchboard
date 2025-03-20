import random

import pytest

from azure_switchboard import Switchboard

from .conftest import COMPLETION_PARAMS


class TestAdvanced:
    """Test advanced features of the switchboard."""

    @pytest.mark.mock_models("gpt-4o-mini")
    async def test_round_robin_selector(self, switchboard: Switchboard):
        switchboard.selector = lambda _, options: random.choice(options)

        for _ in range(100):
            await switchboard.create(**COMPLETION_PARAMS)

        # print(switchboard.get_usage())
