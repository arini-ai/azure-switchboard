import pytest

from azure_switchboard import Switchboard

from .conftest import COMPLETION_PARAMS


class TestAdvanced:
    """Test advanced features of the switchboard."""

    @pytest.mark.mock_models("gpt-4o-mini")
    async def test_a_custom_selector_is_consulted_and_obeyed(
        self, switchboard: Switchboard
    ):
        """A custom selector must both be asked and have its answer used."""
        chosen = []

        def round_robin(options):
            pick = options[len(chosen) % len(options)]
            chosen.append(pick)
            return pick

        switchboard.selector = round_robin

        for _ in range(100):
            await switchboard.chat.completions.create(**COMPLETION_PARAMS)

        assert len(chosen) == 100, "the selector should decide every request"
        # a round robin over three resources should reach all of them
        assert len({d.resource.name for d in chosen}) == 3
        assert all(d.name == "gpt-4o-mini" for d in chosen)
