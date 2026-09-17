from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir


@pytest.fixture
def config_factory():
    def make(mode="dynamic", *overrides):
        config_dir = str(Path(__file__).resolve().parents[1] / "config")
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            return compose(config_name="eplb_trainer", overrides=[f"mode={mode}", *overrides])

    return make
