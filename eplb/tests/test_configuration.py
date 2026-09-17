from pathlib import Path

import pytest
from recipe.eplb.configuration import prepare_eplb_config, validate_eplb_config


@pytest.mark.parametrize("mode", ["baseline", "collect", "static", "dynamic"])
def test_modes_compose_native_megatron_config(config_factory, mode):
    config = config_factory(mode)
    assert config.actor_rollout_ref.actor.strategy == "megatron"
    assert config.actor_rollout_ref.rollout.name == "vllm"
    assert config.actor_rollout_ref.actor.eplb_enable == (mode in ("static", "dynamic"))
    assert config.trainer.eplb_load_collection_enable == (mode == "collect")
    assert config.actor_rollout_ref.actor.megatron.router_replay.mode == "R3"


def test_default_strategy_resolves_outside_workspace(config_factory, monkeypatch, tmp_path):
    config = config_factory()
    monkeypatch.chdir(tmp_path)
    prepare_eplb_config(config)
    assert Path(config.actor_rollout_ref.rollout.eplb_strategy_file_path).is_file()
    assert Path(config.trainer.eplb_load_output_path).is_absolute()


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("actor_rollout_ref.actor.megatron.router_replay.mode=R2", "R3"),
        ("trainer.eplb_update_interval=0", "positive"),
        ("trainer.resume_mode=auto", "checkpoint"),
        ("actor_rollout_ref.actor.megatron.use_distributed_optimizer=false", "distributed_optimizer"),
    ],
)
def test_invalid_combinations_fail_before_ray(config_factory, override, message):
    with pytest.raises(ValueError, match=message):
        prepare_eplb_config(config_factory("dynamic", override))


def test_static_needs_layout(config_factory):
    with pytest.raises(ValueError, match="train_eplb_strategy_path"):
        validate_eplb_config(config_factory("static"))
