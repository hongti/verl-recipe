import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from recipe.eplb.configuration import prepare_eplb_config
from recipe.eplb.ray_trainer import RayEPLBTrainer, _EPLBWeightPublisher
from recipe.eplb.strategy_manager import EPLBStrategyManager

from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def layout(version=1):
    return {
        "strategy_version": version,
        "moe_layer_count": 1,
        "layer_list": [
            {
                "layer_id": 0,
                "device_list": [
                    {"device_id": 0, "device_expert": [0, 1]},
                    {"device_id": 1, "device_expert": [2, 3]},
                ],
            }
        ],
    }


@pytest.fixture
def trainer(config_factory):
    instance = object.__new__(RayEPLBTrainer)
    instance.config = config_factory("dynamic", "trainer.eplb_update_interval=1")
    prepare_eplb_config(instance.config)
    instance.eplb_manager = EPLBStrategyManager(
        {
            "num_experts": 4,
            "training_device_count": 2,
            "rollout_device_count": 2,
            "update_interval": 1,
            "eplb_strategy_file_path": instance.config.actor_rollout_ref.rollout.eplb_strategy_file_path,
        }
    )
    instance.eplb_training_strategy_applied = False
    instance.initial_eplb_strategy_checked = False
    instance.eplb_load_exported = False
    instance.pending_deferred_eplb_strategy = None
    instance._queued_training_strategy = None
    instance._applied_inference_strategy = None
    instance.global_steps = 1
    instance.actor_rollout_wg = SimpleNamespace(expert_weight_and_optimizer_state_relocation=Mock())
    # Mirror init_workers(): publishes go through the layout wrapper, so the
    # assertions below see exactly what the real checkpoint manager would.
    inner = SimpleNamespace(update_weights=Mock())
    instance.publish_mock = inner.update_weights
    instance.checkpoint_manager = _EPLBWeightPublisher(inner, instance._current_inference_eplb_strategy)
    return instance


def test_deferred_layout_only_published_after_actor_success(trainer, monkeypatch):
    strategy = layout()
    trainer.eplb_manager.current_inference_strategy = strategy
    trainer._apply_training_eplb_strategy(strategy, defer=True)
    trainer._update_rollout_weights(1)
    assert trainer.publish_mock.call_args.kwargs["eplb_strategy"] is None
    monkeypatch.setattr(RayPPOTrainer, "_update_actor", lambda self, batch: "updated")
    assert trainer._update_actor(None) == "updated"
    trainer._update_rollout_weights(1)
    published = trainer.publish_mock.call_args.kwargs["eplb_strategy"]
    assert published["strategy_version"] == 1
    assert published["_training_strategy"] == strategy
    assert trainer._queued_training_strategy is None


def test_actor_failure_does_not_commit_queued_layout(trainer, monkeypatch):
    trainer.eplb_manager.current_inference_strategy = layout()
    trainer._apply_training_eplb_strategy(layout(), defer=True)

    def fail(*args):
        raise RuntimeError("migration failed")

    monkeypatch.setattr(RayPPOTrainer, "_update_actor", fail)
    with pytest.raises(RuntimeError, match="migration failed"):
        trainer._update_actor(None)
    assert trainer.eplb_manager.applied_training_strategy is None
    assert trainer._current_inference_eplb_strategy() is None


def test_candidate_does_not_replace_applied_inference_during_warmup(trainer):
    trainer.eplb_manager.current_inference_strategy = layout(1)
    trainer._mark_training_strategy_applied(layout(1))
    trainer.eplb_manager.current_inference_strategy = layout(2)
    assert trainer._current_inference_eplb_strategy()["strategy_version"] == 1


@pytest.mark.parametrize("replies", [None, [], [{"success": True}, {"success": False}]])
def test_static_requires_all_worker_acknowledgements(trainer, replies):
    trainer.actor_rollout_wg.expert_weight_and_optimizer_state_relocation.return_value = replies
    with pytest.raises(RuntimeError, match="acknowledged"):
        trainer._apply_training_eplb_strategy(layout())
    assert not trainer.eplb_training_strategy_applied


def test_load_generation_uses_routes_and_valid_token_mask(trainer):
    batch = SimpleNamespace(
        batch={
            "routed_experts": torch.tensor([[[[0]], [[0]], [[1]], [[3]]]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0]]),
        }
    )
    timing = {}
    trainer._collect_eplb_load(batch, timing)
    assert trainer.pending_deferred_eplb_strategy is not None
    assert trainer.pending_deferred_eplb_strategy["strategy_version"] == 1
    assert "eplb_load_collection" in timing
    assert not trainer.eplb_training_strategy_applied


def test_collect_only_exports_without_migration(trainer, tmp_path):
    trainer.config.actor_rollout_ref.actor.eplb_enable = False
    trainer.config.trainer.eplb_load_collection_enable = True
    trainer.config.trainer.eplb_load_collection_steps = 1
    output = tmp_path / "load.json"
    trainer.config.trainer.eplb_load_output_path = str(output)
    trainer._collect_eplb_load(
        SimpleNamespace(
            batch={
                "routed_experts": torch.tensor([[[[0]], [[3]]]]),
                "attention_mask": torch.tensor([[1, 0]]),
            }
        ),
        {},
    )
    payload = json.loads(output.read_text())
    assert payload["aggregated_load"] == {"0": {"0": 1.0}}
    assert trainer.pending_deferred_eplb_strategy is None
    trainer.actor_rollout_wg.expert_weight_and_optimizer_state_relocation.assert_not_called()


def test_missing_routes_fails_instead_of_silently_disabling_eplb(trainer):
    with pytest.raises(ValueError, match="routed_experts"):
        trainer._collect_eplb_load(SimpleNamespace(batch={}), {})


def test_fit_applies_initial_layout_before_first_publication(trainer, tmp_path, monkeypatch):
    events = []
    strategy_path = tmp_path / "layout.json"
    strategy_path.write_text(json.dumps(layout()))
    trainer.config.actor_rollout_ref.actor.eplb_mode = "Static"
    trainer.config.actor_rollout_ref.actor.train_eplb_strategy_path = str(strategy_path)
    trainer.config.trainer.val_only = True
    trainer.config.trainer.logger = ["console"]
    trainer.train_dataloader = [None]
    trainer._dump_executor = SimpleNamespace(_shutdown=False)
    trainer._load_checkpoint = lambda: events.append("checkpoint")
    trainer._validate = lambda: {"ok": 1}
    trainer._shutdown_dump_executor = Mock()

    def migrate(*args, **kwargs):
        events.append("migration")
        return [{"success": True}, {"success": True}]

    trainer.actor_rollout_wg.expert_weight_and_optimizer_state_relocation.side_effect = migrate

    def publish(*args, **kwargs):
        events.append("publish")
        assert kwargs["eplb_strategy"]["strategy_version"] == 1

    trainer.publish_mock.side_effect = publish
    trainer.fit()
    assert events == ["checkpoint", "migration", "publish"]


def test_original_validation_and_ppo_operations_are_inherited():
    for name in ("_validate", "_compute_old_log_prob", "_compute_values", "_compute_ref_log_prob"):
        assert getattr(RayEPLBTrainer, name) is getattr(RayPPOTrainer, name)


def test_static_null_layout_is_rejected(trainer, tmp_path):
    source = tmp_path / "null.json"
    source.write_text("null")
    trainer.config.actor_rollout_ref.actor.eplb_mode = "Static"
    trainer.config.actor_rollout_ref.actor.train_eplb_strategy_path = str(source)
    with pytest.raises(ValueError, match="non-null"):
        trainer._apply_initial_eplb_strategy_if_needed()
    trainer.actor_rollout_wg.expert_weight_and_optimizer_state_relocation.assert_not_called()


def test_missing_rollout_layout_is_rejected_before_training_migration(trainer, tmp_path):
    source = tmp_path / "train.json"
    source.write_text(json.dumps(layout()))
    trainer.config.actor_rollout_ref.actor.train_eplb_strategy_path = str(source)
    trainer.eplb_manager.rollout_device_count = 1
    with pytest.raises(ValueError, match="rollout placement"):
        trainer._apply_initial_eplb_strategy_if_needed()
    trainer.actor_rollout_wg.expert_weight_and_optimizer_state_relocation.assert_not_called()
