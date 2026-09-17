"""Resolve recipe paths and reject unsupported EPLB combinations early."""

from pathlib import Path

from omegaconf import OmegaConf, open_dict


def validate_eplb_config(config):
    actor = config.actor_rollout_ref.actor
    rollout = config.actor_rollout_ref.rollout
    enabled = actor.get("eplb_enable", False)
    collect = config.trainer.get("eplb_load_collection_enable", False)
    if not enabled and not collect:
        return
    if actor.strategy != "megatron" or rollout.name != "vllm":
        raise ValueError("This EPLB recipe requires actor.strategy=megatron and rollout.name=vllm.")
    if enabled and config.trainer.get("resume_mode", "disable") != "disable":
        raise ValueError("EPLB checkpoint layout restore is not supported yet; use trainer.resume_mode=disable.")
    if actor.eplb_mode not in ("Static", "Dynamic"):
        raise ValueError("actor.eplb_mode must be Static or Dynamic.")
    if enabled and not actor.megatron.use_distributed_optimizer:
        raise ValueError("MindSpeed EPLB requires actor.megatron.use_distributed_optimizer=true.")
    needs_routes = collect or (enabled and actor.eplb_mode == "Dynamic")
    if needs_routes:
        if not rollout.enable_rollout_routing_replay or actor.megatron.router_replay.mode != "R3":
            raise ValueError(
                "Load collection requires rollout routing replay and actor.megatron.router_replay.mode=R3."
            )
        if int(config.trainer.eplb_update_interval) <= 0:
            raise ValueError("trainer.eplb_update_interval must be positive.")
    if collect and (int(config.trainer.eplb_load_collection_steps) <= 0 or not config.trainer.eplb_load_output_path):
        raise ValueError("Load collection needs a positive step count and eplb_load_output_path.")
    if enabled and actor.eplb_mode == "Static" and not actor.train_eplb_strategy_path:
        raise ValueError("Static EPLB requires actor_rollout_ref.actor.train_eplb_strategy_path.")
    for key in (
        "actor_rollout_ref.actor.train_eplb_strategy_path",
        "actor_rollout_ref.rollout.rollout_eplb_strategy_path",
        "actor_rollout_ref.rollout.eplb_strategy_file_path",
    ):
        path = OmegaConf.select(config, key)
        if path and not Path(path).is_file():
            raise ValueError(f"{key} is not a readable file: {path}")


def prepare_eplb_config(config):
    """Set the built-in strategy path without depending on the launch directory."""
    actor = config.actor_rollout_ref.actor
    rollout = config.actor_rollout_ref.rollout
    with open_dict(config):
        if actor.eplb_enable and actor.eplb_mode == "Dynamic" and not rollout.eplb_strategy_file_path:
            rollout.eplb_strategy_file_path = str(Path(__file__).parent / "load_balanced_expert_placement.py")
        for key in (
            "actor_rollout_ref.actor.train_eplb_strategy_path",
            "actor_rollout_ref.rollout.rollout_eplb_strategy_path",
            "actor_rollout_ref.rollout.eplb_strategy_file_path",
            "trainer.eplb_load_output_path",
        ):
            path = OmegaConf.select(config, key)
            if path:
                OmegaConf.update(config, key, str(Path(path).expanduser().resolve()))
    validate_eplb_config(config)
