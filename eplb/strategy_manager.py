"""EPLB strategy policy layer: load accounting, mode helpers and the manager."""

import importlib.util
import json
import logging
import os
from collections import defaultdict
from typing import Any, Optional

import torch

from verl.utils.moe_eplb import (
    EMPTY_PATH_VALUES,
    _to_int,
    load_strategy_from_path,
    validate_eplb_strategy,
)

logger = logging.getLogger(__name__)

EPLB_MODES = {"static": "Static", "dynamic": "Dynamic"}


def _as_detached_tensor(value: Any, name: str) -> torch.Tensor:
    try:
        return value.detach() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be convertible to a tensor") from exc


def _count_routed_experts(
    routed_experts: Any,
    attention_mask: Any,
    num_experts: Optional[int],
) -> torch.Tensor:
    routed_experts = _as_detached_tensor(routed_experts, "routed_experts")
    attention_mask = _as_detached_tensor(attention_mask, "attention_mask")
    if routed_experts.is_nested or attention_mask.is_nested:
        raise ValueError("routed_experts and attention_mask must be dense tensors")
    if routed_experts.dim() != 4:
        raise ValueError(
            f"routed_experts must have shape [batch, sequence, layer, topk], got {tuple(routed_experts.shape)}"
        )
    if attention_mask.dim() != 2:
        raise ValueError(f"attention_mask must have shape [batch, sequence], got {tuple(attention_mask.shape)}")
    if routed_experts.shape[:2] != attention_mask.shape:
        raise ValueError(
            "routed_experts and attention_mask batch/sequence shape mismatch: "
            f"{tuple(routed_experts.shape[:2])} != {tuple(attention_mask.shape)}"
        )

    layer_count, topk = routed_experts.shape[-2:]
    if layer_count == 0 or topk == 0:
        return torch.zeros((layer_count, num_experts or 0), dtype=torch.int64)

    valid_mask = attention_mask.to(device=routed_experts.device, dtype=torch.bool).reshape(-1)
    token_routes = routed_experts.reshape(-1, layer_count, topk)
    if not torch.any(valid_mask):
        return torch.zeros(
            (layer_count, num_experts or 0),
            dtype=torch.int64,
            device=routed_experts.device,
        )

    valid_routes = token_routes[valid_mask]
    min_expert = int(valid_routes.min().item())
    max_expert = int(valid_routes.max().item())
    resolved_num_experts = num_experts if num_experts is not None else max_expert + 1
    if resolved_num_experts <= 0:
        raise ValueError(f"num_experts must be positive, got {resolved_num_experts}")
    if min_expert < 0 or max_expert >= resolved_num_experts:
        raise ValueError(f"expert_id range [{min_expert}, {max_expert}] is outside [0, {resolved_num_experts})")

    # Bound the int64 temporary to one layer instead of converting the full route tensor.
    return torch.stack(
        [
            torch.bincount(
                valid_routes[:, layer_id, :].reshape(-1).to(torch.int64),
                minlength=resolved_num_experts,
            )
            for layer_id in range(layer_count)
        ]
    )


def _resolve_device_count(config: dict[str, Any]) -> Optional[int]:
    if "device_count" in config and config["device_count"] is not None:
        return _to_int(config["device_count"], "device_count")
    if "num_devices" in config and config["num_devices"] is not None:
        return _to_int(config["num_devices"], "num_devices")
    return None


def _resolve_named_device_count(config: dict[str, Any], name: str) -> Optional[int]:
    if name in config and config[name] is not None:
        return _to_int(config[name], name)
    return None


def normalize_eplb_mode(mode: Any) -> str:
    normalized = EPLB_MODES.get(str(mode or "Static").strip().lower())
    if normalized is None:
        raise ValueError("eplb_mode must be either 'Static' or 'Dynamic'.")
    return normalized


def should_collect_eplb_load(
    *,
    eplb_manager_present: bool,
    eplb_enable: bool,
    eplb_mode: Any,
    eplb_load_collection_enable: bool,
) -> bool:
    if not eplb_manager_present:
        return False
    return bool(eplb_load_collection_enable) or (bool(eplb_enable) and normalize_eplb_mode(eplb_mode) == "Dynamic")


def should_defer_eplb_training_strategy(*, eplb_mode: Any, is_initial_strategy: bool) -> bool:
    return normalize_eplb_mode(eplb_mode) == "Dynamic" and not is_initial_strategy


def _canonical_strategy(strategy: Optional[dict[str, Any]]) -> Optional[tuple]:
    if strategy is None:
        return None
    normalized = validate_eplb_strategy(strategy)
    return tuple(
        (layer_id, tuple((device_id, tuple(experts)) for device_id, experts in sorted(devices.items())))
        for layer_id, devices in sorted(normalized.items())
    )


def strategies_equal(left: Optional[dict[str, Any]], right: Optional[dict[str, Any]]) -> bool:
    return _canonical_strategy(left) == _canonical_strategy(right)


def aggregate_history_loads(history_loads: list[dict[int, dict[int, int]]]) -> dict[int, dict[int, int]]:
    aggregated: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for step_load in history_loads:
        for layer_id, expert_load in step_load.items():
            for expert_id, load in expert_load.items():
                aggregated[int(layer_id)][int(expert_id)] += int(load)
    return {layer_id: dict(loads) for layer_id, loads in aggregated.items()}


class EPLBStrategyManager:
    def __init__(self, eplb_config: dict[str, Any]):
        self.eplb_config = eplb_config
        self.num_experts = eplb_config.get("num_experts")
        if self.num_experts is not None:
            self.num_experts = _to_int(self.num_experts, "num_experts")
        self.training_device_count = _resolve_named_device_count(
            eplb_config, "training_device_count"
        ) or _resolve_device_count(eplb_config)
        self.rollout_device_count = (
            _resolve_named_device_count(eplb_config, "rollout_device_count") or self.training_device_count
        )
        self.device_count = self.training_device_count
        self.update_interval = _to_int(eplb_config.get("update_interval", 1), "update_interval")
        self.next_strategy_version = _to_int(eplb_config.get("strategy_version", 1), "strategy_version")
        self.internal_step = 0
        self.global_step = 0
        self.strategy_file_path = eplb_config.get("eplb_strategy_file_path")
        self.strategy_func = None
        self.current_strategy_json = "{}"
        self.current_training_strategy: Optional[dict[str, Any]] = None
        self.current_inference_strategy: Optional[dict[str, Any]] = load_strategy_from_path(
            eplb_config.get("rollout_eplb_strategy_path")
        )
        self.applied_training_strategy: Optional[dict[str, Any]] = None
        self.history_load_data: list[dict[int, dict[int, int]]] = []

        if self.strategy_file_path and os.path.exists(self.strategy_file_path):
            self._load_external_strategy()
        elif self.strategy_file_path:
            logger.warning("No valid EPLB strategy file path provided: %s", self.strategy_file_path)

    def _load_external_strategy(self):
        try:
            module_name = "external_moe_strategy"
            spec = importlib.util.spec_from_file_location(module_name, self.strategy_file_path)
            strategy_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(strategy_module)
            if not hasattr(strategy_module, "eplb_strategy"):
                logger.error("EPLB strategy file is missing an eplb_strategy function.")
                return
            self.strategy_func = strategy_module.eplb_strategy
        except Exception as exc:
            logger.error("Failed to load EPLB strategy file: %s", exc)

    def extract_and_store_load(self, gen_batch_output: Any, global_step: int = 0) -> dict[int, dict[int, int]]:
        self.global_step = global_step
        batch_dict = gen_batch_output.batch
        if "routed_experts" not in batch_dict or "attention_mask" not in batch_dict:
            return {}

        load_tensor = _count_routed_experts(
            batch_dict["routed_experts"],
            batch_dict["attention_mask"],
            self.num_experts,
        )
        result = {
            layer_id: {expert_id: int(load) for expert_id, load in enumerate(layer_load) if load}
            for layer_id, layer_load in enumerate(load_tensor.cpu().tolist())
            if any(layer_load)
        }
        self.history_load_data.append(result)
        self.internal_step += 1
        return result

    def update_strategy(self, force_update: bool = False) -> Optional[dict[str, Any]]:
        if self.internal_step == 0:
            return None
        if not force_update and self.internal_step % self.update_interval != 0:
            return None
        if self.strategy_func is None:
            logger.error("No EPLB strategy function is loaded; skipping this update.")
            return None

        try:
            strategy_version = self.next_strategy_version
            training_strategy = self._generate_strategy_for_target(
                target="training",
                device_count=self.training_device_count,
                strategy_version=strategy_version,
            )
            rollout_strategy = self._generate_rollout_strategy(training_strategy, strategy_version)
            if training_strategy is not None:
                self.current_training_strategy = training_strategy
                self.current_inference_strategy = rollout_strategy
                self.current_strategy_json = json.dumps(
                    {
                        "training_strategy": training_strategy,
                        "rollout_strategy": rollout_strategy,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                self._sync_next_strategy_version(training_strategy, fallback_version=strategy_version)
            self.history_load_data.clear()
            return training_strategy
        except Exception as exc:
            logger.error("Failed to update EPLB strategy: %s", exc)
            self.current_strategy_json = json.dumps({"error": str(exc)}, ensure_ascii=False)
            return None

    def _generate_strategy_for_target(
        self,
        *,
        target: str,
        device_count: Optional[int],
        strategy_version: int,
    ) -> Optional[dict[str, Any]]:
        strategy = self.strategy_func(
            history_loads=self.history_load_data,
            eplb_config={
                **self.eplb_config,
                "num_experts": self.num_experts,
                "device_count": device_count,
                "placement_target": target,
                "strategy_version": strategy_version,
            },
        )
        if strategy is not None:
            validate_eplb_strategy(strategy, num_experts=self.num_experts, device_count=device_count)
        return strategy

    def _generate_rollout_strategy(
        self,
        training_strategy: Optional[dict[str, Any]],
        strategy_version: int,
    ) -> Optional[dict[str, Any]]:
        if self.training_device_count == self.rollout_device_count:
            return training_strategy
        return self._generate_strategy_for_target(
            target="rollout",
            device_count=self.rollout_device_count,
            strategy_version=strategy_version,
        )

    def should_apply_training_strategy(self, strategy: Optional[dict[str, Any]]) -> bool:
        if strategy is None:
            return False
        return not strategies_equal(strategy, self.applied_training_strategy)

    def mark_training_strategy_applied(self, strategy: Optional[dict[str, Any]]):
        self.applied_training_strategy = strategy
        self._sync_next_strategy_version(strategy)
        if self.current_inference_strategy is None:
            if self.training_device_count != self.rollout_device_count:
                raise ValueError(
                    "rollout_eplb_strategy_path must be set or generated when training and rollout EP differ"
                )
            self.current_inference_strategy = strategy

    def _sync_next_strategy_version(
        self,
        strategy: Optional[dict[str, Any]],
        fallback_version: Optional[int] = None,
    ):
        version = fallback_version
        if strategy is not None and strategy.get("strategy_version") is not None:
            version = _to_int(strategy["strategy_version"], "strategy_version")
        if version is not None:
            self.next_strategy_version = max(self.next_strategy_version, version + 1)

    def export_load(self, output_path: str, *, metadata: Optional[dict[str, Any]] = None):
        if output_path is None or str(output_path).strip() in EMPTY_PATH_VALUES:
            raise ValueError("output_path must be set when exporting EPLB load")
        payload = {
            "metadata": metadata or {},
            "num_experts": self.num_experts,
            "sample_count": len(self.history_load_data),
            "history_loads": self.history_load_data,
            "aggregated_load": aggregate_history_loads(self.history_load_data),
        }
        output_dir = os.path.dirname(os.path.abspath(output_path))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def get_current_strategy_json(self) -> str:
        return self.current_strategy_json
