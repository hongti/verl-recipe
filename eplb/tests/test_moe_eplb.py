import json
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from recipe.eplb import strategy_manager

from verl.utils import moe_eplb as core_moe_eplb

# The strategy format lives in verl (worker code turns a strategy into a
# placement); the policy layer lives here. Tests reach both through one alias.
moe_eplb = SimpleNamespace(**{**vars(core_moe_eplb), **vars(strategy_manager)})

EPLBStrategyManager = moe_eplb.EPLBStrategyManager
load_strategy_from_path = moe_eplb.load_strategy_from_path
normalize_eplb_mode = moe_eplb.normalize_eplb_mode
should_collect_eplb_load = moe_eplb.should_collect_eplb_load
should_defer_eplb_training_strategy = moe_eplb.should_defer_eplb_training_strategy
strategies_equal = moe_eplb.strategies_equal
strategy_to_placement = moe_eplb.strategy_to_placement
validate_eplb_strategy = moe_eplb.validate_eplb_strategy

from recipe.eplb import load_balanced_expert_placement  # noqa: E402


class TestEPLBStrategyManager(unittest.TestCase):
    def test_extract_and_store_load_counts_all_valid_tokens(self):
        routed_experts = torch.tensor(
            [
                [
                    [[0, 1], [1, 2]],
                    [[2, 3], [0, 1]],
                    [[1, 1], [2, 2]],
                    [[3, 3], [3, 3]],
                ],
                [
                    [[0, 0], [0, 0]],
                    [[0, 2], [1, 1]],
                    [[2, 2], [0, 3]],
                    [[3, 0], [2, 1]],
                ],
            ],
            dtype=torch.int32,
        )
        attention_mask = torch.tensor(
            [
                [1, 1, 1, 0],
                [0, 1, 1, 1],
            ],
            dtype=torch.bool,
        )
        batch = SimpleNamespace(
            batch={
                "routed_experts": routed_experts,
                "attention_mask": attention_mask,
                "response_mask": attention_mask[:, -2:],
            }
        )
        manager = EPLBStrategyManager({"num_experts": 4, "update_interval": 2})

        load = manager.extract_and_store_load(batch, global_step=7)

        self.assertEqual(
            load,
            {
                0: {0: 3, 1: 3, 2: 4, 3: 2},
                1: {0: 2, 1: 5, 2: 4, 3: 1},
            },
        )
        self.assertEqual(manager.history_load_data, [load])
        self.assertEqual(manager.internal_step, 1)
        self.assertEqual(manager.global_step, 7)

    def test_extract_and_store_load_ignores_padded_expert_ids(self):
        routed_experts = torch.tensor(
            [[[[0, 1]], [[2, 3]], [[255, 255]]]],
            dtype=torch.uint8,
        )
        attention_mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
        batch = SimpleNamespace(
            batch={
                "routed_experts": routed_experts,
                "attention_mask": attention_mask,
            }
        )
        manager = EPLBStrategyManager({"num_experts": 4})

        load = manager.extract_and_store_load(batch)

        self.assertEqual(load, {0: {0: 1, 1: 1, 2: 1, 3: 1}})

    def test_extract_and_store_load_rejects_valid_out_of_range_expert(self):
        batch = SimpleNamespace(
            batch={
                "routed_experts": torch.tensor([[[[0, 4]]]], dtype=torch.int64),
                "attention_mask": torch.tensor([[1]], dtype=torch.bool),
            }
        )
        manager = EPLBStrategyManager({"num_experts": 4})

        with self.assertRaisesRegex(ValueError, "outside"):
            manager.extract_and_store_load(batch)

    def test_update_strategy_uses_external_function_and_clears_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            strategy_file = Path(tmpdir) / "strategy.py"
            strategy_file.write_text(
                textwrap.dedent(
                    """
                    def eplb_strategy(history_loads, eplb_config):
                        return {
                            "strategy_version": 9,
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
                    """
                ),
                encoding="utf-8",
            )
            manager = EPLBStrategyManager(
                {
                    "num_experts": 4,
                    "num_nodes": 1,
                    "n_gpus_per_node": 2,
                    "update_interval": 1,
                    "eplb_strategy_file_path": str(strategy_file),
                }
            )
            manager.history_load_data.append({0: {0: 10}})
            manager.internal_step = 1

            strategy = manager.update_strategy()

        self.assertEqual(strategy["strategy_version"], 9)
        self.assertEqual(manager.history_load_data, [])

    def test_dynamic_update_reuses_training_strategy_when_train_and_rollout_ep_match(self):
        calls = []

        def strategy_func(history_loads, eplb_config):
            calls.append(eplb_config["device_count"])
            device_count = eplb_config["device_count"]
            experts_per_device = eplb_config["num_experts"] // device_count
            return {
                "strategy_version": len(calls),
                "moe_layer_count": 1,
                "layer_list": [
                    {
                        "layer_id": 0,
                        "device_list": [
                            {
                                "device_id": device_id,
                                "device_expert": list(
                                    range(device_id * experts_per_device, (device_id + 1) * experts_per_device)
                                ),
                            }
                            for device_id in range(device_count)
                        ],
                    }
                ],
            }

        manager = EPLBStrategyManager(
            {
                "num_experts": 4,
                "training_device_count": 2,
                "rollout_device_count": 2,
                "update_interval": 1,
            }
        )
        manager.strategy_func = strategy_func
        manager.history_load_data.append({0: {0: 10}})
        manager.internal_step = 1

        strategy = manager.update_strategy()

        self.assertEqual(calls, [2])
        self.assertIs(manager.current_training_strategy, strategy)
        self.assertIs(manager.current_inference_strategy, strategy)
        self.assertEqual(len(manager.current_training_strategy["layer_list"][0]["device_list"]), 2)

    def test_dynamic_update_increments_strategy_version(self):
        versions = []

        def strategy_func(history_loads, eplb_config):
            versions.append(eplb_config["strategy_version"])
            return {
                "strategy_version": eplb_config["strategy_version"],
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

        manager = EPLBStrategyManager(
            {
                "num_experts": 4,
                "training_device_count": 2,
                "rollout_device_count": 2,
                "update_interval": 1,
            }
        )
        manager.strategy_func = strategy_func

        manager.history_load_data.append({0: {0: 10}})
        manager.internal_step = 1
        first_strategy = manager.update_strategy()
        manager.history_load_data.append({0: {1: 10}})
        manager.internal_step = 2
        second_strategy = manager.update_strategy()

        self.assertEqual(versions, [1, 2])
        self.assertEqual(first_strategy["strategy_version"], 1)
        self.assertEqual(second_strategy["strategy_version"], 2)

    def test_dynamic_update_generates_training_and_rollout_strategies_when_ep_differs(self):
        calls = []

        def strategy_func(history_loads, eplb_config):
            calls.append((eplb_config["placement_target"], eplb_config["device_count"]))
            device_count = eplb_config["device_count"]
            experts_per_device = eplb_config["num_experts"] // device_count
            return {
                "strategy_version": len(calls),
                "moe_layer_count": 1,
                "layer_list": [
                    {
                        "layer_id": 0,
                        "device_list": [
                            {
                                "device_id": device_id,
                                "device_expert": list(
                                    range(device_id * experts_per_device, (device_id + 1) * experts_per_device)
                                ),
                            }
                            for device_id in range(device_count)
                        ],
                    }
                ],
            }

        manager = EPLBStrategyManager(
            {
                "num_experts": 4,
                "training_device_count": 2,
                "rollout_device_count": 4,
                "update_interval": 1,
            }
        )
        manager.strategy_func = strategy_func
        manager.history_load_data.append({0: {0: 10}})
        manager.internal_step = 1

        training_strategy = manager.update_strategy()

        self.assertEqual(calls, [("training", 2), ("rollout", 4)])
        self.assertIs(manager.current_training_strategy, training_strategy)
        self.assertIsNot(manager.current_inference_strategy, training_strategy)
        self.assertEqual(len(manager.current_training_strategy["layer_list"][0]["device_list"]), 2)
        self.assertEqual(len(manager.current_inference_strategy["layer_list"][0]["device_list"]), 4)

    def test_export_load_writes_history_and_aggregated_load(self):
        manager = EPLBStrategyManager({"num_experts": 4})
        manager.history_load_data = [
            {0: {0: 1, 1: 2}},
            {0: {1: 3}, 1: {2: 4}},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "load.json"

            manager.export_load(str(output_path), metadata={"steps": [1, 2]})

            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["sample_count"], 2)
        self.assertEqual(payload["metadata"], {"steps": [1, 2]})
        self.assertEqual(payload["aggregated_load"], {"0": {"0": 1, "1": 5}, "1": {"2": 4}})

    def test_loads_rollout_strategy_from_named_strategy_path(self):
        strategy = {
            "strategy_version": 1,
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
        with tempfile.TemporaryDirectory() as tmpdir:
            strategy_path = Path(tmpdir) / "rollout_strategy.json"
            strategy_path.write_text(json.dumps(strategy), encoding="utf-8")

            manager = EPLBStrategyManager({"rollout_eplb_strategy_path": str(strategy_path)})

        self.assertEqual(manager.current_inference_strategy, strategy)

    def test_applied_training_strategy_becomes_rollout_strategy_when_rollout_strategy_is_unset(self):
        strategy = {
            "strategy_version": 2,
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
        manager = EPLBStrategyManager({"num_experts": 4})

        manager.mark_training_strategy_applied(strategy)

        self.assertEqual(manager.current_inference_strategy, strategy)


class TestEPLBModeHelpers(unittest.TestCase):
    def test_normalize_eplb_mode_accepts_only_static_or_dynamic(self):
        self.assertEqual(normalize_eplb_mode("static"), "Static")
        self.assertEqual(normalize_eplb_mode("Dynamic"), "Dynamic")

        with self.assertRaisesRegex(ValueError, "eplb_mode"):
            normalize_eplb_mode("relocate")

    def test_should_collect_load_only_for_dynamic_or_collection_task(self):
        self.assertFalse(
            should_collect_eplb_load(
                eplb_manager_present=True,
                eplb_enable=True,
                eplb_mode="Static",
                eplb_load_collection_enable=False,
            )
        )
        self.assertTrue(
            should_collect_eplb_load(
                eplb_manager_present=True,
                eplb_enable=True,
                eplb_mode="Dynamic",
                eplb_load_collection_enable=False,
            )
        )
        self.assertTrue(
            should_collect_eplb_load(
                eplb_manager_present=True,
                eplb_enable=False,
                eplb_mode="Static",
                eplb_load_collection_enable=True,
            )
        )
        self.assertFalse(
            should_collect_eplb_load(
                eplb_manager_present=False,
                eplb_enable=True,
                eplb_mode="Dynamic",
                eplb_load_collection_enable=True,
            )
        )

    def test_should_defer_training_strategy_only_for_runtime_dynamic(self):
        self.assertFalse(should_defer_eplb_training_strategy(eplb_mode="Static", is_initial_strategy=False))
        self.assertFalse(should_defer_eplb_training_strategy(eplb_mode="Dynamic", is_initial_strategy=True))
        self.assertTrue(should_defer_eplb_training_strategy(eplb_mode="Dynamic", is_initial_strategy=False))


class TestEPLBStrategyValidation(unittest.TestCase):
    def test_validate_strategy_rejects_duplicate_or_missing_experts(self):
        strategy = {
            "strategy_version": 1,
            "moe_layer_count": 1,
            "layer_list": [
                {
                    "layer_id": 0,
                    "device_list": [
                        {"device_id": 0, "device_expert": [0, 1]},
                        {"device_id": 1, "device_expert": [1, 3]},
                    ],
                }
            ],
        }

        with self.assertRaisesRegex(ValueError, "exactly once"):
            validate_eplb_strategy(strategy, num_experts=4, device_count=2)

    def test_validate_strategy_rejects_wrong_device_count(self):
        strategy = {
            "strategy_version": 1,
            "moe_layer_count": 1,
            "layer_list": [
                {
                    "layer_id": 0,
                    "device_list": [
                        {"device_id": 0, "device_expert": [0, 1, 2, 3]},
                    ],
                }
            ],
        }

        with self.assertRaisesRegex(ValueError, "device count"):
            validate_eplb_strategy(strategy, num_experts=4, device_count=2)

    def test_strategy_to_placement_uses_layer_id_not_list_position(self):
        strategy = {
            "strategy_version": 3,
            "moe_layer_count": 2,
            "layer_list": [
                {
                    "layer_id": 2,
                    "device_list": [
                        {"device_id": 0, "device_expert": [2, 3]},
                        {"device_id": 1, "device_expert": [0, 1]},
                    ],
                },
                {
                    "layer_id": 0,
                    "device_list": [
                        {"device_id": 0, "device_expert": [0, 3]},
                        {"device_id": 1, "device_expert": [1, 2]},
                    ],
                },
            ],
        }

        placement = strategy_to_placement(strategy, num_experts=4, device_count=2)

        self.assertEqual(
            placement,
            {
                0: {0: [0, 3], 1: [1, 2]},
                2: {0: [2, 3], 1: [0, 1]},
            },
        )

    def test_load_strategy_from_path_treats_empty_values_as_none(self):
        self.assertIsNone(load_strategy_from_path(""))
        self.assertIsNone(load_strategy_from_path("None"))
        self.assertIsNone(load_strategy_from_path(None))

    def test_strategies_equal_ignores_device_order_but_not_slot_order(self):
        left = {
            "moe_layer_count": 1,
            "layer_list": [
                {
                    "layer_id": 0,
                    "device_list": [
                        {"device_id": 1, "device_expert": [2, 3]},
                        {"device_id": 0, "device_expert": [0, 1]},
                    ],
                }
            ],
        }
        right = {
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
        different_slot_order = {
            "moe_layer_count": 1,
            "layer_list": [
                {
                    "layer_id": 0,
                    "device_list": [
                        {"device_id": 0, "device_expert": [1, 0]},
                        {"device_id": 1, "device_expert": [2, 3]},
                    ],
                }
            ],
        }

        self.assertTrue(strategies_equal(left, right))
        self.assertFalse(strategies_equal(left, different_slot_order))


class TestLoadBalancedExpertPlacement(unittest.TestCase):
    def test_generate_eplb_strategy_places_every_expert_once_per_layer(self):
        strategy = load_balanced_expert_placement.generate_eplb_strategy(
            history_loads=[
                {0: {0: 100, 1: 50, 2: 25, 3: 1}},
                {0: {0: 100, 2: 25}},
            ],
            eplb_config={"num_experts": 4, "device_count": 2, "strategy_version": 5},
        )

        validate_eplb_strategy(strategy, num_experts=4, device_count=2)
        self.assertEqual(strategy["strategy_version"], 5)
        self.assertEqual(strategy["moe_layer_count"], 1)
        for device in strategy["layer_list"][0]["device_list"]:
            self.assertEqual(len(device["device_expert"]), 2)


if __name__ == "__main__":
    unittest.main()
