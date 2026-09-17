# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import sys
import types
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_weight_update_utils():
    module_path = _REPO_ROOT / "verl/workers/rollout/vllm_rollout/weight_update_utils.py"
    spec = importlib.util.spec_from_file_location("weight_update_utils", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


_weight_update_utils = _load_weight_update_utils()


def _load_vllm_rollout_utils():
    """Load vllm_rollout/utils.py with heavyweight deps stubbed.

    Injected ``sys.modules`` entries are restored afterwards so the fakes do not
    leak into other tests; the loaded module keeps working since it binds the
    names it needs at import time.
    """
    module_name = "verl.workers.rollout.vllm_rollout.utils"
    module_path = _REPO_ROOT / "verl/workers/rollout/vllm_rollout/utils.py"

    fake_outputs = types.ModuleType("vllm.outputs")

    class _FakeRequestOutput:
        pass

    fake_outputs.RequestOutput = _FakeRequestOutput
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.outputs = fake_outputs

    fake_vllm_third_party = types.ModuleType("verl.third_party.vllm")
    fake_vllm_third_party.VLLM_SLEEP_LEVEL = 1
    fake_vllm_third_party.get_version = lambda pkg: "0.8.0"

    fake_vllm_utils = types.ModuleType("verl.utils.vllm")

    class _FakeTensorLoRARequest:
        pass

    class _FakeVLLMHijack:
        @staticmethod
        def hijack():
            return None

    fake_vllm_utils.TensorLoRARequest = _FakeTensorLoRARequest
    fake_vllm_utils.VLLMHijack = _FakeVLLMHijack

    fake_vllm_patch = types.ModuleType("verl.utils.vllm.patch")
    fake_vllm_patch.patch_vllm_moe_model_weight_loader = lambda model: None

    fake_vllm_fp8 = types.ModuleType("verl.utils.vllm.vllm_fp8_utils")
    fake_vllm_fp8.apply_vllm_fp8_patches = lambda: None
    fake_vllm_fp8.is_fp8_model = lambda config: False
    fake_vllm_fp8.load_quanted_weights = lambda weights, runner, is_drafter=False: weights

    fake_platform = types.ModuleType("verl.plugin.platform")
    fake_platform.get_platform = lambda: None

    fakes = {
        "vllm": fake_vllm,
        "vllm.outputs": fake_outputs,
        "verl.third_party.vllm": fake_vllm_third_party,
        "verl.utils.vllm": fake_vllm_utils,
        "verl.utils.vllm.patch": fake_vllm_patch,
        "verl.utils.vllm.vllm_fp8_utils": fake_vllm_fp8,
        "verl.plugin.platform": fake_platform,
        "verl.workers.rollout.vllm_rollout.weight_update_utils": _weight_update_utils,
    }

    saved = {name: sys.modules.get(name) for name in fakes}
    try:
        sys.modules.update(fakes)
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        for name, prev in saved.items():
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev
    return module


_vllm_rollout_utils = _load_vllm_rollout_utils()
vLLMColocateWorkerExtension = _vllm_rollout_utils.vLLMColocateWorkerExtension


def test_reorder_eplb_expert_weights_keeps_logical_order_without_training_strategy():
    strategy = {
        "moe_layer_count": 1,
        "layer_list": [
            {
                "layer_id": 0,
                "device_list": [
                    {"device_id": 0, "device_expert": [2, 0]},
                    {"device_id": 1, "device_expert": [3, 1]},
                ],
            }
        ],
    }
    exported_in_logical_order = torch.tensor([[0], [1], [2], [3]], dtype=torch.float32)

    reordered = _vllm_rollout_utils._reorder_eplb_expert_weight_tensor(
        "model.layers.0.mlp.experts.gate_up_proj",
        exported_in_logical_order,
        strategy,
    )

    torch.testing.assert_close(
        reordered,
        exported_in_logical_order,
    )


def test_reorder_eplb_expert_weights_converts_training_order_to_logical_order():
    training_strategy = {
        "moe_layer_count": 1,
        "layer_list": [
            {
                "layer_id": 0,
                "device_list": [
                    {"device_id": 0, "device_expert": [3, 1]},
                    {"device_id": 1, "device_expert": [0, 2]},
                ],
            }
        ],
    }
    rollout_strategy = {
        "moe_layer_count": 1,
        "_training_strategy": training_strategy,
        "layer_list": [
            {
                "layer_id": 0,
                "device_list": [
                    {"device_id": 0, "device_expert": [2, 0]},
                    {"device_id": 1, "device_expert": [3, 1]},
                ],
            }
        ],
    }
    exported_in_training_order = torch.tensor([[30], [10], [0], [20]], dtype=torch.float32)

    reordered = _vllm_rollout_utils._reorder_eplb_expert_weight_tensor(
        "model.layers.0.mlp.experts.down_proj",
        exported_in_training_order,
        rollout_strategy,
    )

    torch.testing.assert_close(
        reordered,
        torch.tensor([[0], [10], [20], [30]], dtype=torch.float32),
    )


def test_reorder_eplb_per_expert_names_from_training_slots_to_logical_experts():
    training_strategy = {
        "moe_layer_count": 1,
        "layer_list": [
            {
                "layer_id": 0,
                "device_list": [
                    {"device_id": 0, "device_expert": [3, 1]},
                    {"device_id": 1, "device_expert": [0, 2]},
                ],
            }
        ],
    }
    rollout_strategy = {
        "moe_layer_count": 1,
        "_training_strategy": training_strategy,
        "layer_list": [
            {
                "layer_id": 0,
                "device_list": [
                    {"device_id": 0, "device_expert": [2, 0]},
                    {"device_id": 1, "device_expert": [3, 1]},
                ],
            }
        ],
    }
    tensor = torch.tensor([1.0])

    updates = _vllm_rollout_utils._reorder_eplb_expert_weight_updates(
        [("model.layers.0.mlp.experts.2.down_proj.weight", tensor)],
        rollout_strategy,
    )

    assert updates == [("model.layers.0.mlp.experts.0.down_proj.weight", tensor)]


class _FakeVllmConfig:
    def __init__(self, speculative_config=None):
        self.speculative_config = speculative_config


class _FakeModelRunner:
    def __init__(self, inner_model, speculative_config=None):
        self.model = inner_model
        self.vllm_config = _FakeVllmConfig(speculative_config=speculative_config)


class _FakeMoeConfig:
    ep_size = 2
    ep_rank = 0
    num_experts = 4


class _FakeFusedMoe:
    def __init__(self):
        self.moe_config = _FakeMoeConfig()
        self._expert_map = torch.full((4,), -1, dtype=torch.int32)
        self.log2phy = torch.empty(4, dtype=torch.int32)
        self.global_expert_map = None

    def update_expert_map(self, new_expert_map):
        self._expert_map = new_expert_map


class _FakeMoeModel(torch.nn.Module):
    def __init__(self, fused_moe):
        super().__init__()
        self.model = types.SimpleNamespace(layers=[types.SimpleNamespace(mlp=types.SimpleNamespace(experts=fused_moe))])
        self.loaded_weights = []

    def load_weights(self, weights):
        expected_expert_map = torch.tensor([-1, 1, 0, -1], dtype=torch.int32)
        torch.testing.assert_close(self.model.layers[0].mlp.experts._expert_map, expected_expert_map)
        self.loaded_weights.append(weights)


def test_vllm_ipc_update_applies_rollout_expert_map_before_loading_params(monkeypatch):
    fake_eplb_utils = types.ModuleType("vllm_ascend.eplb.core.eplb_utils")
    fake_eplb_utils.generate_log2phy_map = lambda global_map, ep_rank: torch.arange(
        global_map.shape[1], dtype=torch.int32
    )

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.__path__ = []
    fake_vllm_platforms = types.ModuleType("vllm.platforms")
    fake_vllm_platforms.current_platform = types.SimpleNamespace(device_type="cpu")
    fake_vllm.platforms = fake_vllm_platforms

    fake_weight_transfer = types.ModuleType("verl.workers.rollout.vllm_rollout.bucketed_weight_transfer")

    class _FakeBucketedWeightReceiver:
        def __init__(self, **kwargs):
            pass

        def receive_weights(self, on_bucket_received):
            on_bucket_received([("model.layers.0.linear.weight", torch.ones(1, dtype=torch.float32))])

    fake_weight_transfer.BucketedWeightReceiver = _FakeBucketedWeightReceiver

    fake_qat = types.ModuleType("verl.utils.qat")
    fake_qat.prepare_qat_for_load_weights = lambda model, device: None
    fake_qat.manual_process_weights_after_loading = lambda model: None

    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.platforms", fake_vllm_platforms)
    monkeypatch.setitem(sys.modules, "vllm_ascend.eplb.core.eplb_utils", fake_eplb_utils)
    monkeypatch.setitem(
        sys.modules,
        "verl.workers.rollout.vllm_rollout.bucketed_weight_transfer",
        fake_weight_transfer,
    )
    monkeypatch.setitem(sys.modules, "verl.utils.qat", fake_qat)

    fused_moe = _FakeFusedMoe()
    model = _FakeMoeModel(fused_moe)
    worker = object.__new__(vLLMColocateWorkerExtension)
    worker.model_runner = _FakeModelRunner(model)
    worker.device = torch.device("cpu")
    worker.local_rank = 0
    worker._is_qat_model = True
    worker._is_modelopt_qat = False
    worker._get_zmq_handle = lambda: "test"
    strategy = {
        "moe_layer_count": 1,
        "layer_list": [
            {
                "layer_id": 0,
                "device_list": [
                    {"device_id": 0, "device_expert": [2, 1]},
                    {"device_id": 1, "device_expert": [3, 0]},
                ],
            }
        ],
    }

    # Simulate a stale cross-RPC cache key colliding with this strategy object.
    worker._eplb_weight_load_map_cache_key = id(strategy)
    worker.update_weights_from_ipc(eplb_strategy=strategy)

    assert model.loaded_weights
