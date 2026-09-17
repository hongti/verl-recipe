from collections import defaultdict
from typing import Any


def _to_int(value: Any, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def aggregate_history_loads(history_loads):
    if isinstance(history_loads, dict):
        history_loads = [history_loads]
    aggregated = defaultdict(lambda: defaultdict(float))
    for step_load in history_loads:
        for layer_id, expert_load in step_load.items():
            layer_id = _to_int(layer_id, "layer_id")
            for expert_id, load in expert_load.items():
                aggregated[layer_id][_to_int(expert_id, "expert_id")] += float(load)
    return {layer_id: dict(expert_load) for layer_id, expert_load in aggregated.items()}


def get_device_count(config):
    if config.get("device_count") is not None:
        device_count = _to_int(config["device_count"], "device_count")
    elif config.get("num_devices") is not None:
        device_count = _to_int(config["num_devices"], "num_devices")
    else:
        num_nodes = _to_int(config.get("num_nodes", 1), "num_nodes")
        npu_per_node = _to_int(config.get("npu_per_node", config.get("n_gpus_per_node", 8)), "npu_per_node")
        device_count = num_nodes * npu_per_node
    if device_count <= 0:
        raise ValueError(f"device_count must be positive, got {device_count}")
    return device_count


def build_device_capacities(num_experts, device_count):
    if num_experts <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}")
    if device_count <= 0:
        raise ValueError(f"device_count must be positive, got {device_count}")
    if device_count > num_experts:
        raise ValueError(f"device_count ({device_count}) cannot exceed num_experts ({num_experts})")
    base = num_experts // device_count
    remainder = num_experts % device_count
    return [base + (1 if device_id < remainder else 0) for device_id in range(device_count)]


def compute_device_loads(placement, expert_load):
    return [sum(expert_load.get(expert_id, 0.0) for expert_id in experts) for experts in placement]


def _spread(loads):
    return max(loads) - min(loads) if loads else 0.0


def refine_placement_by_swap(placement, expert_load, max_iterations=1000):
    if len(placement) <= 1:
        return placement
    refined = [list(experts) for experts in placement]
    device_loads = compute_device_loads(refined, expert_load)

    for _ in range(max_iterations):
        current_spread = _spread(device_loads)
        max_device = max(range(len(device_loads)), key=lambda device_id: device_loads[device_id])
        min_device = min(range(len(device_loads)), key=lambda device_id: device_loads[device_id])
        best_swap = None
        best_loads = None
        best_spread = current_spread

        for high_expert in refined[max_device]:
            high_load = expert_load.get(high_expert, 0.0)
            for low_expert in refined[min_device]:
                low_load = expert_load.get(low_expert, 0.0)
                if high_load <= low_load:
                    continue
                candidate_loads = list(device_loads)
                candidate_loads[max_device] = candidate_loads[max_device] - high_load + low_load
                candidate_loads[min_device] = candidate_loads[min_device] - low_load + high_load
                candidate_spread = _spread(candidate_loads)
                if candidate_spread + 1e-12 < best_spread:
                    best_spread = candidate_spread
                    best_swap = (high_expert, low_expert)
                    best_loads = candidate_loads

        if best_swap is None:
            break

        high_expert, low_expert = best_swap
        refined[max_device].remove(high_expert)
        refined[min_device].remove(low_expert)
        refined[max_device].append(low_expert)
        refined[min_device].append(high_expert)
        device_loads = best_loads

    return refined


def generate_layer_placement(expert_load, num_experts, device_count, max_swap_iterations=1000):
    capacities = build_device_capacities(num_experts, device_count)
    normalized_load = {expert_id: float(expert_load.get(expert_id, 0.0)) for expert_id in range(num_experts)}
    sorted_experts = sorted(range(num_experts), key=lambda expert_id: (-normalized_load[expert_id], expert_id))
    placement = [[] for _ in range(device_count)]
    device_loads = [0.0 for _ in range(device_count)]

    for expert_id in sorted_experts:
        candidates = [
            device_id for device_id, capacity in enumerate(capacities) if len(placement[device_id]) < capacity
        ]
        target_device = min(
            candidates, key=lambda device_id: (device_loads[device_id], len(placement[device_id]), device_id)
        )
        placement[target_device].append(expert_id)
        device_loads[target_device] += normalized_load[expert_id]

    return refine_placement_by_swap(placement, normalized_load, max_iterations=max_swap_iterations)


def generate_eplb_strategy(history_loads, eplb_config):
    num_experts = _to_int(eplb_config.get("num_experts"), "num_experts")
    device_count = get_device_count(eplb_config)
    strategy_version = _to_int(eplb_config.get("strategy_version", 1), "strategy_version")
    max_swap_iterations = _to_int(eplb_config.get("max_swap_iterations", 1000), "max_swap_iterations")
    aggregated_loads = aggregate_history_loads(history_loads)

    layer_list = []
    for layer_id in sorted(aggregated_loads):
        placement = generate_layer_placement(
            aggregated_loads[layer_id],
            num_experts=num_experts,
            device_count=device_count,
            max_swap_iterations=max_swap_iterations,
        )
        layer_list.append(
            {
                "layer_id": layer_id,
                "device_list": [
                    {"device_id": device_id, "device_expert": experts} for device_id, experts in enumerate(placement)
                ],
            }
        )

    return {
        "strategy_version": strategy_version,
        "moe_layer_count": len(layer_list),
        "layer_list": layer_list,
    }


def eplb_strategy(history_loads, eplb_config):
    return generate_eplb_strategy(history_loads, eplb_config)
