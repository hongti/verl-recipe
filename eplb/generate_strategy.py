"""Generate training and rollout layouts from an exported load snapshot."""

import argparse
import json
from pathlib import Path

from verl.utils.moe_eplb import validate_eplb_strategy

from .load_balanced_expert_placement import generate_eplb_strategy


def generate(load_path, output_path, device_count, strategy_version=1):
    payload = json.loads(Path(load_path).read_text(encoding="utf-8"))
    num_experts = int(payload["num_experts"])
    if device_count <= 0 or num_experts % device_count:
        raise ValueError("Training EPLB requires equal expert counts on all EP ranks.")
    history = payload.get("history_loads") or [payload.get("aggregated_load", {})]
    if not any(history):
        raise ValueError("The load file contains no expert observations.")
    strategy = generate_eplb_strategy(
        history,
        {"num_experts": num_experts, "device_count": device_count, "strategy_version": strategy_version},
    )
    validate_eplb_strategy(strategy, num_experts=num_experts, device_count=device_count)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(strategy, indent=2) + "\n", encoding="utf-8")
    return strategy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ep", type=int, required=True)
    parser.add_argument("--version", type=int, default=1)
    args = parser.parse_args()
    generate(args.load, args.output, args.ep, args.version)


if __name__ == "__main__":
    main()
