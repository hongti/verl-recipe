import json

import pytest
from recipe.eplb.generate_strategy import generate


def test_generate_layout_for_distinct_training_and_rollout_ep(tmp_path):
    source = tmp_path / "load.json"
    source.write_text(json.dumps({"num_experts": 4, "history_loads": [{"0": {"0": 100, "1": 10, "2": 5}}]}))
    train = generate(source, tmp_path / "train.json", 2)
    rollout = generate(source, tmp_path / "rollout.json", 1)
    assert len(train["layer_list"][0]["device_list"]) == 2
    assert len(rollout["layer_list"][0]["device_list"]) == 1
    assert sorted(e for d in train["layer_list"][0]["device_list"] for e in d["device_expert"]) == list(range(4))


def test_unequal_training_capacity_rejected(tmp_path):
    source = tmp_path / "load.json"
    source.write_text(json.dumps({"num_experts": 4, "history_loads": [{"0": {"0": 1}}]}))
    with pytest.raises(ValueError, match="equal expert counts"):
        generate(source, tmp_path / "strategy.json", 3)
