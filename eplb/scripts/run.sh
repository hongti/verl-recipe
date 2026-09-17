#!/usr/bin/env bash
set -euo pipefail

mode="${1:-dynamic}"
if [[ $# -gt 0 ]]; then shift; fi
case "$mode" in
  baseline|collect|static|dynamic) ;;
  *) echo "Usage: $0 {baseline|collect|static|dynamic} [Hydra overrides...]" >&2; exit 2 ;;
esac

: "${MODEL_PATH:?Set MODEL_PATH to a supported MoE model}"
: "${TRAIN_FILES:?Set TRAIN_FILES to a training parquet file or Hydra list}"
: "${VAL_FILES:?Set VAL_FILES to a validation parquet file or Hydra list}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
package_parent="$(cd "$script_dir/../../.." && pwd)"
export PYTHONPATH="$package_parent${PYTHONPATH:+:$PYTHONPATH}"

overrides=()
if [[ "$mode" == static ]]; then
  : "${TRAIN_STRATEGY:?Set TRAIN_STRATEGY to the training layout JSON}"
  overrides+=("actor_rollout_ref.actor.train_eplb_strategy_path=$TRAIN_STRATEGY")
fi
if [[ -n "${ROLLOUT_STRATEGY:-}" ]]; then
  overrides+=("actor_rollout_ref.rollout.rollout_eplb_strategy_path=$ROLLOUT_STRATEGY")
fi

exec "${PYTHON:-python3}" -m recipe.eplb.main_eplb \
  "mode=$mode" \
  "actor_rollout_ref.model.path=$MODEL_PATH" \
  "data.train_files=$TRAIN_FILES" \
  "data.val_files=$VAL_FILES" \
  algorithm.adv_estimator=grpo \
  data.train_batch_size=8 \
  "data.filter_overlong_prompts_workers=${FILTER_WORKERS:-16}" \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.n=4 \
  "trainer.nnodes=${NNODES:-1}" \
  "trainer.n_gpus_per_node=${NPUS_PER_NODE:-8}" \
  "actor_rollout_ref.actor.megatron.expert_model_parallel_size=${TRAIN_EP:-8}" \
  "actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP:-8}" \
  "actor_rollout_ref.rollout.expert_parallel_size=${ROLLOUT_EP:-8}" \
  "${overrides[@]}" "$@"
