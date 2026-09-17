# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""EPLB recipe: inherit PPO operations; specialize only orchestration."""

import logging
import uuid
from pprint import pprint

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
    compute_spec_decode_metrics,
)
from verl.trainer.ppo.reward import extract_reward
from verl.trainer.ppo.utils import Role
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.moe_eplb import load_strategy_from_path, validate_eplb_strategy
from verl.utils.skip.skip_manager import SkipManager

from .configuration import validate_eplb_config
from .strategy_manager import (
    EPLBStrategyManager,
    normalize_eplb_mode,
    should_collect_eplb_load,
    should_defer_eplb_training_strategy,
)

# Named to avoid shadowing by the local `logger` (a Tracking instance) inside fit().
_eplb_log = logging.getLogger(__name__)


class _EPLBWeightPublisher:
    """Attach the live EPLB inference layout to every weight publish.

    `RayPPOTrainer._validate()` publishes weights through
    `self.checkpoint_manager.update_weights(global_steps)` and knows nothing about
    EPLB. Wrapping the manager lets that inherited path carry the layout without
    patching the core trainer or monkey patching it at runtime, so rollout never
    serves a layout the trainer has already moved past.
    """

    def __init__(self, inner, strategy_provider):
        self._inner = inner
        self._strategy_provider = strategy_provider

    def update_weights(self, global_steps, *args, eplb_strategy=None, **kwargs):
        if eplb_strategy is None:
            eplb_strategy = self._strategy_provider()
        return self._inner.update_weights(global_steps, *args, eplb_strategy=eplb_strategy, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _build_eplb_config(config, hf_config):
    num_experts = getattr(hf_config, "n_routed_experts", None)
    if num_experts is None:
        num_experts = getattr(hf_config, "num_experts", None)
    training_device_count = OmegaConf.select(
        config,
        "actor_rollout_ref.actor.megatron.expert_model_parallel_size",
        default=None,
    )
    if training_device_count is None:
        training_device_count = OmegaConf.select(
            config,
            "actor_rollout_ref.actor.mindspeed.expert_model_parallel_size",
            default=1,
        )
    rollout_device_count = OmegaConf.select(
        config,
        "actor_rollout_ref.rollout.expert_parallel_size",
        default=training_device_count,
    )
    if rollout_device_count is None:
        rollout_device_count = training_device_count
    return {
        "num_nodes": config.trainer.nnodes,
        "n_gpus_per_node": config.trainer.n_gpus_per_node,
        "eplb_strategy_file_path": config.actor_rollout_ref.rollout.get("eplb_strategy_file_path"),
        "rollout_eplb_strategy_path": config.actor_rollout_ref.rollout.get("rollout_eplb_strategy_path"),
        "num_experts": num_experts,
        "training_device_count": int(training_device_count),
        "rollout_device_count": int(rollout_device_count),
        "update_interval": config.trainer.eplb_update_interval,
        "load_output_path": config.trainer.get("eplb_load_output_path", ""),
    }


class RayEPLBTrainer(RayPPOTrainer):
    """Reuse native workers, datasets, PPO math, validation and checkpoints."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        validate_eplb_config(self.config)
        self.initial_eplb_strategy_checked = False
        self.eplb_training_strategy_applied = False
        self.eplb_load_exported = False
        self.pending_deferred_eplb_strategy = None
        self._queued_training_strategy = None
        self._applied_inference_strategy = None
        eplb_enable = self.config.actor_rollout_ref.actor.get("eplb_enable", False)
        eplb_load_collection_enable = self.config.trainer.get("eplb_load_collection_enable", False)
        if eplb_enable or eplb_load_collection_enable:
            from transformers import AutoConfig

            hf_config = AutoConfig.from_pretrained(
                self.config.actor_rollout_ref.model.path,
                trust_remote_code=self.config.actor_rollout_ref.model.trust_remote_code,
            )
            self.eplb_manager = EPLBStrategyManager(eplb_config=_build_eplb_config(self.config, hf_config))
        else:
            self.eplb_manager = None

        if eplb_enable and self.config.actor_rollout_ref.actor.eplb_mode == "Dynamic":
            if self.eplb_manager.strategy_func is None:
                raise ValueError("Dynamic EPLB strategy could not be loaded.")

    def _current_inference_eplb_strategy(self):
        if self.eplb_manager is None or not self.eplb_training_strategy_applied:
            return None
        inference_strategy = self._applied_inference_strategy
        if not isinstance(inference_strategy, dict):
            return inference_strategy

        import copy

        strategy_for_rollout = copy.deepcopy(inference_strategy)
        training_strategy = self.eplb_manager.applied_training_strategy
        if isinstance(training_strategy, dict):
            strategy_for_rollout["_training_strategy"] = training_strategy
        return strategy_for_rollout

    def init_workers(self):
        super().init_workers()
        # Route the inherited `_validate()` publish path through the layout wrapper.
        self.checkpoint_manager = _EPLBWeightPublisher(
            self.checkpoint_manager,
            self._current_inference_eplb_strategy,
        )

    def _update_rollout_weights(self, global_steps: int):
        self.checkpoint_manager.update_weights(global_steps)

    def _apply_training_eplb_strategy(
        self,
        strategy: dict,
        *,
        defer: bool = False,
        allow_load_offload: bool = False,
    ):
        if self.eplb_manager is not None and not self.eplb_manager.should_apply_training_strategy(strategy):
            return False
        results = self.actor_rollout_wg.expert_weight_and_optimizer_state_relocation(
            strategy,
            defer=defer,
            allow_load_offload=allow_load_offload,
        )
        if defer:
            # Enqueueing is not acknowledgement of migration. The engine executes
            # this inside train_mode, after loading weights and optimizer states.
            self._queued_training_strategy = strategy
        else:
            replies = results if isinstance(results, (list, tuple)) else [results]
            if not replies or any(not isinstance(reply, dict) or not reply.get("success") for reply in replies):
                raise RuntimeError(f"EPLB migration was not acknowledged by all workers: {results!r}")
            self._mark_training_strategy_applied(strategy)
        return True

    def _mark_training_strategy_applied(self, strategy):
        import copy

        self.eplb_manager.mark_training_strategy_applied(strategy)
        self._applied_inference_strategy = copy.deepcopy(self.eplb_manager.current_inference_strategy)
        self.eplb_training_strategy_applied = True
        # One line per actual layout change, not per step and not per rank: without
        # it there is no way to tell which placement a run is on.
        _eplb_log.info(
            "EPLB layout v%s applied at step %s across %s MoE layers",
            strategy.get("strategy_version") if isinstance(strategy, dict) else None,
            self.global_steps,
            strategy.get("moe_layer_count") if isinstance(strategy, dict) else None,
        )

    def _update_actor(self, batch):
        # The backend raises if deferred migration fails. Only a completed actor
        # update permits publishing the matching inference placement.
        result = super()._update_actor(batch)
        if self._queued_training_strategy is not None:
            self._mark_training_strategy_applied(self._queued_training_strategy)
            self._queued_training_strategy = None
        return result

    def _collect_eplb_load(self, batch, timing_raw):
        actor = self.config.actor_rollout_ref.actor
        if not should_collect_eplb_load(
            eplb_manager_present=self.eplb_manager is not None,
            eplb_enable=actor.eplb_enable,
            eplb_mode=actor.eplb_mode,
            eplb_load_collection_enable=self.config.trainer.eplb_load_collection_enable,
        ):
            return
        if "routed_experts" not in batch.batch:
            raise ValueError("EPLB load collection received no routed_experts from rollout.")
        with marked_timer("eplb_load_collection", timing_raw):
            self.eplb_manager.extract_and_store_load(batch, global_step=self.global_steps)
        if (
            self.config.trainer.eplb_load_collection_enable
            and not self.eplb_load_exported
            and self.eplb_manager.internal_step >= self.config.trainer.eplb_load_collection_steps
        ):
            self.eplb_manager.export_load(
                self.config.trainer.eplb_load_output_path,
                metadata={"global_step": self.global_steps},
            )
            self.eplb_load_exported = True
        if actor.eplb_enable and actor.eplb_mode == "Dynamic":
            new_strategy = self.eplb_manager.update_strategy()
            if new_strategy:
                self.pending_deferred_eplb_strategy = new_strategy
            elif self.eplb_manager.internal_step % self.eplb_manager.update_interval == 0:
                raise RuntimeError("Dynamic EPLB policy returned no valid strategy at the update boundary.")

    def _apply_initial_eplb_strategy_if_needed(self, *, update_rollout: bool = True):
        if self.initial_eplb_strategy_checked:
            return False
        self.initial_eplb_strategy_checked = True

        actor_config = self.config.actor_rollout_ref.actor
        if not actor_config.get("eplb_enable", False) or self.eplb_manager is None:
            return False

        initial_strategy = load_strategy_from_path(actor_config.get("train_eplb_strategy_path", ""))
        if initial_strategy is None:
            if actor_config.get("eplb_mode") == "Static":
                raise ValueError("Static EPLB requires a non-null training placement.")
            return False

        manager = self.eplb_manager
        if not validate_eplb_strategy(
            initial_strategy, num_experts=manager.num_experts, device_count=manager.training_device_count
        ):
            raise ValueError("The initial training placement contains no layers.")
        rollout_strategy = manager.current_inference_strategy
        if rollout_strategy is None:
            if manager.training_device_count != manager.rollout_device_count:
                raise ValueError("Different training and rollout EP sizes require a rollout placement file.")
            rollout_strategy = initial_strategy
        if not validate_eplb_strategy(
            rollout_strategy, num_experts=manager.num_experts, device_count=manager.rollout_device_count
        ):
            raise ValueError("The initial rollout placement contains no layers.")

        initial_strategy_applied = self._apply_training_eplb_strategy(
            initial_strategy,
            allow_load_offload=True,
        )
        if initial_strategy_applied and update_rollout:
            self._update_rollout_weights(self.global_steps)
        return initial_strategy_applied

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        if self._dump_executor._shutdown:
            self._init_dump_executor()

        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint and update weights before doing anything
        self._load_checkpoint()
        self._apply_initial_eplb_strategy_if_needed(update_rollout=False)
        self._update_rollout_weights(self.global_steps)

        current_epoch = self.global_steps // len(self.train_dataloader)

        SkipManager.init(self.config)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                self._shutdown_dump_executor()
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        SkipManager.set_step(self.global_steps)

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                actor_config = self.config.actor_rollout_ref.actor
                eplb_mode = normalize_eplb_mode(actor_config.get("eplb_mode", "Static"))
                self._apply_initial_eplb_strategy_if_needed()

                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                rollout_n = self.config.actor_rollout_ref.rollout.n
                gen_batch_output = gen_batch.repeat(repeat_times=rollout_n, interleave=True)

                if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                    # NOTE: REMAX needs one sampled rollout plus one greedy baseline per prompt.
                    # Keep them in a single agent-loop/vLLM request to avoid sending a second
                    # rollout after replicas have been put to sleep, which can leave async vLLM
                    # engines in an invalid state for multi-turn agent workloads.
                    gen_batch_output.non_tensor_batch["__do_sample__"] = np.ones(len(gen_batch_output), dtype=bool)
                    gen_baseline_batch = gen_batch.slice(0, None)
                    gen_baseline_batch.non_tensor_batch["__do_sample__"] = np.zeros(len(gen_baseline_batch), dtype=bool)
                    combined_gen_batch = DataProto.concat([gen_batch_output, gen_baseline_batch])
                    num_sampled_prompts = len(gen_batch_output)
                else:
                    combined_gen_batch = gen_batch_output
                    num_sampled_prompts = len(gen_batch_output)

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if curr_step_profile:
                            self.llm_server_manager.start_profile()
                        combined_gen_output = self.async_rollout_manager.generate_sequences(combined_gen_batch)
                        self.checkpoint_manager.sleep_replicas()
                        if curr_step_profile:
                            self.llm_server_manager.stop_profile()

                        timing_raw.update(combined_gen_output.meta_info["timing"])
                        combined_gen_output.meta_info.pop("timing", None)

                    gen_batch_output = combined_gen_output.slice(0, num_sampled_prompts)
                    if "__do_sample__" in gen_batch_output.non_tensor_batch:
                        gen_batch_output.pop(non_tensor_batch_keys=["__do_sample__"])

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        gen_baseline_output = combined_gen_output.slice(num_sampled_prompts, None)
                        if "__do_sample__" in gen_baseline_output.non_tensor_batch:
                            gen_baseline_output.pop(non_tensor_batch_keys=["__do_sample__"])

                        if self.use_rm and "rm_scores" not in gen_baseline_output.batch.keys():
                            baseline_reward = self._compute_reward_colocate(gen_baseline_output)
                            gen_baseline_output = gen_baseline_output.union(baseline_reward)

                        reward_baseline_tensor = gen_baseline_output.batch["rm_scores"].sum(dim=-1)
                        batch.batch["reward_baselines"] = reward_baseline_tensor

                        del gen_baseline_output
                    del combined_gen_batch, combined_gen_output
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    self._collect_eplb_load(batch, timing_raw)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    # get images_seqlens
                    images_seqlens_all = []
                    for multi_modal_input in batch.non_tensor_batch["multi_modal_inputs"]:
                        if "image_grid_thw" not in multi_modal_input.keys():
                            continue
                        images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
                    batch.meta_info["images_seqlens"] = images_seqlens_all
                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            batch_reward = self._compute_reward_colocate(batch)
                            batch = batch.union(batch_reward)

                        # extract reward_tensor and reward_extra_infos_dict for training
                        reward_tensor, reward_extra_infos_dict = extract_reward(batch)

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                        apply_bypass_mode(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            old_log_prob_metrics = {
                                "actor/entropy": entropy_agg.detach().item(),
                                "perf/mfu/actor_infer": old_log_prob_mfu,
                            }
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            if "routed_experts" in batch.batch and "routed_experts" in old_log_prob.batch:
                                raise ValueError(
                                    "Detected conflicting router replay configuration: "
                                    "router_replay.mode='R2' and enable_rollout_routing_replay=True "
                                    "cannot be enabled simultaneously. "
                                    "The enable_rollout_routing_replay option is only used in R3 mode; "
                                    "it should not be set when using R2 mode."
                                )
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            ref_log_prob = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self._update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup > self.global_steps:
                        # Still in critic warmup, only update weights to wake up rollout replicas.
                        self._update_rollout_weights(self.global_steps)
                    else:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            if self.pending_deferred_eplb_strategy is not None:
                                deferred_eplb_strategy = self.pending_deferred_eplb_strategy
                                self.pending_deferred_eplb_strategy = None
                                self._apply_training_eplb_strategy(
                                    deferred_eplb_strategy,
                                    defer=should_defer_eplb_training_strategy(
                                        eplb_mode=eplb_mode,
                                        is_initial_strategy=False,
                                    ),
                                )
                            actor_output = self._update_actor(batch)

                        # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                        esi_close_to_expiration = should_save_ckpt_esi(
                            max_steps_duration=self.max_steps_duration,
                            redundant_time=self.config.trainer.esi_redundant_time,
                        )
                        # Check if the conditions for saving a checkpoint are met.
                        # The conditions include a mandatory condition (1) and
                        # one of the following optional conditions (2/3/4):
                        # 1. The save frequency is set to a positive value.
                        # 2. It's the last training step.
                        # 3. The current step number is a multiple of the save frequency.
                        # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                        if self.config.trainer.save_freq > 0 and (
                            is_last_step
                            or self.global_steps % self.config.trainer.save_freq == 0
                            or esi_close_to_expiration
                        ):
                            if esi_close_to_expiration:
                                print("Force saving checkpoint: ESI instance expiration approaching.")
                            with marked_timer("save_checkpoint", timing_raw, color="green"):
                                self._save_checkpoint()

                        # update weights from trainer to rollout
                        with marked_timer("update_weights", timing_raw, color="red"):
                            self._update_rollout_weights(self.global_steps)

                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                # GDPO per-component reward metrics
                gdpo_reward_keys = self.config.algorithm.get("gdpo_reward_keys", None)
                if gdpo_reward_keys and self.config.algorithm.adv_estimator in ("gdpo", AdvantageEstimator.GDPO):
                    for key in gdpo_reward_keys:
                        if key in batch.non_tensor_batch:
                            vals = np.asarray(batch.non_tensor_batch[key], dtype=np.float32)
                            metrics[f"gdpo/{key}/mean"] = float(np.mean(vals))
                            metrics[f"gdpo/{key}/std"] = float(np.std(vals))
                            metrics[f"gdpo/{key}/max"] = float(np.max(vals))
                            metrics[f"gdpo/{key}/min"] = float(np.min(vals))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # compute variance proxy metrics
                gradient_norm = metrics.get("actor/grad_norm", None)
                metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # Per-request spec decode metrics.
                metrics.update(
                    compute_spec_decode_metrics(
                        batch.non_tensor_batch.get("spec_num_draft_tokens", None),
                        batch.non_tensor_batch.get("spec_num_accepted_tokens", None),
                        batch.non_tensor_batch.get("spec_num_verify_steps", None),
                    )
                )

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                SkipManager.set_step(self.global_steps)

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    self._shutdown_dump_executor()
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)

        # Ensure dump executor is shut down when training loop ends without reaching is_last_step
        self._shutdown_dump_executor()
