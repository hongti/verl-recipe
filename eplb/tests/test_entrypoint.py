from unittest.mock import Mock

from recipe.eplb import main_eplb

from verl.trainer.main_ppo import TaskRunner


def test_main_selects_eplb_runner_through_native_launcher(config_factory, monkeypatch):
    config = config_factory("baseline")
    selected = []
    actor_class = object()

    def remote(**options):
        def decorate(cls):
            selected.append(cls)
            return actor_class

        return decorate

    launcher = Mock()
    monkeypatch.setattr(main_eplb.ray, "remote", remote)
    monkeypatch.setattr(main_eplb, "auto_set_device", lambda cfg: None)
    monkeypatch.setattr(main_eplb, "migrate_legacy_reward_impl", lambda cfg: cfg)
    monkeypatch.setattr(main_eplb, "run_ppo", launcher)
    main_eplb.main.__wrapped__(config)
    assert selected == [main_eplb.EPLBTaskRunner]
    assert launcher.call_args.kwargs["task_runner_class"] is actor_class


def test_runner_builds_recipe_trainer_and_reuses_dataset_helpers(config_factory, monkeypatch):
    import verl.utils
    import verl.utils.fs

    config = config_factory("baseline", "actor_rollout_ref.rollout.prompt_length=128")
    runner = main_eplb.EPLBTaskRunner()
    helpers = (
        "add_critic_worker",
        "add_reward_model_resource_pool",
        "add_teacher_model_resource_pool",
        "add_ref_policy_worker",
    )
    for name in helpers:
        monkeypatch.setattr(runner, name, Mock())
    monkeypatch.setattr(runner, "add_actor_rollout_worker", Mock(return_value=("worker", "worker_group")))
    monkeypatch.setattr(runner, "init_resource_pool_mgr", Mock(return_value="pool"))
    monkeypatch.setattr(main_eplb, "validate_config", Mock())
    monkeypatch.setattr(verl.utils.fs, "copy_to_local", Mock(return_value="model"))
    monkeypatch.setattr(verl.utils, "hf_tokenizer", Mock(return_value="tokenizer"))
    monkeypatch.setattr(verl.utils, "hf_processor", Mock(return_value="processor"))
    datasets = Mock(side_effect=["train", "val"])
    monkeypatch.setattr(main_eplb, "create_rl_dataset", datasets)
    monkeypatch.setattr(main_eplb, "create_rl_sampler", Mock(return_value="sampler"))
    trainer = Mock()
    factory = Mock(return_value=trainer)
    monkeypatch.setattr(main_eplb, "RayEPLBTrainer", factory)
    runner.run(config)
    assert datasets.call_count == 2
    assert factory.call_args.kwargs["train_dataset"] == "train"
    assert factory.call_args.kwargs["val_dataset"] == "val"
    assert factory.call_args.kwargs["train_sampler"] == "sampler"
    assert [call[0] for call in trainer.method_calls] == ["init_workers", "fit"]


def test_worker_and_resource_construction_are_inherited():
    for name in ("add_actor_rollout_worker", "add_critic_worker", "init_resource_pool_mgr"):
        assert getattr(main_eplb.EPLBTaskRunner, name) is getattr(TaskRunner, name)
