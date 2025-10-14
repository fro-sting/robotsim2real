import optuna
from optuna.pruners import MedianPruner
from humanoid.envs.custom import finetuning_policy_config
from humanoid.utils import get_args, task_registry
import torch
import copy


from optuna.samplers import RandomSampler
from optuna.samplers import TPESampler
from optuna.samplers import CmaEsSampler


import numpy as np
import sys
sys.argv = [
    "optuna_finetune.py",
    "--task=finetuning_policy_ppo",
    "--num_envs", "128",
    "--headless",
]

global_env = None
global_runner = None

def get_global_env_runner():
    global global_env, global_runner
    if global_env is None or global_runner is None:
        args = get_args()
        global_env, env_cfg = task_registry.make_env(name=args.task, args=args)
        global_runner, train_cfg = task_registry.make_alg_runner(env=global_env, name=args.task, args=args)

        # ✅ 保存初始化的 policy 状态，用于后续 reset
        global_runner.initial_policy_state = copy.deepcopy(global_runner.alg.actor_critic.state_dict())

    return global_env, global_runner

def reset_policy_and_optimizer(runner, config):
    # 重新初始化policy参数
    # reset前clone所有会inplace update的buffer
    if hasattr(runner, 'scheduler'):
        runner.scheduler = None
    if hasattr(runner, 'storage'):
        runner.storage.reset()  # 或者 runner.storage = create_new_buffer()
    runner.alg.actor_critic.load_state_dict(copy.deepcopy(runner.initial_policy_state))
    # 重新初始化optimizer
    if hasattr(runner, 'optimizer'):
        del runner.alg.optimizer  # 清掉旧引用
        runner.optimizer = torch.optim.Adam(runner.alg.actor_critic.parameters(), lr=config.algorithm.learning_rate)


def evaluate_policy(runner, env, n_episodes=5):
    """
    用当前policy评估n_episodes集，返回平均reward
    """
    policy = runner.get_inference_policy(device=runner.device)
    rewards = []
    clone_buffers = [
        'episode_phase_buf', 'actions', 'last_actions', 'last_last_actions',
        'last_rigid_state', 'last_dof_vel', 'last_root_vel', 'reset_buf',
        'feet_air_time', 'command_stand'
    ]
    for _ in range(n_episodes):
        
        for buf in clone_buffers:
            if hasattr(env, buf):
                setattr(env, buf, getattr(env, buf).clone())
        if hasattr(env, 'obs_history'):
            for i in range(len(env.obs_history)):
                env.obs_history[i] = env.obs_history[i].clone()
        if hasattr(env, 'critic_history'):
            for i in range(len(env.critic_history)):
                env.critic_history[i] = env.critic_history[i].clone()
        obs, _ = env.reset()
        # reset后再clone一次
        for buf in clone_buffers:
            if hasattr(env, buf):
                setattr(env, buf, getattr(env, buf).clone())
        if hasattr(env, 'obs_history'):
            for i in range(len(env.obs_history)):
                env.obs_history[i] = env.obs_history[i].clone()
        if hasattr(env, 'critic_history'):
            for i in range(len(env.critic_history)):
                env.critic_history[i] = env.critic_history[i].clone()
        done = torch.zeros(env.num_envs, dtype=torch.bool, device=runner.device)
        total_reward = torch.zeros(env.num_envs, device=runner.device)
        while not torch.all(done):
            with torch.no_grad():
                action = policy(obs)
            obs, critic_obs, rews, dones, infos, _ = env.step(action)
            total_reward += rews.squeeze() * (~done)
            done |= dones.squeeze().bool()
        rewards.append(total_reward.detach().cpu().numpy())
    return np.mean(rewards)

def objective(trial):
    # 1. 采样超参数
    lr = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    gamma = trial.suggest_uniform('gamma', 0.95, 0.999)
    lam = trial.suggest_uniform('lam', 0.9, 0.99)
    entropy_coef = trial.suggest_loguniform('entropy_coef', 1e-5, 1e-2)
    # 你可以继续添加其它参数

    # 2. 写入config
    finetuning_policy_config.FineTuningCfgPPO.algorithm.learning_rate = lr
    finetuning_policy_config.FineTuningCfgPPO.algorithm.gamma = gamma
    finetuning_policy_config.FineTuningCfgPPO.algorithm.lam = lam
    finetuning_policy_config.FineTuningCfgPPO.algorithm.entropy_coef = entropy_coef

    # 3. 获取全局环境和runner
    env, ppo_runner = get_global_env_runner()

    # 4. 重新初始化policy参数和optimizer
    reset_policy_and_optimizer(ppo_runner, finetuning_policy_config.FineTuningCfgPPO)

    # 5. 训练一小段
    ppo_runner.learn(num_learning_iterations=50, init_at_random_ep_len=True)

    # 6. 评估
    mean_reward = evaluate_policy(ppo_runner, env, n_episodes=3)
    import gc
    torch.cuda.empty_cache()
    gc.collect()

    trial.report(mean_reward, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()
    
    return mean_reward

if __name__ == '__main__':
    sampler = CmaEsSampler()
    pruner = MedianPruner(n_warmup_steps=0)
    study = optuna.create_study(
        study_name="ppo_finetune_study",
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage="sqlite:///ppo_optuna.db",
        load_if_exists=True
    )
    study.optimize(objective, n_trials=20)
    print('Best params:', study.best_params)