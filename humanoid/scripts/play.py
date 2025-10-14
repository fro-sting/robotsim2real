# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.

import os
import cv2
import numpy as np
from isaacgym import gymapi
from humanoid import LEGGED_GYM_ROOT_DIR

# import isaacgym
from humanoid.envs import *
from humanoid.utils import get_args, export_policy_as_jit, task_registry, Logger
from isaacgym.torch_utils import *
from isaacgym import gymtorch

import torch
from tqdm import tqdm
from datetime import datetime
from scipy.spatial.transform import Rotation as R


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 3)
    env_cfg.sim.max_gpu_contact_pairs = 2**10
    # env_cfg.terrain.mesh_type = 'trimesh'
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_init_terrain_level = 5
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.joint_angle_noise = 0.0
    env_cfg.noise.curriculum = False
    env_cfg.noise.noise_level = 0.5

    train_cfg.seed = 123145
    print("train_cfg.runner_class_name:", train_cfg.runner_class_name)

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.set_camera(env_cfg.viewer.pos, env_cfg.viewer.lookat)

    obs = env.get_observations()

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    stop_state_log = 300  # 提前定义，保证if分支和主流程都能用

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        export_dir = f"policies_{train_cfg.runner.run_name}"
        path = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name, "exported", export_dir)
        os.makedirs(path, exist_ok=True)
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print("Exported policy as jit script to: ", path)

    # 提前初始化渲染和视频相关变量，确保所有分支都能用
    h1 = None
    video = None
    if RENDER:
        camera_properties = gymapi.CameraProperties()
        camera_properties.width = 1920
        camera_properties.height = 1080
        h1 = env.gym.create_camera_sensor(env.envs[0], camera_properties)
        camera_offset = gymapi.Vec3(1, -1, 0.5)
        camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(-0.3, 0.2, 1), np.deg2rad(135))
        actor_handle = env.gym.get_actor_handle(env.envs[0], 0)
        body_handle = env.gym.get_actor_rigid_body_handle(env.envs[0], actor_handle, 0)
        env.gym.attach_camera_to_body(
            h1, env.envs[0], body_handle, gymapi.Transform(camera_offset, camera_rotation), gymapi.FOLLOW_POSITION
        )

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "videos")
        experiment_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "videos", train_cfg.runner.experiment_name)
        dir = os.path.join(
            experiment_dir, datetime.now().strftime("%b%d_%H-%M-%S") + train_cfg.runner.run_name + ".mp4"
        )
        if not os.path.exists(video_dir):
            os.mkdir(video_dir)
        if not os.path.exists(experiment_dir):
            os.mkdir(experiment_dir)
        video = cv2.VideoWriter(dir, fourcc, 50.0, (1920, 1080))

    logger = Logger(env.dt)
    robot_index = 1

    # base+delta组合输出，仅在delta_action_model_ppo任务下启用
    if args.task == "delta_action_model_ppo":

        # 2. 直接加载JIT导出的pretrained policy
        pretrained_policy_path = (
            "/home/wegg/kuavo_rl_asap-main111/RL_train/logs/kuavo_jog/exported/policies_pretraining/policy_1.pt"
        )
        pretrained_policy = torch.jit.load(pretrained_policy_path)
        pretrained_policy.eval()
        pretrained_policy.to(env.device)

        # 3. 获取踝关节索引
        delta_names = ["leg_l3_joint", "leg_l4_joint", "leg_r3_joint", "leg_r4_joint"]
        delta_indices = [env.dof_names.index(name) for name in delta_names]

        # 4. 加载第一条真实数据
        real_data_dir = "/home/wegg/kuavo_rl_asap-main111/RL_train/output_data/partitioned_aligned_npz/combined"
        real_data_files = [f for f in os.listdir(real_data_dir) if f.endswith(".npz")]
        real_data_files.sort()  # 保证顺序一致  
        first_real_data_path = os.path.join(real_data_dir, real_data_files[0])
        real_data = dict(np.load(first_real_data_path, allow_pickle=True))

        dof_pos_buffer = []
        dof_vel_buffer = []
        foot_force_buffer = []
        foot_height_buffer = []
        foot_zvel_buffer = []
        root_pos_buffer = []
        root_eu_ang_buffer = []
        root_ang_vel_buffer = []
        root_lin_vel_buffer = []
        action_buffer = []
        prev_foot_height = None
        for i in tqdm(range(stop_state_log)):
            # === 用真实数据刷新仿真状态 ===
            traj_len = len(real_data["jointpos"])
            idx = i % traj_len
            root_pos = torch.from_numpy(real_data["pos_xyz"][idx : idx + 1]).to(env.device)
            root_eu_ang = real_data["angular_eu_ang"][idx : idx + 1]
            root_quat = torch.from_numpy(R.from_euler("xyz", root_eu_ang).as_quat()).to(env.device)
            dof_pos = torch.from_numpy(real_data["jointpos"][idx : idx + 1, :12]).to(env.device)
            dof_vel = torch.from_numpy(real_data["jointvel"][idx : idx + 1, :12]).to(env.device)
            env.root_states[0, :3] = root_pos + env.env_origins[0] + env.platform_offsets[0]
            env.root_states[0, 3:7] = root_quat
            env.dof_pos[0] = dof_pos
            env.dof_vel[0] = dof_vel
            env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_states))
            env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env.dof_state))

            # 固定命令输入
            env.commands[:, 0] = 2.0  # x方向速度
            env.commands[:, 1] = 0.0  # y方向速度
            env.commands[:, 2] = 0.0  # yaw角速度
            env.commands[:, 3] = 0.0  # heading

            # === 计算 observation ===
            env.compute_observations()

            # === 生成 base action 和 delta action ===
            with torch.no_grad():
                base_action = pretrained_policy(env.obs_buf)
                delta_action = policy(env.obs_buf)

            # === 组合动作（与delta_action_env.py一致） ===
            combined_action = base_action.clone()
            if delta_action.shape[1] == len(delta_indices):
                combined_action[:, delta_indices] += delta_action
            else:
                combined_action += delta_action

            # === step ===
            obs, critic_obs, rews, dones, infos, _ = env.step(combined_action)

            if RENDER:
                env.gym.fetch_results(env.sim, True)
                env.gym.step_graphics(env.sim)
                env.gym.render_all_camera_sensors(env.sim)
                img = env.gym.get_camera_image(env.sim, env.envs[0], h1, gymapi.IMAGE_COLOR)
                img = np.reshape(img, (1080, 1920, 4))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                video.write(img[..., :3])

            logger.log_states(
                {
                    "dof_pos_target0_gym": (
                        env.actions[robot_index, 0].item() * env.cfg.control.action_scale
                        + env.default_dof_pos[0][0].item()
                    ),
                    "dof_pos0_gym": env.dof_pos[robot_index, 0].item(),
                    "dof_pos_target1_gym": (
                        env.actions[robot_index, 1].item() * env.cfg.control.action_scale
                        + env.default_dof_pos[0][1].item()
                    ),
                    "dof_pos1_gym": env.dof_pos[robot_index, 1].item(),
                    "dof_pos_target2_gym": (
                        env.actions[robot_index, 2].item() * env.cfg.control.action_scale
                        + env.default_dof_pos[0][2].item()
                    ),
                    "dof_pos2_gym": env.dof_pos[robot_index, 2].item(),
                    "dof_pos_target3_gym": (
                        env.actions[robot_index, 3].item() * env.cfg.control.action_scale
                        + env.default_dof_pos[0][3].item()
                    ),
                    "dof_pos3_gym": env.dof_pos[robot_index, 3].item(),
                    "dof_pos_target4_gym": (
                        env.actions[robot_index, 4].item() * env.cfg.control.action_scale
                        + env.default_dof_pos[0][4].item()
                    ),
                    "dof_pos4_gym": env.dof_pos[robot_index, 4].item(),
                    "dof_pos_target5_gym": (
                        env.actions[robot_index, 5].item() * env.cfg.control.action_scale
                        + env.default_dof_pos[0][5].item()
                    ),
                    "dof_pos5_gym": env.dof_pos[robot_index, 5].item(),
                    "dof_pos_target6_gym": (
                        env.actions[robot_index, 6].item() * env.cfg.control.action_scale
                        + env.default_dof_pos[0][6].item()
                    ),
                    "dof_pos6_gym": env.dof_pos[robot_index, 6].item(),
                    "dof_pos_target7_gym": (
                        env.actions[robot_index, 7].item() * env.cfg.control.action_scale
                        + env.default_dof_pos[0][7].item()
                    ),
                    "dof_pos7_gym": env.dof_pos[robot_index, 7].item(),
                    "dof_pos_target8_gym": (
                        env.actions[robot_index, 8].item() * env.cfg.control.action_scale
                        + env.default_dof_pos[0][8].item()
                    ),
                    "dof_pos8_gym": env.dof_pos[robot_index, 8].item(),
                    "dof_pos_target9_gym": (
                        env.actions[robot_index, 9].item() * env.cfg.control.action_scale
                        + env.default_dof_pos[0][9].item()
                    ),
                    "dof_pos9_gym": env.dof_pos[robot_index, 9].item(),
                    "dof_pos_target10_gym": (
                        env.actions[robot_index, 10].item() * env.cfg.control.action_scale
                        + env.default_dof_pos[0][10].item()
                    ),
                    "dof_pos10_gym": env.dof_pos[robot_index, 10].item(),
                    "dof_pos_target11_gym": (
                        env.actions[robot_index, 11].item() * env.cfg.control.action_scale
                        + env.default_dof_pos[0][11].item()
                    ),
                    "dof_pos11_gym": env.dof_pos[robot_index, 11].item(),
                    "dof_vel0_gym": env.dof_vel[robot_index, 0].item(),
                    "dof_vel1_gym": env.dof_vel[robot_index, 1].item(),
                    "dof_vel2_gym": env.dof_vel[robot_index, 2].item(),
                    "dof_vel3_gym": env.dof_vel[robot_index, 3].item(),
                    "dof_vel4_gym": env.dof_vel[robot_index, 4].item(),
                    "dof_vel5_gym": env.dof_vel[robot_index, 5].item(),
                    "dof_vel6_gym": env.dof_vel[robot_index, 6].item(),
                    "dof_vel7_gym": env.dof_vel[robot_index, 7].item(),
                    "dof_vel8_gym": env.dof_vel[robot_index, 8].item(),
                    "dof_vel9_gym": env.dof_vel[robot_index, 9].item(),
                    "dof_vel10_gym": env.dof_vel[robot_index, 10].item(),
                    "dof_vel11_gym": env.dof_vel[robot_index, 11].item(),
                    "ref_dof_pos0_gym": env.ref_dof_pos[robot_index, 0].item(),
                    "ref_dof_pos1_gym": env.ref_dof_pos[robot_index, 1].item(),
                    "ref_dof_pos2_gym": env.ref_dof_pos[robot_index, 2].item(),
                    "ref_dof_pos3_gym": env.ref_dof_pos[robot_index, 3].item(),
                    "ref_dof_pos4_gym": env.ref_dof_pos[robot_index, 4].item(),
                    "ref_dof_pos5_gym": env.ref_dof_pos[robot_index, 5].item(),
                    "ref_dof_pos6_gym": env.ref_dof_pos[robot_index, 6].item(),
                    "ref_dof_pos7_gym": env.ref_dof_pos[robot_index, 7].item(),
                    "ref_dof_pos8_gym": env.ref_dof_pos[robot_index, 8].item(),
                    "ref_dof_pos9_gym": env.ref_dof_pos[robot_index, 9].item(),
                    "ref_dof_pos10_gym": env.ref_dof_pos[robot_index, 10].item(),
                    "ref_dof_pos11_gym": env.ref_dof_pos[robot_index, 11].item(),
                    "dof_torque_0": env.torques[robot_index, 0].item(),
                    "dof_torque_1": env.torques[robot_index, 1].item(),
                    "dof_torque_2": env.torques[robot_index, 2].item(),
                    "dof_torque_3": env.torques[robot_index, 3].item(),
                    "dof_torque_4": env.torques[robot_index, 4].item(),
                    "dof_torque_5": env.torques[robot_index, 5].item(),
                    "dof_torque_6": env.torques[robot_index, 6].item(),
                    "dof_torque_7": env.torques[robot_index, 7].item(),
                    "dof_torque_8": env.torques[robot_index, 8].item(),
                    "dof_torque_9": env.torques[robot_index, 9].item(),
                    "dof_torque_10": env.torques[robot_index, 10].item(),
                    "dof_torque_11": env.torques[robot_index, 11].item(),
                    "command_x": env.commands[robot_index, 0].item(),
                    "command_y": env.commands[robot_index, 1].item(),
                    "command_yaw": env.commands[robot_index, 2].item(),
                    "base_vel_x": env.base_lin_vel[robot_index, 0].item(),
                    "base_vel_y": env.base_lin_vel[robot_index, 1].item(),
                    "base_vel_z": env.base_lin_vel[robot_index, 2].item(),
                    "contact_forces_z": env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy(),
                    "contact_vel_z": (
                        torch.norm(env.rigid_state[robot_index, env.feet_indices, 7:10], dim=-1).cpu().numpy()
                    ),
                    "contact_period": [
                        env._get_gait_phase()[robot_index, 0].item(),
                        env._get_gait_phase()[robot_index, 1].item(),
                    ],
                    "base_euler0_gym": env.base_euler_xyz[robot_index, 0].item(),
                    "base_euler1_gym": env.base_euler_xyz[robot_index, 1].item(),
                    "base_euler2_gym": env.base_euler_xyz[robot_index, 2].item(),
                }
            )
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes > 0:
                    logger.log_rewards(infos["episode"], num_episodes)
            dof_pos_buffer.append(env.dof_pos[robot_index].cpu().numpy().copy())
            dof_vel_buffer.append(env.dof_vel[robot_index].cpu().numpy().copy())
            foot_force = env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy().copy()
            foot_force_buffer.append(foot_force)
            foot_height = env.rigid_state[robot_index, env.feet_indices, 2].cpu().numpy().copy()
            foot_height_buffer.append(foot_height)
            if prev_foot_height is not None:
                foot_zvel = (foot_height - prev_foot_height) / env.dt
            else:
                foot_zvel = np.zeros_like(foot_height)
            foot_zvel_buffer.append(foot_zvel)
            prev_foot_height = foot_height.copy()
            root_pos_buffer.append(env.root_states[robot_index, :3].cpu().numpy().copy())
            root_eu_ang_buffer.append(env.base_euler_xyz[robot_index].cpu().numpy().copy())
            root_ang_vel_buffer.append(env.base_ang_vel[robot_index].cpu().numpy().copy())
            root_lin_vel_buffer.append(env.base_lin_vel[robot_index].cpu().numpy().copy())
            action_buffer.append(combined_action[robot_index].detach().cpu().numpy().copy())
        logger.print_rewards()
        logger.plot_states()
        if RENDER:
            video.release()
        np.savez(
            "play_isaac_data.npz",
            dof_pos=np.array(dof_pos_buffer),
            dof_vel=np.array(dof_vel_buffer),
            foot_force=np.array(foot_force_buffer),
            foot_height=np.array(foot_height_buffer),
            foot_zvel=np.array(foot_zvel_buffer),
            root_pos=np.array(root_pos_buffer),
            root_eu_ang=np.array(root_eu_ang_buffer),
            root_ang_vel=np.array(root_ang_vel_buffer),
            root_lin_vel=np.array(root_lin_vel_buffer),
            action=np.array(action_buffer),
        )
        return

    # 采集buffer
    dof_pos_buffer = []
    dof_vel_buffer = []
    foot_force_buffer = []
    foot_height_buffer = []
    foot_zvel_buffer = []
    root_pos_buffer = []
    root_eu_ang_buffer = []
    root_ang_vel_buffer = []
    root_lin_vel_buffer = []
    action_buffer = []
    prev_foot_height = None
    for i in tqdm(range(stop_state_log)):
        actions = policy(obs.detach())

        if FIX_COMMAND:
            env.commands[:, 0] = 2.0  # 1.0
            env.commands[:, 1] = 0.0
            env.commands[:, 2] = 0.0
            env.commands[:, 3] = 0.0
        # actions = torch.zeros_like(actions)
        obs, critic_obs, rews, dones, infos, _ = env.step(actions.detach())

        if RENDER:
            env.gym.fetch_results(env.sim, True)
            env.gym.step_graphics(env.sim)
            env.gym.render_all_camera_sensors(env.sim)
            img = env.gym.get_camera_image(env.sim, env.envs[0], h1, gymapi.IMAGE_COLOR)
            img = np.reshape(img, (1080, 1920, 4))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            video.write(img[..., :3])

        logger.log_states(
            {
                "dof_pos_target0_gym": (
                    env.actions[robot_index, 0].item() * env.cfg.control.action_scale + env.default_dof_pos[0][0].item()
                ),
                "dof_pos0_gym": env.dof_pos[robot_index, 0].item(),
                "dof_pos_target1_gym": (
                    env.actions[robot_index, 1].item() * env.cfg.control.action_scale + env.default_dof_pos[0][1].item()
                ),
                "dof_pos1_gym": env.dof_pos[robot_index, 1].item(),
                "dof_pos_target2_gym": (
                    env.actions[robot_index, 2].item() * env.cfg.control.action_scale + env.default_dof_pos[0][2].item()
                ),
                "dof_pos2_gym": env.dof_pos[robot_index, 2].item(),
                "dof_pos_target3_gym": (
                    env.actions[robot_index, 3].item() * env.cfg.control.action_scale + env.default_dof_pos[0][3].item()
                ),
                "dof_pos3_gym": env.dof_pos[robot_index, 3].item(),
                "dof_pos_target4_gym": (
                    env.actions[robot_index, 4].item() * env.cfg.control.action_scale + env.default_dof_pos[0][4].item()
                ),
                "dof_pos4_gym": env.dof_pos[robot_index, 4].item(),
                "dof_pos_target5_gym": (
                    env.actions[robot_index, 5].item() * env.cfg.control.action_scale + env.default_dof_pos[0][5].item()
                ),
                "dof_pos5_gym": env.dof_pos[robot_index, 5].item(),
                "dof_pos_target6_gym": (
                    env.actions[robot_index, 6].item() * env.cfg.control.action_scale + env.default_dof_pos[0][6].item()
                ),
                "dof_pos6_gym": env.dof_pos[robot_index, 6].item(),
                "dof_pos_target7_gym": (
                    env.actions[robot_index, 7].item() * env.cfg.control.action_scale + env.default_dof_pos[0][7].item()
                ),
                "dof_pos7_gym": env.dof_pos[robot_index, 7].item(),
                "dof_pos_target8_gym": (
                    env.actions[robot_index, 8].item() * env.cfg.control.action_scale + env.default_dof_pos[0][8].item()
                ),
                "dof_pos8_gym": env.dof_pos[robot_index, 8].item(),
                "dof_pos_target9_gym": (
                    env.actions[robot_index, 9].item() * env.cfg.control.action_scale + env.default_dof_pos[0][9].item()
                ),
                "dof_pos9_gym": env.dof_pos[robot_index, 9].item(),
                "dof_pos_target10_gym": (
                    env.actions[robot_index, 10].item() * env.cfg.control.action_scale
                    + env.default_dof_pos[0][10].item()
                ),
                "dof_pos10_gym": env.dof_pos[robot_index, 10].item(),
                "dof_pos_target11_gym": (
                    env.actions[robot_index, 11].item() * env.cfg.control.action_scale
                    + env.default_dof_pos[0][11].item()
                ),
                "dof_pos11_gym": env.dof_pos[robot_index, 11].item(),
                "dof_vel0_gym": env.dof_vel[robot_index, 0].item(),
                "dof_vel1_gym": env.dof_vel[robot_index, 1].item(),
                "dof_vel2_gym": env.dof_vel[robot_index, 2].item(),
                "dof_vel3_gym": env.dof_vel[robot_index, 3].item(),
                "dof_vel4_gym": env.dof_vel[robot_index, 4].item(),
                "dof_vel5_gym": env.dof_vel[robot_index, 5].item(),
                "dof_vel6_gym": env.dof_vel[robot_index, 6].item(),
                "dof_vel7_gym": env.dof_vel[robot_index, 7].item(),
                "dof_vel8_gym": env.dof_vel[robot_index, 8].item(),
                "dof_vel9_gym": env.dof_vel[robot_index, 9].item(),
                "dof_vel10_gym": env.dof_vel[robot_index, 10].item(),
                "dof_vel11_gym": env.dof_vel[robot_index, 11].item(),
                "ref_dof_pos0_gym": env.ref_dof_pos[robot_index, 0].item(),
                "ref_dof_pos1_gym": env.ref_dof_pos[robot_index, 1].item(),
                "ref_dof_pos2_gym": env.ref_dof_pos[robot_index, 2].item(),
                "ref_dof_pos3_gym": env.ref_dof_pos[robot_index, 3].item(),
                "ref_dof_pos4_gym": env.ref_dof_pos[robot_index, 4].item(),
                "ref_dof_pos5_gym": env.ref_dof_pos[robot_index, 5].item(),
                "ref_dof_pos6_gym": env.ref_dof_pos[robot_index, 6].item(),
                "ref_dof_pos7_gym": env.ref_dof_pos[robot_index, 7].item(),
                "ref_dof_pos8_gym": env.ref_dof_pos[robot_index, 8].item(),
                "ref_dof_pos9_gym": env.ref_dof_pos[robot_index, 9].item(),
                "ref_dof_pos10_gym": env.ref_dof_pos[robot_index, 10].item(),
                "ref_dof_pos11_gym": env.ref_dof_pos[robot_index, 11].item(),
                "dof_torque_0": env.torques[robot_index, 0].item(),
                "dof_torque_1": env.torques[robot_index, 1].item(),
                "dof_torque_2": env.torques[robot_index, 2].item(),
                "dof_torque_3": env.torques[robot_index, 3].item(),
                "dof_torque_4": env.torques[robot_index, 4].item(),
                "dof_torque_5": env.torques[robot_index, 5].item(),
                "dof_torque_6": env.torques[robot_index, 6].item(),
                "dof_torque_7": env.torques[robot_index, 7].item(),
                "dof_torque_8": env.torques[robot_index, 8].item(),
                "dof_torque_9": env.torques[robot_index, 9].item(),
                "dof_torque_10": env.torques[robot_index, 10].item(),
                "dof_torque_11": env.torques[robot_index, 11].item(),
                "command_x": env.commands[robot_index, 0].item(),
                "command_y": env.commands[robot_index, 1].item(),
                "command_yaw": env.commands[robot_index, 2].item(),
                "base_vel_x": env.base_lin_vel[robot_index, 0].item(),
                "base_vel_y": env.base_lin_vel[robot_index, 1].item(),
                "base_vel_z": env.base_lin_vel[robot_index, 2].item(),
                "contact_forces_z": env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy(),
                "contact_vel_z": torch.norm(env.rigid_state[robot_index, env.feet_indices, 7:10], dim=-1).cpu().numpy(),
                "contact_period": [
                    env._get_gait_phase()[robot_index, 0].item(),
                    env._get_gait_phase()[robot_index, 1].item(),
                ],
                "base_euler0_gym": env.base_euler_xyz[robot_index, 0].item(),
                "base_euler1_gym": env.base_euler_xyz[robot_index, 1].item(),
                "base_euler2_gym": env.base_euler_xyz[robot_index, 2].item(),
                # 'base_ang0_gazebo': env_gazebo.base_ang_vel[robot_index, 0].item(),
                # 'base_ang1_gazebo': env_gazebo.base_ang_vel[robot_index, 1].item(),
                # 'base_ang2_gazebo': env_gazebo.base_ang_vel[robot_index, 2].item(),
            }
        )
        # ====================== Log states ======================
        if infos["episode"]:
            num_episodes = torch.sum(env.reset_buf).item()
            if num_episodes > 0:
                logger.log_rewards(infos["episode"], num_episodes)

        # 采集数据
        dof_pos_buffer.append(env.dof_pos[robot_index].cpu().numpy().copy())
        dof_vel_buffer.append(env.dof_vel[robot_index].cpu().numpy().copy())
        # 足底和高度
        foot_force = env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy().copy()
        foot_force_buffer.append(foot_force)
        foot_height = env.rigid_state[robot_index, env.feet_indices, 2].cpu().numpy().copy()
        foot_height_buffer.append(foot_height)
        # 脚z向速度
        if prev_foot_height is not None:
            foot_zvel = (foot_height - prev_foot_height) / env.dt
        else:
            foot_zvel = np.zeros_like(foot_height)
        foot_zvel_buffer.append(foot_zvel)
        prev_foot_height = foot_height.copy()
        # 根部
        root_pos_buffer.append(env.root_states[robot_index, :3].cpu().numpy().copy())
        root_eu_ang_buffer.append(env.base_euler_xyz[robot_index].cpu().numpy().copy())
        root_ang_vel_buffer.append(env.base_ang_vel[robot_index].cpu().numpy().copy())
        root_lin_vel_buffer.append(env.base_lin_vel[robot_index].cpu().numpy().copy())
        # 动作
        action_buffer.append(actions[robot_index].detach().cpu().numpy().copy())

    logger.print_rewards()
    logger.plot_states()

    if RENDER:
        video.release()

    # 保存为npz
    np.savez(
        "play_isaac_data.npz",
        dof_pos=np.array(dof_pos_buffer),
        dof_vel=np.array(dof_vel_buffer),
        foot_force=np.array(foot_force_buffer),
        foot_height=np.array(foot_height_buffer),
        foot_zvel=np.array(foot_zvel_buffer),
        root_pos=np.array(root_pos_buffer),
        root_eu_ang=np.array(root_eu_ang_buffer),
        root_ang_vel=np.array(root_ang_vel_buffer),
        root_lin_vel=np.array(root_lin_vel_buffer),
        action=np.array(action_buffer),
    )


# python humanoid/scripts/play.py --task=bruce_ppo  --load_run May24_17-57-23_newurdf_newpos_newcom_delay
# python humanoid/scripts/play.py --task=bruce_teacher_ppo  --load_run Apr28_15-10-28_teacher_policy

# python humanoid/scripts/play.py --task=humanoid_ppo --load_run May24_17-57-23_newurdf_newpos_newcom_delay
# python humanoid/scripts/play.py --task=himbruce_ppo --load_run Apr12_18-00-58_laginit2.0_oldpos
if __name__ == "__main__":
    EXPORT_POLICY = True
    RENDER = True
    FIX_COMMAND = True
    args = get_args()
    play(args)
