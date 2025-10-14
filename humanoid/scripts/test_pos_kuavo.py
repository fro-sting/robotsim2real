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

import torch
from tqdm import tqdm
from datetime import datetime
from isaacgym import gymtorch, gymapi, gymutil


def get_euler_xyz_tensor(quat):
    r, p, w = get_euler_xyz(quat)
    # stack r, p, w in dim1
    euler_xyz = torch.stack((r, p, w), dim=1)
    euler_xyz[euler_xyz > np.pi] -= 2 * np.pi
    return euler_xyz


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = 1  # min(env_cfg.env.num_envs, 10)
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
    env_cfg.env.use_only_ref_actions = True
    env_cfg.asset.fix_base_link = True

    train_cfg.seed = 123145
    print("train_cfg.runner_class_name:", train_cfg.runner_class_name)

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.set_camera(env_cfg.viewer.pos, env_cfg.viewer.lookat)

    env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_LEFT, "previous")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_RIGHT, "next")

    data = dict(np.load("mpc_pose/ref_np_trot_mpc_1112_a.npz", allow_pickle=True))

    data["dof_pos"][:, [4, 5, 10, 11]] *= 0.0
    data["dof_vel"][:, [4, 5, 10, 11]] *= 0.0
    # data["dof_pos"][:, 0] += 0.03
    # data["dof_pos"][:, 6] -= 0.03
    data["root_eu_ang"][:, [0, 1, 2, 3, 4, 6]] *= 0.0
    data["root_eu_ang"][:, 3] += 1.0
    data["root_ang_vel"][:] *= 0.0
    data["root_lin_vel"][:, [0, 1]] *= 0.0

    ref_phase_dof_pos = data["dof_pos"]
    ref_phase_root_pos = data["root_pos"]
    ref_phase_root_eu_ang = data["root_eu_ang"]

    ref_phase_dof_pos = torch.from_numpy(ref_phase_dof_pos).to(env.device)
    ref_phase_root_pos = torch.from_numpy(ref_phase_root_pos).to(env.device)
    ref_phase_root_eu_ang = torch.from_numpy(ref_phase_root_eu_ang).to(env.device)

    ref_phase_root_eu_ang[:, 0:4] = quat_from_euler_xyz(
        ref_phase_root_eu_ang[:, 6], ref_phase_root_eu_ang[:, 5], ref_phase_root_eu_ang[:, 4]
    )
    data["root_eu_ang"] = np.asarray(ref_phase_root_eu_ang.cpu().numpy())

    # ref_phase_root_pos[:, 2] += 0.01
    # data["root_pos"] = np.asarray(ref_phase_root_pos.cpu().numpy())

    env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = env_ids.to(dtype=torch.int32)

    foot_height_buffer = []
    foot_zvel_buffer = []
    dof_pos_buffer = []

    last_pos = None
    last_foot_z = None

    idx = 0
    for i in tqdm(range(ref_phase_dof_pos.shape[0])):
        idx %= ref_phase_dof_pos.shape[0]

        root_states = torch.zeros(1, 13, dtype=torch.float, device=env.device, requires_grad=False)
        # root_states[0, 0] += ref_phase_root_pos[idx][0]
        # root_states[0, 1] += ref_phase_root_pos[idx][1]
        root_states[0, 2] += ref_phase_root_pos[idx][2] + 0.02
        root_states[0, 3] += ref_phase_root_eu_ang[idx][0]
        root_states[0, 4] += ref_phase_root_eu_ang[idx][1]
        root_states[0, 5] += ref_phase_root_eu_ang[idx][2]
        root_states[0, 6] += ref_phase_root_eu_ang[idx][3]

        env.gym.set_actor_root_state_tensor_indexed(
            env.sim, gymtorch.unwrap_tensor(root_states), gymtorch.unwrap_tensor(env_ids), len(env_ids)
        )

        dof_pos_ = torch.zeros(1, 12, dtype=torch.float, device=env.device, requires_grad=False)
        dof_pos_[:] = ref_phase_dof_pos[idx]

        dof_state = torch.stack([dof_pos_, torch.zeros_like(dof_pos_)], dim=-1).squeeze().repeat(env.num_envs, 1)
        env.gym.set_dof_state_tensor_indexed(
            env.sim, gymtorch.unwrap_tensor(dof_state), gymtorch.unwrap_tensor(env_ids), len(env_ids)
        )

        env.gym.fetch_results(env.sim, True)

        env.gym.simulate(env.sim)
        env.gym.fetch_results(env.sim, True)

        env.gym.refresh_dof_state_tensor(env.sim)
        env.gym.refresh_actor_root_state_tensor(env.sim)
        env.gym.refresh_net_contact_force_tensor(env.sim)
        env.gym.refresh_rigid_body_state_tensor(env.sim)

        # ---------------- re modify
        dof_pos_[:, 4] -= get_euler_xyz_tensor(env.rigid_state[:, env.feet_indices[0], 3:7])[:, 1]
        dof_pos_[:, 10] -= get_euler_xyz_tensor(env.rigid_state[:, env.feet_indices[1], 3:7])[:, 1]

        dof_pos_[:, 5] -= get_euler_xyz_tensor(env.rigid_state[:, env.feet_indices[0], 3:7])[:, 0]
        dof_pos_[:, 11] -= get_euler_xyz_tensor(env.rigid_state[:, env.feet_indices[1], 3:7])[:, 0]

        dof_state = torch.stack([dof_pos_, torch.zeros_like(dof_pos_)], dim=-1).squeeze().repeat(env.num_envs, 1)
        env.gym.set_dof_state_tensor_indexed(
            env.sim, gymtorch.unwrap_tensor(dof_state), gymtorch.unwrap_tensor(env_ids), len(env_ids)
        )

        env.gym.fetch_results(env.sim, True)

        env.gym.simulate(env.sim)
        env.gym.fetch_results(env.sim, True)

        env.gym.refresh_dof_state_tensor(env.sim)
        env.gym.refresh_actor_root_state_tensor(env.sim)
        env.gym.refresh_net_contact_force_tensor(env.sim)
        env.gym.refresh_rigid_body_state_tensor(env.sim)
        # # -------------------------

        foot_height_buffer.append(env.rigid_state[:, env.feet_indices, 2][0].cpu().numpy() - 0.02)
        dof_pos_buffer.append(dof_pos_[0].cpu().numpy())
        # foot_height_buffer.append(env.rigid_state[:, env.ankle_indices, 2][0].cpu().numpy() - 0.02)

        if i == 0:
            last_foot_z = env.rigid_state[:, env.feet_indices, 2][0].cpu().numpy()
            last_pos = dof_pos_[0].cpu().numpy()

        foot_zvel = (env.rigid_state[:, env.feet_indices, 2][0].cpu().numpy() - last_foot_z) / 0.01
        foot_zvel_buffer.append(foot_zvel)
        dof_vel_ = (dof_pos_[0].cpu().numpy() - last_pos) / 0.01
        data["dof_vel"][i, 4] += dof_vel_[4]
        data["dof_vel"][i, 5] += dof_vel_[5]
        data["dof_vel"][i, 10] += dof_vel_[10]
        data["dof_vel"][i, 11] += dof_vel_[11]

        last_foot_z = env.rigid_state[:, env.feet_indices, 2][0].cpu().numpy()
        last_pos = dof_pos_[0].cpu().numpy()

        # update the viewer
        env.gym.step_graphics(env.sim)
        env.gym.draw_viewer(env.viewer, env.sim, True)

        # Wait for dt to elapse in real time.
        # This synchronizes the physics simulation with the rendering rate.
        env.gym.sync_frame_time(env.sim)

        idx += 1

        for evt in env.gym.query_viewer_action_events(env.viewer):
            if evt.action == "previous" and evt.value > 0:
                idx -= 1
            elif evt.action == "next" and evt.value > 0:
                idx += 1

            print(idx)

    # data["root_eu_ang"][:, [0,1,2,3,4,6]] *=0.
    # data["root_eu_ang"][:, 3] +=1.
    # ref_phase_root_eu_ang = torch.from_numpy(data["root_eu_ang"]).to(env.device)
    # ref_phase_root_eu_ang[:, 0:4] = quat_from_euler_xyz(ref_phase_root_eu_ang[:,6], ref_phase_root_eu_ang[:,5], ref_phase_root_eu_ang[:,4])
    # data["root_eu_ang"] = np.asarray(ref_phase_root_eu_ang.cpu().numpy())

    foot_height_buffer = np.array(foot_height_buffer)
    dof_pos_buffer = np.array(dof_pos_buffer)
    foot_zvel_buffer = np.array(foot_zvel_buffer)

    ref_np = {
        "dof_pos": dof_pos_buffer,  # data["dof_pos"],
        "dof_vel": data["dof_vel"],
        # "dof_torque":data["dof_torque"],
        "foot_force": data["foot_force"],
        "foot_height": foot_height_buffer,
        "foot_zvel": foot_zvel_buffer,
        "root_pos": data["root_pos"],
        "root_eu_ang": data["root_eu_ang"],
        "root_ang_vel": data["root_ang_vel"],
        "root_lin_vel": data["root_lin_vel"],
    }

    print(data["dof_pos"].shape, foot_height_buffer.shape, dof_pos_buffer.shape)
    np.savez_compressed("mpc_pose/ref_np_trot_mpc_1112_c", **ref_np)

    import csv

    with open("ref_np_trot_mpc_1112_c.csv", mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        row_name = (
            [name_ + "pos" for name_ in env.dof_names]
            + [name_ + "vel" for name_ in env.dof_names]
            + ["foot_force_l", "foot_force_r"]
            + ["foot_height_l", "foot_height_r"]
            + ["foot_zvel_l", "foot_zvel_r"]
            + ["root_pos_0", "root_pos_1", "root_pos_2"]
            + [
                "root_eu_quat_0",
                "root_eu_quat_1",
                "root_eu_quat_2",
                "root_eu_quat_3",
                "root_eu_ang_0",
                "root_eu_ang_1",
                "root_eu_ang_2",
            ]
            + ["root_ang_vel_0", "root_ang_vel_1", "root_ang_vel_2"]
            + ["root_lin_vel_0", "root_lin_vel_1", "root_lin_vel_2"]
        )

        writer.writerow(row_name)

        for i in range(data["dof_pos"].shape[0]):
            writer.writerow(
                list(dof_pos_buffer[i])
                + list(data["dof_vel"][i])
                + list(data["foot_force"][i])
                + list(foot_height_buffer[i])
                + list(foot_zvel_buffer[i])
                + list(data["root_pos"][i])
                + list(data["root_eu_ang"][i])
                + list(data["root_ang_vel"][i])
                + list(data["root_lin_vel"][i])
            )


# python humanoid/scripts/test_pos_kuavo.py --task=kuavo_ppo
# python humanoid/scripts/play.py --task=humanoid_ppo --load_run Mar06_13-32-35_v1

if __name__ == "__main__":
    RENDER = True
    args = get_args()
    play(args)
