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


import math
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.envs import KuavoCfg
import torch
from datetime import datetime


class cmd:
    vx = 2.0
    vy = 0.0
    dyaw = 0.0
    stand = 0.0


def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat

    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)

    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)

    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)

    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z])


def get_obs(data):
    """Extracts an observation from the mujoco data structure"""
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    # print('get_obs q:', q)
    # print('get_obs dq:', dq)
    quat = data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = data.sensor("angular-velocity").data.astype(np.double)
    gvec = r.apply(np.array([0.0, 0.0, -1.0]), inverse=True).astype(np.double)
    return (q, dq, quat, v, omega, gvec)


def pd_control(target_q, default_dof_pos, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q + default_dof_pos - q) * kp + (target_dq - dq) * kd


# === 施加外力扰动的函数和开关 ===
ENABLE_PUSH = False  # 设置为True时推，为False时不推


def run_mujoco(policy, cfg, out_file):
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.
        out_file: The output file name for saving the data.

    Returns:
        None
    """
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt

    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_step(model, data)
    viewer = mujoco_viewer.MujocoViewer(model, data)

    target_q = np.zeros((cfg.env.num_actions + 6), dtype=np.double)
    action = np.zeros((cfg.env.num_actions), dtype=np.double)

    hist_obs = deque()
    for _ in range(cfg.env.frame_stack):
        hist_obs.append(np.zeros([1, cfg.env.num_single_obs], dtype=np.double))

    count_lowlevel = 0

    default_dof_pos = np.array([0.0, 0.0, -0.47, 0.86, -0.44, 0.0, 0.0, 0.0, -0.47, 0.86, -0.44, 0.0, 0, 0, 0, 0, 0, 0])

    # default_dof_pos = np.array([0., 0., -0.66, 1.0, -0.48, 0.,
    #                             0., 0., -0.66, 1.0, -0.48, 0., ])

    # 采集数据的buffer
    dof_pos_buffer = []
    dof_vel_buffer = []
    foot_force_buffer = []
    foot_height_buffer = []
    foot_zvel_buffer = []
    root_pos_buffer = []
    root_eu_ang_buffer = []
    root_ang_vel_buffer = []
    root_lin_vel_buffer = []
    action_buffer = []  # 新增action采集
    joint_pos_global_buffer = []  # 只采集action相关关节的全局xyz

    # 获取关节名顺序
    joint_names = [
        "leg_l1_joint",
        "leg_l2_joint",
        "leg_l3_joint",
        "leg_l4_joint",
        "leg_l5_joint",
        "leg_l6_joint",
        "leg_r1_joint",
        "leg_r2_joint",
        "leg_r3_joint",
        "leg_r4_joint",
        "leg_r5_joint",
        "leg_r6_joint",
    ]
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]
    dof_indices = [model.jnt_dofadr[jid] for jid in joint_ids]

    # 只采集action相关关节
    num_action_joints = cfg.env.num_actions
    action_joint_body_ids = [model.jnt_bodyid[jid] for jid in joint_ids[:num_action_joints]]

    # 获取site id
    left_foot_site_id = model.site("left_foot_site")
    right_foot_site_id = model.site("right_foot_site")

    # 获取基座 id
    root_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")

    # 调试：打印所有body名称和id
    print("=== 所有body名称和id ===")
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        print(f"body {i}: {name}")
    print(f"base_link id: {root_body_id}")
    print("========================")

    prev_left_foot_pos = None
    prev_right_foot_pos = None
    dt = cfg.sim_config.dt

    push_force = np.zeros(6)  # 初始化为零向量
    push_step_counter = 0

    def apply_random_push(data, root_body_id, force):
        """使用mj_applyFT对base_link施加指定外力"""
        if force is not None:
            # mj_applyFT(model, data, force, torque, point, body, qfrc_target)
            # force: 3D力向量, torque: 3D力矩向量, point: 施加点(世界坐标系), body: body id
            force_vec = force[:3].reshape(3, 1).astype(np.float64)  # 转换为3x1列向量
            torque_vec = force[3:6].reshape(3, 1).astype(np.float64)  # 转换为3x1列向量
            point = data.xpos[root_body_id].reshape(3, 1).astype(np.float64)  # 转换为3x1列向量
            mujoco.mj_applyFT(model, data, force_vec, torque_vec, point, root_body_id, data.qfrc_applied)
        else:
            # 清零外力
            data.xfrc_applied[:] = 0

    for _ in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):
        # Obtain an observation
        q_, dq, quat, v, omega, gvec = get_obs(data)

        eu_ang = quaternion_to_euler_array(quat)
        eu_ang[eu_ang > math.pi] -= 2 * math.pi

        q = np.array(data.actuator_length)
        dq = np.array(data.actuator_velocity)
        if count_lowlevel % cfg.sim_config.decimation == 0:
            obs = np.zeros([1, cfg.env.num_single_obs], dtype=np.float32)
            obs[0, 0] = math.sin(
                2 * math.pi * count_lowlevel * cfg.sim_config.dt / cfg.rewards.cycle_time + math.pi * 0.5
            )
            obs[0, 1] = math.cos(
                2 * math.pi * count_lowlevel * cfg.sim_config.dt / cfg.rewards.cycle_time + math.pi * 0.5
            )
            obs[0, 2] = cmd.vx * cfg.normalization.obs_scales.lin_vel
            obs[0, 3] = cmd.vy * cfg.normalization.obs_scales.lin_vel
            obs[0, 4] = cmd.dyaw * cfg.normalization.obs_scales.ang_vel
            obs[0, 5] = cmd.stand
            obs[0, 6:18] = (
                q[: cfg.env.num_actions] - default_dof_pos[: cfg.env.num_actions]
            ) * cfg.normalization.obs_scales.dof_pos
            obs[0, 18:30] = dq[: cfg.env.num_actions] * cfg.normalization.obs_scales.dof_vel
            obs[0, 30:42] = action
            obs[0, 42:45] = omega
            obs[0, 45:47] = eu_ang[0:2]

            obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)
            hist_obs.append(obs)
            hist_obs.popleft()

            policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
            for i in range(cfg.env.frame_stack):
                policy_input[0, i * cfg.env.num_single_obs : (i + 1) * cfg.env.num_single_obs] = hist_obs[i][0, :]
            action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
            action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)
            action_buffer.append(action.copy())

            # 打印action
            # print(f"step={count_lowlevel}, action={action}")

            target_q[: cfg.env.num_actions] = action * cfg.control.action_scale

            target_q[cfg.env.num_actions] = (
                -math.sin(2 * math.pi * count_lowlevel * cfg.sim_config.dt / cfg.rewards.cycle_time + math.pi * 0.5)
                * 0.9
            )
            target_q[cfg.env.num_actions + 1] = 0.0
            target_q[cfg.env.num_actions + 2] = -110 * np.pi / 180.0
            target_q[cfg.env.num_actions + 3] = (
                math.sin(2 * math.pi * count_lowlevel * cfg.sim_config.dt / cfg.rewards.cycle_time + math.pi * 0.5)
                * 0.9
            )
            target_q[cfg.env.num_actions + 4] = 0.0
            target_q[cfg.env.num_actions + 5] = -110 * np.pi / 180.0

            jointpos_sensor_names = [name.replace("_joint", "_pos") for name in joint_names]
            jointvel_sensor_names = [name.replace("_joint", "_vel") for name in joint_names]
            # 在这里同步采集其他数据
            dof_pos = np.array([data.sensor(name).data.copy()[0] for name in jointpos_sensor_names])
            dof_vel = np.array([data.sensor(name).data.copy()[0] for name in jointvel_sensor_names])
            dof_pos_buffer.append(dof_pos)
            dof_vel_buffer.append(dof_vel)

            # 采集脚site的力
            left_foot_force = data.sensor("left_foot_force").data.copy()
            right_foot_force = data.sensor("right_foot_force").data.copy()
            # 取反z向力，统一为"向上为正"
            foot_force_buffer.append([-left_foot_force[2], -right_foot_force[2]])

            # 采集脚site的位置
            left_foot_pos = data.sensor("left_foot_pos").data.copy()
            right_foot_pos = data.sensor("right_foot_pos").data.copy()
            foot_height_buffer.append([left_foot_pos[2], right_foot_pos[2]])

            # 采集脚site的z向速度
            if prev_left_foot_pos is not None and prev_right_foot_pos is not None:
                left_foot_zvel = (left_foot_pos[2] - prev_left_foot_pos[2]) / dt
                right_foot_zvel = (right_foot_pos[2] - prev_right_foot_pos[2]) / dt
            else:
                left_foot_zvel = 0.0
                right_foot_zvel = 0.0
            foot_zvel_buffer.append([left_foot_zvel, right_foot_zvel])
            prev_left_foot_pos = left_foot_pos.copy()
            prev_right_foot_pos = right_foot_pos.copy()

            # 采集根部位置
            root_pos_buffer.append(data.xpos[root_body_id].copy())
            # 采集根部欧拉角
            quat = data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.double)
            root_eu_ang_buffer.append(quaternion_to_euler_array(quat))
            # 采集根部角速度
            root_ang_vel_buffer.append(data.sensor("angular-velocity").data.copy())
            # 采集根部线速度
            root_lin_vel_buffer.append(data.qvel[:3].copy())

            # 采集action相关关节的全局位置
            joint_pos_global = []
            for body_id in action_joint_body_ids:
                pos = data.xpos[body_id].copy()  # shape (3,)
                joint_pos_global.append(pos)
            joint_pos_global_buffer.append(joint_pos_global)  # shape: [帧, action关节数, 3]

            # 打印dof_pos
            # print(f"step={count_lowlevel}, dof_pos={dof_pos}")

            # === 在200-300步施加推力 ===
            if ENABLE_PUSH and 200 <= count_lowlevel <= 300:
                if count_lowlevel == 200:  # 第200步生成推力
                    push_force = np.zeros(6)
                    push_force[0] = np.random.uniform(0, 3)  # x方向推力，单位N（只往前推）
                    push_force[1] = 0  # y方向不施加推力
                    # push_force[2] = np.random.uniform(-2, 2)   # z方向推力（如需）
                    print(f"Step {count_lowlevel}: 生成初始推力 {push_force[:3]}, 持续到第300步")
                # 持续施加推力直到第300步
                data.xfrc_applied[:] = 0  # 先清零，避免累加
                apply_random_push(data, root_body_id, push_force)
                if count_lowlevel % 10 == 0:  # 每10步打印一次推力状态
                    print(f"Step {count_lowlevel}: 施加推力 {push_force[:3]}")
                    # 打印机器人位置
                    base_pos = data.xpos[root_body_id]
                    print(f"Step {count_lowlevel}: 机器人位置 {base_pos}")
            else:
                # 其他时候清零外力
                apply_random_push(data, root_body_id, None)

        viewer.render()

        count_lowlevel += 1
        target_dq = np.zeros((cfg.env.num_actions + 6), dtype=np.double)

        # Generate PD control
        tau = pd_control(
            target_q, default_dof_pos, q, cfg.robot_config.kps, target_dq, dq, cfg.robot_config.kds
        )  # Calc torques
        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques

        data.ctrl = tau

        mujoco.mj_step(model, data)

        # print('policy action:', action)

    viewer.close()

    # 保存为npz文件
    np.savez(
        out_file,
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
        joint_pos_global=np.array(joint_pos_global_buffer),  # 只保存action相关关节的全局位置
    )

    # 打印所有 Mujoco 模型中的关节名和索引
    for i in range(model.njnt):
        print(f"joint {i}:", mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i), "qpos adr:", model.jnt_qposadr[i])


#  python humanoid/scripts/sim2sim_bruce.py --load_model /home/bigeast/humanoid-gym/logs/XBot_ppo/exported/policies/policy_1.pt
if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Deployment script.")
    parser.add_argument("--load_model", type=str, required=True, help="Run to load from.")
    parser.add_argument("--terrain", action="store_true", help="terrain or plane")
    args = parser.parse_args()

    class Sim2simCfg(KuavoCfg):
        class sim_config:
            if args.terrain:
                mujoco_model_path = f"{LEGGED_GYM_ROOT_DIR}/resources/robots/XBot/mjcf/XBot-L-terrain.xml"
            else:
                mujoco_model_path = f"{LEGGED_GYM_ROOT_DIR}/resources/robots/biped_s44/xml/scene.xml"
            sim_duration = 74.17
            dt = 0.001
            decimation = 10

        class robot_config:
            kps = np.array(
                [60, 60, 100, 150, 15, 15, 60, 60, 100, 150, 15, 15, 100, 100, 100, 100, 100, 100], dtype=np.double
            )
            kds = np.array(
                [5.0, 5.0, 5.0, 6.0, 5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 5.0, 5.0, 10, 10, 10, 10, 10, 10], dtype=np.double
            )
            tau_limit = (
                np.array(
                    [60.0, 60.0, 60.0, 80.0, 20.0, 20.0, 60.0, 60.0, 60.0, 80.0, 20.0, 20.0, 20, 20, 20, 20, 20, 20]
                )
                * 10.0
            )

    policy = torch.jit.load(args.load_model)
    # 根据模型路径自动命名输出文件
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M")
    if "finetuning" in args.load_model:
        save_dir = "output_data/finetuning_data"
        out_file = f"{current_datetime}_sim2sim_mujoco_finetuning_data1.npz"
    else:
        save_dir = "output_data/pretrained_data"
        out_file = f"{current_datetime}_sim2sim_mujoco_data1.npz"

    os.makedirs(save_dir, exist_ok=True)
    out_file = os.path.join(save_dir, out_file)

    run_mujoco(policy, Sim2simCfg(), out_file=out_file)
