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


from humanoid.envs.base.legged_robot_config import LeggedRobotCfg

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi
import numpy as np

import torch
from humanoid.envs import LeggedRobot

from humanoid.utils.terrain import HumanoidTerrain

# from collections import deque
from humanoid.utils.dr_utils import (
    get_property_setter_map,
    get_property_getter_map,
    get_default_setter_args,
    apply_random_samples,
    check_buckets,
    generate_random_samples,
)


def get_euler_xyz_tensor(quat):
    r, p, w = get_euler_xyz(quat)
    # stack r, p, w in dim1
    euler_xyz = torch.stack((r, p, w), dim=1)
    euler_xyz[euler_xyz > np.pi] -= 2 * np.pi
    return euler_xyz


class KuavoSuspendedFreeEnv(LeggedRobot):
    """
    XBotLFreeEnv is a class that represents a custom environment for a legged robot.

    Args:
        cfg (LeggedRobotCfg): Configuration object for the legged robot.
        sim_params: Parameters for the simulation.
        physics_engine: Physics engine used in the simulation.
        sim_device: Device used for the simulation.
        headless: Flag indicating whether the simulation should be run in headless mode.

    Attributes:
        last_feet_z (float): The z-coordinate of the last feet position.
        feet_height (torch.Tensor): Tensor representing the height of the feet.
        sim (gymtorch.GymSim): The simulation object.
        terrain (HumanoidTerrain): The terrain object.
        up_axis_idx (int): The index representing the up axis.
        command_input (torch.Tensor): Tensor representing the command input.
        privileged_obs_buf (torch.Tensor): Tensor representing the privileged observations buffer.
        obs_buf (torch.Tensor): Tensor representing the observations buffer.
        obs_history (collections.deque): Deque containing the history of observations.
        critic_history (collections.deque): Deque containing the history of critic observations.

    Methods:
        _push_robots(): Randomly pushes the robots by setting a randomized base velocity.
        _get_phase(): Calculates the phase of the gait cycle.
        _get_gait_phase(): Calculates the gait phase.
        compute_ref_state(): Computes the reference state.
        create_sim(): Creates the simulation, terrain, and environments.
        _get_noise_scale_vec(cfg): Sets a vector used to scale the noise added to the observations.
        step(actions): Performs a simulation step with the given actions.
        compute_observations(): Computes the observations.
        reset_idx(env_ids): Resets the environment for the specified environment IDs.
    """

    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.last_feet_z = 0.03
        self.period = round(self.cfg.rewards.cycle_time / self.dt)
        self.episode_phase_buf = torch.zeros_like(self.episode_length_buf)
        self.num_aux = self.cfg.env.num_aux

        data = [
            dict(np.load("mpc_pose/play_cmu_0401_b.npz", allow_pickle=True)),
            dict(np.load("mpc_pose/play_cmu_0401_b.npz", allow_pickle=True)),
        ]

        ref_phase_dof_pos = []
        ref_phase_dof_vel = []
        ref_phase_foot_force = []
        ref_phase_foot_height = []
        ref_phase_foot_zvel = []
        ref_phase_root_pos = []
        ref_phase_root_eu_ang = []
        ref_phase_root_ang_vel = []
        ref_phase_root_lin_vel = []

        for data_np in data:
            ref_phase_dof_pos.append(data_np["dof_pos"][:4000, :])
            ref_phase_dof_vel.append(data_np["dof_vel"][:4000, :])
            ref_phase_foot_force.append(data_np["foot_force"][:4000, :])
            ref_phase_foot_height.append(data_np["foot_height"][:4000, :])
            ref_phase_foot_zvel.append(data_np["foot_zvel"][:4000, :])
            ref_phase_root_pos.append(data_np["root_pos"][:4000, :])
            ref_phase_root_eu_ang.append(data_np["root_eu_ang"][:4000, :])
            ref_phase_root_ang_vel.append(data_np["root_ang_vel"][:4000, :])
            ref_phase_root_lin_vel.append(data_np["root_lin_vel"][:4000, :])

            ref_phase_foot_height[-1][:, 0] -= ref_phase_foot_height[-1][19, 0]
            ref_phase_foot_height[-1][:, 1] -= ref_phase_foot_height[-1][50, 1]
            ref_phase_root_pos[-1] -= ref_phase_root_pos[-1][17, 2]

        self.ref_phase_dof_pos = np.asarray(ref_phase_dof_pos, dtype=np.float32)
        self.ref_phase_dof_vel = np.asarray(ref_phase_dof_vel, dtype=np.float32)
        self.ref_phase_foot_force = np.asarray(ref_phase_foot_force, dtype=np.float32)
        self.ref_phase_foot_height = np.asarray(ref_phase_foot_height, dtype=np.float32)
        self.ref_phase_foot_zvel = np.asarray(ref_phase_foot_zvel, dtype=np.float32)
        self.ref_phase_root_pos = np.asarray(ref_phase_root_pos, dtype=np.float32)
        self.ref_phase_root_eu_ang = np.asarray(ref_phase_root_eu_ang, dtype=np.float32)
        self.ref_phase_root_ang_vel = np.asarray(ref_phase_root_ang_vel, dtype=np.float32)
        self.ref_phase_root_lin_vel = np.asarray(ref_phase_root_lin_vel, dtype=np.float32)

        self.ref_phase_dof_pos = torch.from_numpy(self.ref_phase_dof_pos).to(self.device).to(torch.float)
        self.ref_phase_dof_vel = torch.from_numpy(self.ref_phase_dof_vel).to(self.device).to(torch.float)
        self.ref_phase_foot_force = torch.from_numpy(self.ref_phase_foot_force).to(self.device).to(torch.float)
        self.ref_phase_foot_height = torch.from_numpy(self.ref_phase_foot_height).to(self.device).to(torch.float)
        self.ref_phase_foot_zvel = torch.from_numpy(self.ref_phase_foot_zvel).to(self.device).to(torch.float)
        self.ref_phase_root_pos = torch.from_numpy(self.ref_phase_root_pos).to(self.device).to(torch.float)
        self.ref_phase_root_eu_ang = torch.from_numpy(self.ref_phase_root_eu_ang).to(self.device).to(torch.float)
        self.ref_phase_root_ang_vel = torch.from_numpy(self.ref_phase_root_ang_vel).to(self.device).to(torch.float)
        self.ref_phase_root_lin_vel = torch.from_numpy(self.ref_phase_root_lin_vel).to(self.device).to(torch.float)

        self.feet_height = torch.zeros((self.num_envs, 2), device=self.device)
        self.command_stand = torch.zeros((self.num_envs, 1), device=self.device)
        self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))
        self.compute_observations()

    def _push_robots(self):
        """Random pushes the robots. Emulates an impulse by setting a randomized base velocity."""
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        max_push_angular = self.cfg.domain_rand.max_push_ang_vel
        self.rand_push_force[:, :2] = torch_rand_float(
            -max_vel, max_vel, (self.num_envs, 2), device=self.device
        )  # lin vel x/y
        self.root_states[:, 7:9] = self.rand_push_force[:, :2]

        self.rand_push_torque = torch_rand_float(
            -max_push_angular, max_push_angular, (self.num_envs, 3), device=self.device
        )

        self.root_states[:, 10:13] = self.rand_push_torque

        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _disturbance_robots(self):
        """Random add disturbance force to the robots."""
        disturbance = torch_rand_float(
            self.cfg.domain_rand.disturbance_range[0],
            self.cfg.domain_rand.disturbance_range[1],
            (self.num_envs, 3),
            device=self.device,
        )
        self.disturbance[:, 0, :] = disturbance
        self.gym.apply_rigid_body_force_tensors(
            self.sim, forceTensor=gymtorch.unwrap_tensor(self.disturbance), space=gymapi.CoordinateSpace.LOCAL_SPACE
        )

    def _get_phase(self):
        cycle_time = self.cfg.rewards.cycle_time
        phase = self.episode_phase_buf * self.dt / cycle_time
        return phase

    def _get_gait_phase(self):
        # return float mask 1 is stance, 0 is swing
        ref_idx_ = torch.where(self.commands[:, 0] > 1.0, 1, 0)

        stance_mask = torch.zeros((self.num_envs, 2), device=self.device)

        stance_mask[torch.abs(self.ref_phase_foot_force[ref_idx_, self.episode_phase_buf]) > 100.0] = 1

        # stance_mask[torch.abs(self.ref_phase_foot_height[ref_idx_, self.episode_phase_buf]) < 0.005] = 1

        # phase = self._get_phase()
        # sin_pos = torch.sin(2 * torch.pi * phase)
        # # Add double support phase
        # stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
        # # left foot stance
        # stance_mask[:, 0] = sin_pos >= 0
        # # right foot stance
        # stance_mask[:, 1] = sin_pos < 0
        # # Double support phase
        # stance_mask[torch.abs(sin_pos) < 0.1] = 1

        return stance_mask

    def compute_ref_state(self):
        ref_idx_ = torch.where(self.commands[:, 0] > 1.0, 1, 0)

        self.ref_action = self.ref_phase_dof_pos[ref_idx_, self.episode_phase_buf]
        self.ref_dof_pos = self.ref_phase_dof_pos[ref_idx_, self.episode_phase_buf]

    def create_sim(self):
        """Creates simulation, terrain and environments"""
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params
        )
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ["heightfield", "trimesh"]:
            self.terrain = HumanoidTerrain(self.cfg.terrain, self.num_envs)
        if mesh_type == "plane":
            self._create_ground_plane()
        elif mesh_type == "heightfield":
            self._create_heightfield()
        elif mesh_type == "trimesh":
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def _get_noise_scale_vec(self, cfg):
        """Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros(self.cfg.env.num_single_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_vec[0:6] = 0.0  # commands
        noise_vec[6:18] = noise_scales.dof_pos * self.obs_scales.dof_pos
        noise_vec[18:30] = noise_scales.dof_vel * self.obs_scales.dof_vel
        noise_vec[30:42] = 0.0  # previous actions
        noise_vec[42:45] = noise_scales.ang_vel * self.obs_scales.ang_vel  # ang vel
        noise_vec[45:47] = noise_scales.quat * self.obs_scales.quat  # euler x,y
        return noise_vec

    def step(self, actions):
        if self.cfg.env.use_only_ref_actions:
            actions = self.ref_action
        else:
            if self.cfg.env.use_ref_actions:
                actions = self.ref_action
        return super().step(actions)

    def _compute_torques(self, actions):
        """Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        p_gains = self.p_gains * self.Kp_factors
        d_gains = self.d_gains * self.Kd_factors

        if self.cfg.env.use_only_ref_actions:
            torques = p_gains * (actions - self.dof_pos) - d_gains * self.dof_vel
        else:
            torques = p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos) - d_gains * self.dof_vel

        if self.cfg.domain_rand.randomize_motor_strength:
            motor_strength_factors = torch_rand_float(
                self.cfg.domain_rand.motor_strength_range[0],
                self.cfg.domain_rand.motor_strength_range[1],
                (self.num_envs, self.num_actions),
                device=self.device,
            )
            torques *= motor_strength_factors

        if self.cfg.domain_rand.randomize_torque_rfi:
            torques = (
                torques
                + (torch.rand_like(torques) * 2.0 - 1.0)
                * self.cfg.domain_rand.rfi_lim
                * self._rfi_lim_scale
                * self.torque_limits
            )

        # print('self.dof_vel_limits:',self.dof_vel_limits)
        # print('self.torque_limits:',self.torque_limits)
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def compute_observations(self):
        phase = self._get_phase()
        self.compute_ref_state()

        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)

        stance_mask = self._get_gait_phase()
        contact_mask = self.contact_forces[:, self.feet_indices, 2] > 5.0

        self.command_input = torch.cat((sin_pos, cos_pos, self.commands[:, :3] * self.commands_scale), dim=1)

        q = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        dq = self.dof_vel * self.obs_scales.dof_vel

        diff = self.dof_pos - self.ref_dof_pos

        self.privileged_obs_buf = torch.cat(
            (
                self.command_input,  # 2 + 3
                self.command_stand,
                (self.dof_pos - self.default_joint_pd_target) * self.obs_scales.dof_pos,  # 12
                (self.dof_vel * self.obs_scales.dof_vel),  # 12
                self.actions,  # 12
                diff,  # 12
                self.base_lin_vel * self.obs_scales.lin_vel,  # 3
                self.base_ang_vel * self.obs_scales.ang_vel,  # 3
                self.base_euler_xyz * self.obs_scales.quat,  # 3
                self.rand_push_force[:, :2] / self.cfg.domain_rand.max_push_vel_xy,  # 2
                self.rand_push_torque / self.cfg.domain_rand.max_push_ang_vel,  # 3
                self.disturbance[:, 0, :] / self.cfg.domain_rand.disturbance_range[1],  # 3
                (self.friction_coeffs - self.cfg.domain_rand.friction_range[0])
                / (self.cfg.domain_rand.friction_range[1] - self.cfg.domain_rand.friction_range[0]),  # 1
                (self.restitution_coeffs - self.cfg.domain_rand.restitution_range[0])
                / (self.cfg.domain_rand.restitution_range[1] - self.cfg.domain_rand.restitution_range[0]),  # 1
                (self.joint_friction_coeffs - self.cfg.domain_rand.joint_friction_range[0])
                / (self.cfg.domain_rand.joint_friction_range[1] - self.cfg.domain_rand.joint_friction_range[0]),  # 10
                (self.joint_armature_coeffs - self.cfg.domain_rand.joint_armature_range[0])
                / (self.cfg.domain_rand.joint_armature_range[1] - self.cfg.domain_rand.joint_armature_range[0]),  # 10
                (self.Kp_factors - self.cfg.domain_rand.kp_range[0])
                / (self.cfg.domain_rand.kp_range[1] - self.cfg.domain_rand.kp_range[0]),  # 10
                (self.Kd_factors - self.cfg.domain_rand.kd_range[0])
                / (self.cfg.domain_rand.kd_range[1] - self.cfg.domain_rand.kd_range[0]),  # 10
                (self.payload - self.cfg.domain_rand.payload_mass_range[0])
                / (self.cfg.domain_rand.payload_mass_range[1] - self.cfg.domain_rand.payload_mass_range[0]),  # 1
                (self.com_displacement - self.cfg.domain_rand.com_displacement_range[0])
                / (
                    self.cfg.domain_rand.com_displacement_range[1] - self.cfg.domain_rand.com_displacement_range[0]
                ),  # 3
                # (self.root_states_delay_steps.unsqueeze(1) - self.cfg.domain_rand.root_states_delay_range[0])/(self.cfg.domain_rand.root_states_delay_range[1] - self.cfg.domain_rand.root_states_delay_range[0]),  # 1
                # (self.joint_states_delay_steps.unsqueeze(1) - self.cfg.domain_rand.joint_states_delay_range[0])/(self.cfg.domain_rand.joint_states_delay_range[1] - self.cfg.domain_rand.joint_states_delay_range[0]),  # 1
                stance_mask,  # 2
                contact_mask,  # 2
            ),
            dim=-1,
        )

        base_euler_xy_noise = torch.zeros((self.num_envs, 2), device=self.device)
        base_euler_xy_noise[:] = self.base_euler_xyz[:, :2]

        if self.cfg.domain_rand.root_states_delay:
            root_states_ = self.root_states_history[
                self.root_states_delay_steps, torch.arange(self.num_envs, device=self.device), :
            ]
            base_ang_vel_ = quat_rotate_inverse(root_states_[:, 3:7], root_states_[:, 10:13])
            base_euler_xy_noise[:] = get_euler_xyz_tensor(root_states_[:, 3:7])[:, :2]
        else:
            base_ang_vel_ = self.base_ang_vel

        if self.cfg.domain_rand.joint_states_delay:
            dos_pos_ = self.joint_states_history[
                self.joint_states_delay_steps, torch.arange(self.num_envs, device=self.device), : self.num_actions
            ]
            dos_vel_ = self.joint_states_history[
                self.joint_states_delay_steps, torch.arange(self.num_envs, device=self.device), self.num_actions :
            ]

            q = (dos_pos_ - self.default_dof_pos) * self.obs_scales.dof_pos
            dq = dos_vel_ * self.obs_scales.dof_vel

        base_euler_xy_noise[:, 1] += self.euler_y_zero_pos[:, 0]

        obs_buf = torch.cat(
            (
                self.command_input,  # 5 = 2D(sin cos) + 3D(vel_x, vel_y, aug_vel_yaw)
                self.command_stand,
                q,  # 12D
                dq,  # 12D
                self.actions,  # 12D
                base_ang_vel_ * self.obs_scales.ang_vel,  # 3
                base_euler_xy_noise * self.obs_scales.quat,  # 2
            ),
            dim=-1,
        )

        # if self.cfg.terrain.measure_heights:
        #     heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
        #     self.privileged_obs_buf = torch.cat((self.obs_buf, heights), dim=-1)

        if self.add_noise:
            # obs_now = obs_buf.clone() + torch.randn_like(obs_buf) * self.noise_scale_vec * self.cfg.noise.noise_level
            obs_now = (
                obs_buf.clone() + (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec * self.cfg.noise.noise_level
            )
        else:
            obs_now = obs_buf.clone()
        self.obs_history.append(obs_now)
        self.critic_history.append(self.privileged_obs_buf)

        obs_buf_all = torch.stack([self.obs_history[i] for i in range(self.obs_history.maxlen)], dim=1)  # N,T,K

        self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)  # N, T*K
        self.privileged_obs_buf = torch.cat([self.critic_history[i] for i in range(self.cfg.env.c_frame_stack)], dim=1)

    def gen_aux(self):
        contact_mask = self.contact_forces[:, self.feet_indices, 2] > 5.0

        q = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        dq = self.dof_vel * self.obs_scales.dof_vel

        self.aux = torch.cat(
            (
                q,
                dq,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_euler_xyz[:, :2] * self.obs_scales.quat,
                contact_mask,
            ),
            dim=-1,
        )

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0

    # ================================================ Rewards ================================================== #
    def _reward_joint_pos(self):
        """
        Calculates the reward based on the difference between the current joint positions and the target joint positions.
        """
        joint_pos = self.dof_pos.clone()
        pos_target = self.ref_dof_pos.clone()
        diff = joint_pos - pos_target
        r = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)
        return r

    def _reward_feet_distance(self):
        """
        Calculates the reward based on the distance between the feet. Penalize feet get close to each other or too far away.
        """
        foot_pos = self.rigid_state[:, self.feet_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist
        fd += (self.command_stand[:, 0] > 0.0) * 0.01
        max_df = self.cfg.rewards.max_dist
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.0)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 200) + torch.exp(-torch.abs(d_max) * 200)) / 2

    def _reward_knee_distance(self):
        """
        Calculates the reward based on the distance between the knee of the humanoid.
        """
        foot_pos = self.rigid_state[:, self.knee_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist
        fd += (self.command_stand[:, 0] > 0.0) * 0.01
        max_df = self.cfg.rewards.max_dist / 2
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.0)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 200) + torch.exp(-torch.abs(d_max) * 200)) / 2

    def _reward_foot_slip(self):
        """
        Calculates the reward for minimizing foot slip. The reward is based on the contact forces
        and the speed of the feet. A contact threshold is used to determine if the foot is in contact
        with the ground. The speed of the foot is calculated and scaled by the contact condition.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0
        foot_speed_norm = torch.norm(self.rigid_state[:, self.feet_indices, 7:9], dim=2)
        rew = torch.sqrt(foot_speed_norm)
        rew *= contact
        return torch.sum(rew, dim=1)

    def _reward_feet_air_time(self):
        """
        Calculates the reward for feet air time, promoting longer steps. This is achieved by
        checking the first contact with the ground after being in the air. The air time is
        limited to a maximum value for reward calculation.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0
        stance_mask = self._get_gait_phase()
        self.contact_filt = torch.logical_or(torch.logical_or(contact, stance_mask), self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.0) * self.contact_filt
        self.feet_air_time += self.dt
        air_time = self.feet_air_time.clamp(0, 0.6) * first_contact
        self.feet_air_time *= ~self.contact_filt
        return air_time.sum(dim=1)

    def _reward_feet_contact_number(self):
        """
        Calculates a reward based on the number of feet contacts aligning with the gait phase.
        Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0
        stance_mask = self._get_gait_phase()
        reward = torch.where(contact == stance_mask, 1, -0.3)
        return torch.mean(reward, dim=1)

    def _reward_orientation(self):
        """
        Calculates the reward for maintaining a flat base orientation. It penalizes deviation
        from the desired base orientation using the base euler angles and the projected gravity vector.
        """
        quat_mismatch = torch.exp(-torch.sum(torch.abs(self.base_euler_xyz[:, :2]), dim=1) * 10)
        orientation = torch.exp(-torch.norm(self.projected_gravity[:, :2], dim=1) * 20)
        return (quat_mismatch + orientation) / 2.0

    def _reward_foot_orientation(self):
        """
        Calculates the reward for maintaining a flat base orientation. It penalizes deviation
        from the desired base orientation using the base euler angles and the projected gravity vector.
        """

        foot_quat = self.rigid_state[:, self.feet_indices, 3:7]
        foot_euler_xyz_0 = get_euler_xyz_tensor(foot_quat[:, 0, :])
        foot_euler_xyz_1 = get_euler_xyz_tensor(foot_quat[:, 1, :])

        quat_mismatch_0 = torch.exp(-torch.sum(torch.abs(foot_euler_xyz_0[:, :2]), dim=1) * 10)
        quat_mismatch_1 = torch.exp(-torch.sum(torch.abs(foot_euler_xyz_1[:, :2]), dim=1) * 10)
        return (quat_mismatch_0 + quat_mismatch_1) / 2.0

    def _reward_feet_contact_forces(self):
        """
        Calculates the reward for keeping contact forces within a specified range. Penalizes
        high contact forces on the feet.
        """
        return torch.sum(
            (
                torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - self.cfg.rewards.max_contact_force
            ).clip(0, 2000),
            dim=1,
        )

    def _reward_feet_contact_vel(self):
        """
        Calculates the reward for minimizing foot slip. The reward is based on the contact forces
        and the speed of the feet. A contact threshold is used to determine if the foot is in contact
        with the ground. The speed of the foot is calculated and scaled by the contact condition.
        """
        ref_idx_ = torch.where(self.commands[:, 0] > 1.0, 1, 0)

        # stance_current = torch.zeros((self.num_envs, 2), device=self.device)
        # stance_next = torch.zeros((self.num_envs, 2), device=self.device)
        # stance_current[torch.abs(self.ref_phase_foot_force[ref_idx_, self.episode_phase_buf]) < 5.] = 1
        # stance_next[torch.abs(self.ref_phase_foot_force[ref_idx_, self.episode_phase_buf+3]) > 5.] = 1

        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0

        foot_speed_norm = torch.norm(self.rigid_state[:, self.feet_indices, 7:10], dim=2)

        rew = foot_speed_norm
        rew *= contact
        return torch.sum(rew, dim=1)

    def _reward_contact_no_vel(self):
        # Penalize contact with no velocity
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.0
        feet_vel = self.rigid_state[:, self.feet_indices, 7:10]
        contact_feet_vel = feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1, 2))

    def _reward_default_joint_pos(self):
        """
        Calculates the reward for keeping joint positions close to default positions, with a focus
        on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
        """
        joint_diff = self.dof_pos - self.default_joint_pd_target
        left_yaw_roll = joint_diff[:, :2]
        right_yaw_roll = joint_diff[:, 6:8]
        yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
        yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)
        return torch.exp(-yaw_roll * 100) - 0.01 * torch.norm(joint_diff, dim=1)

    def _reward_base_height(self):
        """
        Calculates the reward based on the robot's base height. Penalizes deviation from a target base height.
        The reward is computed based on the height difference between the robot's base and the average height
        of its feet when they are in contact with the ground.
        """
        measured_heights = torch.minimum(
            self.rigid_state[:, self.feet_indices[0], 2], self.rigid_state[:, self.feet_indices[1], 2]
        )

        base_height = self.root_states[:, 2] - (measured_heights - 0.05)
        return torch.exp(-torch.abs(base_height - self.cfg.rewards.base_height_target) * 100)

    def _reward_base_acc(self):
        """
        Computes the reward based on the base's acceleration. Penalizes high accelerations of the robot's base,
        encouraging smoother motion.
        """
        root_acc = self.last_root_vel - self.root_states[:, 7:13]
        rew = torch.exp(-torch.norm(root_acc, dim=1) * 3)
        return rew

    def _reward_vel_mismatch_exp(self):
        """
        Computes a reward based on the mismatch in the robot's linear and angular velocities.
        Encourages the robot to maintain a stable velocity by penalizing large deviations.
        """
        lin_mismatch = torch.exp(-torch.square(self.base_lin_vel[:, 2]) * 10)
        ang_mismatch = torch.exp(-torch.norm(self.base_ang_vel[:, :2], dim=1) * 5.0)

        c_update = (lin_mismatch + ang_mismatch) / 2.0

        return c_update

    def _reward_track_vel_hard(self):
        """
        Calculates a reward for accurately tracking both linear and angular velocity commands.
        Penalizes deviations from specified linear and angular velocity targets.
        """
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.norm(self.commands[:, :2] - self.base_lin_vel[:, :2], dim=1)
        lin_vel_error_exp = torch.exp(-lin_vel_error * 10)

        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.abs(self.commands[:, 2] - self.base_ang_vel[:, 2])
        ang_vel_error_exp = torch.exp(-ang_vel_error * 10)

        linear_error = 0.2 * (lin_vel_error + ang_vel_error)

        return (lin_vel_error_exp + ang_vel_error_exp) / 2.0 - linear_error

    def _reward_tracking_lin_vel(self):
        """
        Tracks linear velocity commands along the xy axes.
        Calculates a reward based on how closely the robot's linear velocity matches the commanded values.
        """
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error * self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        """
        Tracks angular velocity commands for yaw rotation.
        Computes a reward based on how closely the robot's angular velocity matches the commanded yaw values.
        """

        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error * self.cfg.rewards.tracking_sigma)

    def _reward_feet_clearance(self):
        """
        Calculates reward based on the clearance of the swing leg from the ground during movement.
        Encourages appropriate lift of the feet during the swing phase of the gait.
        """
        # Compute feet contact mask
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0

        # Get the z-position of the feet and compute the change in z-position
        feet_z = self.rigid_state[:, self.feet_indices, 2] - 0.03
        delta_z = feet_z - self.last_feet_z
        self.feet_height += delta_z
        self.last_feet_z = feet_z

        # Compute swing mask
        swing_mask = 1 - self._get_gait_phase()

        # feet height should be closed to target feet height at the peak
        rew_pos = torch.abs(self.feet_height - self.cfg.rewards.target_feet_height) < 0.01
        rew_pos = torch.sum(rew_pos * swing_mask, dim=1)
        self.feet_height *= ~contact
        return rew_pos

    def _reward_low_speed(self):
        """
        Rewards or penalizes the robot based on its speed relative to the commanded speed.
        This function checks if the robot is moving too slow, too fast, or at the desired speed,
        and if the movement direction matches the command.
        """
        # Calculate the absolute value of speed and command for comparison
        absolute_speed = torch.abs(self.base_lin_vel[:, 0])
        absolute_command = torch.abs(self.commands[:, 0])

        # Define speed criteria for desired range
        speed_too_low = absolute_speed < 0.5 * absolute_command
        speed_too_high = absolute_speed > 1.2 * absolute_command
        speed_desired = ~(speed_too_low | speed_too_high)

        # Check if the speed and command directions are mismatched
        sign_mismatch = torch.sign(self.base_lin_vel[:, 0]) != torch.sign(self.commands[:, 0])

        # Initialize reward tensor
        reward = torch.zeros_like(self.base_lin_vel[:, 0])

        # Assign rewards based on conditions
        # Speed too low
        reward[speed_too_low] = -1.0
        # Speed too high
        reward[speed_too_high] = 0.0
        # Speed within desired range
        reward[speed_desired] = 1.2
        # Sign mismatch has the highest priority
        reward[sign_mismatch] = -2.0
        return reward * (self.commands[:, 0].abs() > 0.1)

    def _reward_torques(self):
        """
        Penalizes the use of high torques in the robot's joints. Encourages efficient movement by minimizing
        the necessary force exerted by the motors.
        """
        torques_ = self.torques.clone()
        torques_0 = (torch.abs(torques_[:, [5, 11]]) - 15.0).clip(0, 100)
        torques_1 = (torch.abs(torques_[:, [4, 10]]) - 20.0).clip(0, 100)
        torques_2 = (torch.abs(torques_[:, [1, 7]]) - 20.0).clip(0, 100)
        return torch.sum(torch.square(torques_), dim=1) + torch.sum(
            torch.square(torques_0 * 50) + torch.square(torques_1 * 50) + torch.square(torques_2 * 10), dim=1
        )

    def _reward_dof_vel(self):
        """
        Penalizes high velocities at the degrees of freedom (DOF) of the robot. This encourages smoother and
        more controlled movements.
        """
        dof_vel_ = (torch.abs(self.dof_vel[:, [3, 9]]) - 10.5).clip(0, 100) * 3.0
        return torch.sum(torch.square(dof_vel_), dim=1)

    def _reward_dof_acc(self):
        """
        Penalizes high accelerations at the robot's degrees of freedom (DOF). This is important for ensuring
        smooth and stable motion, reducing wear on the robot's mechanical parts.
        """
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_energy(self):
        """
        Penalizes the use of high torques in the robot's joints. Encourages efficient movement by minimizing
        the necessary force exerted by the motors.
        """
        return torch.sum(torch.abs(self.torques * self.dof_vel), dim=1)

    def _reward_collision(self):
        """
        Penalizes collisions of the robot with the environment, specifically focusing on selected body parts.
        This encourages the robot to avoid undesired contact with objects or surfaces.
        """
        return torch.sum(
            1.0 * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1
        )

    def _reward_action_smoothness(self):
        """
        Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
        This is important for achieving fluid motion and reducing mechanical stress.
        """
        term_1 = torch.sum(torch.square(self.last_actions - self.actions), dim=1)
        term_2 = torch.sum(torch.square(self.actions + self.last_last_actions - 2 * self.last_actions), dim=1)
        term_3 = 0.05 * torch.sum(torch.abs(self.actions), dim=1)

        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0
        command_stand_contact = (self.command_stand[:, 0] > 0.0) & contact[:, 0] & contact[:, 1]

        term_1 = torch.where(command_stand_contact, term_1 * 20.0, term_1 * 2.0)
        term_2 = torch.where(command_stand_contact, term_2 * 20.0, term_2 * 2.0)

        return term_1 + term_2 + term_3

    def _reward_imition_Torso_orientation(self):
        """
        imition reward: Torso orientation
        """
        ref_idx_ = torch.where(self.commands[:, 0] > 1.0, 1, 0)
        ref_xy = self.ref_phase_root_eu_ang[ref_idx_, self.episode_phase_buf][:, 4:6]
        current_xy = self.base_euler_xyz[:, :2]
        ref_xy_stand_ = ref_xy * 0.0
        # ref_xy_stand_[:, 0] += -0.0024
        ref_xy_stand_[:, 1] += 0.046

        ref_xy = torch.where(self.command_stand > 0.0, ref_xy_stand_, ref_xy)

        # reward = torch.exp(-torch.sum(torch.square(current_xy - ref_xy), dim=1) * 20)

        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0
        command_stand_contact = (self.command_stand[:, 0] > 0.0) & contact[:, 0] & contact[:, 1]
        reward = torch.where(
            command_stand_contact,
            torch.exp(-torch.sum(torch.square(current_xy - ref_xy), dim=1) * 50),
            torch.exp(-torch.sum(torch.square(current_xy - ref_xy), dim=1) * 20),
        )

        return reward

    def _reward_imition_Linear_velocity_xy(self):
        """
        imition reward: Linear velocity xy
        """
        ref_idx_ = torch.where(self.commands[:, 0] > 1.0, 1, 0)
        ref_xy = self.ref_phase_root_lin_vel[ref_idx_, self.episode_phase_buf][:, :2] * 0.0
        current_xy = self.base_lin_vel[:, :2]

        reward = torch.exp(-torch.sum(torch.square(current_xy - ref_xy), dim=1) * 8)
        return reward

    def _reward_imition_Linear_velocity_z(self):
        """
        imition reward: Linear velocity z
        """
        ref_idx_ = torch.where(self.commands[:, 0] > 1.0, 1, 0)
        ref_z = self.ref_phase_root_lin_vel[ref_idx_, self.episode_phase_buf][:, 2:]
        current_z = self.base_lin_vel[:, 2:]

        ref_z = torch.where(self.command_stand > 0.0, ref_z * 0.0, ref_z)

        reward = torch.exp(-torch.sum(torch.square(current_z - ref_z), dim=1) * 8)
        return reward

    def _reward_imition_Angular_velocity_xy(self):
        """
        imition reward: Angular velocity xy
        """
        ref_idx_ = torch.where(self.commands[:, 0] > 1.0, 1, 0)
        ref_xy = self.ref_phase_root_ang_vel[ref_idx_, self.episode_phase_buf][:, :2]
        current_xy = self.base_ang_vel[:, :2]

        diff = current_xy - ref_xy
        diff[:, 0] *= 1.2
        diff[:, 1] *= 0.8
        reward = torch.exp(-torch.sum(torch.square(current_xy - ref_xy), dim=1) * 2)
        return reward

    def _reward_imition_Angular_velocity_z(self):
        """
        imition reward: Angular velocity z
        """
        ref_idx_ = torch.where(self.commands[:, 0] > 1.0, 1, 0)
        ref_z = self.ref_phase_root_ang_vel[ref_idx_, self.episode_phase_buf][:, 2:] * 0.0
        current_z = self.base_ang_vel[:, 2:]

        reward = torch.exp(-torch.sum(torch.square(current_z - ref_z), dim=1) * 2)
        return reward

    def _reward_imition_Leg_joint_positions(self):
        """
        imition reward: Leg joint positions
        """
        ref_idx_ = torch.where(self.commands[:, 0] > 1.0, 1, 0)
        ref_pos = self.ref_phase_dof_pos[ref_idx_, self.episode_phase_buf].clone()
        current_pos = self.dof_pos
        stand_pos_ = ref_pos * 0.0
        # stand_pos_[:, 0] += -0.017
        # stand_pos_[:, 1] += -0.001
        stand_pos_[:, 2] += -0.633
        stand_pos_[:, 3] += 1.138
        stand_pos_[:, 4] += -0.552
        # stand_pos_[:, 5] += 0.019

        # stand_pos_[:, 6] += 0.0227
        # stand_pos_[:, 7] += 0.001
        stand_pos_[:, 8] += -0.631
        stand_pos_[:, 9] += 1.136
        stand_pos_[:, 10] += -0.551
        # stand_pos_[:, 11] += -0.02

        ref_pos = torch.where(self.command_stand > 0.0, stand_pos_, ref_pos)

        # reward = -torch.sum(torch.square(current_pos - ref_pos), dim=1)
        diff = current_pos - ref_pos
        diff = torch.where(self.commands[:, 0:1] > 1.0, diff * 0.6, diff)
        reward = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)

        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0
        command_stand_contact = (self.command_stand[:, 0] > 0.0) & contact[:, 0] & contact[:, 1]
        reward = torch.where(command_stand_contact, reward * 2.0, reward)
        return reward

    def _reward_imition_Leg_joint_velocities(self):
        """
        imition reward: Leg joint velocities
        """
        ref_idx_ = torch.where(self.commands[:, 0] > 1.0, 1, 0)
        ref_vel = self.ref_phase_dof_vel[ref_idx_, self.episode_phase_buf].clone().clip(-10.0, 10.0)
        current_vel = self.dof_vel

        ref_vel = torch.where(self.command_stand > 0.0, ref_vel * 0.0, ref_vel)

        error = torch.square(current_vel - ref_vel)
        # error[:, [4,5,10,11]] *= 0.1
        # error[:, [0,1,6,7]] *= 0.1

        reward = -torch.sum(error, dim=1)
        reward = torch.where(self.command_stand[:, 0] > 0.0, reward * 20.0, reward)
        return reward

    def _reward_imition_Torso_height(self):
        """
        imition reward: imition_Torso_height
        """
        ref_idx_ = torch.where(self.commands[:, 0] > 1.0, 1, 0)

        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0

        self.last_base_height[:] = torch.where(
            torch.logical_and(contact[:, 0], (self.episode_phase_buf % self.period) == 17),
            self.root_states[:, 2],
            self.last_base_height[:],
        )

        ref_z = self.ref_phase_root_pos[ref_idx_, self.episode_phase_buf][:, 2]
        current_z = self.root_states[:, 2].clone()
        current_z[:] -= self.last_base_height[:]

        ref_z = torch.where(self.command_stand[:, 0] > 0.0, ref_z * 0.0, ref_z)
        # current_z = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        error = torch.square(current_z - ref_z)
        reward = torch.exp(-error * 100.0)

        return reward

    def _reward_imition_Foot_height(self):
        """
        imition reward: imition_Foot_height
        """
        ref_idx_ = torch.where(self.commands[:, 0] > 1.0, 1, 0)

        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0

        self.last_feet_height[:, 0] = torch.where(
            torch.logical_and(contact[:, 0], (self.episode_phase_buf % self.period) == 19),
            self.rigid_state[:, self.feet_indices[0], 2],
            self.last_feet_height[:, 0],
        )
        self.last_feet_height[:, 1] = torch.where(
            torch.logical_and(contact[:, 1], (self.episode_phase_buf % self.period) == 50),
            self.rigid_state[:, self.feet_indices[1], 2],
            self.last_feet_height[:, 1],
        )

        ref_z = self.ref_phase_foot_height[ref_idx_, self.episode_phase_buf]
        current_z = self.rigid_state[:, self.feet_indices, 2].clone()
        current_z[:, 0] -= self.last_feet_height[:, 0]
        current_z[:, 1] -= self.last_feet_height[:, 1]
        # current_z = self.rigid_state[:, self.ankle_indices, 2]

        # left_z = torch.mean(self.rigid_state[:, [self.feet_indices[0]], 2] - self.measured_heights, dim=1)
        # right_z = torch.mean(self.rigid_state[:, [self.feet_indices[1]], 2] - self.measured_heights, dim=1)
        # error = torch.square(left_z - ref_z[:, 0]) + torch.square(right_z - ref_z[:, 1])

        ref_z = torch.where(self.command_stand > 0.0, ref_z * 0.0, ref_z)

        error = current_z - ref_z

        error = torch.square(torch.norm(error, dim=1))
        # reward = torch.exp(-error * 1000.)
        reward = torch.where(self.commands[:, 0] > 1.0, torch.exp(-error * 100.0), torch.exp(-error * 1000.0))
        return reward

    def _reward_imition_Foot_vel(self):
        """
        imition reward: imition_Foot_vel
        """
        ref_idx_ = torch.where(self.commands[:, 0] > 1.0, 1, 0)

        ref_zvel = self.ref_phase_foot_zvel[ref_idx_, self.episode_phase_buf]
        current_zvel = self.rigid_state[:, self.feet_indices, 9].clone()

        ref_zvel = torch.where(self.command_stand > 0.0, ref_zvel * 0.0, ref_zvel)
        reward = -torch.sum(torch.square(current_zvel - ref_zvel), dim=1)
        return reward

    def _reward_imition_Contact(self):
        """
        imition reward: Contact
        """
        # ref_idx_ = torch.where(self.commands[:, 0] > 1.0, 1, 0)

        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0
        stance_mask = self._get_gait_phase()

        stance_mask = torch.where(self.command_stand > 0.0, stance_mask * 0.0 + 1.0, stance_mask)
        reward = torch.where(contact == stance_mask, 1, 0.0)
        reward = torch.sum(reward, dim=1)
        return reward

    def _reward_imition_Torque(self):
        """
        imition reward: Torque
        """

        ref_torque = self.ref_phase_dof_torque[self.episode_phase_buf]
        current_torque = self.torques

        error = torch.sum(torch.square(current_torque - ref_torque), dim=1)
        reward = torch.exp(-error * 0.0001)
        return reward

    def _reward_imition_Survival(self):
        """
        imition reward: Survival
        """

        return 1.0
