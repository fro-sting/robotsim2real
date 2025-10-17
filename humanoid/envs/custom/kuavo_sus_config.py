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


from humanoid.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import numpy as np


class KuavoSuspendedCfg(LeggedRobotCfg):
    """
    Configuration class for the XBotL humanoid robot.
    """

    class env(LeggedRobotCfg.env):
        # change the observation dim
        frame_stack = 15
        c_frame_stack = 3
        num_single_obs = 47
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = 129
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        num_actions = 12
        num_envs = 4096
        episode_length_s = 24  # episode length in seconds
        use_ref_actions = False
        use_only_ref_actions = False
        num_aux = 34

    class safety:
        # safety factors
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 1.0

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/biped_s44/urdf/biped_s44.urdf"

        name = "kuavo"
        foot_name = "6_link"
        ankle_name = "5_link"
        knee_name = "4_link"

        terminate_after_contacts_on = [
            # 'dummy_link',
            "base_link",
            "leg_l1_link",
            "leg_l2_link",
            "leg_l3_link",
            "leg_l4_link",
            "leg_l5_link",
            "leg_r1_link",
            "leg_r2_link",
            "leg_r3_link",
            "leg_r4_link",
            "leg_r5_link",
        ]

        # terminate_after_contacts_on = ['base_link']
        penalize_contacts_on = [
            # 'dummy_link',
            "base_link",
            "leg_l1_link",
            "leg_l2_link",
            "leg_l3_link",
            "leg_l4_link",
            "leg_l5_link",
            "leg_r1_link",
            "leg_r2_link",
            "leg_r3_link",
            "leg_r4_link",
            "leg_r5_link",
        ]

        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "trimesh"
        # mesh_type = 'trimesh'
        curriculum = False
        # rough terrain only:
        measure_heights = False
        measured_points_x = [-0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15]  # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15]
        static_friction = 0.6
        dynamic_friction = 0.6
        terrain_length = 8.0
        terrain_width = 8.0
        num_rows = 20  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        max_init_terrain_level = 10  # starting curriculum state
        # plane; obstacles; uniform; slope_up; slope_down, stair_up, stair_down
        terrain_proportions = [0.4, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0]
        restitution = 0.0

    class noise:
        add_noise = True
        noise_level = 1.0  # scales other values

        class noise_scales:
            dof_pos = 0.05
            dof_vel = 0.5
            ang_vel = 0.2
            lin_vel = 0.05
            quat = 0.02
            height_measurements = 0.1

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.0]

        default_joint_angles = {
            "leg_l1_joint": 0.0,
            "leg_l2_joint": 0.0,
            "leg_l3_joint": 0.0,
            "leg_l4_joint": 0.0,
            "leg_l5_joint": -0.1,
            "leg_l6_joint": 0.0,
            "leg_r1_joint": 0.0,
            "leg_r2_joint": 0.0,
            "leg_r3_joint": 0.0,
            "leg_r4_joint": 0.0,
            "leg_r5_joint": 0.0,
            "leg_r6_joint": 0.0,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        stiffness = {
            "leg_l1_joint": 60,
            "leg_l2_joint": 60,
            "leg_l3_joint": 100,
            "leg_l4_joint": 150,
            "leg_l5_joint": 15,
            "leg_l6_joint": 15,
            "leg_r1_joint": 60,
            "leg_r2_joint": 60,
            "leg_r3_joint": 100,
            "leg_r4_joint": 150,
            "leg_r5_joint": 15,
            "leg_r6_joint": 15,
        }
        damping = {
            "leg_l1_joint": 5.0,
            "leg_l2_joint": 5.0,
            "leg_l3_joint": 5.0,
            "leg_l4_joint": 6.0,
            "leg_l5_joint": 5.0,
            "leg_l6_joint": 5.0,
            "leg_r1_joint": 5.0,
            "leg_r2_joint": 5.0,
            "leg_r3_joint": 5.0,
            "leg_r4_joint": 6.0,
            "leg_r5_joint": 5.0,
            "leg_r6_joint": 5.0,
        }

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10  # 100hz

    class sim(LeggedRobotCfg.sim):
        dt = 0.001  # 1000 Hz
        substeps = 1  # 2
        up_axis = 1  # 0 is y, 1 is z

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.5  # [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**24  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2

    class domain_rand:
        randomize_payload_mass = True
        payload_mass_range = [-5.0, 5.0]

        randomize_com_displacement = True
        com_displacement_range = [-0.05, 0.05]

        randomize_link_mass = True
        link_mass_range = [0.8, 1.2]

        randomize_friction = True
        friction_range = [0.1, 2.0]

        randomize_restitution = True
        restitution_range = [0.0, 0.5]

        randomize_motor_strength = True
        motor_strength_range = [0.8, 1.2]

        randomize_joint_friction = True
        joint_friction_range = [0.02, 0.06]

        randomize_joint_armature = True
        joint_armature_range = [0.02, 0.06]

        disturbance = True
        disturbance_range = [-600.0, 600.0]
        disturbance_s = 3

        push_robots = True
        push_interval_s = 6
        max_push_vel_xy = 0.5
        max_push_ang_vel = 0.5

        randomize_kp = True
        kp_range = [0.8, 1.2]

        randomize_kd = True
        kd_range = [0.8, 1.2]

        action_delay = True
        action_delay_range = [10, 40]

        root_states_delay = True
        root_states_delay_range = [2, 10]

        joint_states_delay = True
        joint_states_delay_range = [2, 5]

        randomize_euler_y_zero_pos = True
        euler_y_zero_pos_range = [-0.05, 0.05]

        randomize_torque_rfi = False
        rfi_lim = 0.1
        randomize_rfi_lim = True
        rfi_lim_range = [0.5, 1.5]

    class commands(LeggedRobotCfg.commands):
        # Vers: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 8.0  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [-0.3, 2.1]  # min max [m/s]
            lin_vel_y = [-0.2, 0.2]  # min max [m/s]
            ang_vel_yaw = [-0.3, 0.3]  # min max [rad/s]
            heading = [-3.14, 3.14]

    class rewards:
        base_height_target = 0.82
        min_dist = 0.15
        max_dist = 1.0

        hip_pos_scale = 0.17  # rad
        knee_pos_scale = 0.34
        ankle_pos_scale = 0.17

        # put some settings here for LLM parameter tuning
        target_joint_pos_scale = 0.17  # rad
        target_feet_height = 0.06  # m
        cycle_time = 0.6  # sec
        # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards = False
        # tracking reward = exp(error*sigma)
        tracking_sigma = 5
        max_contact_force = 900  # Forces above this value are penalized

        class scales:
            # reference motion tracking
            # joint_pos = 1.6
            # feet_clearance = 1.
            # feet_contact_number = 1.2
            # # gait
            # feet_air_time = 1.
            # foot_slip = -0.05
            feet_distance = 0.5
            knee_distance = 0.5
            # # contact
            feet_contact_forces = -0.02
            # feet_contact_vel = -0.1
            contact_no_vel = -5.0
            # # vel tracking
            tracking_lin_vel = 2.5
            tracking_ang_vel = 2.1
            # vel_mismatch_exp = 0.5  # lin_z; ang x,y
            low_speed = 0.5
            track_vel_hard = 0.5
            # # base pos
            # default_joint_pos = 0.5
            # orientation = 1.
            # # foot_orientation = 0.2
            # base_height = 0.2
            # base_acc = 0.2
            # energy
            action_smoothness = -0.005
            torques = -1e-5
            dof_vel = -2e-1
            dof_acc = -1e-7
            energy = -1e-3
            collision = -1.0

            # reference motion imition
            imition_Torso_orientation = 0.5
            # imition_Linear_velocity_xy = 3.
            imition_Linear_velocity_z = 0.5
            imition_Angular_velocity_xy = 1.0
            # imition_Angular_velocity_z = 1.0
            imition_Leg_joint_positions = 5.0
            imition_Leg_joint_velocities = 1e-3
            imition_Torso_height = 1.0
            imition_Foot_height = 2.0
            # imition_Foot_vel = 0.01
            imition_Contact = 2.0
            # imition_Torque = 1.
            imition_Survival = 1.0

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 1.0
            dof_pos = 1.0
            dof_vel = 0.05
            quat = 1.0
            height_measurements = 5.0

        clip_observations = 18.0
        clip_actions = 18.0


class KuavoSuspendedCfgPPO(LeggedRobotCfgPPO):
    seed = 5
    runner_class_name = "OnPolicyRunner"  # DWLOnPolicyRunner

    class policy:
        init_noise_std = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) * 1.2
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.001
        learning_rate = 1e-5
        num_learning_epochs = 2
        gamma = 0.995
        lam = 0.95
        num_mini_batches = 4

    class runner:
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 60  # per iteration
        max_iterations = 60000  # number of policy updates

        # logging
        save_interval = 50  # Please check for potential savings every `save_interval` iterations.
        experiment_name = "kuavo_jog"
        run_name = "pretraining"
        # Load and resume
        resume = True
        # 不能瞎改，因为验证action_real+delta_action的时候也会用到，要保证这个model是导出pretrained policy的model
        load_run = "Apr07_09-25-44"  # -1 = last run
        # 不能瞎改，因为验证action_real+delta_action的时候也会用到，要保证这个model是导出pretrained policy的model
        checkpoint = -1  # -1 = last saved model
        save_config = "kuavo_config.py"
