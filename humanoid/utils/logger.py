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

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value
import csv


class Logger:
    def __init__(self, dt):
        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.dt = dt
        self.num_episodes = 0
        self.plot_process = None

    def log_state(self, key, value):
        self.state_log[key].append(value)

    def log_states(self, dict):
        for key, value in dict.items():
            self.log_state(key, value)

    def log_rewards(self, dict, num_episodes):
        for key, value in dict.items():
            if "rew" in key:
                self.rew_log[key].append(value.item() * num_episodes)
        self.num_episodes += num_episodes

    def reset(self):
        self.state_log.clear()
        self.rew_log.clear()

    def plot_states(self):
        # self.plot_process = Process(target=self._plot)
        self._plot()
        # self.plot_process.start()

    def _plot(self):
        nb_rows = 8
        nb_cols = 9
        fig, axs = plt.subplots(nb_rows, nb_cols)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value) * self.dt, len(value))
            break
        log = self.state_log

        # dof_name = ['leg_l1_joint', 'leg_l2_joint', 'leg_l3_joint', 'leg_l4_joint', 'leg_l5_joint', 'leg_l6_joint',
        #             'leg_r1_joint', 'leg_r2_joint', 'leg_r3_joint', 'leg_r4_joint', 'leg_r5_joint', 'leg_r6_joint']

        # with open('v4pro_jog_1104.csv', mode='w', newline='') as csvfile:
        #     writer = csv.writer(csvfile)

        #     row_name = [name_ + '_pos' for name_ in dof_name] + \
        #                 [name_ + '_vel' for name_ in dof_name] + \
        #                 [name_ + '_torque' for name_ in dof_name]

        #     writer.writerow(row_name)

        #     print('len(log["dof_pos"+str(0)+"_gym"]):',len(log["dof_pos"+str(0)+"_gym"]))
        #     for i in range(len(log["dof_pos"+str(0)+"_gym"])):

        #         dof_pos_buffer = []
        #         dof_vel_buffer = []
        #         dof_torque_buffer = []

        #         for j in range(12):
        #             dof_pos_buffer.append(log["dof_pos"+str(j)+"_gym"][i])
        #             dof_vel_buffer.append(log["dof_vel"+str(j)+"_gym"][i])
        #             dof_torque_buffer.append(log["dof_torque_"+str(j)][i])

        #         writer.writerow(dof_pos_buffer + \
        #                         dof_vel_buffer + \
        #                         dof_torque_buffer)

        # plot base vel x
        a = axs[0, nb_cols - 1]
        if log["base_vel_x"]:
            a.plot(time, log["base_vel_x"], label="measured")
        if log["command_x"]:
            a.plot(time, log["command_x"], label="commanded")
        a.set(xlabel="time [s]", ylabel="base lin vel [m/s]", title="Base velocity x")
        a.legend()
        # plot base vel y
        a = axs[1, nb_cols - 1]
        if log["base_vel_y"]:
            a.plot(time, log["base_vel_y"], label="measured")
        if log["command_y"]:
            a.plot(time, log["command_y"], label="commanded")
        a.set(xlabel="time [s]", ylabel="base lin vel [m/s]", title="Base velocity y")
        a.legend()
        # plot base vel z
        a = axs[3, nb_cols - 1]
        if log["base_vel_z"]:
            a.plot(time, log["base_vel_z"], label="measured")
        a.set(xlabel="time [s]", ylabel="base lin vel [m/s]", title="Base velocity z")
        a.legend()
        # plot contact forces
        a = axs[4, nb_cols - 1]
        if log["contact_forces_z"]:
            forces = np.array(log["contact_forces_z"])
            for i in range(forces.shape[1]):
                a.plot(time, forces[:, i], label=f"force {i}")
        a.set(xlabel="time [s]", ylabel="Forces z [N]", title="Vertical Contact forces")
        a.legend()
        # plot contact vel
        a = axs[5, nb_cols - 1]
        if log["contact_period"]:
            forces = np.array(log["contact_period"])
            for i in range(forces.shape[1]):
                a.plot(time, forces[:, i], label=f"period {i}")
        a.set(xlabel="time [s]", ylabel="vel z [N]", title="Vertical Contact period")
        a.legend()
        # plot contact vel
        a = axs[6, nb_cols - 1]
        if log["contact_vel_z"]:
            forces = np.array(log["contact_vel_z"])
            for i in range(forces.shape[1]):
                a.plot(time, forces[:, i], label=f"vel {i}")
        a.set(xlabel="time [s]", ylabel="vel z [N]", title="Vertical Contact vel")
        a.legend()
        # plot torque/vel curves
        a = axs[7, nb_cols - 1]
        if log["dof_vel0_gym"] != [] and log["dof_torque_0"] != []:
            a.plot(log["dof_vel0_gym"], log["dof_torque_0"], "x", label="measured")
        a.set(xlabel="Joint vel [rad/s]", ylabel="Joint Torque [Nm]", title="Torque/velocity curves")
        a.legend()

        for i in range(3):
            # plot base vel yaw
            a = axs[nb_rows - 1, 3 + i]
            if log["base_ang" + str(i) + "_gym"]:
                a.plot(time, log["base_ang" + str(i) + "_gym"], label="measured")
            if i == 0:
                if log["command_yaw_gym"]:
                    a.plot(time, log["command_yaw_gym"], label="commanded")
            a.set(xlabel="time [s]", ylabel="base ang vel [rad/s]", title="Base ang" + str(i) + "_gym")
            a.legend()

            # plot base vel yaw
            a = axs[nb_rows - 2, 3 + i]
            if log["base_ang" + str(i) + "_gazebo"]:
                a.plot(time, log["base_ang" + str(i) + "_gazebo"], label="measured")
            if i == 0:
                if log["command_yaw_gazebo"]:
                    a.plot(time, log["command_yaw_gazebo"], label="commanded")
            a.set(xlabel="time [s]", ylabel="base ang vel [rad/s]", title="Base ang" + str(i) + "_gym")
            a.legend()

        for i in range(3):
            a = axs[nb_rows - 1, i]
            if log["base_euler" + str(i) + "_gym"]:
                a.plot(time, log["base_euler" + str(i) + "_gym"], label="measured")
            a.set(xlabel="time [s]", ylabel="base euler [rad/s]", title="Base euler" + str(i) + "_gym")
            a.legend()

            a = axs[nb_rows - 2, i]
            if log["base_euler" + str(i) + "_gazebo"]:
                a.plot(time, log["base_euler" + str(i) + "_gazebo"], label="measured")
            a.set(xlabel="time [s]", ylabel="base euler [rad/s]", title="Base euler" + str(i) + "_gazebo")
            a.legend()

        for i in range(6):
            a = axs[i, 0]
            if log["dof_pos" + str(i) + "_gym"]:
                a.plot(time, log["dof_pos" + str(i) + "_gym"], label="measured")
            if log["dof_pos_target" + str(i) + "_gym"]:
                a.plot(time, log["dof_pos_target" + str(i) + "_gym"], label="target")
            if log["ref_dof_pos" + str(i) + "_gym"]:
                a.plot(time, log["ref_dof_pos" + str(i) + "_gym"], label="ref")

            a.set(xlabel="time [s]", ylabel="Position [rad]", title="DOF Position" + str(i))
            a.legend()

        for i in range(6):
            a = axs[i, 1]
            if log["dof_pos" + str(6 + i) + "_gym"]:
                a.plot(time, log["dof_pos" + str(6 + i) + "_gym"], label="measured")
            if log["dof_pos_target" + str(6 + i) + "_gym"]:
                a.plot(time, log["dof_pos_target" + str(6 + i) + "_gym"], label="target")
            if log["ref_dof_pos" + str(6 + i) + "_gym"]:
                a.plot(time, log["ref_dof_pos" + str(6 + i) + "_gym"], label="ref")

            a.set(xlabel="time [s]", ylabel="Position [rad]", title="DOF Position" + str(6 + i))
            a.legend()

        for i in range(6):
            a = axs[i, 2]
            if log["dof_vel" + str(i) + "_gym"]:
                a.plot(time, log["dof_vel" + str(i) + "_gym"], label="measured")
            if log["dof_vel_target" + str(i) + "_gym"]:
                a.plot(time, log["dof_vel_target" + str(i) + "_gym"], label="target")
            if log["ref_dof_vel" + str(i) + "_gym"]:
                a.plot(time, log["ref_dof_vel" + str(i) + "_gym"], label="ref")
            a.set(xlabel="time [s]", ylabel="Velocity [rad/s]", title="Joint Velocity" + str(i))
            a.legend()

        for i in range(6):
            a = axs[i, 3]
            if log["dof_vel" + str(6 + i) + "_gym"]:
                a.plot(time, log["dof_vel" + str(6 + i) + "_gym"], label="measured")
            if log["dof_vel_target" + str(6 + i) + "_gym"]:
                a.plot(time, log["dof_vel_target" + str(6 + i) + "_gym"], label="target")
            if log["ref_dof_vel" + str(6 + i) + "_gym"]:
                a.plot(time, log["ref_dof_vel" + str(6 + i) + "_gym"], label="ref")
            a.set(xlabel="time [s]", ylabel="Velocity [rad/s]", title="Joint Velocity" + str(6 + i))
            a.legend()

        for i in range(6):
            a = axs[i, 4]
            if log["dof_torque_" + str(i)]:
                a.plot(time, log["dof_torque_" + str(i)], label="measured")
            if log["ref_dof_torque" + str(i) + "_gym"]:
                a.plot(time, log["ref_dof_torque" + str(i) + "_gym"], label="ref")
            a.set(xlabel="time [s]", ylabel="Joint Torque [Nm]", title="Torque " + str(i))
            a.legend()

        for i in range(6):
            a = axs[i, 5]
            if log["dof_torque_" + str(6 + i)]:
                a.plot(time, log["dof_torque_" + str(6 + i)], label="measured")
            if log["ref_dof_torque" + str(6 + i) + "_gym"]:
                a.plot(time, log["ref_dof_torque" + str(6 + i) + "_gym"], label="ref")
            a.set(xlabel="time [s]", ylabel="Joint Torque [Nm]", title="Torque " + str(6 + i))
            a.legend()

        #########

        for i in range(5):
            a = axs[i, 6]
            if log["dof_vel" + str(i) + "_gazebo"]:
                a.plot(time, log["dof_vel" + str(i) + "_gazebo"], label="measured")
            if log["dof_vel_target" + str(i) + "_gazebo"]:
                a.plot(time, log["dof_vel_target" + str(i) + "_gazebo"], label="target")
            a.set(xlabel="time [s]", ylabel="Velocity [rad/s]", title="Joint Velocity" + str(i) + "_gazebo")
            a.legend()

        for i in range(5):
            a = axs[i, 7]
            if log["dof_vel" + str(5 + i) + "_gazebo"]:
                a.plot(time, log["dof_vel" + str(5 + i) + "_gazebo"], label="measured")
            if log["dof_vel_target" + str(5 + i) + "_gazebo"]:
                a.plot(time, log["dof_vel_target" + str(5 + i) + "_gazebo"], label="target")
            a.set(xlabel="time [s]", ylabel="Velocity [rad/s]", title="Joint Velocity" + str(5 + i) + "_gazebo")
            a.legend()

        plt.show()

    def print_rewards(self):
        print("Average rewards per second:")
        for key, values in self.rew_log.items():
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")

    def __del__(self):
        if self.plot_process is not None:
            self.plot_process.kill()
