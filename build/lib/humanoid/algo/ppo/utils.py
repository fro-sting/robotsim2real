# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
import numpy as np


def split_and_pad_trajectories(tensor, dones):
    """Splits trajectories at done indices. Then concatenates them and pads with zeros up to the length og the longest trajectory.
    Returns masks corresponding to valid parts of the trajectories
    Example:
        Input: [ [a1, a2, a3, a4 | a5, a6],
                 [b1, b2 | b3, b4, b5 | b6]
                ]

        Output:[ [a1, a2, a3, a4], | [  [True, True, True, True],
                 [a5, a6, 0, 0],   |    [True, True, False, False],
                 [b1, b2, 0, 0],   |    [True, True, False, False],
                 [b3, b4, b5, 0],  |    [True, True, True, False],
                 [b6, 0, 0, 0]     |    [True, False, False, False],
                ]                  | ]

    Assumes that the input has the following dimension order: [time, number of envs, additional dimensions]
    """
    dones = dones.clone()
    dones[-1] = 1
    # Permute the buffers to have order (num_envs, num_transitions_per_env, ...), for correct reshaping
    flat_dones = dones.transpose(1, 0).reshape(-1, 1)

    # Get length of trajectory by counting the number of successive not done elements
    done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero()[:, 0]))
    trajectory_lengths = done_indices[1:] - done_indices[:-1]
    trajectory_lengths_list = trajectory_lengths.tolist()
    # Extract the individual trajectories
    trajectories = torch.split(tensor.transpose(1, 0).flatten(0, 1), trajectory_lengths_list)
    padded_trajectories = torch.nn.utils.rnn.pad_sequence(trajectories)

    trajectory_masks = trajectory_lengths > torch.arange(0, tensor.shape[0], device=tensor.device).unsqueeze(1)
    return padded_trajectories, trajectory_masks


def unpad_trajectories(trajectories, masks):
    """Does the inverse operation of  split_and_pad_trajectories()"""
    # Need to transpose before and after the masking to have proper reshaping
    return (
        trajectories.transpose(1, 0)[masks.transpose(1, 0)]
        .view(-1, trajectories.shape[0], trajectories.shape[-1])
        .transpose(1, 0)
    )


# TODO: this is probably a better case for inheritance than for a wrapper
# Gives an interface to exploit mirror symmetry
class SymmetricEnv:
    def __init__(self, mirrored_obs=None, mirrored_act=None, clock_inds=None, obs_fn=None, act_fn=None):
        assert (bool(mirrored_act) ^ bool(act_fn)) and (bool(mirrored_obs) ^ bool(obs_fn)), (
            "You must provide either mirror indices or a mirror function, but not both, for              observation"
            " and action."
        )

        if mirrored_act:
            self.act_mirror_matrix = torch.Tensor(_get_symmetry_matrix(mirrored_act))

        elif act_fn:
            assert callable(act_fn), "Action mirror function must be callable"
            self.mirror_action = act_fn

        if mirrored_obs:
            self.obs_mirror_matrix = torch.Tensor(_get_symmetry_matrix(mirrored_obs))

        elif obs_fn:
            assert callable(obs_fn), "Observation mirror function must be callable"
            self.mirror_observation = obs_fn

        self.clock_inds = clock_inds
        self.base_obs_len = 47

    def __getattr__(self, attr):
        return getattr(self.env, attr)

    def mirror_action(self, action):
        return action @ self.act_mirror_matrix

    def mirror_observation(self, obs):
        return obs @ self.obs_mirror_matrix

    # To be used when there is a clock in the observation. In this case, the mirrored_obs vector inputted
    # when the SymmeticEnv is created should not move the clock input order. The indices of the obs vector
    # where the clocks are located need to be inputted.
    def mirror_clock_observation(self, obs):
        # print("obs.shape = ", obs.shape)
        # print("obs_mirror_matrix.shape = ", self.obs_mirror_matrix.shape)
        mirror_obs_batch = torch.zeros_like(obs, device=obs.device)
        history_len = 15  # FIX HISTORY-OF-STATES LENGTH TO 1 FOR NOW
        for block in range(history_len):
            obs_ = obs[:, self.base_obs_len * block : self.base_obs_len * (block + 1)]
            mirror_obs = obs_ @ self.obs_mirror_matrix
            clock = mirror_obs[:, self.clock_inds]
            for i in range(np.shape(clock)[1]):
                mirror_obs[:, self.clock_inds[i]] = torch.sin(torch.arcsin(clock[:, i]) + torch.pi)
            mirror_obs_batch[:, self.base_obs_len * block : self.base_obs_len * (block + 1)] = mirror_obs
        return mirror_obs_batch


def _get_symmetry_matrix(mirrored):
    numel = len(mirrored)
    mat = np.zeros((numel, numel))

    for i, j in zip(np.arange(numel), np.abs(np.array(mirrored).astype(int))):
        mat[i, j] = np.sign(mirrored[i])

    return mat
