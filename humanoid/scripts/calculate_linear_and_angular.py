#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#   这个程序
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.signal import butter, filtfilt
import os

# 1. 读取数据
data_slam_vel = np.load('/home/zl/ASAP_data_leju/slam/20250731/selected_topics_2025-08-01-09-51-41.npz')
print(list(data_slam_vel.keys()))
twist_radar = data_slam_vel['bodytwistofslambaseframe']  # (N, 6)
twist_radar_timestamps = data_slam_vel['bodytwistofslambaseframetimestamps']  # (N,)

# 读取状态估计数据
joint_pos = data_slam_vel['jointpos']
joint_vel = data_slam_vel['jointvel']
joint_pos_timestamps = data_slam_vel['jointpostimestamps']

# 跟关节位置数据
data_slam_pose = np.load('/home/zl/ASAP_data_leju/slam/20250731/base_link_trajectory_20250801_095712.npz')
pos_xyz = data_slam_pose['pos_xyz']
euler = data_slam_pose['angular_eu_ang']
root_pose_timestamps = data_slam_pose['timestamps']

# 时间戳对齐，状态估计向slam对齐，插值
print(f"SLAM速度数据帧数: {len(twist_radar_timestamps)}")
print(f"状态估计数据帧数: {len(joint_pos_timestamps)}")
print(f"SLAM位姿数据帧数: {len(root_pose_timestamps)}")
print(f"SLAM速度时间范围: {twist_radar_timestamps[0]:.3f} - {twist_radar_timestamps[-1]:.3f}")
print(f"状态估计时间范围: {joint_pos_timestamps[0]:.3f} - {joint_pos_timestamps[-1]:.3f}")
print(f"SLAM位姿时间范围: {root_pose_timestamps[0]:.3f} - {root_pose_timestamps[-1]:.3f}")

# 以root_pose_timestamps为基准，将所有数据插值对齐
N_aligned = len(root_pose_timestamps)

# 插值joint_pos到root_pose_timestamps
joint_pos_interp = np.zeros((N_aligned, joint_pos.shape[1]))
for i in range(joint_pos.shape[1]):
    joint_pos_interp[:, i] = np.interp(root_pose_timestamps, joint_pos_timestamps, joint_pos[:, i])

# 插值joint_vel到root_pose_timestamps
joint_vel_interp = np.zeros((N_aligned, joint_vel.shape[1]))
for i in range(joint_vel.shape[1]):
    joint_vel_interp[:, i] = np.interp(root_pose_timestamps, joint_pos_timestamps, joint_vel[:, i])

# 插值twist_radar到root_pose_timestamps
twist_radar_interp = np.zeros((N_aligned, 6))
for i in range(6):
    twist_radar_interp[:, i] = np.interp(root_pose_timestamps, twist_radar_timestamps, twist_radar[:, i])

print(f"对齐后数据形状:")
print(f"  joint_pos_interp: {joint_pos_interp.shape}")
print(f"  joint_vel_interp: {joint_vel_interp.shape}")
print(f"  twist_radar_interp: {twist_radar_interp.shape}")
print(f"  pos_xyz: {pos_xyz.shape}")
print(f"  euler: {euler.shape}")

# 2. T_base_from_radar
T_base_from_radar = np.eye(4)
T_base_from_radar[:3, :3] = np.array([
    [0.9737,  0,  0.2277],
    [0.,     -1,   0],
    [-0.2277, 0,  -0.9737]
])
T_base_from_radar[:3, 3] = np.array([0.00328008, 0.0, -0.6890155])

N = N_aligned  # 使用对齐后的数据长度
v_body = np.zeros((N, 3))
w_body = np.zeros((N, 3))


# 4. 变换速度
for i in range(N):
    # 求T_base_from_radar的逆矩阵
    R = T_base_from_radar[:3, :3]
    t = T_base_from_radar[:3, 3]
    T_radar_from_base = np.eye(4)
    T_radar_from_base[:3, :3] = R.T
    T_radar_from_base[:3, 3] = -R.T @ t
    R = T_radar_from_base[:3, :3]
    p = T_radar_from_base[:3, 3]
    # 反对称矩阵
    def hat(vec):
        return np.array([
            [0, -vec[2], vec[1]],
            [vec[2], 0, -vec[0]],
            [-vec[1], vec[0], 0]
        ])
    # 构建Ad矩阵
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = R.T
    Ad[3:, 3:] = R.T
    Ad[:3, 3:] = -R.T @ hat(p)
    # twist_radar_interp[i] 是在radar系下的速度，变换到base系
    twist_base = np.linalg.inv(Ad) @ twist_radar_interp[i]
    v_base = twist_base[:3]
    w_base = twist_base[3:]
    # 这里计算的是rootlink的body vel
    v_body[i] = v_base
    w_body[i] = w_base

# 对速度数据进行滤波处理
window_length = 41  # 增大滤波窗口长度，必须是奇数
polyorder = 2       # 降低多项式阶数，增加平滑效果

# 对v_body进行滤波
v_body_filtered = np.zeros_like(v_body)
w_body_filtered = np.zeros_like(w_body)
for i in range(3):
    v_body_filtered[:, i] = savgol_filter(v_body[:, i], window_length, polyorder)
    w_body_filtered[:, i] = savgol_filter(w_body[:, i], window_length, polyorder)

# 对pos_xyz的z分量进行高通滤波，滤除低频部分
# 高通滤波参数
cutoff_freq = 0.5  # 提高截止频率，更狠的滤波
sampling_freq = 1.0 / np.mean(np.diff(root_pose_timestamps))  # 采样频率
nyquist_freq = sampling_freq / 2.0
normalized_cutoff = cutoff_freq / nyquist_freq
# 设计高通滤波器
b, a = butter(4, normalized_cutoff, btype='high', analog=False)
# 对z位置进行高通滤波，保持基准高度
z_baseline = 0.84  # 基准高度
z_relative = pos_xyz[:, 2] - z_baseline  # 相对高度
z_relative_filtered = filtfilt(b, a, z_relative)  # 高通滤波
pos_z_filtered = z_relative_filtered + z_baseline  # 恢复基准高度

# 对y和z位置进行低通滤波，滤除尖锐部分
# 低通滤波参数
lowpass_cutoff = 6  # 提高截止频率，不那么狠的低通滤波
normalized_lowpass_cutoff = lowpass_cutoff / nyquist_freq
# 设计低通滤波器
b_low, a_low = butter(3, normalized_lowpass_cutoff, btype='low', analog=False)
# 对y位置进行低通滤波
pos_y_filtered = filtfilt(b_low, a_low, pos_xyz[:, 1])
# 对z位置进行低通滤波（在已高通滤波的基础上）
pos_z_lowpass_filtered = filtfilt(b_low, a_low, pos_z_filtered)

# 组合滤波后的位置数据
pos_xyz_filtered = np.zeros_like(pos_xyz)
pos_xyz_filtered[:, 0] = pos_xyz[:, 0]  # x位置保持原始数据
pos_xyz_filtered[:, 1] = pos_y_filtered  # y位置使用低通滤波结果
pos_xyz_filtered[:, 2] = pos_z_lowpass_filtered  # z位置使用高通+低通滤波结果

# 保存对齐后的数据
# 创建输出目录
output_dir = '/home/zl/Downloads/kuavo-rl-train-run/RL_train/output_data/predeal'
os.makedirs(output_dir, exist_ok=True)

# 保存数据到指定目录
output_path = os.path.join(output_dir, 'preprocessed_slam_data.npz')
np.savez(output_path,
         timestamps=root_pose_timestamps,
         joint_pos=joint_pos_interp,
         joint_vel=joint_vel_interp,
         v_body=v_body_filtered,
         w_body=w_body_filtered,
         pos_xyz=pos_xyz_filtered,
         euler=euler)

print(f"已保存对齐后的数据到 {output_path}")

# 只画前20000帧
draw_len = min(40000, N)

# 画线速度对比
fig1, axs1 = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
vel_labels = ['vx', 'vy', 'vz']
for i in range(3):
    axs1[i].plot(v_body[:draw_len, i], label='Original', zorder=1, linewidth=1, alpha=0.7)
    axs1[i].plot(v_body_filtered[:draw_len, i], label='Filtered', zorder=2, linewidth=2)
    axs1[i].set_ylabel(vel_labels[i] + ' (m/s)')
    axs1[i].grid(True, alpha=0.3)
    axs1[i].legend()
    axs1[i].set_title(f'{vel_labels[i]} Linear Velocity Comparison')
axs1[2].set_xlabel('Frame')
fig1.suptitle('Base Linear Velocity in Map Frame (Original vs Filtered)')
plt.tight_layout(rect=[0, 0, 1, 0.96])

# 画角速度对比
fig2, axs2 = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
ang_labels = ['wx', 'wy', 'wz']
for i in range(3):
    axs2[i].plot(w_body[:draw_len, i], label='Original', zorder=1, linewidth=1, alpha=0.7)
    axs2[i].plot(w_body_filtered[:draw_len, i], label='Filtered', zorder=2, linewidth=2)
    axs2[i].set_ylabel(ang_labels[i] + ' (rad/s)')
    axs2[i].grid(True, alpha=0.3)
    axs2[i].legend()
    axs2[i].set_title(f'{ang_labels[i]} Angular Velocity Comparison')
axs2[2].set_xlabel('Frame')
fig2.suptitle('Base Angular Velocity in Map Frame (Original vs Filtered)')
plt.tight_layout(rect=[0, 0, 1, 0.96])

# 画位置数据
fig4, axs4 = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
pos_labels = ['x', 'y', 'z']
for i in range(3):
    if i == 1:  # y位置显示滤波前后对比
        axs4[i].plot(pos_xyz[:draw_len, i], label='Original', zorder=1, linewidth=1, alpha=0.7)
        axs4[i].plot(pos_y_filtered[:draw_len], label='Low-pass Filtered', zorder=2, linewidth=2)
    elif i == 2:  # z位置显示多重滤波对比
        axs4[i].plot(pos_xyz[:draw_len, i], label='Original', zorder=1, linewidth=1, alpha=0.5)
        axs4[i].plot(pos_z_filtered[:draw_len], label='High-pass Filtered', zorder=2, linewidth=1, alpha=0.7)
        axs4[i].plot(pos_z_lowpass_filtered[:draw_len], label='High+Low-pass Filtered', zorder=3, linewidth=2)
    else:  # x位置只显示原始数据
        axs4[i].plot(pos_xyz[:draw_len, i], label='Position', zorder=2, linewidth=2)
    axs4[i].set_ylabel(pos_labels[i] + ' (m)')
    axs4[i].grid(True, alpha=0.3)
    axs4[i].legend()
    axs4[i].set_title(f'{pos_labels[i]} Position in Map Frame')
axs4[2].set_xlabel('Frame')
fig4.suptitle('Base Position in Map Frame (y: Low-pass, z: High+Low-pass)')
plt.tight_layout(rect=[0, 0, 1, 0.96])

# 画欧拉角数据
fig5, axs5 = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
euler_labels = ['roll', 'pitch', 'yaw']
for i in range(3):
    # 转换为角度显示
    euler_deg = np.rad2deg(euler[:draw_len, i])
    axs5[i].plot(euler_deg, label='Euler Angles', zorder=2, linewidth=2)
    axs5[i].set_ylabel(euler_labels[i] + ' (deg)')
    axs5[i].grid(True, alpha=0.3)
    axs5[i].legend()
    axs5[i].set_title(f'{euler_labels[i]} Euler Angle')
axs5[2].set_xlabel('Frame')
fig5.suptitle('Base Euler Angles in Map Frame')
plt.tight_layout(rect=[0, 0, 1, 0.96])

# 画关节位置数据
fig6, axs6 = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
joint_names = ['Hip_pitch', 'knee_pitch', 'Ankle_roll', 'Ankle_pitch']
for i in range(4):
    axs6[i].plot(joint_pos_interp[:draw_len, i+2], label='Joint Position', linewidth=2)
    axs6[i].set_ylabel('Position (rad)')
    axs6[i].set_title(joint_names[i])
    axs6[i].grid(True, alpha=0.3)
    axs6[i].legend()
# 设置x轴标签
axs6[3].set_xlabel('Frame')
fig6.suptitle('Joint Positions (Aligned Data)')
plt.tight_layout(rect=[0, 0, 1, 0.96])

# 画关节速度数据
fig7, axs7 = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
for i in range(4):
    axs7[i].plot(joint_vel_interp[:draw_len, i+2], label='Joint Velocity', linewidth=2)
    axs7[i].set_ylabel('Velocity (rad/s)')
    axs7[i].set_title(joint_names[i])
    axs7[i].grid(True, alpha=0.3)
    axs7[i].legend()
# 设置x轴标签
axs7[3].set_xlabel('Frame')
fig7.suptitle('Joint Velocities (Aligned Data)')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()





