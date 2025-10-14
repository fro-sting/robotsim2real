import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import glob
import os

# 1. 读取所有 real 数据，统计周期
real_dir = "/home/zl/Downloads/kuavo-rl-train-run/RL_train/output_data/partitioned_aligned_npz/slam_new"
real_files = sorted(glob.glob(os.path.join(real_dir, "partitioned_aligned_*.npz")))
dt = 0.01

def calc_period(data, dt):
    peaks, _ = find_peaks(data, distance=60)
    if len(peaks) < 2:
        return None
    intervals = np.diff(peaks)
    mean_interval = np.mean(intervals)
    period = mean_interval * dt
    return period, peaks

# 统计所有real数据的周期
periods = []
all_real_data = []
for f in real_files:
    real_data = np.load(f)["joint_pos"][:, 3]  # 以第4个关节为例
    all_real_data.append(real_data)
    period, peaks = calc_period(real_data, dt)
    if period is not None:
        periods.append(period)
        print(f"{os.path.basename(f)} 周期: {period:.4f}s, 峰数: {len(peaks)}")
    else:
        print(f"{os.path.basename(f)} 未检测到足够峰值，跳过")
if not periods:
    raise RuntimeError("没有检测到任何周期，检查数据！")
mean_period = np.mean(periods)
# mean_period = 1.4
print(f"所有real数据平均周期: {mean_period:.4f} 秒")

# 2. 用平均周期做相位映射
cycle_time_ref = 0.6  # ref数据周期直接指定
ref_period_frames = int(round(cycle_time_ref / dt))

compare_keys = [
    ("dof_pos", "joint_pos", "Joint Position"),
    ("dof_vel", "joint_vel", "Joint Velocity"),
    ("root_pos", "pos_xyz", "Root Position"),
    ("root_eu_ang", "euler", "Root Euler Angle"),
    ("root_ang_vel", "w_body", "Root Angular Velocity"),
    ("root_lin_vel", "v_body", "Root Linear Velocity"),
]

partition_indices_to_plot = [1, 3, 5, 7]

for ref_key, real_key, title_en in compare_keys:
    ref_data = np.load("/home/zl/Downloads/kuavo-rl-train-run/RL_train/humanoid/mpc_pose/play_cmu_0401_b.npz")[ref_key]
    all_real_data_full = [np.load(f)[real_key] for f in real_files]

    # 针对root_eu_ang特殊处理，只取ref的第6~8列
    if ref_key == "root_eu_ang":
        ref_data = ref_data[:, 4:7]
    # Only compare first min(columns, 6) dims to avoid too many plots
    max_dim = min(ref_data.shape[1] if ref_data.ndim > 1 else 1, 6)
    # 统一长度
    max_plot_len = 1000
    # 取所有real数据的每个col的最小长度
    real_cols_all = []
    for col in range(max_dim):
        real_cols = [d[:, col] if d.ndim > 1 else d[:] for d in all_real_data_full]
        if partition_indices_to_plot is None:
            all_real_data = real_cols
        else:
            all_real_data = [real_cols[i] for i in partition_indices_to_plot]
        real_cols_all.append(all_real_data)
    # 取所有real数据和ref的最小长度
    min_len = min([len(d) for col_data in real_cols_all for d in col_data] + [ref_data.shape[0], max_plot_len])
    sim_indices = np.arange(min_len)
    ref_period_frames = int(round(cycle_time_ref / dt))
    ref_indices_env = np.round((sim_indices * dt / mean_period) * ref_period_frames).astype(int)
    ref_indices_env = np.clip(ref_indices_env, 0, min_len - 1)

    # 画原始对齐
    fig, axes = plt.subplots(max_dim, 1, figsize=(14, 4 * max_dim), sharex=True)
    if max_dim == 1:
        axes = [axes]
    for col in range(max_dim):
        ax = axes[col]
        ref_col = ref_data[:, col] if ref_data.ndim > 1 else ref_data[:]
        ref_col = ref_col[:min_len]
        all_real_data = real_cols_all[col]
        all_real_data = [d[:min_len] for d in all_real_data]
        real_indices = partition_indices_to_plot if partition_indices_to_plot is not None else range(len(all_real_data))
        ax.plot(sim_indices, ref_col, label="ref_data", linewidth=2.5)
        for i, real_data in zip(real_indices, all_real_data):
            ax.plot(sim_indices, real_data, label=f"real_data_{i+1}", alpha=0.7)
        ax.set_ylabel("Value", fontsize=14, labelpad=8)
        # ax.set_title(f'{title_en} (original alignment) - dim {col}', fontsize=15, pad=8)
        ax.grid(True, linewidth=1.2, alpha=0.5)
        ax.legend(fontsize=12)
        ax.tick_params(axis="both", which="major", labelsize=12, width=2, length=5)
    plt.xlabel("Step", fontsize=14, labelpad=8)
    plt.suptitle(f"{title_en} All real_data vs. ref_data (original alignment)", fontsize=18, y=1.02)
    plt.tight_layout()
    plt.show()

    # 画phase-mapped对齐
    fig, axes = plt.subplots(max_dim, 1, figsize=(14, 4 * max_dim), sharex=True)
    if max_dim == 1:
        axes = [axes]
    for col in range(max_dim):
        ax = axes[col]
        ref_col = ref_data[:, col] if ref_data.ndim > 1 else ref_data[:]
        ref_col = ref_col[:min_len]
        all_real_data = real_cols_all[col]
        all_real_data = [d[:min_len] for d in all_real_data]
        real_indices = partition_indices_to_plot if partition_indices_to_plot is not None else range(len(all_real_data))
        ax.plot(sim_indices, ref_col[ref_indices_env], label="ref_data (phase-mapped)", linewidth=2.5)
        for i, real_data in zip(real_indices, all_real_data):
            ax.plot(sim_indices, real_data, label=f"real_data_{i+1}", alpha=0.7, linewidth=2.5)
        ax.set_ylabel("Value", fontsize=14, labelpad=8)
        # ax.set_title(f'{title_en} (phase-mapped, avg period) - dim {col}', fontsize=15, pad=8)
        ax.grid(True, linewidth=1.2, alpha=0.5)
        ax.legend(fontsize=12)
        ax.tick_params(axis="both", which="major", labelsize=12, width=2, length=5)
    plt.xlabel("Step", fontsize=14, labelpad=8)
    plt.suptitle(f"{title_en} All real_data vs. ref_data (phase-mapped, avg period)", fontsize=18, y=1.02)
    plt.tight_layout()
    plt.show()

