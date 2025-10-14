import numpy as np
import matplotlib.pyplot as plt

import os
import glob
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter

# 导入完整npz文件
ref_npz = np.load(
    "/home/zl/Downloads/kuavo-rl-train-run/RL_train/humanoid/mpc_pose/play_cmu_0401_b.npz", allow_pickle=True
)
# ================== 初步对比 ==================
slam_data = np.load(
    "/home/zl/Downloads/kuavo-rl-train-run/RL_train/output_data/predeal/preprocessed_slam_data.npz",
    allow_pickle=True,
)

ref_data_dof_pos = ref_npz["dof_pos"]
ref_data_dof_vel = ref_npz["dof_vel"]
ref_data_foot_force = ref_npz["foot_force"]
ref_data_foot_height = ref_npz["foot_height"]
ref_data_foot_zvel = ref_npz["foot_zvel"]
ref_data_root_pos = ref_npz["root_pos"]
ref_data_root_eu_ang = ref_npz["root_eu_ang"]
ref_data_root_ang_vel = ref_npz["root_ang_vel"]
ref_data_root_lin_vel = ref_npz["root_lin_vel"]

slam_data_joint_pos = slam_data["joint_pos"]
slam_data_joint_vel = slam_data["joint_vel"]
slam_data_root_pos = slam_data["pos_xyz"]
slam_data_root_eu_ang = slam_data["euler"]
slam_data_root_eu_ang = slam_data_root_eu_ang.copy()
slam_data_root_eu_ang[:, 1] -= 0.42  # y方向欧拉角减去0.42
slam_data_root_lin_vel = slam_data["v_body"]
slam_data_root_ang_vel = slam_data["w_body"]

# 1. 打印shape
print("slam_data_root_pos shape:", slam_data_root_pos.shape)
print("ref_data_root_pos shape:", ref_data_root_pos.shape)

slam_eu_xyz = slam_data_root_eu_ang[:, :3]
ref_eu_xyz = ref_data_root_eu_ang[:, 4:7]

# 3. 画root_pos对比
fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
labels = ["x", "y", "z"]
for i in range(3):
    ax = axes[i]
    if slam_data_root_pos.shape[1] > i:
        ax.plot(slam_data_root_pos[:, i], label="slam pos_xyz " + labels[i], linewidth=1.5)
    if ref_data_root_pos.shape[1] > i:
        ax.plot(ref_data_root_pos[:, i], label="ref root_pos " + labels[i], linewidth=1.5)
    ax.set_ylabel(labels[i], fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.5)
axes[-1].set_xlabel("Frame", fontsize=12)
fig.suptitle("root_pos: real vs slam vs ref", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# 4. 画欧拉角对比（XYZ）
fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
labels = ["roll (x)", "pitch (y)", "yaw (z)"]
for i in range(3):
    ax = axes[i]
    if slam_eu_xyz.shape[1] > i:
        ax.plot(slam_eu_xyz[:, i], label="slam angle_eu_ang " + labels[i], linewidth=1.5)
    if ref_eu_xyz.shape[1] > i:
        ax.plot(ref_eu_xyz[:, i], label="ref root_eu_ang col4-6 " + labels[i], linewidth=1.5)
    ax.set_ylabel(labels[i] + " (rad)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.5)
fig.suptitle("root_eu_ang (XYZ): real vs slam vs ref", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# 关节角度对比
joint_indices = [2, 3, 4, 5]
fig, axes = plt.subplots(4, 1, figsize=(20, 24))
axes = axes.flatten()
# 定义要画的帧范围
start_frame = 0
end_frame = 31000
# 画图
for idx, joint in enumerate(joint_indices):
    ax = axes[idx]
    ax.plot(slam_data_joint_pos[start_frame:end_frame, joint], label=f"real_data jointpos[:,{joint}]", linewidth=1.5)
    ax.plot(ref_data_dof_pos[start_frame:end_frame, joint], label=f"ref_data dof_pos[:,{joint}]", linewidth=1.5)
    ax.set_ylabel("Joint Value", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, linewidth=1.0, alpha=0.5)
    ax.tick_params(axis="both", which="major", labelsize=10, width=1.5, length=4)
    # 修改x轴刻度，显示5500-6150帧范围
    xticks = np.arange(start_frame, end_frame, 1000)
    ax.set_xticks(xticks - start_frame)  # 减去start_frame因为plot时已经切片了
    ax.set_xticklabels(xticks, rotation=30)
plt.subplots_adjust(hspace=0.4, bottom=0.08)
plt.show()

# ================== 分割 ==================
do_partition = False  # 只在为True时才执行分割

if do_partition:
    # 分割输入文件和slam文件
    slam_path = "/home/zl/Downloads/kuavo-rl-train-run/RL_train/output_data/predeal/preprocessed_slam_data.npz"
    save_dir_slam = "/home/zl/Downloads/kuavo-rl-train-run/RL_train/output_data/partitioned_npz/slam_new"
    
    os.makedirs(save_dir_slam, exist_ok=True)
    data_slam = np.load(slam_path, allow_pickle=True)
    keys_slam = list(data_slam.keys())

    # 统一分割区间
    split_ranges = [
        (16500, 19900),
        (24700, 27800)
    ]

    # 分割slam数据
    for idx, (start, end) in enumerate(split_ranges):
        part_dict_slam = {}
        for key in keys_slam:
            arr = data_slam[key]
            if arr.ndim == 2:
                part_arr = arr[start:end, :]
            else:
                part_arr = arr[start:end]
            part_dict_slam[key] = part_arr
        save_path_slam = os.path.join(save_dir_slam, f"partition_{idx+1}.npz")
        np.savez(save_path_slam, **part_dict_slam)
        print(f"Saved slam: {save_path_slam}, shape: {part_dict_slam[keys_slam[0]].shape}")
    print("Custom partition completed!")

# ================== Initial Frame Alignment ==================

do_align_plot = True # Only execute alignment visualization when True
# Partition slam data numbering
partition_indices_to_plot = [0, 1, 2, 3]  # Set to None to plot all, or like [0,2,4] to plot only specified partitions
partition_discard_starts = [287, 72, 117, 149]
# partition_indices_to_plot = [0, 1, 2, 3]  # Set to None to plot all, or like [0,2,4] to plot only specified partitions
# partition_discard_starts = [277, 60, 107, 139]

if do_align_plot:
    partition_slam_dir = "/home/zl/Downloads/kuavo-rl-train-run/RL_train/output_data/partitioned_npz/slam_new"
    partition_slam_files = sorted(glob.glob(os.path.join(partition_slam_dir, "partition_*.npz")))
    partition_slam_names = [os.path.basename(f) for f in partition_slam_files]

    # First load all partition data joint_pos
    all_joint_pos = []
    for file in partition_slam_files:
        data = np.load(file, allow_pickle=True)
        if "joint_pos" in data:
            all_joint_pos.append(data["joint_pos"])  # shape: (N_i, 12)

    if not all_joint_pos:
        print("Warning: No partition files containing joint_pos found, cannot plot!")
    else:
        # New: Save remaining part of each partition data to partitioned_aligned_npz
        aligned_save_dir_slam = (
            "/home/zl/Downloads/kuavo-rl-train-run/RL_train/output_data/partitioned_aligned_npz/slam_new"
        )
        os.makedirs(aligned_save_dir_slam, exist_ok=True)
        # Slam data alignment
        for i in range(len(partition_slam_files)):
            # Base discard for this partition
            discard_base = partition_discard_starts[i] if i < len(partition_discard_starts) else 0
            # Additional 10-frame discard for non-joint data
            discard_extra = discard_base + 10
            data_slam = np.load(partition_slam_files[i], allow_pickle=True)
            aligned_dict_slam = {}
            for key in data_slam.keys():
                arr = data_slam[key]
                # For joint_pos and joint_vel, use base discard; others use base+10
                if key in ["joint_pos", "joint_vel"]:
                    d_n = discard_base
                else:
                    d_n = discard_extra
                if isinstance(arr, np.ndarray) and arr.shape[0] > d_n:
                    aligned_dict_slam[key] = arr[d_n:]
                else:
                    aligned_dict_slam[key] = arr
            
            # aligned_dict_slam["euler"] = aligned_dict_slam["euler"].copy()
            # # Savitzky-Golay滤波参数
            # window_length = 17  # 必须为奇数，可根据数据长度调整
            # polyorder = 3
            # for col in range(aligned_dict_slam["euler"].shape[1]):
            #     aligned_dict_slam["euler"][:, col] = savgol_filter(
            #     aligned_dict_slam["euler"][:, col], window_length, polyorder, mode="interp"
            #     )
            # Subtract 0.42 from euler y value before saving
            if i >= 2:
                aligned_dict_slam["euler"] = aligned_dict_slam["euler"].copy()
                aligned_dict_slam["euler"][:, 1] -= 0.42  # Subtract 0.42 from y (pitch) component
            
            aligned_save_path_slam = os.path.join(aligned_save_dir_slam, f"partitioned_aligned_slam_{i+1}.npz")
            np.savez(aligned_save_path_slam, **aligned_dict_slam)
            print(
                f"Saved aligned slam partition file: {aligned_save_path_slam}, shape:"
                f" {list(aligned_dict_slam.values())[0].shape}"
            )

        plot_len = 2500  # Customizable, only plot first N frames
        fig, axes = plt.subplots(4, 1, figsize=(20, 24))
        axes = axes.flatten()
        for idx, joint in enumerate([2, 3, 4, 5]):
            ax = axes[idx]
            ax.plot(ref_data_dof_pos[:plot_len, joint], label=f"ref_data dof_pos[:,{joint}]", linewidth=1.5)
            # Select partition curves to plot
            if partition_indices_to_plot is None:
                indices = range(len(all_joint_pos))
            else:
                indices = partition_indices_to_plot
            for d_idx, i in enumerate(indices):
                if i < len(all_joint_pos):
                    discard_n = partition_discard_starts[d_idx] if d_idx < len(partition_discard_starts) else 0
                    ax.plot(
                        all_joint_pos[i][discard_n : discard_n + plot_len, joint],
                        label=f"{partition_slam_names[i]} joint_pos[:,{joint}] (discard {discard_n})",
                        linewidth=1.5,
                    )
            # Calculate max_len considering discard
            max_len = min(
                plot_len,
                max(
                    all_joint_pos[i][
                        partition_discard_starts[d_idx] if d_idx < len(partition_discard_starts) else 0 :
                    ].shape[0]
                    for d_idx, i in enumerate(indices)
                    if i < len(all_joint_pos)
                ),
            )
            ax.legend(fontsize=12)
            ax.grid(True, linewidth=1.0, alpha=0.5)
            ax.tick_params(axis="both", which="major", labelsize=16, width=2, length=6)
            if idx != len(axes) - 1:
                ax.set_xticklabels([])
            else:
                xticks = np.arange(0, max_len, 50)
                ax.set_xticks(xticks)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.subplots_adjust(bottom=0.15)
        plt.show()

    # Partition slam data
    all_pos_xyz_slam = []
    all_euler_slam = []
    for file in partition_slam_files:
        data = np.load(file, allow_pickle=True)
        if "pos_xyz" in data:
            all_pos_xyz_slam.append(data["pos_xyz"])  # shape: (N_i, 3)
        if "euler" in data:
            all_euler_slam.append(data["euler"])  # shape: (N_i, 3)

    if not all_pos_xyz_slam:
        print("Warning: No partition files containing pos_xyz found, cannot plot!")
    else:
        fig, axes = plt.subplots(3, 1, figsize=(20, 18))
        axes = axes.flatten()
        if partition_indices_to_plot is None:
            indices_slam = range(len(all_pos_xyz_slam))
        else:
            indices_slam = partition_indices_to_plot
        plot_len = 1000  # Customizable
        for idx, dim in enumerate(["x", "y", "z"]):
            ax = axes[idx]
            ax.plot(ref_data_root_pos[:plot_len, idx], label=f"ref root_pos {dim}", linewidth=1.5)
            for d_idx, i in enumerate(indices_slam):
                if i < len(all_pos_xyz_slam):
                    discard_n = partition_discard_starts[d_idx] if d_idx < len(partition_discard_starts) else 0
                    ax.plot(
                        all_pos_xyz_slam[i][discard_n : discard_n + plot_len, idx],
                        label=f"partition_slam_{i+1} pos_xyz[:,{idx}] (discard {discard_n})",
                        linewidth=1.5,
                    )
            max_len = min(
                plot_len,
                max(
                    all_pos_xyz_slam[i][
                        partition_discard_starts[d_idx] if d_idx < len(partition_discard_starts) else 0 :
                    ].shape[0]
                    for d_idx, i in enumerate(indices_slam)
                    if i < len(all_pos_xyz_slam)
                ),
            )
            ax.legend(fontsize=12)
            ax.grid(True, linewidth=1.0, alpha=0.5)
            ax.tick_params(axis="both", which="major", labelsize=16, width=2, length=6)
            if idx != len(axes) - 1:
                ax.set_xticklabels([])
            else:
                xticks = np.arange(0, max_len, 50)
                ax.set_xticks(xticks)

    # Plot euler vs ref root_eu_ang comparison
    if not all_euler_slam:
        print("Warning: No partition files containing euler found, cannot plot!")
    else:
        fig, axes = plt.subplots(3, 1, figsize=(20, 18))
        axes = axes.flatten()
        if partition_indices_to_plot is None:
            indices_slam = range(len(all_euler_slam))
        else:
            indices_slam = partition_indices_to_plot
        plot_len = 3000  # Customizable
        for idx, label in enumerate(["roll", "pitch", "yaw"]):
            ax = axes[idx]
            ax.plot(
                ref_data_root_eu_ang[:plot_len, 4 + idx], label=f"ref root_eu_ang col{4+idx} ({label})", linewidth=1.5
            )
            for d_idx, i in enumerate(indices_slam):
                if i < len(all_euler_slam):
                    discard_n = partition_discard_starts[d_idx] if d_idx < len(partition_discard_starts) else 0
                    ax.plot(
                        all_euler_slam[i][discard_n : discard_n + plot_len, idx],
                        label=f"partition_slam_{i+1} euler[:,{idx}] (discard {discard_n})",
                        linewidth=1.5,
                    )
            max_len = min(
                plot_len,
                max(
                    all_euler_slam[i][
                        partition_discard_starts[d_idx] if d_idx < len(partition_discard_starts) else 0 :
                    ].shape[0]
                    for d_idx, i in enumerate(indices_slam)
                    if i < len(all_euler_slam)
                ),
            )
            ax.legend(fontsize=12)
            ax.grid(True, linewidth=1.0, alpha=0.5)
            ax.tick_params(axis="both", which="major", labelsize=16, width=2, length=6)
            if idx != len(axes) - 1:
                ax.set_xticklabels([])
            else:
                xticks = np.arange(0, max_len, 100)
                ax.set_xticks(xticks)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.subplots_adjust(bottom=0.15)
        plt.suptitle("Aligned slam partitioned pos_xyz vs ref root_pos comparison", fontsize=18)
        plt.show()

# Comparison after alignment
compare_aligned = True  # Set to True to execute comparison between aligned data and ref
compare_plot_len = 2000  # Customizable comparison frame count
compare_partition_indices_to_plot = [0, 1, 2, 3]  # Set to None to compare all, or like [0,2,4] to compare only specified partitions

if compare_aligned:
    # Slam aligned data comparison
    aligned_dir_slam = "/home/zl/Downloads/kuavo-rl-train-run/RL_train/output_data/partitioned_aligned_npz/slam_new"
    aligned_files_slam = sorted(glob.glob(os.path.join(aligned_dir_slam, "partitioned_aligned_slam_*.npz")))
    aligned_names_slam = [os.path.basename(f) for f in aligned_files_slam]
    aligned_data_list_slam = []
    for f in aligned_files_slam:
        npz = np.load(f, allow_pickle=True)
        data = {}
        for k in npz.keys():
            arr = npz[k]
            data[k] = arr
        aligned_data_list_slam.append(data)
    # Only select partitions to compare
    if compare_partition_indices_to_plot is None:
        indices = range(len(aligned_data_list_slam))
    else:
        indices = compare_partition_indices_to_plot
    # Define comparison keys and ref keys
    compare_keys_slam = [
        ("joint_pos", "dof_pos"),
        ("joint_vel", "dof_vel"),
        ("pos_xyz", "root_pos"),
        ("euler", "root_eu_ang"),
        ("v_body", "root_lin_vel"),
        ("w_body", "root_ang_vel"),
    ]
    for slam_key, ref_key in compare_keys_slam:
        ref_arr = ref_npz[ref_key]
        # Only take selected partitions
        slam_arrs = [aligned_data_list_slam[i][slam_key] for i in indices if slam_key in aligned_data_list_slam[i]]
        slam_names = [aligned_names_slam[i] for i in indices if slam_key in aligned_data_list_slam[i]]
        min_len = min([arr.shape[0] for arr in slam_arrs] + [ref_arr.shape[0]]) if slam_arrs else 0
        plot_len = min(compare_plot_len, min_len)
        # 只画joint_pos和joint_vel的2,3,4,5列
        if slam_key in ["joint_pos", "joint_vel"] and ref_arr.ndim > 1 and ref_arr.shape[1] >= 6:
            cols = [2, 3, 4, 5]
        elif slam_key == "pos_xyz" and ref_arr.ndim > 1 and ref_arr.shape[1] >= 3:
            cols = [0, 1, 2]
        elif slam_key == "euler" and ref_arr.ndim > 1 and ref_arr.shape[1] >= 7:
            cols = [4, 5, 6]
        else:
            cols = list(range(ref_arr.shape[1])) if ref_arr.ndim > 1 else [0]
        fig, axes = plt.subplots(len(cols), 1, figsize=(20, 5 * len(cols)))
        if len(cols) == 1:
            axes = [axes]
        for idx, j in enumerate(cols):
            ax = axes[idx]
            # Plot ref
            if ref_arr.ndim > 1:
                ax.plot(ref_arr[:plot_len, j], label=f"ref {ref_key}[:,{j}]", color="black", linewidth=1.5)
            else:
                ax.plot(ref_arr[:plot_len], label=f"ref {ref_key}", color="black", linewidth=1.5)
            # Plot slam
            for name, arr in zip(slam_names, slam_arrs):
                if arr.ndim > 1 and (slam_key != "euler" or j - 4 < arr.shape[1]):
                    arr_col = j - 4 if (slam_key == "euler" and j >= 4) else j
                    if arr_col >= 0 and arr_col < arr.shape[1]:
                        ax.plot(arr[:plot_len, arr_col], label=f"{name} {slam_key}[:,{arr_col}]", linewidth=1.5)
                elif arr.ndim == 1 and j == 0:
                    ax.plot(arr[:plot_len], label=f"{name} {slam_key}", linewidth=1.5)
            ax.legend(fontsize=10)
            ax.grid(True, linewidth=1.0, alpha=0.5)
            ax.tick_params(axis="both", which="major", labelsize=12, width=1.5, length=4)
            ax.set_xticks(np.arange(0, plot_len, 50))
        plt.tight_layout()
        plt.show()