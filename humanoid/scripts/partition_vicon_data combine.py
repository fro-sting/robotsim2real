import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]  # 指定中文字体
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号
import os
import glob
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter

def smooth_data(data, window_length=11, polyorder=3):
    """
    对数据进行平滑化处理
    使用Savitzky-Golay滤波器进行平滑化
    
    Args:
        data: 输入数据（1维数组）
        window_length: 窗口长度（必须是奇数）
        polyorder: 多项式阶数
    
    Returns:
        smoothed_data: 平滑化后的数据
    """
    if len(data) < window_length:
        # 如果数据长度小于窗口长度，使用较小的窗口
        window_length = len(data) if len(data) % 2 == 1 else len(data) - 1
        if window_length < 3:
            return data  # 数据太短，直接返回原数据
    
    try:
        return savgol_filter(data, window_length, polyorder)
    except:
        # 如果Savitzky-Golay滤波失败，使用简单的移动平均
        return moving_average_smooth(data)

def moving_average_smooth(data, window_size=5):
    """
    简单的移动平均平滑化
    """
    if len(data) < window_size:
        return data
    
    smoothed = np.zeros_like(data)
    half_window = window_size // 2
    
    # 处理边界
    for i in range(half_window):
        smoothed[i] = np.mean(data[:i + half_window + 1])
        smoothed[-(i+1)] = np.mean(data[-(i + half_window + 1):])
    
    # 处理中间部分
    for i in range(half_window, len(data) - half_window):
        smoothed[i] = np.mean(data[i - half_window:i + half_window + 1])
    
    return smoothed

# 导入完整npz文件
ref_npz = np.load(
    "/home/zl/Downloads/kuavo-rl-train-run/RL_train/humanoid/mpc_pose/play_cmu_0401_b.npz", allow_pickle=True
)

# ================== 初步对比 ==================
real_sampled_npz = np.load(
    "/home/zl/ASAP_data_leju/vicon/2025-07-25-11-51-51.npz", allow_pickle=True
)
real_pos_vicon = np.load(
    "/home/zl/Downloads/kuavo-rl-train-run/RL_train/output_data/vicon/processed_vicon_data.npz",
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

real_data_dof_pos = real_pos_vicon["jointpos"]
real_data_dof_vel = real_pos_vicon["jointvel"]
real_data_root_pos = real_sampled_npz["posxyz"]
real_data_root_eu_ang = real_sampled_npz["anglezyx"]
real_data_root_ang_vel = real_sampled_npz["angularvelzyx"]
real_data_root_lin_vel = real_sampled_npz["linearvel"]

vicon_data_root_pos = real_pos_vicon["root_pos"]
viocn_data_root_eu_ang = real_pos_vicon["root_euler"]
vicon_data_root_quaternion_rot = real_pos_vicon["quaternion_rot"]
vicon_data_root_vel_world = real_pos_vicon["root_vel_world"]
vicon_data_root_vel_body = real_pos_vicon["root_vel_body"]
vicon_data_root_omega_body = real_pos_vicon["root_omega_body"]
vicon_data_root_omega_world = real_pos_vicon["root_omega_world"]


# 1. 打印shape
print("real_data_root_pos shape:", real_data_root_pos.shape)
print("vicon_data_root_pos shape:", vicon_data_root_pos.shape)
print("ref_data_root_pos shape:", ref_data_root_pos.shape)

# 2. ZYX欧拉角转XYZ欧拉角
real_eu_xyz = R.from_euler("zyx", real_data_root_eu_ang).as_euler("xyz")
vicon_eu_xyz = viocn_data_root_eu_ang[:, :3]
ref_eu_xyz = ref_data_root_eu_ang[:, 4:7]

# 3. 画root_pos对比
fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
labels = ["x", "y", "z"]
for i in range(3):
    ax = axes[i]
    if real_data_root_pos.shape[1] > i:
        ax.plot(real_data_root_pos[:, i], label="real posxyz " + labels[i], linewidth=1.5)
    if vicon_data_root_pos.shape[1] > i:
        ax.plot(vicon_data_root_pos[:, i], label="vicon pos_xyz " + labels[i], linewidth=1.5)
    if ref_data_root_pos.shape[1] > i:
        ax.plot(ref_data_root_pos[:, i], label="ref root_pos " + labels[i], linewidth=1.5)
    ax.set_ylabel(labels[i], fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.5)
    ax.set_xticks(np.arange(0, real_data_dof_pos.shape[0], 2500))
    ax.set_xticklabels(np.arange(0, real_data_dof_pos.shape[0], 2500), rotation=30)
axes[-1].set_xlabel("帧", fontsize=12)
fig.suptitle("root_pos: real vs slam vs ref", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# 4. 画欧拉角对比（XYZ）
fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
labels = ["roll (x)", "pitch (y)", "yaw (z)"]
for i in range(3):
    ax = axes[i]
    if real_eu_xyz.shape[1] > i:
        ax.plot(real_eu_xyz[:, i], label="real anglezyx→xyz " + labels[i], linewidth=1.5)
    if vicon_eu_xyz.shape[1] > i:
        ax.plot(vicon_eu_xyz[:, i], label="vicon angle_eu_ang " + labels[i], linewidth=1.5)
    if ref_eu_xyz.shape[1] > i:
        ax.plot(ref_eu_xyz[:, i], label="ref root_eu_ang col4-6 " + labels[i], linewidth=1.5)
    ax.set_ylabel(labels[i] + " (rad)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.5)
    ax.set_xticks(np.arange(0, real_data_dof_pos.shape[0], 2500))
    ax.set_xticklabels(np.arange(0, real_data_dof_pos.shape[0], 2500), rotation=30)
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
    ax.plot(real_data_dof_pos[start_frame:end_frame, joint], label=f"real_data jointpos[:,{joint}]", linewidth=1.5)
    # ax.plot(ref_data_dof_pos[start_frame:end_frame, joint], label=f"ref_data dof_pos[:,{joint}]", linewidth=1.5)
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
do_partition = True  # 只在为True时才执行分割

if do_partition:
    # 分割vicon文件
    vicon_path = "/home/zl/Downloads/kuavo-rl-train-run/RL_train/output_data/vicon/processed_vicon_data.npz"
    save_dir_vicon = "/home/zl/Downloads/kuavo-rl-train-run/RL_train/output_data/partitioned_npz/vicon"
    os.makedirs(save_dir_vicon, exist_ok=True)

    data_vicon = np.load(vicon_path, allow_pickle=True)
    keys_slam = list(data_vicon.keys())

    # 统一分割区间
    split_ranges = [
        (2650, 3150),
        (5700, 6150),
        (8900, 9300),
        (12000, 12550),
        (15100, 15750),
        (18300, 18820),
        (21600, 22150),
        (24700, 25180),
        (27800, 28500),
    ]

    # 分割slam数据
    for idx, (start, end) in enumerate(split_ranges):
        part_dict_slam = {}
        for key in keys_slam:
            arr = data_vicon[key]
            if arr.ndim == 2:
                part_arr = arr[start:end, :]
            else:
                part_arr = arr[start:end]
            part_dict_slam[key] = part_arr
        save_path_slam = os.path.join(save_dir_vicon, f"partition_{idx+1}.npz")
        np.savez(save_path_slam, **part_dict_slam)
        print(f"Saved slam: {save_path_slam}, shape: {part_dict_slam[keys_slam[0]].shape}")

    print("自定义分割完成！")

# ================== 找到最大值并拼接 ==================

do_align_plot = True  # 只有为True时才执行对齐可视化
# 分割real和slam数据统一编号
partition_indices_to_plot = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # 设为None画全部，或如[0,2,4]只画指定分割

if do_align_plot:
    partition_vicon_dir = "/home/zl/Downloads/kuavo-rl-train-run/RL_train/output_data/partitioned_npz/vicon"
    partition_vicon_files = sorted(glob.glob(os.path.join(partition_vicon_dir, "partition_*.npz")))
    partition_vicon_names = [os.path.basename(f) for f in partition_vicon_files]

    # 先把所有分割数据的joint_pos都读进来
    all_joint_pos = []
    for file in partition_vicon_files:
        data = np.load(file, allow_pickle=True)
        if "jointpos" in data:
            all_joint_pos.append(data["jointpos"])  # shape: (N_i, 12)

    if not all_joint_pos:
        print("警告：没有找到任何包含 jointpos 的分割文件，无法绘图！")
    else:
        # 画分割的joint pos数据
        joint_indices = [2, 3, 4, 5]
        fig, axes = plt.subplots(4, 1, figsize=(20, 24))
        axes = axes.flatten()
        
        # 选择要画的分割曲线
        if partition_indices_to_plot is None:
            indices = range(len(all_joint_pos))
        else:
            indices = partition_indices_to_plot
            
        for idx, joint in enumerate(joint_indices):
            ax = axes[idx]
            
            # 画每个分割文件的数据
            for d_idx, i in enumerate(indices):
                if i < len(all_joint_pos):
                    joint_data = all_joint_pos[i][:, joint]
                    ax.plot(
                        joint_data,
                        label=f"{partition_vicon_names[i]} jointpos[:,{joint}]",
                        linewidth=1.5,
                    )
                    # 计算前200帧的最大值
                    front_200_max_idx = np.argmax(joint_data[:200])
                    front_200_max_val = joint_data[front_200_max_idx]
                    ax.plot(front_200_max_idx, front_200_max_val, 'ro', markersize=8)
                    ax.annotate(f'front_200_max\nframe={front_200_max_idx}\nval={front_200_max_val:.3f}', 
                                xy=(front_200_max_idx, front_200_max_val), xytext=(10, 10),
                                textcoords='offset points', ha='left', va='bottom',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
                    
                    # 计算后200帧的最大值
                    back_200_max_idx = np.argmax(joint_data[-150:]) + len(joint_data) - 150
                    back_200_max_val = joint_data[back_200_max_idx]
                    ax.plot(back_200_max_idx, back_200_max_val, 'mo', markersize=8)
                    ax.annotate(f'back_200_max\nframe={back_200_max_idx}\nval={back_200_max_val:.3f}', 
                                xy=(back_200_max_idx, back_200_max_val), xytext=(10, -10),
                                textcoords='offset points', ha='left', va='top',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='pink', alpha=0.7),
                                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            
            ax.set_ylabel(f"Joint {joint} Value", fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, linewidth=1.0, alpha=0.5)
            ax.tick_params(axis="both", which="major", labelsize=10, width=1.5, length=4)
            
            # 设置x轴刻度
            if idx == len(axes) - 1:  # 只在最后一个子图显示x轴标签
                ax.set_xlabel("帧", fontsize=12)
                # 计算最大长度用于x轴刻度
                max_len = max(
                    all_joint_pos[i].shape[0]
                    for i in indices
                    if i < len(all_joint_pos)
                )
                xticks = np.arange(0, max_len, 50)
                ax.set_xticks(xticks)
                ax.set_xticklabels(xticks, rotation=30)
            else:
                ax.set_xticklabels([])
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.subplots_adjust(bottom=0.15)
        plt.suptitle("分割的Vicon Joint Position数据", fontsize=16)
        plt.show()

# ================== 拼接数据 ==================
do_concatenate = True  # 只有为True时才执行拼接

if do_concatenate:
    print("开始重复拼接单个数据段...")

    # ================== 用户可配置参数 ==================
    # 要重复拼接的分割文件索引 (0表示partition_1.npz, 1表示partition_2.npz, ...)
    segment_to_repeat_idx = 0
    # 重复拼接的次数
    repeat_count = 10
    # ===============================================

    partition_vicon_dir = "/home/zl/Downloads/kuavo-rl-train-run/RL_train/output_data/partitioned_npz/vicon"
    partition_vicon_files = sorted(glob.glob(os.path.join(partition_vicon_dir, "partition_*.npz")))

    if segment_to_repeat_idx >= len(partition_vicon_files):
        raise ValueError(f"错误：指定的分割文件索引 {segment_to_repeat_idx} 超出范围 (共 {len(partition_vicon_files)} 个文件)")

    # 加载要重复的分割文件
    file_to_repeat = partition_vicon_files[segment_to_repeat_idx]
    print(f"将使用文件 '{os.path.basename(file_to_repeat)}' 进行重复拼接 {repeat_count} 次。")
    
    partition_data = np.load(file_to_repeat, allow_pickle=True)
    all_keys = list(partition_data.keys())

    # 计算关键帧 (以joint 2为例)
    joint_data = partition_data['jointpos'][:, 2]
    # 前200帧最大值作为起始帧
    start_frame = np.argmax(joint_data[:200])
    # 后150帧最大值作为结束帧
    end_frame = np.argmax(joint_data[-150:]) + len(joint_data) - 150
    print(f"使用的关键帧范围: {start_frame} -> {end_frame}")

    concatenated_data = {}
    for key in all_keys:
        # 提取关键帧之间的数据段
        if partition_data[key].ndim == 2:
            base_segment = partition_data[key][start_frame:end_frame+1, :]
        else:
            base_segment = partition_data[key][start_frame:end_frame+1]

        # 对基础段进行平滑处理
        if base_segment.ndim == 2:
            smoothed_base_segment = np.zeros_like(base_segment)
            for col in range(base_segment.shape[1]):
                smoothed_base_segment[:, col] = smooth_data(base_segment[:, col])
        else:
            smoothed_base_segment = smooth_data(base_segment)
            
        repeated_arrays = []
        for i in range(repeat_count):
            if key == 'root_pos' and i > 0:
                # 对于root_pos，只对第一列（x方向）进行位置对齐
                prev_end_x = repeated_arrays[-1][-1, 0]
                current_start_x = smoothed_base_segment[0, 0]
                x_offset = prev_end_x - current_start_x
                
                adjusted_segment = smoothed_base_segment.copy()
                adjusted_segment[:, 0] += x_offset
                # y和z方向保持不变
                repeated_arrays.append(adjusted_segment)
            else:
                # 其他数据或第一段直接添加
                repeated_arrays.append(smoothed_base_segment)

        # 拼接所有重复的段
        if repeated_arrays:
            if repeated_arrays[0].ndim == 2:
                concatenated_data[key] = np.vstack(repeated_arrays)
            else:
                concatenated_data[key] = np.concatenate(repeated_arrays)
    
    # 保存拼接后的数据
    save_path = "/home/zl/Downloads/kuavo-rl-train-run/RL_train/output_data/vicon/concatenated/concatenated_vicon_data.npz"
    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, **concatenated_data)
    
    print(f"拼接完成！保存到: {save_path}")
    print("拼接后的数据形状:")
    for key, data in concatenated_data.items():
        print(f"  {key}: {data.shape}")
    
    # 验证拼接结果 - 画出拼接后的joint pos数据
    if 'jointpos' in concatenated_data:
        joint_indices = [2, 3, 4, 5]
        fig, axes = plt.subplots(4, 1, figsize=(20, 24))
        axes = axes.flatten()
        
        for idx, joint in enumerate(joint_indices):
            ax = axes[idx]
            ax.plot(concatenated_data['jointpos'][:, joint], 
                   label=f"concatenated jointpos[:,{joint}]", linewidth=1.5)
            
            # 标记每个段的边界
            segment_length = end_frame - start_frame + 1
            for i in range(1, repeat_count):
                 ax.axvline(x=i * segment_length, color='red', linestyle='--', alpha=0.7)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.subplots_adjust(bottom=0.15)
        plt.suptitle("Vicon Joint Position after repeating concatenation", fontsize=16)
        plt.show()
    
    # 验证拼接结果 - 画出拼接后的root_pos数据
    if 'root_pos' in concatenated_data:
        fig, axes = plt.subplots(3, 1, figsize=(20, 18))
        axes = axes.flatten()
        
        labels = ['x', 'y', 'z']
        for idx, dim in enumerate(labels):
            ax = axes[idx]
            ax.plot(concatenated_data['root_pos'][:, idx], 
                   label=f"concatenated root_pos {dim}", linewidth=1.5)
            
            # 标记每个段的边界
            segment_length = end_frame - start_frame + 1
            for i in range(1, repeat_count):
                 ax.axvline(x=i * segment_length, color='red', linestyle='--', alpha=0.7)
            
            ax.set_ylabel(f"Position {dim}", fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, linewidth=1.0, alpha=0.5)
            ax.tick_params(axis="both", which="major", labelsize=10, width=1.5, length=4)
            
            if idx == len(axes) - 1:
                ax.set_xlabel("step", fontsize=12)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.subplots_adjust(bottom=0.15)
        plt.suptitle("Vicon Root Position after repeating concatenation", fontsize=16)
        plt.show()
    
    # 验证拼接结果 - 画出拼接后的root_vel_world数据
    if 'root_vel_world' in concatenated_data:
        fig, axes = plt.subplots(3, 1, figsize=(20, 18))
        axes = axes.flatten()
        labels = ['vx', 'vy', 'vz']
        for idx, dim in enumerate(labels):
            ax = axes[idx]
            ax.plot(concatenated_data['root_vel_world'][:, idx], 
                   label=f"concatenated root_vel_world {dim}", linewidth=1.5)
            # 标记每个段的边界
            segment_length = end_frame - start_frame + 1
            for i in range(1, repeat_count):
                 ax.axvline(x=i * segment_length, color='red', linestyle='--', alpha=0.7)
            
            ax.set_ylabel(f"Velocity {dim}", fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, linewidth=1.0, alpha=0.5)
            ax.tick_params(axis="both", which="major", labelsize=10, width=1.5, length=4)
            
            if idx == len(axes) - 1:
                ax.set_xlabel("step", fontsize=12)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.subplots_adjust(bottom=0.15)
        plt.suptitle("Vicon Root Velocity (World Frame) after repeating concatenation", fontsize=16)
        plt.show()

# ================== 对齐 ==================

do_align_with_ref = True  # 只有为True时才执行与ref的对齐
discard_frames = 34  # 手动抛弃前几帧

if do_align_with_ref:
    print("开始与ref数据对齐...")
    
    # 加载拼接后的数据
    concatenated_path = "/home/zl/Downloads/kuavo-rl-train-run/RL_train/output_data/vicon/concatenated/concatenated_vicon_data.npz"
    concatenated_data = np.load(concatenated_path, allow_pickle=True)
    
    # 抛弃前几帧
    aligned_data = {}
    for key in concatenated_data.keys():
        if concatenated_data[key].ndim == 2:
            aligned_data[key] = concatenated_data[key][discard_frames:, :]
        else:
            aligned_data[key] = concatenated_data[key][discard_frames:]
    
    print(f"抛弃前{discard_frames}帧后的数据形状:")
    for key, data in aligned_data.items():
        print(f"  {key}: {data.shape}")
    
    # 对root_omega_world和root_omega_body进行滤波处理
    if 'root_omega_world' in aligned_data:
        print("对root_omega_world进行滤波...")
        filtered_omega_world = np.zeros_like(aligned_data['root_omega_world'])
        for col in range(aligned_data['root_omega_world'].shape[1]):
            # 使用更强的滤波参数
            filtered_omega_world[:, col] = smooth_data(aligned_data['root_omega_world'][:, col], window_length=21, polyorder=3)
        aligned_data['root_omega_world'] = filtered_omega_world
    
    if 'root_omega_body' in aligned_data:
        print("对root_omega_body进行滤波...")
        filtered_omega_body = np.zeros_like(aligned_data['root_omega_body'])
        for col in range(aligned_data['root_omega_body'].shape[1]):
            # 使用更强的滤波参数
            filtered_omega_body[:, col] = smooth_data(aligned_data['root_omega_body'][:, col], window_length=21, polyorder=3)
        aligned_data['root_omega_body'] = filtered_omega_body
    
    # 保存对齐后的数据
    aligned_save_path = "/home/zl/Downloads/kuavo-rl-train-run/RL_train/output_data/vicon/aligned_concatenated_vicon_data.npz"
    np.savez(aligned_save_path, **aligned_data)
    print(f"对齐后的数据保存到: {aligned_save_path}")
    
    # 验证对齐效果 - 画出对齐后的数据与ref对比
    if 'jointpos' in aligned_data and 'root_pos' in aligned_data:
        # 画joint pos对比
        joint_indices = [2, 3, 4, 5]
        fig, axes = plt.subplots(4, 1, figsize=(20, 24))
        axes = axes.flatten()
        
        plot_len = min(2000, len(aligned_data['jointpos']), len(ref_data_dof_pos))  # 只画前2000帧
        
        for idx, joint in enumerate(joint_indices):
            ax = axes[idx]
            ax.plot(aligned_data['jointpos'][:plot_len, joint], 
                   label=f"aligned jointpos[:,{joint}]", linewidth=1.5)
            ax.plot(ref_data_dof_pos[:plot_len, joint], 
                   label=f"ref dof_pos[:,{joint}]", linewidth=1.5, linestyle='--')
            
            ax.set_ylabel(f"Joint {joint} Value", fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, linewidth=1.0, alpha=0.5)
            ax.tick_params(axis="both", which="major", labelsize=10, width=1.5, length=4)
            
            if idx == len(axes) - 1:
                ax.set_xlabel("step", fontsize=12)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.subplots_adjust(bottom=0.15)
        plt.suptitle("Aligned Joint Position vs Reference", fontsize=16)
        plt.show()
        
        # 画root pos对比
        fig, axes = plt.subplots(3, 1, figsize=(20, 18))
        axes = axes.flatten()
        
        labels = ['x', 'y', 'z']
        for idx, dim in enumerate(labels):
            ax = axes[idx]
            ax.plot(aligned_data['root_pos'][:plot_len, idx], 
                   label=f"aligned root_pos {dim}", linewidth=1.5)
            ax.plot(ref_data_root_pos[:plot_len, idx], 
                   label=f"ref root_pos {dim}", linewidth=1.5, linestyle='--')
            
            ax.set_ylabel(f"Position {dim}", fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, linewidth=1.0, alpha=0.5)
            ax.tick_params(axis="both", which="major", labelsize=10, width=1.5, length=4)
            
            if idx == len(axes) - 1:
                ax.set_xlabel("step", fontsize=12)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.subplots_adjust(bottom=0.15)
        plt.suptitle("Aligned Root Position vs Reference", fontsize=16)
        plt.show()
        
        # 画root vel对比
        if 'root_vel_world' in aligned_data:
            fig, axes = plt.subplots(3, 1, figsize=(20, 18))
            axes = axes.flatten()
            
            labels = ['vx', 'vy', 'vz']
            for idx, dim in enumerate(labels):
                ax = axes[idx]
                ax.plot(aligned_data['root_vel_world'][:plot_len, idx], 
                       label=f"aligned root_vel_world {dim}", linewidth=1.5)
                ax.plot(ref_data_root_lin_vel[:plot_len, idx], 
                       label=f"ref root_lin_vel {dim}", linewidth=1.5, linestyle='--')
                
                ax.set_ylabel(f"Velocity {dim}", fontsize=12)
                ax.legend(fontsize=10)
                ax.grid(True, linewidth=1.0, alpha=0.5)
                ax.tick_params(axis="both", which="major", labelsize=10, width=1.5, length=4)
                
                if idx == len(axes) - 1:
                    ax.set_xlabel("step", fontsize=12)
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.subplots_adjust(bottom=0.15)
            plt.suptitle("Aligned Root Velocity vs Reference", fontsize=16)
            plt.show()
        
        # 画root euler对比
        if 'root_euler' in aligned_data:
            fig, axes = plt.subplots(3, 1, figsize=(20, 18))
            axes = axes.flatten()
            
            labels = ['roll', 'pitch', 'yaw']
            for idx, dim in enumerate(labels):
                ax = axes[idx]
                ax.plot(aligned_data['root_euler'][:plot_len, idx], 
                       label=f"aligned root_euler {dim}", linewidth=1.5)
                ax.plot(ref_data_root_eu_ang[:plot_len, 4+idx], 
                       label=f"ref root_eu_ang {dim}", linewidth=1.5, linestyle='--')
                
                ax.set_ylabel(f"Euler Angle {dim} (rad)", fontsize=12)
                ax.legend(fontsize=10)
                ax.grid(True, linewidth=1.0, alpha=0.5)
                ax.tick_params(axis="both", which="major", labelsize=10, width=1.5, length=4)
                
                if idx == len(axes) - 1:
                    ax.set_xlabel("step", fontsize=12)
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.subplots_adjust(bottom=0.15)
            plt.suptitle("Aligned Root Euler Angles vs Reference", fontsize=16)
            plt.show()
        
        # 画root omega world对比
        if 'root_omega_world' in aligned_data:
            fig, axes = plt.subplots(3, 1, figsize=(20, 18))
            axes = axes.flatten()
            
            labels = ['ωx', 'ωy', 'ωz']
            for idx, dim in enumerate(labels):
                ax = axes[idx]
                ax.plot(aligned_data['root_omega_world'][:plot_len, idx], 
                       label=f"aligned root_omega_world {dim}", linewidth=1.5)
                ax.plot(ref_data_root_ang_vel[:plot_len, idx], 
                       label=f"ref root_ang_vel {dim}", linewidth=1.5, linestyle='--')
                
                ax.set_ylabel(f"Angular Velocity {dim} (rad/s)", fontsize=12)
                ax.legend(fontsize=10)
                ax.grid(True, linewidth=1.0, alpha=0.5)
                ax.tick_params(axis="both", which="major", labelsize=10, width=1.5, length=4)
                
                if idx == len(axes) - 1:
                    ax.set_xlabel("step", fontsize=12)
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.subplots_adjust(bottom=0.15)
            plt.suptitle("Aligned Root Angular Velocity (World Frame) vs Reference", fontsize=16)
            plt.show()


