import os
import numpy as np
import matplotlib.pyplot as plt

# ===== 控制画图开关 =====
PLOT_REAL_INDICES = None  # 控制画哪几条真实数据，例如[0,1,2]表示画前3条，[1,3]表示画第2和第4条，None表示画所有数据

# ===== 批量加载pretrained policy生成的real data文件，并保存去除前N帧和归一化后的数据 =====
real_data_skip_frames = 0  # 控制对比时real data跳过的帧数
pretrained_data_dir = "/home/zl/deploy_jog/kuavo-RL/output_data"
real_data_dealed_dir = "/home/zl/Downloads/kuavo-rl-train-run/RL_train/output_data/real_data_dealed"
os.makedirs(real_data_dealed_dir, exist_ok=True)
pretrained_file_list = [
    os.path.join(pretrained_data_dir, f) for f in os.listdir(pretrained_data_dir) if f.endswith(".npz")
]
real_data_dealed_list = []
for file in sorted(pretrained_file_list):
    data = np.load(file, allow_pickle=True)
    data_dealed = {k: v[real_data_skip_frames:] for k, v in data.items()}
    # 归一化处理（与ref一致）
    if "foot_height" in data_dealed:
        data_dealed["foot_height"][:, 0] -= data_dealed["foot_height"][19, 0]
        data_dealed["foot_height"][:, 1] -= data_dealed["foot_height"][50, 1]
    # if 'root_pos' in data_dealed:
    #     data_dealed['root_pos'][:, 2] -= data_dealed['root_pos'][17, 2]
    # 取反dof_pos和dof_vel的第6维和第12维
    for key in ["dof_pos", "dof_vel"]:
        if key in data_dealed:
            if data_dealed[key].shape[1] >= 12:
                data_dealed[key][:, 5] *= -1  # 第6维
                data_dealed[key][:, 11] *= -1  # 第12维
    base = os.path.basename(file)
    name, ext = os.path.splitext(base)
    out_path = os.path.join(real_data_dealed_dir, name + "_dealed.npz")
    np.savez(out_path, **data_dealed)
    real_data_dealed_list.append({"file": file, "data": data_dealed})

# 加载ref数据
ref_data = np.load(
    "/home/zl/Downloads/kuavo-rl-train-run/RL_train/humanoid/mpc_pose/play_cmu_0401_b.npz", allow_pickle=True
)

# 全部拷贝为可写变量
ref_dof_pos = ref_data["dof_pos"].copy()
ref_dof_vel = ref_data["dof_vel"].copy()
ref_foot_force = ref_data["foot_force"].copy()
ref_foot_height = ref_data["foot_height"].copy()
ref_foot_zvel = ref_data["foot_zvel"].copy()
ref_root_pos = ref_data["root_pos"].copy()
ref_root_eu_ang = ref_data["root_eu_ang"].copy()
ref_root_ang_vel = ref_data["root_ang_vel"].copy()
ref_root_lin_vel = ref_data["root_lin_vel"].copy()

# 归一化处理
ref_foot_height[:, 0] -= ref_foot_height[19, 0]
ref_foot_height[:, 1] -= ref_foot_height[50, 1]
ref_root_pos[:, 2] -= ref_root_pos[17, 2]

# 画出处理后的真实数据和ref数据进行对比
real_data_plot_start = 0  # 控制对比时从第几帧开始画图
real_data_plot_end = None  # 控制对比时结束帧，None为到末尾

# 创建包含所有数据的对比图
# 确定要画的real数据列表
if PLOT_REAL_INDICES is None:
    plot_real_list = real_data_dealed_list
else:
    plot_real_list = [real_data_dealed_list[i] for i in PLOT_REAL_INDICES if i < len(real_data_dealed_list)]

# 1. dof_pos对比（一张大图，12个子图，每个子图包含所有real数据和ref数据）
plt.figure(figsize=(15, 10))
for j in range(min(real_data_dealed_list[0]["data"]["joint_pos"].shape[1], ref_dof_pos.shape[1])):  # 关节数
    plt.subplot(3, 4, j + 1)
    x_range = np.arange(ref_dof_pos.shape[0])[slice(real_data_plot_start, real_data_plot_end)]
    plt.plot(
        x_range,
        ref_dof_pos[slice(real_data_plot_start, real_data_plot_end), j],
        label="ref",
        linewidth=2,
        color="black",
    )

    for i, real_item in enumerate(plot_real_list):
        # 找到在原始列表中的索引
        original_index = real_data_dealed_list.index(real_item)
        real_name = f"real{original_index+1}"
        real_dof_pos = real_item["data"]["joint_pos"]
        plot_slice = slice(real_data_plot_start, real_data_plot_end)
        x_range_real = np.arange(real_dof_pos.shape[0])[plot_slice]
        plt.plot(x_range_real, real_dof_pos[plot_slice, j], label=f"{real_name}", alpha=0.7)

    plt.title(f"dof_pos_{j}")
    plt.legend(fontsize=6)
plt.suptitle("Selected real data vs ref dof_pos")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(
    f"/home/zl/Downloads/kuavo-rl-train-run/RL_train/pictures/selected_real_vs_ref_dof_pos.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# 2. dof_vel对比
plt.figure(figsize=(15, 10))
for j in range(min(real_data_dealed_list[0]["data"]["joint_vel"].shape[1], ref_dof_vel.shape[1])):
    plt.subplot(3, 4, j + 1)
    x_range = np.arange(ref_dof_vel.shape[0])[slice(real_data_plot_start, real_data_plot_end)]
    plt.plot(
        x_range,
        ref_dof_vel[slice(real_data_plot_start, real_data_plot_end), j],
        label="ref",
        linewidth=2,
        color="black",
    )

    for i, real_item in enumerate(plot_real_list):
        original_index = real_data_dealed_list.index(real_item)
        real_name = f"real{original_index+1}"
        real_dof_vel = real_item["data"]["joint_vel"]
        plot_slice = slice(real_data_plot_start, real_data_plot_end)
        x_range_real = np.arange(real_dof_vel.shape[0])[plot_slice]
        plt.plot(x_range_real, real_dof_vel[plot_slice, j], label=f"{real_name}", alpha=0.7)

    plt.title(f"dof_vel_{j}")
    plt.legend(fontsize=6)
plt.suptitle("Selected real data vs ref dof_vel")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(
    f"/home/zl/Downloads/kuavo-rl-train-run/RL_train/pictures/selected_real_vs_ref_dof_vel.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# 4. root_pos对比
plt.figure(figsize=(15, 5))
for j in range(min(real_data_dealed_list[0]["data"]["pos_xyz"].shape[1], ref_root_pos.shape[1])):
    plt.subplot(1, 3, j + 1)
    x_range = np.arange(ref_root_pos.shape[0])[slice(real_data_plot_start, real_data_plot_end)]
    plt.plot(
        x_range,
        ref_root_pos[slice(real_data_plot_start, real_data_plot_end), j],
        label="ref",
        linewidth=2,
        color="black",
    )

    for i, real_item in enumerate(plot_real_list):
        original_index = real_data_dealed_list.index(real_item)
        real_name = f"real{original_index+1}"
        real_root_pos = real_item["data"]["pos_xyz"]
        plot_slice = slice(real_data_plot_start, real_data_plot_end)
        x_range_real = np.arange(real_root_pos.shape[0])[plot_slice]
        plt.plot(x_range_real, real_root_pos[plot_slice, j], label=f"{real_name}", alpha=0.7)

    plt.title(f"root_pos_{j}")
    plt.legend(fontsize=8)
plt.suptitle("Selected real data vs ref root_pos")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(
    f"/home/zl/Downloads/kuavo-rl-train-run/RL_train/pictures/selected_real_vs_ref_root_pos.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# 5. root_eu_ang对比
plt.figure(figsize=(15, 5))
for j in range(min(real_data_dealed_list[0]["data"]["angle_zyx"].shape[1], ref_root_eu_ang.shape[1])):
    plt.subplot(1, 3, j + 1)
    x_range = np.arange(ref_root_eu_ang.shape[0])[slice(real_data_plot_start, real_data_plot_end)]
    plt.plot(
        x_range,
        ref_root_eu_ang[slice(real_data_plot_start, real_data_plot_end), j],
        label="ref",
        linewidth=2,
        color="black",
    )

    for i, real_item in enumerate(plot_real_list):
        original_index = real_data_dealed_list.index(real_item)
        real_name = f"real{original_index+1}"
        real_root_eu_ang = real_item["data"]["angle_zyx"]
        plot_slice = slice(real_data_plot_start, real_data_plot_end)
        x_range_real = np.arange(real_root_eu_ang.shape[0])[plot_slice]
        plt.plot(x_range_real, real_root_eu_ang[plot_slice, j], label=f"{real_name}", alpha=0.7)

    plt.title(f"root_eu_ang_{j}")
    plt.legend(fontsize=8)
plt.suptitle("Selected real data vs ref root_eu_ang")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(
    f"/home/zl/Downloads/kuavo-rl-train-run/RL_train/pictures/selected_real_vs_ref_root_eu_ang.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
