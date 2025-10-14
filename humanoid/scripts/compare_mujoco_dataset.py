import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ===== 批量加载pretrained policy生成的real data文件，并保存去除前N帧和归一化后的数据 =====
real_data_skip_frames = 2161  # 先降采样再去除前200帧
pretrained_data_dir = "/home/zl/deploy_jog/kuavo-RL/output_data"
real_data_dealed_dir = "/home/zl/Downloads/kuavo-rl-train-run/RL_train/output_data/real_data_dealed"
os.makedirs(real_data_dealed_dir, exist_ok=True)
pretrained_file_list = [
    os.path.join(pretrained_data_dir, f) for f in os.listdir(pretrained_data_dir) if f.endswith(".npz")
]
real_data_dealed_list = []
for file in sorted(pretrained_file_list):
    data = np.load(file, allow_pickle=True)
    # 先降采样（每10帧取1帧）
    data_downsampled = {k: v[::10] for k, v in data.items()}
    # 再去除前200帧
    data_dealed = {k: v[real_data_skip_frames:] for k, v in data_downsampled.items()}
    base = os.path.basename(file)
    name, ext = os.path.splitext(base)
    out_path = os.path.join(real_data_dealed_dir, name + "_dealed.npz")
    np.savez(out_path, **data_dealed)
    real_data_dealed_list.append({"file": file, "data": data_dealed})

# 加载ref数据
ref_data = np.load(
    "/home/zl/Downloads/kuavo-rl-train-run/RL_train/humanoid/mpc_pose/play_cmu_0401_b.npz", allow_pickle=True
)
# 加载通过eval_recon_data.py生成的ref每个关节的xyz数据
ref_joint_pos_global = np.load(
    "/home/zl/Downloads/kuavo-rl-train-run/RL_train/ref_fk_joint_pos_global_by_mujoco.npz", allow_pickle=True
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
ref_joint_pos_global = ref_joint_pos_global["joint_pos_global"].copy()

# 归一化处理
ref_foot_height[:, 0] -= ref_foot_height[19, 0]
ref_foot_height[:, 1] -= ref_foot_height[50, 1]
# ref_root_pos[:, 0] -= 10
# ref_root_pos[:, 1] -= 4
# ref_root_pos[:, 2] -= ref_root_pos[17, 2]

# 加载Isaac Gym数据
isaac_data_o = np.load("/home/zl/Downloads/kuavo-rl-train-run/RL_train/play_isaac_data.npz", allow_pickle=True)

isaac_data = {}
for k in isaac_data_o.files:
    arr = isaac_data_o[k]
    # 沿第0维拼接两次，得到10个周期
    arr_repeat = np.concatenate([arr, arr], axis=0)
    isaac_data[k] = arr_repeat

# 加载sim2sim+delta action数据，并去除前200帧左右机器人在mujoco下落并初始化的数据
sim2sim_add_delta_action_data = np.load(
    "/home/zl/Downloads/kuavo-rl-train-run/RL_train/output_data/add_delta_action_data/20250619_1413_sim2sim_add_delta_action_data1.npz",
    allow_pickle=True,
)
sim2sim_add_delta_action_data_dealed = {k: v[217:] for k, v in sim2sim_add_delta_action_data.items()}

# ===== 批量加载finetuning policy生成的所有数据文件 =====
finetuning_data_dir = "/home/zl/Downloads/kuavo-rl-train-run/RL_train/output_data/finetuning_data"
finetuning_file_list = [
    os.path.join(finetuning_data_dir, f) for f in os.listdir(finetuning_data_dir) if f.endswith(".npz")
]
sim2sim_finetuning_data_dealed_list = []
for file in sorted(finetuning_file_list):
    data = np.load(file, allow_pickle=True)
    # 这里默认去除前220帧（可根据实际情况调整）
    data_dealed = {k: v[220:] for k, v in data.items()}
    sim2sim_finetuning_data_dealed_list.append({"file": file, "data": data_dealed})
# 现在sim2sim_finetuning_data_dealed_list是一个列表，每个元素是{'file': 文件名, 'data': 数据字典}
# 后续可用于循环分析和绘图

ref_len = ref_dof_pos.shape[0]
sim_len = real_data_dealed_list[0]["data"]["joint_pos"].shape[0]
isaac_len = isaac_data["dof_pos"].shape[0]


joint_names = [
    "l_leg_roll",
    "l_leg_yaw",
    "l_leg_pitch",
    "l_knee",
    "l_foot_pitch",
    "l_foot_roll",
    "r_leg_roll",
    "r_leg_yaw",
    "r_leg_pitch",
    "r_knee",
    "r_foot_pitch",
    "r_foot_roll",
]

# sim2sim/ref 对比帧范围
start_frame = 0  # 从第几帧开始对比
end_frame = 400  # 对比多少帧，可根据需要调整
plot_slice = slice(start_frame, end_frame)

# isaacgym 对比帧范围
isaac_start = 400
isaac_end = 800
isaac_slice = slice(isaac_start, isaac_end)

# 只对比有效的8个关节（去掉l_leg_roll, l_leg_yaw, r_leg_roll, r_leg_yaw）
valid_idx = [2, 3, 4, 5, 8, 9, 10, 11]  # 对应joint_names的下标
valid_joint_names = [joint_names[i] for i in valid_idx]

# ========== 批量画dof_pos对比 ==========
plt.figure(figsize=(12, 6))
for i, idx in enumerate(valid_idx):
    plt.subplot(2, 4, i + 1)
    plt.plot(real_data_dealed_list[0]["data"]["joint_pos"][plot_slice, idx], label="sim2sim")
    # plt.plot(sim2sim_add_delta_action_data_dealed['dof_pos'][plot_slice, idx], label='sim2sim+delta')
    # for j, finetune_item in enumerate(sim2sim_finetuning_data_dealed_list):
    #     plt.plot(finetune_item['data']['dof_pos'][plot_slice, idx], label=f'finetune{j+1}', alpha=0.7)
    plt.plot(ref_dof_pos[plot_slice, idx], label="ref")
    plt.title(f"dof_pos: {valid_joint_names[i]}")
    plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("/home/zl/Downloads/kuavo-rl-train-run/RL_train/pictures/dof_pos_compare_multi_finetune.png")
plt.show()  # 可选，交互式查看

# ========== 批量画dof_vel对比 ==========
plt.figure(figsize=(12, 6))
for i, idx in enumerate(valid_idx):
    plt.subplot(2, 4, i + 1)
    plt.plot(real_data_dealed_list[0]["data"]["joint_vel"][plot_slice, idx], label="sim2sim")
    # plt.plot(sim2sim_add_delta_action_data_dealed['dof_vel'][plot_slice, idx], label='sim2sim+delta')
    # for j, finetune_item in enumerate(sim2sim_finetuning_data_dealed_list):
    #     plt.plot(finetune_item['data']['dof_vel'][plot_slice, idx], label=f'finetune{j+1}', alpha=0.7)
    plt.plot(ref_dof_vel[plot_slice, idx], label="ref")
    plt.title(f"dof_vel: {valid_joint_names[i]}")
    plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("/home/zl/Downloads/kuavo-rl-train-run/RL_train/pictures/dof_vel_compare_multi_finetune.png")
plt.show()  # 可选，交互式查看

# # ========== 批量画foot_height/foot_force/foot_zvel对比 ==========
# fields = ['foot_height', 'foot_force', 'foot_zvel']
# ref_fields = [ref_foot_height, ref_foot_force, ref_foot_zvel]
# plt.figure(figsize=(12, 9))
# for row, field in enumerate(fields):
#     for col in range(2):
#         plt.subplot(3, 2, row*2 + col + 1)
#         plt.plot(real_data_dealed_list[0]['data'][field][plot_slice, col], label='sim2sim')
#         plt.plot(sim2sim_add_delta_action_data_dealed[field][plot_slice, col], label='sim2sim+delta')
#         for j, finetune_item in enumerate(sim2sim_finetuning_data_dealed_list):
#             plt.plot(finetune_item['data'][field][plot_slice, col], label=f'finetune{j+1}', alpha=0.7)
#         plt.plot(ref_fields[row][plot_slice, col], label='ref')
#         plt.title(f'{field}: foot {col}')
#         plt.legend(fontsize=8)
# plt.tight_layout()
# plt.savefig('/home/zl/Downloads/kuavo-rl-train-run/RL_train/pictures/foot_fields_compare_multi_finetune.png')

# # ========== 批量画root_pos/root_eu_ang对比 ==========
# root_fields = ['root_pos', 'root_eu_ang']
# ref_root_fields = [ref_root_pos, ref_root_eu_ang]
# plt.figure(figsize=(14, 8))
# for row, field in enumerate(root_fields):
#     if field == 'root_eu_ang':
#         plot_indices = [0, 1, 2] # r,p,y
#         labels = ['roll', 'pitch', 'yaw']
#     else:
#         plot_indices = [0, 1, 2]
#         labels = [f'{field}_{i}' for i in range(3)]
#     for col, idx in enumerate(plot_indices):
#         plt.subplot(2, 3, row*3 + col + 1)
#         plt.plot(real_data_dealed_list[0]['data'][field][plot_slice, col], label='sim2sim')
#         plt.plot(sim2sim_add_delta_action_data_dealed[field][plot_slice, col], label='sim2sim+delta')
#         for j, finetune_item in enumerate(sim2sim_finetuning_data_dealed_list):
#             plt.plot(finetune_item['data'][field][plot_slice, col], label=f'finetune{j+1}', alpha=0.7)
#         if field == 'root_pos':
#             plt.plot(ref_root_pos[plot_slice, idx], label='ref')
#         else:
#             plt.plot(ref_root_fields[row][plot_slice, idx], label='ref')
#         plt.title(f'{field}: {labels[col]}')
#         plt.legend(fontsize=8)
# plt.tight_layout()
# plt.savefig('/home/zl/Downloads/kuavo-rl-train-run/RL_train/pictures/root_fields_compare_multi_finetune.png')

# # ========== 批量画root_ang_vel/root_lin_vel对比 ==========
# root_fields = ['root_ang_vel', 'root_lin_vel']
# ref_root_fields = [ref_root_ang_vel, ref_root_lin_vel]
# plt.figure(figsize=(14, 8))
# for row, field in enumerate(root_fields):
#     labels = [f'{field}_{i}' for i in range(3)]
#     for col in range(3):
#         plt.subplot(2, 3, row*3 + col + 1)
#         plt.plot(real_data_dealed_list[0]['data'][field][plot_slice, col], label='sim2sim')
#         plt.plot(sim2sim_add_delta_action_data_dealed[field][plot_slice, col], label='sim2sim+delta')
#         for j, finetune_item in enumerate(sim2sim_finetuning_data_dealed_list):
#             plt.plot(finetune_item['data'][field][plot_slice, col], label=f'finetune{j+1}', alpha=0.7)
#         plt.plot(ref_root_fields[row][plot_slice, col], label='ref')
#         plt.title(f'{field}: {labels[col]}')
#         plt.legend(fontsize=8)
# plt.tight_layout()
# plt.savefig('/home/zl/Downloads/kuavo-rl-train-run/RL_train/pictures/root_vel_fields_compare_multi_finetune.png')

# 比较各关节全局坐标
# plt.figure(figsize=(14, 8))  # 宽度从 28 缩小到 14；保持高度 8
# for xyz in range(3):
#     for joint_idx in range(6):
#         ax = plt.subplot(3, 6, xyz * 6 + joint_idx + 1)
#         ax.plot(
#             real_data_dealed_list[0]['data']['joint_pos_global'][plot_slice, joint_idx, xyz],
#             label='pretrained policy'
#         )
#         # ax.plot(
#         #     sim2sim_add_delta_action_data_dealed['joint_pos_global'][plot_slice, joint_idx, xyz],
#         #     label='sim2sim+delta'
#         # )
#         # 批量finetune
#         for j, finetune_item in enumerate(sim2sim_finetuning_data_dealed_list):
#             ax.plot(
#                 finetune_item['data']['joint_pos_global'][plot_slice, joint_idx, xyz],
#                 label=f'finetune{j+1}', alpha=0.7
#             )
#         # ref_joint_pos_global 展平后是 [N, 36]，每个关节3维按顺序排列
#         ax.plot(
#             ref_joint_pos_global[plot_slice, joint_idx * 3 + xyz],
#             label='ref'
#         )

#         # 只有第一行显示关节名称
#         if xyz == 0:
#             ax.set_title(joint_names[joint_idx], fontsize=10)
#         # 只有第一列显示坐标轴标签
#         if joint_idx == 0:
#             ax.set_ylabel('xyz'[xyz], fontsize=10)
#         # 只在第一个子图显示图例
#         if xyz == 0 and joint_idx == 0:
#             ax.legend(fontsize=8)

# plt.tight_layout()
# plt.savefig('/home/zl/Downloads/kuavo-rl-train-run/RL_train/pictures/joint_pos_global_compare_6joints.png')
# plt.show()  # 可选，交互式查看

# # ===================== 误差指标计算 =====================
# print("\n==== 误差指标统计 (sim2sim / sim2sim+delta / sim2sim+finetuning vs ref, 基于全局xyz) ====")

# # 1. 取最小长度
# min_len1 = min(real_data_dealed_list[0]['data']['joint_pos_global'].shape[0], ref_joint_pos_global.shape[0])
# min_len2 = min(sim2sim_add_delta_action_data_dealed['joint_pos_global'].shape[0], ref_joint_pos_global.shape[0])
# min_len_finetune = [min(item['data']['joint_pos_global'].shape[0], ref_joint_pos_global.shape[0]) for item in sim2sim_finetuning_data_dealed_list]

# # 2. 取数据
# # sim2sim
# dof_pos1 = real_data_dealed_list[0]['data']['joint_pos_global'][:min_len1]
# root_pos1 = real_data_dealed_list[0]['data']['root_pos'][:min_len1]
# dof_pos_ref1 = ref_joint_pos_global[:min_len1].reshape(-1, 12, 3)
# root_pos_ref1 = ref_root_pos[:min_len1]
# # sim2sim+delta
# dof_pos2 = sim2sim_add_delta_action_data_dealed['joint_pos_global'][:min_len2]
# root_pos2 = sim2sim_add_delta_action_data_dealed['root_pos'][:min_len2]
# dof_pos_ref2 = ref_joint_pos_global[:min_len2].reshape(-1, 12, 3)
# root_pos_ref2 = ref_root_pos[:min_len2]
# # finetune
# finetune_dof_pos = [item['data']['joint_pos_global'][:min_len] for item, min_len in zip(sim2sim_finetuning_data_dealed_list, min_len_finetune)]
# finetune_root_pos = [item['data']['root_pos'][:min_len] for item, min_len in zip(sim2sim_finetuning_data_dealed_list, min_len_finetune)]
# finetune_dof_pos_ref = [ref_joint_pos_global[:min_len].reshape(-1, 12, 3) for min_len in min_len_finetune]
# finetune_root_pos_ref = [ref_root_pos[:min_len] for min_len in min_len_finetune]

# # 3. 计算指标
# # Eg-mpjpe
# Eg_mpjpe1 = np.mean(np.linalg.norm(dof_pos1 - dof_pos_ref1, axis=2)) * 1000
# Eg_mpjpe2 = np.mean(np.linalg.norm(dof_pos2 - dof_pos_ref2, axis=2)) * 1000
# Eg_mpjpe_finetune = [np.mean(np.linalg.norm(dof_pos - dof_pos_ref, axis=2)) * 1000 for dof_pos, dof_pos_ref in zip(finetune_dof_pos, finetune_dof_pos_ref)]

# # Empjpe
# Empjpe1 = np.mean(np.linalg.norm((dof_pos1 - root_pos1[:, None, :]) - (dof_pos_ref1 - root_pos_ref1[:, None, :]), axis=2)) * 1000
# Empjpe2 = np.mean(np.linalg.norm((dof_pos2 - root_pos2[:, None, :]) - (dof_pos_ref2 - root_pos_ref2[:, None, :]), axis=2)) * 1000
# Empjpe_finetune = [np.mean(np.linalg.norm((dof_pos - root_pos[:, None, :]) - (dof_pos_ref - root_pos_ref[:, None, :]), axis=2)) * 1000 for dof_pos, root_pos, dof_pos_ref, root_pos_ref in zip(finetune_dof_pos, finetune_root_pos, finetune_dof_pos_ref, finetune_root_pos_ref)]

# # Eacc
# def get_acc(arr):
#     return arr[2:] - 2*arr[1:-1] + arr[:-2]

# acc_sim1 = get_acc(dof_pos1)
# acc_ref1 = get_acc(dof_pos_ref1)
# Eacc1 = np.mean(np.linalg.norm(acc_sim1 - acc_ref1, axis=2)) * 1000

# acc_sim2 = get_acc(dof_pos2)
# acc_ref2 = get_acc(dof_pos_ref2)
# Eacc2 = np.mean(np.linalg.norm(acc_sim2 - acc_ref2, axis=2)) * 1000

# Eacc_finetune = []
# for dof_pos, dof_pos_ref in zip(finetune_dof_pos, finetune_dof_pos_ref):
#     acc_sim = get_acc(dof_pos)
#     acc_ref = get_acc(dof_pos_ref)
#     Eacc_finetune.append(np.mean(np.linalg.norm(acc_sim - acc_ref, axis=2)) * 1000)

# # Evel（root_pos三维不变）
# vel_sim1 = root_pos1[1:] - root_pos1[:-1]
# vel_ref1 = root_pos_ref1[1:] - root_pos_ref1[:-1]
# Evel1 = np.mean(np.linalg.norm(vel_sim1 - vel_ref1, axis=1)) * 1000

# vel_sim2 = root_pos2[1:] - root_pos2[:-1]
# vel_ref2 = root_pos_ref2[1:] - root_pos_ref2[:-1]
# Evel2 = np.mean(np.linalg.norm(vel_sim2 - vel_ref2, axis=1)) * 1000

# Evel_finetune = []
# for root_pos, root_pos_ref in zip(finetune_root_pos, finetune_root_pos_ref):
#     vel_sim = root_pos[1:] - root_pos[:-1]
#     vel_ref = root_pos_ref[1:] - root_pos_ref[:-1]
#     Evel_finetune.append(np.mean(np.linalg.norm(vel_sim - vel_ref, axis=1)) * 1000)

# # 4. 打印对比
# print(f"Eg-mpjpe (全局关节误差, mm):")
# print(f"  sim2sim:           {Eg_mpjpe1:.2f}")
# print(f"  sim2sim+delta:     {Eg_mpjpe2:.2f}")
# for file, val in zip([os.path.basename(item['file']) for item in sim2sim_finetuning_data_dealed_list], Eg_mpjpe_finetune):
#     print(f"  finetune:{file:20s} {val:.2f}")

# print(f"Empjpe (root相对关节误差, mm):")
# print(f"  sim2sim:           {Empjpe1:.2f}")
# print(f"  sim2sim+delta:     {Empjpe2:.2f}")
# for file, val in zip([os.path.basename(item['file']) for item in sim2sim_finetuning_data_dealed_list], Empjpe_finetune):
#     print(f"  finetune:{file:20s} {val:.2f}")

# print(f"Eacc (加速度误差, mm/frame²):")
# print(f"  sim2sim:           {Eacc1:.2f}")
# print(f"  sim2sim+delta:     {Eacc2:.2f}")
# for file, val in zip([os.path.basename(item['file']) for item in sim2sim_finetuning_data_dealed_list], Eacc_finetune):
#     print(f"  finetune:{file:20s} {val:.2f}")

# print(f"Evel (root速度误差, mm/frame):")
# print(f"  sim2sim:           {Evel1:.2f}")
# print(f"  sim2sim+delta:     {Evel2:.2f}")
# for file, val in zip([os.path.basename(item['file']) for item in sim2sim_finetuning_data_dealed_list], Evel_finetune):
#     print(f"  finetune:{file:20s} {val:.2f}")

# # ===== 画成4幅柱状图，方便比较（批量finetune） =====
# labels = ['pretrained\npolicy', 'pretrained\npolicy+delta\naction'] + [f'checkpoint{j+1}' for j in range(len(sim2sim_finetuning_data_dealed_list))]

# Eg_mpjpe_list = [Eg_mpjpe1, Eg_mpjpe2] + Eg_mpjpe_finetune
# Empjpe_list = [Empjpe1, Empjpe2] + Empjpe_finetune
# Eacc_list = [Eacc1, Eacc2] + Eacc_finetune
# Evel_list = [Evel1, Evel2] + Evel_finetune

# fig, axs = plt.subplots(2, 2, figsize=(max(14, 2*len(labels)), 10))

# bar_width = 0.6
# label_fontsize = 18
# xtick_fontsize = 12
# ytick_fontsize = 16
# value_fontsize = 12
# title_fontsize = 20

# axs[0, 0].bar(labels, Eg_mpjpe_list, color=['#4C72B0', '#55A868'] + ['#C44E52']*len(Eg_mpjpe_finetune), width=bar_width)
# axs[0, 0].set_title('Eg-mpjpe (mm)', fontsize=title_fontsize)
# axs[0, 0].set_ylabel('mm', fontsize=label_fontsize)
# axs[0, 0].set_ylim(0, max(Eg_mpjpe_list)*1.2)
# axs[0, 0].tick_params(axis='x', labelsize=xtick_fontsize, rotation=30)
# axs[0, 0].tick_params(axis='y', labelsize=ytick_fontsize)

# axs[0, 1].bar(labels, Empjpe_list, color=['#4C72B0', '#55A868'] + ['#C44E52']*len(Empjpe_finetune), width=bar_width)
# axs[0, 1].set_title('Empjpe (mm)', fontsize=title_fontsize)
# axs[0, 1].set_ylabel('mm', fontsize=label_fontsize)
# axs[0, 1].set_ylim(0, max(Empjpe_list)*1.2)
# axs[0, 1].tick_params(axis='x', labelsize=xtick_fontsize, rotation=30)
# axs[0, 1].tick_params(axis='y', labelsize=ytick_fontsize)

# axs[1, 0].bar(labels, Eacc_list, color=['#4C72B0', '#55A868'] + ['#C44E52']*len(Eacc_finetune), width=bar_width)
# axs[1, 0].set_title('Eacc (mm/frame²)', fontsize=title_fontsize)
# axs[1, 0].set_ylabel('mm/frame²', fontsize=label_fontsize)
# axs[1, 0].set_ylim(0, max(Eacc_list)*1.2)
# axs[1, 0].tick_params(axis='x', labelsize=xtick_fontsize, rotation=30)
# axs[1, 0].tick_params(axis='y', labelsize=ytick_fontsize)

# axs[1, 1].bar(labels, Evel_list, color=['#4C72B0', '#55A868'] + ['#C44E52']*len(Evel_finetune), width=bar_width)
# axs[1, 1].set_title('Evel (mm/frame)', fontsize=title_fontsize)
# axs[1, 1].set_ylabel('mm/frame', fontsize=label_fontsize)
# axs[1, 1].set_ylim(0, max(Evel_list)*1.2)
# axs[1, 1].tick_params(axis='x', labelsize=xtick_fontsize, rotation=30)
# axs[1, 1].tick_params(axis='y', labelsize=ytick_fontsize)

# for ax, err_list in zip(axs.flat, [Eg_mpjpe_list, Empjpe_list, Eacc_list, Evel_list]):
#     for i, v in enumerate(ax.patches):
#         ax.text(v.get_x() + v.get_width() / 2, v.get_height() + 0.01*max(err_list), f'{v.get_height():.2f}',
#                 ha='center', va='bottom', fontsize=value_fontsize)

# plt.tight_layout()
# plt.savefig('/home/zl/Downloads/kuavo-rl-train-run/RL_train/pictures/error_compare_multi_finetune.png')
# plt.show()
