# 计算ref数据正运动学，得到各关节全局坐标位置

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

# 1. 加载Mujoco模型
xml_path = "/home/zl/Downloads/kuavo-rl-train-run/RL_train/resources/robots/biped_s44/xml/biped_s44.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# 2. 加载ref数据
ref_path = "/home/zl/Downloads/kuavo-rl-train-run/RL_train/humanoid/mpc_pose/play_cmu_0401_b.npz"
ref_data = np.load(ref_path, allow_pickle=True)
ref_qpos = ref_data["dof_pos"]  # shape: [N, 12]
ref_root_pos = ref_data["root_pos"]  # shape: [N, 3]
ref_eu_ang = ref_data["root_eu_ang"]  # shape: [N, 7]，我们要[:, [1,3,5]]

# 根部坐标归一化 - 减去固定偏移量
# ref_root_pos[:, 0] -= 10  # x坐标减去10
# ref_root_pos[:, 1] -= 4   # y坐标减去4
# ref_root_pos[:, 2] -= ref_root_pos[17, 2]  # z坐标归一化（保持原来的逻辑）

# 3. 关节名顺序（与sim2sim_kuavo.py一致）
joint_names = [
    "leg_l1_joint",
    "leg_l2_joint",
    "leg_l3_joint",
    "leg_l4_joint",
    "leg_l5_joint",
    "leg_l6_joint",
    "leg_r1_joint",
    "leg_r2_joint",
    "leg_r3_joint",
    "leg_r4_joint",
    "leg_r5_joint",
    "leg_r6_joint",
]
joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]
body_ids = [model.jnt_bodyid[jid] for jid in joint_ids]

# 4. 根部自由度
N = ref_qpos.shape[0]
qpos_full = np.zeros((N, model.nq))
qpos_full[:, 0:3] = ref_root_pos  # 根部位置
# 取欧拉角（xyz顺序）
euler_xyz = ref_eu_ang[:, [4, 5, 6]]  # shape: [N, 3]
# 批量欧拉角转四元数
rot = R.from_euler("xyz", euler_xyz)
quat_xyzw = rot.as_quat()  # shape: [N, 4], [x, y, z, w]
# Mujoco需要[w, x, y, z]
qpos_full[:, 3] = quat_xyzw[:, 3]
qpos_full[:, 4] = quat_xyzw[:, 0]
qpos_full[:, 5] = quat_xyzw[:, 1]
qpos_full[:, 6] = quat_xyzw[:, 2]
qpos_full[:, 7:19] = ref_qpos[:, :12]

# 5. 采集每帧各关节body的全局xyz
joint_pos_global = []
for i in range(N):
    data.qpos[:] = qpos_full[i]
    mujoco.mj_forward(model, data)
    frame_xyz = []
    for body_id in body_ids:
        pos = data.xpos[body_id].copy()
        frame_xyz.append(pos)
    joint_pos_global.append(frame_xyz)
joint_pos_global = np.array(joint_pos_global)  # [N, 12, 3]
joint_pos_global_flat = joint_pos_global.reshape(N, 12 * 3)  # [N, 36]

# 6. 保存
np.savez("ref_fk_joint_pos_global_by_mujoco.npz", joint_pos_global=joint_pos_global_flat)
print("已保存ref_fk_joint_pos_global_by_mujoco.npz，shape:", joint_pos_global_flat.shape)
