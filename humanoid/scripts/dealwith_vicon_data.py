import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# 加载数据
vicon_data = np.load("/home/zl/ASAP_data_leju/vicon/2025-07-25-11-51-51.npz")

baselink_pos = vicon_data['BaseLinkposeroot_pos']  # (N, 3)
baselink_quat_ang = vicon_data['BaseLinkposeroot_quat_ang']  # (N,)
timestamps = vicon_data['BaseLinkposetimestamps']  # (N, 3)
timestamps_ros = vicon_data['jointpostimestamps']  # (N, 3)
jointpos = vicon_data['jointpos']
jointvel = vicon_data['jointvel']

# 绘制原始xyz位置
plt.figure(figsize=(10, 6))
for i, name in enumerate(['x', 'y', 'z']):
    plt.plot(timestamps, baselink_pos[:, i], label=f'pos_{name}')
plt.legend()
plt.xlabel('time (s)')
plt.ylabel('position (m)')
plt.title('BaseLink Position (Original Vicon Frame)')
plt.show()

# 目标：x前 = 原z，y左 = 原x，z上 = 原y
root_pos = np.stack([baselink_pos[:, 2], baselink_pos[:, 0], baselink_pos[:, 1]], axis=1)

def rot_transfer(quaternion):
    # 1. 合并所有Y轴旋转: -90度(坐标系变换) - 39.64度(pitch补偿) = -129.64度
    ry_combined = R.from_euler('y', -129.64, degrees=True).as_matrix()
    # 2. 将vicon的四元数转为旋转矩阵
    rot_matrix = R.from_quat(quaternion).as_matrix()
    # 3. 应用总的旋转变换 (右乘)
    rot_new = rot_matrix @ ry_combined
    # 转回四元数
    quaternion_rot = R.from_matrix(rot_new).as_quat()  # [x, y, z, w]
    return rot_new, quaternion_rot

# 由四元数计算角速度
def omega(quaternion, quaternion_prev, dt):
    dq = (quaternion - quaternion_prev) / dt
    x, y, z, w = quaternion
    omega = 2 * np.mat([[w, x, y, z],
                        [-x, w, z, -y],
                        [-y, -z, w, x],
                        [-z, y, -x, w]]) * np.vstack(dq)
    return np.asarray(omega[1:4]).reshape(-1)

# 逐个元素计算位置差分、时间差分和世界系速度
N = root_pos.shape[0]
delta_pos = np.zeros((N, 3))
delta_t = np.zeros(N)
root_vel_world = np.zeros((N, 3))
root_quaternion_rot = np.zeros((N, 4))
root_vel_body = np.zeros((N, 3))
root_euler = np.zeros((N, 3))  # 添加欧拉角数组
root_omega_world = np.zeros((N, 3))  # 添加world omega数组
root_omega_body = np.zeros((N, 3))  # 添加body omega数组

# 移动平均滤波窗口大小
window_size = 3

for i in range(1, N):
    delta_pos[i] = root_pos[i] - root_pos[i-1]
    delta_t[i] = timestamps[i] - timestamps[i-1]
    dt = 0.011  # 对应约90Hz
    if delta_t[i] < dt:
        delta_t[i] = dt
    root_vel_world[i] = delta_pos[i] / delta_t[i]
    
    # 对速度进行移动平均滤波
    if i >= window_size:
        start_idx = i - window_size + 1
        root_vel_world[i] = np.mean(root_vel_world[start_idx:i+1], axis=0)

    R_ab, quaternion_rot = rot_transfer(baselink_quat_ang[i])
    R_ab_prev, quaternion_rot_prev = rot_transfer(baselink_quat_ang[i - 1])
    root_omega_world[i] = omega(quaternion_rot, quaternion_rot_prev, dt)
    root_quaternion_rot[i] = quaternion_rot

    # 计算欧拉角 (XYZ顺序)
    root_euler[i] = R.from_quat(quaternion_rot).as_euler('xyz', degrees=False)
    
    # 计算body omega (将世界系角速度转换到机体系)
    root_omega_body[i] = R_ab.T @ root_omega_world[i] 

    root_vel_body[i] = R_ab.T @ root_vel_world[i]

root_vel_world[root_vel_world > 2] = 2
root_vel_world[root_vel_world < -2] = -2
root_vel_body[root_vel_body > 2] = 2
root_vel_body[root_vel_body < -2] = -2
# 绘制速度对比图
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
labels = ['x', 'y', 'z']

for i in range(3):
    axes[i].plot(timestamps, root_vel_world[:, i], label=f'world_vel_{labels[i]}', linewidth=2)
    axes[i].plot(timestamps, root_vel_body[:, i], '--', label=f'body_vel_{labels[i]}', linewidth=2)
    axes[i].set_ylabel(f'velocity {labels[i]} (m/s)')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

axes[2].set_xlabel('time (s)')
plt.suptitle('Velocity Comparison: World vs Body Frame')
plt.tight_layout()
plt.show()

# 数据对齐：将所有数据向timestamps_ros对齐
# 以从静止到运动的一瞬间作为起始点，以ros时间戳为基准，将vicon时间向ros时间对齐
print("开始数据对齐...")
print(f"BaseLink时间戳范围: {timestamps[0]} - {timestamps[-1]}")
print(f"关节时间戳范围: {timestamps_ros[0]} - {timestamps_ros[-1]}")

# 使用关节时间戳作为基准
target_timestamps = timestamps_ros

# 对齐所有数据
root_pos_aligned = np.zeros((len(target_timestamps), 3))
quaternion_rot_aligned = np.zeros((len(target_timestamps), 4))
root_vel_world_aligned = np.zeros((len(target_timestamps), 3))
root_vel_body_aligned = np.zeros((len(target_timestamps), 3))
root_euler_aligned = np.zeros((len(target_timestamps), 3))
root_omega_body_aligned = np.zeros((len(target_timestamps), 3))
root_omega_world_aligned = np.zeros((len(target_timestamps), 3))

# 对每个维度进行插值
for i in range(3):  # x, y, z
    root_pos_aligned[:, i] = np.interp(target_timestamps, timestamps, root_pos[:, i])
    root_vel_world_aligned[:, i] = np.interp(target_timestamps, timestamps, root_vel_world[:, i])
    root_omega_world_aligned[:, i] = np.interp(target_timestamps, timestamps, root_omega_world[:, i])
    root_vel_body_aligned[:, i] = np.interp(target_timestamps, timestamps, root_vel_body[:, i])
    root_euler_aligned[:, i] = np.interp(target_timestamps, timestamps, root_euler[:, i])
    root_omega_body_aligned[:, i] = np.interp(target_timestamps, timestamps, root_omega_body[:, i])

# 对四元数进行插值（需要特殊处理）
for i in range(4):  # x, y, z, w
    quaternion_rot_aligned[:, i] = np.interp(target_timestamps, timestamps, root_quaternion_rot[:, i])
# 归一化四元数
for i in range(len(quaternion_rot_aligned)):
    quaternion_rot_aligned[i] = quaternion_rot_aligned[i] / np.linalg.norm(quaternion_rot_aligned[i])

print("数据对齐完成！")

# 画出最终欧拉角
plt.figure(figsize=(10, 6))
for i, name in enumerate(['x', 'y']):
    plt.plot(target_timestamps, root_euler_aligned[:, i], label=f'euler_{name}')
plt.legend()
plt.xlabel('time (s)')
plt.ylabel('euler (rad)')
plt.title('BaseLink Euler (Original Vicon Frame)')
plt.show()

# 打印最终欧拉角y的第一个值
print(f"最终欧拉角y的第一个值: {root_euler_aligned[0, 1]}")

# 创建输出目录
output_dir = "/home/zl/Downloads/kuavo-rl-train-run/RL_train/output_data/vicon"
# os.makedirs(output_dir, exist_ok=True)

# 准备保存的数据
output_data = {
    'root_pos': root_pos_aligned,
    'quaternion_rot': quaternion_rot_aligned,
    'root_vel_world': root_vel_world_aligned,
    'root_vel_body': root_vel_body_aligned,
    'root_euler': root_euler_aligned,
    'root_omega_world': root_omega_world_aligned,
    'root_omega_body': root_omega_body_aligned,
    'timestamps': target_timestamps,
    'timestamps_ros': timestamps_ros,
    'jointpos': jointpos,
    'jointvel': jointvel
}

# 保存为npz文件
output_file = os.path.join(output_dir, "processed_vicon_data.npz")
np.savez(output_file, **output_data)

print(f"数据已保存到: {output_file}")
print("保存的数据包括:")
for key in output_data.keys():
    print(f"  - {key}: {output_data[key].shape}")



















