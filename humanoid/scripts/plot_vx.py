import numpy as np
import matplotlib.pyplot as plt

# 加载数据
data = np.load('/home/zl/Downloads/kuavo-rl-train-run/RL_train/output_data/finetuning_data/20250804_0124_sim2sim_mujoco_finetuning_data1.npz')

# 尝试常见的线速度键名
if 'root_lin_vel' in data:
    v = data['root_lin_vel']
elif 'linear_vel' in data:
    v = data['linear_vel']
else:
    print("未找到线速度相关的键！可用键有：", list(data.keys()))
    exit(1)

# 画图
fig, axes = plt.subplots(v.shape[1], 1, figsize=(12, 3 * v.shape[1]), sharex=True)
if v.shape[1] == 1:
    axes = [axes]
for i in range(v.shape[1]):
    axes[i].plot(v[:, i], label=f'v_body[:, {i}]')
    axes[i].set_ylabel(f'v_{i} (m/s)')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)
axes[-1].set_xlabel('Frame')
fig.suptitle('Linear Velocity (Each Column)')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()