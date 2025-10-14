import cma
import numpy as np
from isaacgym import gymtorch
from humanoid.envs import *
from humanoid.utils import get_args, task_registry
import torch
from tqdm import tqdm
import math
import mujoco
from collections import deque
from scipy.spatial.transform import Rotation as R
from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.envs import KuavoCfg
import matplotlib.pyplot as plt
import os
from datetime import datetime
import mujoco_viewer
import random
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from typing import Any
from typing import Dict
import seaborn as sns

SEED = 42
mujoco_see = False
N_TRIALS = 100  # 尝试100次不同的超参数组合
N_STARTUP_TRIALS = 5  # 前5次是随机采样，用于TPE算法“热身”


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 下面两行可选，保证更彻底的确定性（但可能影响性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class cmd:
    vx = 2.0
    vy = 0.0
    dyaw = 0.
    stand = 0.

def quaternion_to_euler_array(quat):
    x, y, z, w = quat
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    return np.array([roll_x, pitch_y, yaw_z])

def get_obs_mujoco(data):
    '''Extracts an observation from the mujoco data structure'''
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = data.sensor('angular-velocity').data.astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    return (q, dq, quat, v, omega, gvec)

def pd_control(target_q, default_dof_pos, q, kp, target_dq, dq, kd):
    return (target_q + default_dof_pos - q) * kp + (target_dq - dq) * kd


class Sim2simCfg(KuavoCfg):
    
    class sim_config:
        mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/biped_s44/xml/scene.xml'
        sim_duration = 74.17
        dt = 0.001
        decimation = 10

    class robot_config:
        kps = np.array([60, 60, 100, 150, 15, 15, 60, 60, 100, 150, 15, 15, 100, 100, 100, 100, 100, 100], dtype=np.double)
        kds = np.array([5.0, 5.0, 5.0, 6.0, 5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 5.0, 5.0, 10, 10, 10, 10, 10, 10], dtype=np.double)
        tau_limit = np.array([100., 25.5, 84., 250., 40., 30., 100., 25.5, 84., 250., 40., 30., 20, 20, 20, 20, 20, 20])

class Sim2RealCMAOptimizer:
    def __init__(self, 
                 initial_params, 
                 sigma0, 
                 real_data, 
                 env, 
                 policy, 
                 jit_policy,
                 param_names=None, 
                 max_iter=50):
        self.initial_params = np.array(initial_params)
        self.sigma0 = sigma0
        self.real_data = real_data
        self.env = env
        self.policy = policy
        self.jit_policy = jit_policy
        self.param_names = param_names
        self.max_iter = max_iter
        self.best_params = None
        self.best_score = float('inf')
        self.mujoco_data = None

        self.save_dir = f"data_comparison/data_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.save_dir, exist_ok=True)
        # 启用交互模式 (interactive mode)
        #plt.ion()       
        # 定义要计算的维度索引
        self.full_range = set(range(42))
        self.dims_to_calcu = {0,3,6,9,12,15}
        self.dims_to_ignore = self.full_range - self.dims_to_calcu 
        self.draw_plt = True
        if self.draw_plt:
            self.fig, self.axes = plt.subplots(len(self.dims_to_calcu), 2, figsize=(10, 3 * len(self.dims_to_calcu)))

    def plot_mujoco_isaac_torque_comparison(self, mujoco_data, isaac_data, filename='mujoco_isaac_torque_comparison.png'):
        """绘制Mujoco和Isaac Gym的扭矩对比"""
        # 数据结构：[joint_pos(12), joint_vel(12), action(12), base_vel(3), world_vel(3), actual_torques(12)]
        num_joints = 12
        
        # 提取动作扭矩（策略输出）
        action_torques_muj = mujoco_data[:, 2*num_joints:3*num_joints]  # 24:36
        action_torques_isaac = isaac_data[:, 2*num_joints:3*num_joints]  
        
        # 提取实际扭矩
        actual_torques_muj = mujoco_data[:, -num_joints:]  # 最后12列
        actual_torques_isaac = isaac_data[:, -num_joints:]  
        
        joint_names = [
            'leg_l1', 'leg_l2', 'leg_l3', 'leg_l4', 'leg_l5', 'leg_l6',
            'leg_r1', 'leg_r2', 'leg_r3', 'leg_r4', 'leg_r5', 'leg_r6'
        ]
        
        # 创建大图，显示所有对比
        fig, axes = plt.subplots(4, 3, figsize=(18, 16))
        fig.suptitle('Mujoco vs Isaac Gym: Action & Actual Torques Comparison', fontsize=16)
        
        time_steps_muj = np.arange(len(mujoco_data))
        time_steps_isaac = np.arange(len(isaac_data))
        
        for i in range(num_joints):
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            
            # # 绘制动作扭矩对比
            # ax.plot(time_steps_muj, action_torques_muj[:, i], 'b-', 
            #         label='Mujoco Action', alpha=0.7, linewidth=1.5)
            # ax.plot(time_steps_isaac, action_torques_isaac[:, i], 'r-', 
            #         label='Isaac Action', alpha=0.7, linewidth=1.5)
            
            # 绘制实际扭矩对比
            ax.plot(time_steps_muj, actual_torques_muj[:, i], 'b--', 
                    label='Real', alpha=0.7, linewidth=1.5)
            ax.plot(time_steps_isaac, actual_torques_isaac[:, i], 'r--', 
                    label='Simdata in Isaac', alpha=0.7, linewidth=1.5)
            
            ax.set_title(f'{joint_names[i]}', fontsize=10)
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Torque [Nm]')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            
            # 计算统计信息
            min_len = min(len(action_torques_muj), len(action_torques_isaac))
            
            # 动作扭矩差异
            action_mae = np.mean(np.abs(action_torques_muj[:min_len, i] - action_torques_isaac[:min_len, i]))
            action_corr = np.corrcoef(action_torques_muj[:min_len, i], action_torques_isaac[:min_len, i])[0, 1]
            
            # 实际扭矩差异
            actual_mae = np.mean(np.abs(actual_torques_muj[:min_len, i] - actual_torques_isaac[:min_len, i]))
            actual_corr = np.corrcoef(actual_torques_muj[:min_len, i], actual_torques_isaac[:min_len, i])[0, 1]
            
            # 显示统计信息
            stats_text = f'Act: MAE={action_mae:.2f}, R={action_corr:.3f}\nReal: MAE={actual_mae:.2f}, R={actual_corr:.3f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                    verticalalignment='top', fontsize=8, 
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
    
    
    def plot_torque_velocity_curves(self, mujoco_data, isaac_data, filename='torque_velocity_curves.png'):
        """绘制关节力矩与速度的关系曲线，包含四象限扭矩-速度特性曲线"""
        # 提取关节速度和动作（对应力矩）
        num_joints = 12
        joint_vel_muj = mujoco_data[:, num_joints:2*num_joints]  # 关节速度
        action_muj = mujoco_data[:, -num_joints:]   # 动作（对应力矩）
        
        joint_vel_isaac = isaac_data[:, num_joints:2*num_joints]
        action_isaac = isaac_data[:, -num_joints:]
        
        joint_names = [
            'leg_l1', 'leg_l2', 'leg_l3', 'leg_l4', 'leg_l5', 'leg_l6',
            'leg_r1', 'leg_r2', 'leg_r3', 'leg_r4', 'leg_r5', 'leg_r6'
        ]
        
        # 获取当前优化参数（用于绘制理论曲线）
        params = getattr(self, 'best_params', {})
        
        # 修改：使用独立的象限速度阈值
        speed_threshold_l3_q1 = params.get('speed_threshold_l3_q1', 5.0)
        speed_threshold_l3_q3 = params.get('speed_threshold_l3_q3', 5.0)
        speed_threshold_l4_q1 = params.get('speed_threshold_l4_q1', 7.0)
        speed_threshold_l4_q3 = params.get('speed_threshold_l4_q3', 7.0)
        
        max_speed_l3 = abs(params.get('angle_vel_l3_top', 10.0))
        max_speed_l4 = abs(params.get('angle_vel_l4_top', 12.0))
        torque_l3_top = params.get('torque_l3_top', 35.0)
        torque_l3_bottom = params.get('torque_l3_bottom', -35.0)
        torque_l4_top = params.get('torque_l4_top', 150.0)
        torque_l4_bottom = params.get('torque_l4_bottom', -150.0)
        
        # 创建速度范围用于理论曲线
        theoretical_speeds = np.linspace(-15, 15, 200)
        
        # 计算l3和l4的四象限理论扭矩曲线（使用新的独立阈值）
        l3_theory_top, l3_theory_bottom = self.calculate_asymmetric_four_quadrant_torque_curve(
            theoretical_speeds, torque_l3_top, torque_l3_bottom, 
            speed_threshold_l3_q1, speed_threshold_l3_q3, max_speed_l3
        )
        l4_theory_top, l4_theory_bottom = self.calculate_asymmetric_four_quadrant_torque_curve(
            theoretical_speeds, torque_l4_top, torque_l4_bottom,
            speed_threshold_l4_q1, speed_threshold_l4_q3, max_speed_l4
        )
        
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle('Torque vs Velocity: Data & Asymmetric Four-Quadrant Theory Comparison', fontsize=16)
        
        for i in range(num_joints):
            row = i // 4
            col = i % 4
            ax = axes[row, col]
            
            # 绘制Mujoco的力矩-速度散点图
            ax.scatter(joint_vel_muj[:, i], action_muj[:, i], 
                    alpha=0.6, s=1, label='Real', color='blue')
            
            # 绘制Isaac Gym的力矩-速度散点图
            ax.scatter(joint_vel_isaac[:, i], action_isaac[:, i], 
                    alpha=0.6, s=1, label='Simdata in Isaac', color='red')
            
            # 添加四象限理论扭矩限制曲线（仅对L3和L4关节）
            if i == 2 or i == 8:  # leg_l3, leg_r3
                # 绘制四象限理论曲线
                ax.plot(theoretical_speeds, l3_theory_top, 'g-', linewidth=2, 
                    label='L3 Theory Upper', alpha=0.8)
                ax.plot(theoretical_speeds, l3_theory_bottom, 'g--', linewidth=2, 
                    label='L3 Theory Lower', alpha=0.8)
                
                # 添加Q2、Q4象限的固定水平线（用不同颜色和线型突出显示）
                ax.axhline(y=torque_l3_top, color='red', linestyle='-', linewidth=2, alpha=0.7,
                        label=f'Q2 Fixed Limit ({torque_l3_top:.1f})', xmin=0, xmax=0.5)
                ax.axhline(y=torque_l3_bottom, color='red', linestyle='-', linewidth=2, alpha=0.7,
                        label=f'Q4 Fixed Limit ({torque_l3_bottom:.1f})', xmin=0.5, xmax=1)
                
                # 添加象限分界线
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
                ax.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
                
                # 添加象限标注
                ax.text(7.5, torque_l3_top*0.8, 'Q1\n(Dynamic)', fontsize=8, ha='center', 
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
                ax.text(-7.5, torque_l3_top*0.8, 'Q2\n(Fixed)', fontsize=8, ha='center',
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
                ax.text(-7.5, torque_l3_bottom*0.8, 'Q3\n(Dynamic)', fontsize=8, ha='center',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
                ax.text(7.5, torque_l3_bottom*0.8, 'Q4\n(Fixed)', fontsize=8, ha='center',
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
                
                # 修改：添加独立的象限阈值线
                ax.axvline(x=speed_threshold_l3_q1, color='orange', linestyle=':', alpha=0.7, 
                        label=f'Q1 Threshold ({speed_threshold_l3_q1})')
                ax.axvline(x=-speed_threshold_l3_q3, color='purple', linestyle=':', alpha=0.7,
                        label=f'Q3 Threshold ({speed_threshold_l3_q3})')
                ax.axvline(x=max_speed_l3, color='brown', linestyle=':', alpha=0.7, 
                        label=f'Max Speed ({max_speed_l3})')
                ax.axvline(x=-max_speed_l3, color='brown', linestyle=':', alpha=0.7)
                
            elif i == 3 or i == 9:  # leg_l4, leg_r4
                # 绘制四象限理论曲线
                ax.plot(theoretical_speeds, l4_theory_top, 'g-', linewidth=2, 
                    label='L4 Theory Upper', alpha=0.8)
                ax.plot(theoretical_speeds, l4_theory_bottom, 'g--', linewidth=2, 
                    label='L4 Theory Lower', alpha=0.8)
                
                # 添加Q2、Q4象限的固定水平线
                ax.axhline(y=torque_l4_top, color='red', linestyle='-', linewidth=2, alpha=0.7,
                        label=f'Q2 Fixed Limit ({torque_l4_top:.1f})')
                ax.axhline(y=torque_l4_bottom, color='red', linestyle='-', linewidth=2, alpha=0.7,
                        label=f'Q4 Fixed Limit ({torque_l4_bottom:.1f})')
                
                # 添加象限分界线
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
                ax.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
                
                # 添加象限标注
                ax.text(7.5, torque_l4_top*0.8, 'Q1\n(Dynamic)', fontsize=8, ha='center', 
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
                ax.text(-7.5, torque_l4_top*0.8, 'Q2\n(Fixed)', fontsize=8, ha='center',
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
                ax.text(-7.5, torque_l4_bottom*0.8, 'Q3\n(Dynamic)', fontsize=8, ha='center',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
                ax.text(7.5, torque_l4_bottom*0.8, 'Q4\n(Fixed)', fontsize=8, ha='center',
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
                
                # 修改：添加独立的象限阈值线
                ax.axvline(x=speed_threshold_l4_q1, color='orange', linestyle=':', alpha=0.7, 
                        label=f'Q1 Threshold ({speed_threshold_l4_q1})')
                ax.axvline(x=-speed_threshold_l4_q3, color='purple', linestyle=':', alpha=0.7,
                        label=f'Q3 Threshold ({speed_threshold_l4_q3})')
                ax.axvline(x=max_speed_l4, color='brown', linestyle=':', alpha=0.7, 
                        label=f'Max Speed ({max_speed_l4})')
                ax.axvline(x=-max_speed_l4, color='brown', linestyle=':', alpha=0.7)
            
            # 添加零线
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
            
            ax.set_xlabel('Joint Velocity [rad/s]')
            ax.set_ylabel('Joint Torque [Nm]')
            ax.set_title(f'{joint_names[i]}')
            
            # 只在有理论曲线的关节显示完整图例
            if i == 2 or i == 3:  # leg_l3, leg_l4
                ax.legend(fontsize=7, loc='best', ncol=2)
            else:
                ax.legend(['Real', 'Simdata in Isaac'], fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # 计算并显示数据统计
            if len(joint_vel_muj[:, i]) > 0 and len(action_muj[:, i]) > 0:
                corr_real = np.corrcoef(joint_vel_muj[:, i], action_muj[:, i])[0, 1]
                corr_sim = np.corrcoef(joint_vel_isaac[:, i], action_isaac[:, i])[0, 1]
                
                # 显示相关性信息
                stats_text = f'Corr Real: {corr_real:.3f}\nCorr Sim: {corr_sim:.3f}'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                    verticalalignment='top', fontsize=8, 
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        plt.tight_layout()
        
        # 保存主图
        plt.savefig(os.path.join(self.save_dir, filename), dpi=300, bbox_inches='tight')
        
        # 创建L3和L4的详细对比图
        fig2, axes2 = plt.subplots(2, 2, figsize=(15, 10))
        fig2.suptitle('L3 & L4 Joints: Asymmetric Four-Quadrant Torque-Velocity Analysis', fontsize=16)
        
        key_joints = [2, 3, 8, 9]  # leg_l3, leg_l4, leg_r3, leg_r4
        joint_labels = ['Left Hip Pitch (l3)', 'Left Knee (l4)', 'Right Hip Pitch (r3)', 'Right Knee (r4)']
        
        for idx, joint_idx in enumerate(key_joints):
            row = idx // 2
            col = idx % 2
            ax = axes2[row, col]
            
            # 绘制散点数据
            ax.scatter(joint_vel_muj[:, joint_idx], action_muj[:, joint_idx], 
                    alpha=0.3, s=1, label='Real', color='blue')
            ax.scatter(joint_vel_isaac[:, joint_idx], action_isaac[:, joint_idx], 
                    alpha=0.3, s=1, label='Simdata in Isaac', color='red')
            
            # 绘制理论曲线
            if joint_idx in [2, 8]:  # L3 joints
                ax.plot(theoretical_speeds, l3_theory_top, 'g-', linewidth=3, 
                    label='Dynamic Upper Limit', alpha=0.9)
                ax.plot(theoretical_speeds, l3_theory_bottom, 'g--', linewidth=3, 
                    label='Dynamic Lower Limit', alpha=0.9)
                
                # Q2、Q4象限的固定水平线（更突出显示）
                ax.axhline(y=torque_l3_top, color='red', linestyle='-', linewidth=3, alpha=0.8,
                        label=f'Q2 Fixed Limit ({torque_l3_top:.1f})')
                ax.axhline(y=torque_l3_bottom, color='red', linestyle='-', linewidth=3, alpha=0.8,
                        label=f'Q4 Fixed Limit ({torque_l3_bottom:.1f})')
                
                # 修改：添加独立的象限阈值线
                ax.axvline(x=speed_threshold_l3_q1, color='orange', linestyle=':', linewidth=2, alpha=0.8, 
                        label=f'Q1 Threshold ({speed_threshold_l3_q1})')
                ax.axvline(x=-speed_threshold_l3_q3, color='purple', linestyle=':', linewidth=2, alpha=0.8,
                        label=f'Q3 Threshold ({speed_threshold_l3_q3})')
                ax.axvline(x=max_speed_l3, color='brown', linestyle=':', linewidth=2, alpha=0.8, 
                        label=f'Max Speed (±{max_speed_l3})')
                ax.axvline(x=-max_speed_l3, color='brown', linestyle=':', linewidth=2, alpha=0.8)
                
            elif joint_idx in [3, 9]:  # L4 joints
                ax.plot(theoretical_speeds, l4_theory_top, 'g-', linewidth=3, 
                    label='Dynamic Upper Limit', alpha=0.9)
                ax.plot(theoretical_speeds, l4_theory_bottom, 'g--', linewidth=3, 
                    label='Dynamic Lower Limit', alpha=0.9)
                
                # Q2、Q4象限的固定水平线
                ax.axhline(y=torque_l4_top, color='red', linestyle='-', linewidth=3, alpha=0.8,
                        label=f'Q2 Fixed Limit ({torque_l4_top:.1f})')
                ax.axhline(y=torque_l4_bottom, color='red', linestyle='-', linewidth=3, alpha=0.8,
                        label=f'Q4 Fixed Limit ({torque_l4_bottom:.1f})')
                
                # 修改：添加独立的象限阈值线
                ax.axvline(x=speed_threshold_l4_q1, color='orange', linestyle=':', linewidth=2, alpha=0.8, 
                        label=f'Q1 Threshold ({speed_threshold_l4_q1})')
                ax.axvline(x=-speed_threshold_l4_q3, color='purple', linestyle=':', linewidth=2, alpha=0.8,
                        label=f'Q3 Threshold ({speed_threshold_l4_q3})')
                ax.axvline(x=max_speed_l4, color='brown', linestyle=':', linewidth=2, alpha=0.8, 
                        label=f'Max Speed (±{max_speed_l4})')
                ax.axvline(x=-max_speed_l4, color='brown', linestyle=':', linewidth=2, alpha=0.8)
            
            # 添加象限分界线
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.5, linewidth=1)
            ax.axvline(x=0, color='k', linestyle='-', alpha=0.5, linewidth=1)
            
            ax.set_xlabel('Joint Velocity [rad/s]')
            ax.set_ylabel('Joint Torque [Nm]')
            ax.set_title(f'{joint_labels[idx]} - Asymmetric Four Quadrant Model')
            ax.legend(fontsize=8, loc='best', ncol=2)
            ax.grid(True, alpha=0.3)
            
            # 设置合理的坐标轴范围
            ax.set_xlim(-15, 15)
            if joint_idx in [2, 8]:  # L3
                ax.set_ylim(-70, 70)
            else:  # L4
                ax.set_ylim(-200, 100)
            
            # 修改：添加非对称四象限模型说明
            if joint_idx in [2, 8]:  # L3
                model_text = f'Asymmetric Four-Quadrant:\nQ1: Q1_thresh={speed_threshold_l3_q1}\nQ3: Q3_thresh={speed_threshold_l3_q3}\nQ2,Q4: Fixed'
            else:  # L4
                model_text = f'Asymmetric Four-Quadrant:\nQ1: Q1_thresh={speed_threshold_l4_q1}\nQ3: Q3_thresh={speed_threshold_l4_q3}\nQ2,Q4: Fixed'
                
            ax.text(0.98, 0.02, model_text, transform=ax.transAxes, 
                verticalalignment='bottom', horizontalalignment='right', fontsize=8, 
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
        
        plt.tight_layout()
        
        # 保存详细对比图
        detailed_filename = filename.replace('.png', '_detailed_asymmetric_four_quadrant.png')
        plt.savefig(os.path.join(self.save_dir, detailed_filename), dpi=300, bbox_inches='tight')
        
        plt.close('all')  # 关闭所有图形
        print(f"Asymmetric four-quadrant torque-velocity curves saved: {filename} and {detailed_filename}")

    def plot_individual_torque_velocity_analysis(self, mujoco_data, isaac_data):
        """详细分析每个关节的力矩-速度特性"""
        num_joints = 12
        joint_vel_muj = mujoco_data[:, num_joints:2*num_joints]
        action_muj = mujoco_data[:, 2*num_joints:3*num_joints]
        
        joint_vel_isaac = isaac_data[:, num_joints:2*num_joints]
        action_isaac = isaac_data[:, 2*num_joints:3*num_joints]
        
           # 修正关节名称数组
        joint_names = [
            'leg_l1', 'leg_l2', 'leg_l3', 'leg_l4', 'leg_l5', 'leg_l6',
            'leg_r1', 'leg_r2', 'leg_r3', 'leg_r4', 'leg_r5', 'leg_r6'
        ]
        
        # 选择几个关键关节进行详细分析
        key_joints = [2, 3, 8, 9]  # leg_l3, leg_l4, leg_r3, leg_r4
        joint_labels = ['Left Hip Pitch (l3)', 'Left Knee (l4)', 'Right Hip Pitch (r3)', 'Right Knee (r4)']
    
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Detailed Torque-Velocity Analysis for Key Joints', fontsize=16)
        
        for idx, joint_idx in enumerate(key_joints):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]
            
            # 计算速度区间的平均力矩
            vel_bins = np.linspace(-10, 10, 20)
            muj_torque_means = []
            isaac_torque_means = []
            bin_centers = []
            
            for i in range(len(vel_bins)-1):
                vel_min, vel_max = vel_bins[i], vel_bins[i+1]
                
                # Mujoco数据
                mask_muj = (joint_vel_muj[:, joint_idx] >= vel_min) & (joint_vel_muj[:, joint_idx] < vel_max)
                if np.sum(mask_muj) > 0:
                    muj_torque_means.append(np.mean(action_muj[mask_muj, joint_idx]))
                else:
                    muj_torque_means.append(np.nan)
                
                # Isaac Gym数据
                mask_isaac = (joint_vel_isaac[:, joint_idx] >= vel_min) & (joint_vel_isaac[:, joint_idx] < vel_max)
                if np.sum(mask_isaac) > 0:
                    isaac_torque_means.append(np.mean(action_isaac[mask_isaac, joint_idx]))
                else:
                    isaac_torque_means.append(np.nan)
                
                bin_centers.append((vel_min + vel_max) / 2)
            
            # 绘制平均力矩曲线
            ax.plot(bin_centers, muj_torque_means, 'o-', label='Real', color='blue', linewidth=2)
            ax.plot(bin_centers, isaac_torque_means, 's-', label='Simdata in Isaac', color='red', linewidth=2)
            
            # 添加散点图作为背景
            ax.scatter(joint_vel_muj[:, joint_idx], action_muj[:, joint_idx], 
                    alpha=0.1, s=0.5, color='blue')
            ax.scatter(joint_vel_isaac[:, joint_idx], action_isaac[:, joint_idx], 
                    alpha=0.1, s=0.5, color='red')
            
            ax.set_xlabel('Joint Velocity [rad/s]')
            ax.set_ylabel('Average Joint Torque [Nm]')
            ax.set_title(f'{joint_labels[idx]}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'detailed_torque_velocity_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def save_and_visualize_data(self, real_data, sim_data, params=None):
        """修正函数参数名称和数据处理逻辑"""
        print("=== 函数入口数据检查 ===")

        torque_real_entry = real_data[:, -12:]
        torque_sim_entry = sim_data[:, -12:]
        

            # 检查入口处的speeds数据（基座速度和世界速度）
        # print(f"\n=== 入口speeds数据检查 ===")
        
        # # 基座速度数据 (第36-39列)
        # base_vel_real_entry = real_data[:, 36:39]
        # base_vel_sim_entry = sim_data[:, 36:39]
        # print(f"入口基座速度real数据:")
        # print(f"  形状: {base_vel_real_entry.shape}")
        # print(f"  范围: [{base_vel_real_entry.min():.6f}, {base_vel_real_entry.max():.6f}]")
        # print(f"  前5行数据:")
        # print(base_vel_real_entry[:5])
        
        # print(f"入口基座速度sim数据:")
        # print(f"  形状: {base_vel_sim_entry.shape}")
        # print(f"  范围: [{base_vel_sim_entry.min():.6f}, {base_vel_sim_entry.max():.6f}]")
        # print(f"  前5行数据:")
        # print(base_vel_sim_entry[:5])
        
       
    
        
        # 切片数据（去掉前200步）
        real_data = real_data[200:]
        sim_data = sim_data[200:]
        
        # print(f"\n=== 切片后数据检查 ===")
        # print(f"切片后real_data形状: {real_data.shape}")
        # print(f"切片后real_data范围: [{real_data.min():.6f}, {real_data.max():.6f}]")
        
        # 切片后再次检查关节位置
        joint_pos_real_after_slice = real_data[:, :12]
        #print(f"切片后joint_pos_real范围: [{joint_pos_real_after_slice.min():.6f}, {joint_pos_real_after_slice.max():.6f}]")
        
        # 保存原始数据
        np.save(os.path.join(self.save_dir, 'real_data.npy'), real_data)
        np.save(os.path.join(self.save_dir, 'sim_data.npy'), sim_data)

        # 如果有对齐后的数据，使用对齐后的
        if hasattr(self, 'last_aligned_sim_data') and hasattr(self, 'last_aligned_real_data'):
            print(f"\n=== 使用对齐后的数据 ===")
            print(f"对齐前real_data形状: {real_data.shape}")
            print(f"对齐前real_data joint_pos范围: [{real_data[:, :12].min():.6f}, {real_data[:, :12].max():.6f}]")
            
            real_data = self.last_aligned_real_data
            sim_data = self.last_aligned_sim_data
            
            print(f"对齐后real_data形状: {real_data.shape}")
            print(f"对齐后real_data joint_pos范围: [{real_data[:, :12].min():.6f}, {real_data[:, :12].max():.6f}]")
            
            np.save(os.path.join(self.save_dir, 'sim_data_aligned.npy'), sim_data)
            np.save(os.path.join(self.save_dir, 'real_data_aligned.npy'), real_data)
            print(f"已保存对齐后的数据，时移: {getattr(self, 'last_delay', None)}")

        if params is not None:
            np.save(os.path.join(self.save_dir, 'best_params.npy'), params)

        # 数据解析
        num_joints = 12
        joint_names = [
            'leg_l1', 'leg_l2', 'leg_l3', 'leg_l4', 'leg_l5', 'leg_l6',
            'leg_r1', 'leg_r2', 'leg_r3', 'leg_r4', 'leg_r5', 'leg_r6'
        ]
        
        # 数据提取
        joint_pos_real = real_data[:, :num_joints]
        joint_vel_real = real_data[:, num_joints:2*num_joints] 
        action_real = real_data[:, 2*num_joints:3*num_joints]
        
        joint_pos_sim = sim_data[:, :num_joints]
        joint_vel_sim = sim_data[:, num_joints:2*num_joints]
        action_sim = sim_data[:, 2*num_joints:3*num_joints]

        
        
        # 基座和世界线速度
        base_lin_vel_real = real_data[:, 36:39]
        base_lin_vel_sim = sim_data[:, 36:39]
        world_lin_vel_real = real_data[:, 39:42]
        world_lin_vel_sim = sim_data[:, 39:42]

        # 调用绘图函数（修正参数名称）
        self._plot_joint_comparison(joint_pos_real, joint_pos_sim, joint_names, 
                                'Joint Positions (Real vs Sim)', 'joint_positions_comparison.png')
        self._plot_joint_comparison(joint_vel_real, joint_vel_sim, joint_names,
                                'Joint Velocities (Real vs Sim)', 'joint_velocities_comparison.png')
        self._plot_joint_comparison(action_real, action_sim, joint_names,
                                'Actions (Real vs Sim)', 'actions_comparison.png')
        
        vel_names = ['vx', 'vy', 'vz']
        self._plot_velocity_comparison(base_lin_vel_real, base_lin_vel_sim, vel_names,
                                'Base Linear Velocity', 'base_linear_velocity_comparison.png')
        self._plot_velocity_comparison(world_lin_vel_real, world_lin_vel_sim, vel_names,
                                'World Linear Velocity', 'world_linear_velocity_comparison.png')
        
        
        
    
        # 其他绘图函数...
        self.plot_torque_velocity_curves(real_data, sim_data)
        self.plot_individual_torque_velocity_analysis(real_data, sim_data)
    
        # 新增：绘制扭矩对比图
        self.plot_mujoco_isaac_torque_comparison(real_data, sim_data)
    
        self._plot_distribution_comparison(real_data, sim_data)
        self._generate_data_report(real_data, sim_data, params)
        print(f"数据和图表已保存到: {self.save_dir}")


    def _plot_velocity_comparison(self, muj_data, isaac_data, vel_names, title, filename):
        """绘制速度数据对比图"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'{title} Comparison: Mujoco vs Isaac Gym', fontsize=16)
        
        time_steps_muj = np.arange(len(muj_data))
        time_steps_isaac = np.arange(len(isaac_data))
        
        for i in range(3):
            ax = axes[i]
            ax.plot(time_steps_muj, muj_data[:, i], 'b-', label='Real', alpha=0.7, linewidth=1.5)
            ax.plot(time_steps_isaac, isaac_data[:, i], 'r--', label='Simdata in Isaac', alpha=0.7, linewidth=1.5)
            
            ax.set_title(f'{vel_names[i]}', fontsize=12)
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Velocity (m/s)' if 'Linear' in title else 'Angular Velocity (rad/s)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # 计算相关性
            min_len = min(len(muj_data), len(isaac_data))
            correlation = np.corrcoef(muj_data[:min_len, i], isaac_data[:min_len, i])[0, 1]
            ax.text(0.02, 0.98, f'Corr: {correlation:.3f}', transform=ax.transAxes, 
                verticalalignment='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_joint_comparison(self, muj_data, isaac_data, joint_names, title, filename):
        """绘制关节数据对比图"""
        num_joints = len(joint_names)
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        fig.suptitle(f'{title} Comparison: Mujoco vs Isaac Gym', fontsize=16)
        
        time_steps_muj = np.arange(len(muj_data))
        time_steps_isaac = np.arange(len(isaac_data))
        
        for i in range(num_joints):
            row = i // 4
            col = i % 4
            ax = axes[row, col]
            
            # 绘制两条曲线
            ax.plot(time_steps_muj, muj_data[:, i], 'b-', label='Real', alpha=0.7, linewidth=1.5)
            ax.plot(time_steps_isaac, isaac_data[:, i], 'r--', label='Simdata in Isaac', alpha=0.7, linewidth=1.5)
            
            ax.set_title(f'{joint_names[i]}', fontsize=10)
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            
            # 计算相关性
            min_len = min(len(muj_data), len(isaac_data))
            correlation = np.corrcoef(muj_data[:min_len, i], isaac_data[:min_len, i])[0, 1]
            ax.text(0.02, 0.98, f'Corr: {correlation:.3f}', transform=ax.transAxes, 
                verticalalignment='top', fontsize=8, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_distribution_comparison(self, muj_data, isaac_data):
        """绘制数据分布对比"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Data Distribution Comparison', fontsize=16)
        
        # 选择几个代表性的维度进行分布比较
        dimensions = [3, 9, 15, 21, 27, 33]  # 每种数据类型选2个关节
        dim_names = ['L1_pos', 'R1_pos', 'L1_vel', 'R1_vel', 'L1_action', 'R1_action']
        
        for idx, (dim, name) in enumerate(zip(dimensions, dim_names)):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            # 绘制直方图
            ax.hist(muj_data[:, dim], bins=50, alpha=0.6, label='Real', color='blue', density=True)
            ax.hist(isaac_data[:, dim], bins=50, alpha=0.6, label='Simdata in Isaac', color='red', density=True)
            
            ax.set_title(f'{name} Distribution')
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'distribution_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _generate_data_report(self, muj_data, isaac_data, params):
        """生成数据分析报告"""
        min_len = min(len(muj_data), len(isaac_data))
        muj_data_aligned = muj_data[:min_len]
        isaac_data_aligned = isaac_data[:min_len]
        
        # 计算各种统计指标
        mse = np.mean((muj_data_aligned - isaac_data_aligned)**2, axis=0)
        mae = np.mean(np.abs(muj_data_aligned - isaac_data_aligned), axis=0)
        correlations = [np.corrcoef(muj_data_aligned[:, i], isaac_data_aligned[:, i])[0, 1] 
                       for i in range(muj_data_aligned.shape[1])]
        
        # 获取最后一次计算的距离分数
        last_score = getattr(self, 'last_distance_score', 'N/A')
        if isinstance(last_score, float):
            last_score_str = f"{last_score:.6f}"
        else:
            last_score_str = last_score
        # 生成报告
        report = f"""
# Data Comparison Report
Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Optimization Results
Best Parameters: {params}
Distance Score (wd + mmd): {last_score_str}
Data Length: Mujoco={len(muj_data)}, Isaac Gym={len(isaac_data)}

## Statistical Summary
Mean MSE across all dimensions: {np.mean(mse):.6f}
Mean MAE across all dimensions: {np.mean(mae):.6f}
Mean Correlation across all dimensions: {np.mean(correlations):.6f}

## Dimension-wise Analysis
{'Dim':<4} {'MSE':<12} {'MAE':<12} {'Correlation':<12} {'Type':<15}
{'-'*70}
"""
        
        dim_types = ['pos']*12 + ['vel']*12 + ['action']*12 + ['base_vel']*3 + ['world_vel']*3
        for i in range(min(len(mse), 42)):  # 加上速度为39维
            report += f"{i:<4} {mse[i]:<12.6f} {mae[i]:<12.6f} {correlations[i]:<12.6f} {dim_types[i]:<15}\n"
        
        # 保存报告
        with open(os.path.join(self.save_dir, 'comparison_report.txt'), 'w') as f:
            f.write(report)
        
        print(f"生成的分析报告:")
        print(f"- 平均MSE: {np.mean(mse):.6f}")
        print(f"- 平均MAE: {np.mean(mae):.6f}")
        print(f"- 平均相关性: {np.mean(correlations):.6f}")

    # def update_friction_coeffs(self, env, params):
    #     """更新环境的关节摩擦系数 - 只优化 l3, l4, r3, r4"""
    #     # 只更新特定关节的摩擦系数
    #     env.joint_friction_coeffs[2] = params['joint_friction']  # leg_l3
    #     env.joint_friction_coeffs[3] = params['joint_friction']  # leg_l4
    #     env.joint_friction_coeffs[8] = params['joint_friction']  # leg_r3
    #     env.joint_friction_coeffs[9] = params['joint_friction']  # leg_r4
        
    #     # 立即应用到所有环境
    #     env_ids = torch.arange(env.num_envs, device=env.device)
    #     env.refresh_actor_dof_shape_props(env_ids)

    def update_friction_coeffs(self, env, params):
        """更新环境的关节摩擦系数 - 使用统一值"""
        if hasattr(env, 'joint_friction_coeffs'):
            # 使用l3的摩擦系数作为统一值
            friction_value = params.get('joint_friction_l3', params.get('joint_friction', 0.007))
            env.joint_friction_coeffs[:] = friction_value
            
            # 立即应用到所有环境
            env_ids = torch.arange(env.num_envs, device=env.device)
            if hasattr(env, 'refresh_actor_dof_shape_props'):
                env.refresh_actor_dof_shape_props(env_ids)
        else:
            print("Warning: Environment does not support friction coefficient updates")
            
    def update_torque_limits(self, env, params):
        """更新环境的关节扭矩限制 - 只优化 l3, l4, r3, r4"""
        dof_props = env.gym.get_actor_dof_properties(env.envs[0], 0)
        
        # l3 (髋关节俯仰) - 索引 2, 8
        effort_limit_l3 = max(abs(params['torque_l3_top']), abs(params['torque_l3_bottom']))
        dof_props['effort'][2] = effort_limit_l3  # leg_l3
        dof_props['effort'][8] = effort_limit_l3  # leg_r3
        
        # l4 (膝关节) - 索引 3, 9  
        effort_limit_l4 = max(abs(params['torque_l4_top']), abs(params['torque_l4_bottom']))
        dof_props['effort'][3] = effort_limit_l4  # leg_l4
        dof_props['effort'][9] = effort_limit_l4  # leg_r4

        # 更新 torque_limits 张量
        env.torque_limits[2, 0] = params['torque_l3_bottom']  # leg_l3 最小扭矩
        env.torque_limits[2, 1] = params['torque_l3_top']     # leg_l3 最大扭矩
        env.torque_limits[8, 0] = params['torque_l3_bottom']  # leg_r3 最小扭矩
        env.torque_limits[8, 1] = params['torque_l3_top']     # leg_r3 最大扭矩
        
        env.torque_limits[3, 0] = params['torque_l4_bottom']  # leg_l4 最小扭矩
        env.torque_limits[3, 1] = params['torque_l4_top']     # leg_l4 最大扭矩
        env.torque_limits[9, 0] = params['torque_l4_bottom']  # leg_r4 最小扭矩
        env.torque_limits[9, 1] = params['torque_l4_top']     # leg_r4 最大扭矩
        print(f"Updated torque limits: l3 [{params['torque_l3_bottom']}, {params['torque_l3_top']}], ")
        # 应用到所有环境
        for i in range(env.num_envs):
            env.gym.set_actor_dof_properties(env.envs[i], 0, dof_props)

    # def update_dynamic_torque_limits(self, env, params):
    #     """
    #     更新环境的动态关节扭矩限制 - 简化四象限模型
    #     Q1, Q3: 根据速度动态调整（驱动象限）
    #     Q2: 固定正扭矩上限水平线（制动象限）
    #     Q4: 固定负扭矩下限水平线（制动象限）
        
    #     Args:
    #         env: Isaac Gym环境
    #         params: 参数字典，包含：
    #             - torque_l3_top/bottom: l3关节的扭矩上下限
    #             - torque_l4_top/bottom: l4关节的扭矩上下限  
    #             - speed_threshold_l3: l3关节的速度阈值
    #             - speed_threshold_l4: l4关节的速度阈值
    #             - max_speed_l3: l3关节的最大速度
    #             - max_speed_l4: l4关节的最大速度
    #             - q2_torque_limit: Q2象限固定扭矩上限 (可选)
    #             - q4_torque_limit: Q4象限固定扭矩下限 (可选)
    #     """
    #     import torch
        
    #     # 获取当前关节速度
    #     current_joint_vel = env.dof_vel  # shape: (num_envs, num_dofs)
        
    #     # 提取参数
    #     torque_l3_top = params.get('torque_l3_top', 35.0)
    #     torque_l3_bottom = params.get('torque_l3_bottom', -35.0)
    #     torque_l4_top = params.get('torque_l4_top', 150.0)
    #     torque_l4_bottom = params.get('torque_l4_bottom', -150.0)
        
    #     speed_threshold_l3 = params.get('speed_threshold_l3', 5.0)  # rad/s
    #     speed_threshold_l4 = params.get('speed_threshold_l4', 7.0)  # rad/s
        
    #     max_speed_l3 = params.get('max_speed_l3', 10.0)  # rad/s
    #     max_speed_l4 = params.get('max_speed_l4', 12.0)  # rad/s
        
    #     # 制动象限的固定扭矩限制（可以设为较小值）
    #     q2_torque_limit_l3 = params.get('q2_torque_limit_l3', torque_l3_top * 0.7)  # Q2象限固定上限
    #     q4_torque_limit_l3 = params.get('q4_torque_limit_l3', torque_l3_bottom * 0.7)  # Q4象限固定下限
    #     q2_torque_limit_l4 = params.get('q2_torque_limit_l4', torque_l4_top * 0.7)
    #     q4_torque_limit_l4 = params.get('q4_torque_limit_l4', torque_l4_bottom * 0.7)
        
    #     # 调试信息（减少频率）
    #     if not hasattr(self, '_dynamic_torque_print_counter'):
    #         self._dynamic_torque_print_counter = 0
        
    #     def calculate_simplified_four_quadrant_torque_limits(joint_vel, fixed_torque_top, fixed_torque_bottom, 
    #                                                     speed_threshold, max_speed, q2_limit, q4_limit):
    #         """
    #         计算简化的四象限动态扭矩限制
            
    #         Args:
    #             joint_vel: 关节速度 (tensor, 可以是正负值)
    #             fixed_torque_top: 固定正向扭矩限制（Q1象限用）
    #             fixed_torque_bottom: 固定负向扭矩限制（Q3象限用）
    #             speed_threshold: 速度阈值
    #             max_speed: 最大速度
    #             q2_limit: Q2象限固定正扭矩限制
    #             q4_limit: Q4象限固定负扭矩限制
                
    #         Returns:
    #             (dynamic_torque_top, dynamic_torque_bottom): 动态扭矩上下限
    #         """
    #         abs_vel = torch.abs(joint_vel)
            
    #         # 避免除零错误
    #         if max_speed <= speed_threshold:
    #             return torch.tensor(fixed_torque_top, device=joint_vel.device), torch.tensor(fixed_torque_bottom, device=joint_vel.device)
            
    #         # 根据速度方向确定象限
    #         if joint_vel >= 0:
    #             # 正速度：Q1和Q4象限
                
    #             # Q1象限：正速度 + 正扭矩（驱动）- 使用动态扭矩限制
    #             if abs_vel < speed_threshold:
    #                 q1_torque_top = fixed_torque_top
    #             else:
    #                 # 线性衰减
    #                 scale_factor = max(0.0, 1.0 - (abs_vel - speed_threshold) / (max_speed - speed_threshold))
    #                 q1_torque_top = fixed_torque_top * scale_factor
                
    #             # Q4象限：正速度 + 负扭矩（制动）- 使用固定扭矩限制
    #             q4_torque_bottom = q4_limit  # 固定水平线
                
    #             return q1_torque_top, q4_torque_bottom
                
    #         else:
    #             # 负速度：Q2和Q3象限
                
    #             # Q2象限：负速度 + 正扭矩（制动）- 使用固定扭矩限制
    #             q2_torque_top = q2_limit  # 固定水平线
                
    #             # Q3象限：负速度 + 负扭矩（驱动）- 使用动态扭矩限制
    #             if abs_vel < speed_threshold:
    #                 q3_torque_bottom = fixed_torque_bottom
    #             else:
    #                 # 线性衰减
    #                 scale_factor = max(0.0, 1.0 - (abs_vel - speed_threshold) / (max_speed - speed_threshold))
    #                 q3_torque_bottom = fixed_torque_bottom * scale_factor
                
    #             return q2_torque_top, q3_torque_bottom
        
    #     # 检查 torque_limits 的维度并相应处理
    #     if len(env.torque_limits.shape) == 2:
    #         # torque_limits shape: (num_dofs, 2) - 没有环境维度
    #         if self._dynamic_torque_print_counter == 0:
    #             print("Using 2D torque_limits (num_dofs, 2) with simplified four-quadrant control")
    #             print("Q1, Q3: Dynamic limits based on speed")
    #             print("Q2, Q4: Fixed horizontal limits for braking")
            
    #         # 计算所有环境的平均速度或使用第一个环境的速度
    #         if current_joint_vel.shape[0] > 1:
    #             reference_vel = current_joint_vel[0]
    #         else:
    #             reference_vel = current_joint_vel[0]
            
    #         # l3关节 (索引 2, 8)
    #         l3_left_vel = reference_vel[2]  # leg_l3
    #         l3_right_vel = reference_vel[8]  # leg_r3
            
    #         # 计算l3关节的简化四象限动态扭矩限制
    #         l3_left_top, l3_left_bottom = calculate_simplified_four_quadrant_torque_limits(
    #             l3_left_vel, torque_l3_top, torque_l3_bottom, speed_threshold_l3, max_speed_l3,
    #             q2_torque_limit_l3, q4_torque_limit_l3
    #         )
    #         l3_right_top, l3_right_bottom = calculate_simplified_four_quadrant_torque_limits(
    #             l3_right_vel, torque_l3_top, torque_l3_bottom, speed_threshold_l3, max_speed_l3,
    #             q2_torque_limit_l3, q4_torque_limit_l3
    #         )
            
    #         # l4关节 (索引 3, 9)
    #         l4_left_vel = reference_vel[3]  # leg_l4
    #         l4_right_vel = reference_vel[9]  # leg_r4
            
    #         # 计算l4关节的简化四象限动态扭矩限制
    #         l4_left_top, l4_left_bottom = calculate_simplified_four_quadrant_torque_limits(
    #             l4_left_vel, torque_l4_top, torque_l4_bottom, speed_threshold_l4, max_speed_l4,
    #             q2_torque_limit_l4, q4_torque_limit_l4
    #         )
    #         l4_right_top, l4_right_bottom = calculate_simplified_four_quadrant_torque_limits(
    #             l4_right_vel, torque_l4_top, torque_l4_bottom, speed_threshold_l4, max_speed_l4,
    #             q2_torque_limit_l4, q4_torque_limit_l4
    #         )
            
    #         # 更新环境的扭矩限制张量 (2D版本)
    #         env.torque_limits[2, 0] = l3_left_bottom    # leg_l3 最小扭矩
    #         env.torque_limits[2, 1] = l3_left_top       # leg_l3 最大扭矩
    #         env.torque_limits[8, 0] = l3_right_bottom   # leg_r3 最小扭矩
    #         env.torque_limits[8, 1] = l3_right_top      # leg_r3 最大扭矩
            
    #         env.torque_limits[3, 0] = l4_left_bottom    # leg_l4 最小扭矩
    #         env.torque_limits[3, 1] = l4_left_top       # leg_l4 最大扭矩
    #         env.torque_limits[9, 0] = l4_right_bottom   # leg_r4 最小扭矩
    #         env.torque_limits[9, 1] = l4_right_top      # leg_r4 最大扭矩
            
    #         # 调试输出（每100步一次）
    #         if self._dynamic_torque_print_counter % 100 == 0:
    #             # print(f"Simplified four-quadrant torque limits updated:")
    #             # print(f"  L3 Left  - vel: {l3_left_vel:.3f}, limits: [{l3_left_bottom:.1f}, {l3_left_top:.1f}]")
    #             # print(f"  L3 Right - vel: {l3_right_vel:.3f}, limits: [{l3_right_bottom:.1f}, {l3_right_top:.1f}]")
    #             # print(f"  L4 Left  - vel: {l4_left_vel:.3f}, limits: [{l4_left_bottom:.1f}, {l4_left_top:.1f}]")
    #             # print(f"  L4 Right - vel: {l4_right_vel:.3f}, limits: [{l4_right_bottom:.1f}, {l4_right_top:.1f}]")
                
    #             # 显示当前是哪个象限
    #             def get_quadrant(vel, torque_top, torque_bottom):
    #                 if vel >= 0:
    #                     return "Q1 (Drive)" if torque_top > abs(torque_bottom) else "Q4 (Brake)"
    #                 else:
    #                     return "Q2 (Brake)" if torque_top > abs(torque_bottom) else "Q3 (Drive)"
                
    #             print(f"  Current quadrants - L3L: {get_quadrant(l3_left_vel, l3_left_top, l3_left_bottom)}, "
    #                 f"L3R: {get_quadrant(l3_right_vel, l3_right_top, l3_right_bottom)}")
            
    #     elif len(env.torque_limits.shape) == 3:
    #         # torque_limits shape: (num_envs, num_dofs, 2) - 有环境维度
    #         if self._dynamic_torque_print_counter == 0:
    #             print("Using 3D torque_limits (num_envs, num_dofs, 2) with simplified four-quadrant control")
            
    #         # 更新每个环境的扭矩限制
    #         for env_idx in range(env.num_envs):
    #             # l3关节 (索引 2, 8)
    #             l3_left_vel = current_joint_vel[env_idx, 2]  # leg_l3
    #             l3_right_vel = current_joint_vel[env_idx, 8]  # leg_r3
                
    #             # 计算l3关节的简化四象限动态扭矩限制
    #             l3_left_top, l3_left_bottom = calculate_simplified_four_quadrant_torque_limits(
    #                 l3_left_vel, torque_l3_top, torque_l3_bottom, speed_threshold_l3, max_speed_l3,
    #                 q2_torque_limit_l3, q4_torque_limit_l3
    #             )
    #             l3_right_top, l3_right_bottom = calculate_simplified_four_quadrant_torque_limits(
    #                 l3_right_vel, torque_l3_top, torque_l3_bottom, speed_threshold_l3, max_speed_l3,
    #                 q2_torque_limit_l3, q4_torque_limit_l3
    #             )
                
    #             # l4关节 (索引 3, 9)
    #             l4_left_vel = current_joint_vel[env_idx, 3]  # leg_l4
    #             l4_right_vel = current_joint_vel[env_idx, 9]  # leg_r4
                
    #             # 计算l4关节的简化四象限动态扭矩限制
    #             l4_left_top, l4_left_bottom = calculate_simplified_four_quadrant_torque_limits(
    #                 l4_left_vel, torque_l4_top, torque_l4_bottom, speed_threshold_l4, max_speed_l4,
    #                 q2_torque_limit_l4, q4_torque_limit_l4
    #             )
    #             l4_right_top, l4_right_bottom = calculate_simplified_four_quadrant_torque_limits(
    #                 l4_right_vel, torque_l4_top, torque_l4_bottom, speed_threshold_l4, max_speed_l4,
    #                 q2_torque_limit_l4, q4_torque_limit_l4
    #             )
                
    #             # 更新环境的扭矩限制张量 (3D版本)
    #             env.torque_limits[env_idx, 2, 0] = l3_left_bottom    # leg_l3 最小扭矩
    #             env.torque_limits[env_idx, 2, 1] = l3_left_top       # leg_l3 最大扭矩
    #             env.torque_limits[env_idx, 8, 0] = l3_right_bottom   # leg_r3 最小扭矩
    #             env.torque_limits[env_idx, 8, 1] = l3_right_top      # leg_r3 最大扭矩
                
    #             env.torque_limits[env_idx, 3, 0] = l4_left_bottom    # leg_l4 最小扭矩
    #             env.torque_limits[env_idx, 3, 1] = l4_left_top       # leg_l4 最大扭矩
    #             env.torque_limits[env_idx, 9, 0] = l4_right_bottom   # leg_r4 最小扭矩
    #             env.torque_limits[env_idx, 9, 1] = l4_right_top      # leg_r4 最大扭矩
                
    #     else:
    #         print(f"Unexpected torque_limits shape: {env.torque_limits.shape}")
    #         return
        
    #     self._dynamic_torque_print_counter += 1
    def update_dynamic_torque_limits(self, env, params):
        """
        更新环境的动态关节扭矩限制 - 支持不同象限的独立速度阈值
        
        Args:
            env: Isaac Gym环境
            params: 参数字典，包含：
                - torque_l3_top/bottom: l3关节的扭矩上下限
                - torque_l4_top/bottom: l4关节的扭矩上下限  
                - speed_threshold_l3_q1: l3关节Q1象限的速度阈值
                - speed_threshold_l3_q3: l3关节Q3象限的速度阈值
                - speed_threshold_l4_q1: l4关节Q1象限的速度阈值
                - speed_threshold_l4_q3: l4关节Q3象限的速度阈值
                - angle_vel_l3_top: l3关节的最大速度限制
                - angle_vel_l4_top: l4关节的最大速度限制
        """
        import torch
        
        # 获取当前关节速度
        current_joint_vel = env.dof_vel  # shape: (num_envs, num_dofs)
        
        # 提取参数
        torque_l3_top = params.get('torque_l3_top', 35.0)
        torque_l3_bottom = params.get('torque_l3_bottom', -35.0)
        torque_l4_top = params.get('torque_l4_top', 150.0)
        torque_l4_bottom = params.get('torque_l4_bottom', -150.0)
        
        # 独立的象限速度阈值
        speed_threshold_l3_q1 = params.get('speed_threshold_l3_q1', 5.0)  # Q1象限（正速度+正扭矩）
        speed_threshold_l3_q3 = params.get('speed_threshold_l3_q3', 5.0)  # Q3象限（负速度+负扭矩）
        speed_threshold_l4_q1 = params.get('speed_threshold_l4_q1', 7.0)  # Q1象限
        speed_threshold_l4_q3 = params.get('speed_threshold_l4_q3', 7.0)  # Q3象限
        
        # 最大速度限制（仍然统一）
        max_speed_l3 = abs(params.get('angle_vel_l3_top', 10.0))
        max_speed_l4 = abs(params.get('angle_vel_l4_top', 12.0))
        
        # 调试信息（减少频率）
        if not hasattr(self, '_dynamic_torque_print_counter'):
            self._dynamic_torque_print_counter = 0
        
        def calculate_asymmetric_dynamic_torque_limits(joint_vel, fixed_torque_top, fixed_torque_bottom, 
                                                    threshold_q1, threshold_q3, max_speed):
            """
            计算支持不同象限独立阈值的动态扭矩限制
            
            Args:
                joint_vel: 关节速度 (tensor, 可以是正负值)
                fixed_torque_top: 固定正向扭矩限制
                fixed_torque_bottom: 固定负向扭矩限制
                threshold_q1: Q1象限（正速度）的速度阈值
                threshold_q3: Q3象限（负速度）的速度阈值  
                max_speed: 最大速度（对应angle_vel_*_top）
                
            Returns:
                (dynamic_torque_top, dynamic_torque_bottom): 动态扭矩上下限
            """
            abs_vel = torch.abs(joint_vel)
            
            # 避免除零错误
            if max_speed <= max(threshold_q1, threshold_q3):
                return torch.tensor(fixed_torque_top, device=joint_vel.device), torch.tensor(fixed_torque_bottom, device=joint_vel.device)
            
            # 根据速度方向选择对应的阈值
            if joint_vel >= 0:
                # 正速度：使用Q1象限的阈值
                threshold = threshold_q1
            else:
                # 负速度：使用Q3象限的阈值  
                threshold = threshold_q3
            
            # 计算速度相关的扭矩衰减系数
            if abs_vel < threshold:
                # 低速区域：使用固定扭矩限制
                scale_factor = 1.0
            else:
                # 高速区域：线性衰减
                # 从 (threshold, 1.0) 线性衰减到 (max_speed, 0.0)
                scale_factor = max(0.0, 1.0 - (abs_vel - threshold) / (max_speed - threshold))
            
            # 计算动态扭矩限制
            dynamic_torque_top = fixed_torque_top * scale_factor
            dynamic_torque_bottom = fixed_torque_bottom * scale_factor
            
            return dynamic_torque_top, dynamic_torque_bottom
        
        # 检查 torque_limits 的维度并相应处理
        if len(env.torque_limits.shape) == 2:
            # torque_limits shape: (num_dofs, 2) - 没有环境维度
            if self._dynamic_torque_print_counter == 0:
                print("Using 2D torque_limits (num_dofs, 2) with asymmetric dynamic control")
                print("Q1 and Q3 quadrants use independent speed thresholds")
                print(f"L3: Q1_threshold={speed_threshold_l3_q1}, Q3_threshold={speed_threshold_l3_q3}, max_speed={max_speed_l3}")
                print(f"L4: Q1_threshold={speed_threshold_l4_q1}, Q3_threshold={speed_threshold_l4_q3}, max_speed={max_speed_l4}")
            
            # 计算所有环境的平均速度或使用第一个环境的速度
            if current_joint_vel.shape[0] > 1:
                reference_vel = current_joint_vel[0]
            else:
                reference_vel = current_joint_vel[0]
            
            # l3关节 (索引 2, 8)
            l3_left_vel = reference_vel[2]  # leg_l3
            l3_right_vel = reference_vel[8]  # leg_r3
            
            # 计算l3关节的非对称动态扭矩限制
            l3_left_top, l3_left_bottom = calculate_asymmetric_dynamic_torque_limits(
                l3_left_vel, torque_l3_top, torque_l3_bottom, 
                speed_threshold_l3_q1, speed_threshold_l3_q3, max_speed_l3
            )
            l3_right_top, l3_right_bottom = calculate_asymmetric_dynamic_torque_limits(
                l3_right_vel, torque_l3_top, torque_l3_bottom,
                speed_threshold_l3_q1, speed_threshold_l3_q3, max_speed_l3
            )
            
            # l4关节 (索引 3, 9)
            l4_left_vel = reference_vel[3]  # leg_l4
            l4_right_vel = reference_vel[9]  # leg_r4
            
            # 计算l4关节的非对称动态扭矩限制
            l4_left_top, l4_left_bottom = calculate_asymmetric_dynamic_torque_limits(
                l4_left_vel, torque_l4_top, torque_l4_bottom,
                speed_threshold_l4_q1, speed_threshold_l4_q3, max_speed_l4
            )
            l4_right_top, l4_right_bottom = calculate_asymmetric_dynamic_torque_limits(
                l4_right_vel, torque_l4_top, torque_l4_bottom,
                speed_threshold_l4_q1, speed_threshold_l4_q3, max_speed_l4
            )
            
            # 更新环境的扭矩限制张量 (2D版本)
            env.torque_limits[2, 0] = l3_left_bottom    # leg_l3 最小扭矩
            env.torque_limits[2, 1] = l3_left_top       # leg_l3 最大扭矩
            env.torque_limits[8, 0] = l3_right_bottom   # leg_r3 最小扭矩
            env.torque_limits[8, 1] = l3_right_top      # leg_r3 最大扭矩
            
            env.torque_limits[3, 0] = l4_left_bottom    # leg_l4 最小扭矩
            env.torque_limits[3, 1] = l4_left_top       # leg_l4 最大扭矩
            env.torque_limits[9, 0] = l4_right_bottom   # leg_r4 最小扭矩
            env.torque_limits[9, 1] = l4_right_top      # leg_r4 最大扭矩
            
        elif len(env.torque_limits.shape) == 3:
            # torque_limits shape: (num_envs, num_dofs, 2) - 有环境维度
            if self._dynamic_torque_print_counter == 0:
                print("Using 3D torque_limits (num_envs, num_dofs, 2) with asymmetric dynamic control")
                print(f"L3: Q1_threshold={speed_threshold_l3_q1}, Q3_threshold={speed_threshold_l3_q3}, max_speed={max_speed_l3}")
                print(f"L4: Q1_threshold={speed_threshold_l4_q1}, Q3_threshold={speed_threshold_l4_q3}, max_speed={max_speed_l4}")
            
            # 更新每个环境的扭矩限制
            for env_idx in range(env.num_envs):
                # l3关节 (索引 2, 8)
                l3_left_vel = current_joint_vel[env_idx, 2]  # leg_l3
                l3_right_vel = current_joint_vel[env_idx, 8]  # leg_r3
                
                # 计算l3关节的非对称动态扭矩限制
                l3_left_top, l3_left_bottom = calculate_asymmetric_dynamic_torque_limits(
                    l3_left_vel, torque_l3_top, torque_l3_bottom,
                    speed_threshold_l3_q1, speed_threshold_l3_q3, max_speed_l3
                )
                l3_right_top, l3_right_bottom = calculate_asymmetric_dynamic_torque_limits(
                    l3_right_vel, torque_l3_top, torque_l3_bottom,
                    speed_threshold_l3_q1, speed_threshold_l3_q3, max_speed_l3
                )
                
                # l4关节 (索引 3, 9)
                l4_left_vel = current_joint_vel[env_idx, 3]  # leg_l4
                l4_right_vel = current_joint_vel[env_idx, 9]  # leg_r4
                
                # 计算l4关节的非对称动态扭矩限制
                l4_left_top, l4_left_bottom = calculate_asymmetric_dynamic_torque_limits(
                    l4_left_vel, torque_l4_top, torque_l4_bottom,
                    speed_threshold_l4_q1, speed_threshold_l4_q3, max_speed_l4
                )
                l4_right_top, l4_right_bottom = calculate_asymmetric_dynamic_torque_limits(
                    l4_right_vel, torque_l4_top, torque_l4_bottom,
                    speed_threshold_l4_q1, speed_threshold_l4_q3, max_speed_l4
                )
                
                # 更新环境的扭矩限制张量 (3D版本)
                env.torque_limits[env_idx, 2, 0] = l3_left_bottom    # leg_l3 最小扭矩
                env.torque_limits[env_idx, 2, 1] = l3_left_top       # leg_l3 最大扭矩
                env.torque_limits[env_idx, 8, 0] = l3_right_bottom   # leg_r3 最小扭矩
                env.torque_limits[env_idx, 8, 1] = l3_right_top      # leg_r3 最大扭矩
                
                env.torque_limits[env_idx, 3, 0] = l4_left_bottom    # leg_l4 最小扭矩
                env.torque_limits[env_idx, 3, 1] = l4_left_top       # leg_l4 最大扭矩
                env.torque_limits[env_idx, 9, 0] = l4_right_bottom   # leg_r4 最小扭矩
                env.torque_limits[env_idx, 9, 1] = l4_right_top      # leg_r4 最大扭矩
                
        else:
            print(f"Unexpected torque_limits shape: {env.torque_limits.shape}")
            return
        
        self._dynamic_torque_print_counter += 1

    def calculate_asymmetric_four_quadrant_torque_curve(self, speeds, torque_top, torque_bottom, 
                                                        threshold_q1, threshold_q3, max_speed):
        """
        计算支持不同象限独立阈值的四象限动态扭矩曲线
        Q1: 正速度+正扭矩（驱动） - 使用threshold_q1
        Q2: 负速度+正扭矩（制动） - 使用固定的torque_top
        Q3: 负速度+负扭矩（驱动） - 使用threshold_q3
        Q4: 正速度+负扭矩（制动） - 使用固定的torque_bottom
        """
        torque_top_curve = []
        torque_bottom_curve = []
        
        # 调试：打印输入参数
        if not hasattr(self, '_curve_debug_printed'):
            print(f"计算理论曲线参数: torque_top={torque_top}, torque_bottom={torque_bottom}")
            print(f"threshold_q1={threshold_q1}, threshold_q3={threshold_q3}, max_speed={max_speed}")
            self._curve_debug_printed = True
        
        for speed in speeds:
            abs_speed = abs(speed)
            
            if speed >= 0:
                # 正速度：Q1和Q4象限
                
                # Q1象限：正速度 + 正扭矩（驱动）- 使用Q1阈值的动态扭矩限制
                if abs_speed < threshold_q1:
                    q1_torque_top = torque_top
                else:
                    # 线性衰减：从threshold_q1到max_speed，扭矩从torque_top衰减到0
                    if max_speed > threshold_q1:
                        scale_factor = max(0.0, 1.0 - (abs_speed - threshold_q1) / (max_speed - threshold_q1))
                    else:
                        scale_factor = 1.0 if abs_speed <= threshold_q1 else 0.0
                    q1_torque_top = torque_top * scale_factor
                
                # Q4象限：正速度 + 负扭矩（制动）- 使用固定扭矩限制
                q4_torque_bottom = torque_bottom  # 固定使用torque_bottom
                
                torque_top_curve.append(q1_torque_top)
                torque_bottom_curve.append(q4_torque_bottom)
                
            else:
                # 负速度：Q2和Q3象限
                
                # Q2象限：负速度 + 正扭矩（制动）- 使用固定扭矩限制
                q2_torque_top = torque_top  # 固定使用torque_top
                
                # Q3象限：负速度 + 负扭矩（驱动）- 使用Q3阈值的动态扭矩限制
                if abs_speed < threshold_q3:
                    q3_torque_bottom = torque_bottom
                else:
                    # 线性衰减：从threshold_q3到max_speed，扭矩从torque_bottom衰减到0
                    if max_speed > threshold_q3:
                        scale_factor = max(0.0, 1.0 - (abs_speed - threshold_q3) / (max_speed - threshold_q3))
                    else:
                        scale_factor = 1.0 if abs_speed <= threshold_q3 else 0.0
                    q3_torque_bottom = torque_bottom * scale_factor
                
                torque_top_curve.append(q2_torque_top)
                torque_bottom_curve.append(q3_torque_bottom)
        
        # 调试：打印一些关键点的值
        if not hasattr(self, '_curve_values_printed'):
            mid_idx = len(speeds) // 2  # 速度为0的点
            print(f"理论曲线关键点:")
            print(f"  速度=0: upper={torque_top_curve[mid_idx]}, lower={torque_bottom_curve[mid_idx]}")
            
            # 找到Q1阈值点
            q1_idx = np.argmin(np.abs(np.array(speeds) - threshold_q1))
            print(f"  速度={threshold_q1}: upper={torque_top_curve[q1_idx]}, lower={torque_bottom_curve[q1_idx]}")
            
            # 找到Q3阈值点  
            q3_idx = np.argmin(np.abs(np.array(speeds) + threshold_q3))
            print(f"  速度=-{threshold_q3}: upper={torque_top_curve[q3_idx]}, lower={torque_bottom_curve[q3_idx]}")
            
            self._curve_values_printed = True
        
        return np.array(torque_top_curve), np.array(torque_bottom_curve)

    def calculate_four_quadrant_torque_curve(self ,speeds, torque_top, torque_bottom, threshold, max_speed):
        """
        计算四象限动态扭矩曲线
        Q1: 正速度+正扭矩（驱动） - 动态限制
        Q2: 负速度+正扭矩（制动） - 使用固定的torque_top
        Q3: 负速度+负扭矩（驱动） - 动态限制  
        Q4: 正速度+负扭矩（制动） - 使用固定的torque_bottom
        """
        torque_top_curve = []
        torque_bottom_curve = []
        
        for speed in speeds:
            abs_speed = abs(speed)
            
            if speed >= 0:
                # 正速度：Q1和Q4象限
                
                # Q1象限：正速度 + 正扭矩（驱动）- 使用动态扭矩限制
                if abs_speed < threshold:
                    q1_torque_top = torque_top
                else:
                    # 线性衰减
                    scale_factor = max(0.0, 1.0 - (abs_speed - threshold) / (max_speed - threshold))
                    q1_torque_top = torque_top * scale_factor
                
                # Q4象限：正速度 + 负扭矩（制动）- 使用固定扭矩限制
                q4_torque_bottom = torque_bottom  # 固定使用torque_bottom
                
                torque_top_curve.append(q1_torque_top)
                torque_bottom_curve.append(q4_torque_bottom)
                
            else:
                # 负速度：Q2和Q3象限
                
                # Q2象限：负速度 + 正扭矩（制动）- 使用固定扭矩限制
                q2_torque_top = torque_top  # 固定使用torque_top
                
                # Q3象限：负速度 + 负扭矩（驱动）- 使用动态扭矩限制
                if abs_speed < threshold:
                    q3_torque_bottom = torque_bottom
                else:
                    # 线性衰减
                    scale_factor = max(0.0, 1.0 - (abs_speed - threshold) / (max_speed - threshold))
                    q3_torque_bottom = torque_bottom * scale_factor
                
                torque_top_curve.append(q2_torque_top)
                torque_bottom_curve.append(q3_torque_bottom)
        
        return np.array(torque_top_curve), np.array(torque_bottom_curve)

    def plot_all_real_data_torque_velocity_curves(self, all_real_data, sim_data=None, filename='all_real_data_torque_velocity_curves.png'):
        """
        绘制所有9条real_data的L3、L4、R3、R4关节扭矩-速度曲线
        包含非对称四象限扭矩限制理论曲线
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # 关节索引和名称
        key_joints = [2, 3, 8, 9]  # leg_l3, leg_l4, leg_r3, leg_r4
        joint_labels = ['Left Hip Pitch (L3)', 'Left Knee (L4)', 'Right Hip Pitch (R3)', 'Right Knee (R4)']
        
        # 获取当前优化参数（用于绘制理论曲线）
        params = getattr(self, 'best_params', {})
        
        # 修改：使用独立的象限速度阈值
        speed_threshold_l3_q1 = params.get('speed_threshold_l3_q1', 5.0)
        speed_threshold_l3_q3 = params.get('speed_threshold_l3_q3', 5.0)
        speed_threshold_l4_q1 = params.get('speed_threshold_l4_q1', 7.0)
        speed_threshold_l4_q3 = params.get('speed_threshold_l4_q3', 7.0)
        
        # 修改：使用独立阈值对应的最大速度，如果没有就使用angle_vel参数
        max_speed_l3 = abs(params.get('angle_vel_l3_top', 10.0))
        max_speed_l4 = abs(params.get('angle_vel_l4_top', 12.0))
        
        torque_l3_top = params.get('torque_l3_top', 35.0)
        torque_l3_bottom = params.get('torque_l3_bottom', -35.0)
        torque_l4_top = params.get('torque_l4_top', 150.0)
        torque_l4_bottom = params.get('torque_l4_bottom', -150.0)
        
        print(f"非对称四象限扭矩限制参数:")
        print(f"L3: Q1_thresh={speed_threshold_l3_q1}, Q3_thresh={speed_threshold_l3_q3}, max_speed={max_speed_l3}")
        print(f"L4: Q1_thresh={speed_threshold_l4_q1}, Q3_thresh={speed_threshold_l4_q3}, max_speed={max_speed_l4}")
        print(f"L3 torque: [{torque_l3_bottom}, {torque_l3_top}]")
        print(f"L4 torque: [{torque_l4_bottom}, {torque_l4_top}]")
        
        # 创建速度范围用于理论曲线
        theoretical_speeds = np.linspace(-15, 15, 300)  # 增加点数以获得更平滑的曲线
        
        # 修改：使用非对称四象限扭矩曲线计算函数
        l3_theory_top, l3_theory_bottom = self.calculate_asymmetric_four_quadrant_torque_curve(
            theoretical_speeds, torque_l3_top, torque_l3_bottom, 
            speed_threshold_l3_q1, speed_threshold_l3_q3, max_speed_l3
        )
        l4_theory_top, l4_theory_bottom = self.calculate_asymmetric_four_quadrant_torque_curve(
            theoretical_speeds, torque_l4_top, torque_l4_bottom,
            speed_threshold_l4_q1, speed_threshold_l4_q3, max_speed_l4
        )
        
        # 调试：打印理论曲线的一些关键点
        print(f"\n=== 理论曲线调试信息 ===")
        print(f"L3理论曲线在速度0处: upper={l3_theory_top[150]:.2f}, lower={l3_theory_bottom[150]:.2f}")
        print(f"L3理论曲线在速度{speed_threshold_l3_q1}处的索引: {np.argmin(np.abs(theoretical_speeds - speed_threshold_l3_q1))}")
        idx_q1 = np.argmin(np.abs(theoretical_speeds - speed_threshold_l3_q1))
        idx_q3 = np.argmin(np.abs(theoretical_speeds + speed_threshold_l3_q3))
        print(f"L3理论曲线在Q1阈值处: upper={l3_theory_top[idx_q1]:.2f}, lower={l3_theory_bottom[idx_q1]:.2f}")
        print(f"L3理论曲线在Q3阈值处: upper={l3_theory_top[idx_q3]:.2f}, lower={l3_theory_bottom[idx_q3]:.2f}")
        
        # 创建2x2的子图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'All Real Data Runs - L3 & L4 Joints Torque-Velocity Analysis\n({len(all_real_data)} datasets) - Asymmetric Four Quadrant Model', fontsize=16)
        
        # 定义颜色
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        for idx, joint_idx in enumerate(key_joints):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]
            
            print(f"绘制关节 {joint_labels[idx]} (索引 {joint_idx})")
            
            # 绘制所有real_data的散点图
            for run_index, data_info in all_real_data.items():
                data = data_info['data']
                
                # 提取关节速度和扭矩（使用最后12列作为扭矩）
                joint_vel = data[:, 12:24]  # 关节速度
                joint_torque = data[:, -12:]  # 实际扭矩（最后12列）
                
                # 获取当前关节的数据
                vel_data = joint_vel[:, joint_idx]
                torque_data = joint_torque[:, joint_idx]
                
                # 使用不同颜色绘制每条数据
                color = colors[run_index % len(colors)]
                ax.scatter(vel_data, torque_data, 
                        alpha=0.4, s=0.8, 
                        color=color, 
                        label=f'Real Run {run_index}')
            
            # 如果有仿真数据，也绘制出来
            if sim_data is not None:
                sim_joint_vel = sim_data[:, 12:24]
                sim_joint_torque = sim_data[:, -12:]
                
                sim_vel_data = sim_joint_vel[:, joint_idx]
                sim_torque_data = sim_joint_torque[:, joint_idx]
                
                ax.scatter(sim_vel_data, sim_torque_data, 
                        alpha=0.6, s=1.5, 
                        color='red', marker='x',
                        label='Isaac Gym Sim')
            
            # 绘制非对称四象限理论扭矩限制曲线
            if joint_idx in [2, 8]:  # L3 joints
                ax.plot(theoretical_speeds, l3_theory_top, 'g-', linewidth=3, 
                    label='L3 Theory Upper', alpha=0.9)
                ax.plot(theoretical_speeds, l3_theory_bottom, 'g--', linewidth=3, 
                    label='L3 Theory Lower', alpha=0.9)
                
                # 添加象限分割线和标注
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
                ax.axvline(x=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
                
                # 添加象限标注
                ax.text(7.5, torque_l3_top*0.8, 'Q1\n(Drive)', fontsize=10, ha='center', 
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
                ax.text(-7.5, torque_l3_top*0.8, 'Q2\n(Brake)', fontsize=10, ha='center',
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
                ax.text(-7.5, torque_l3_bottom*0.8, 'Q3\n(Drive)', fontsize=10, ha='center',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
                ax.text(7.5, torque_l3_bottom*0.8, 'Q4\n(Brake)', fontsize=10, ha='center',
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
                
                # 修改：添加独立的象限阈值线
                ax.axvline(x=speed_threshold_l3_q1, color='orange', linestyle=':', linewidth=2, alpha=0.8, 
                        label=f'Q1 Threshold ({speed_threshold_l3_q1})')
                ax.axvline(x=-speed_threshold_l3_q3, color='purple', linestyle=':', linewidth=2, alpha=0.8,
                        label=f'Q3 Threshold ({speed_threshold_l3_q3})')
                ax.axvline(x=max_speed_l3, color='brown', linestyle=':', linewidth=2, alpha=0.8, 
                        label=f'Max Speed ({max_speed_l3})')
                ax.axvline(x=-max_speed_l3, color='brown', linestyle=':', linewidth=2, alpha=0.8)
                
                # 突出显示固定扭矩线（Q2、Q4象限）
                ax.axhline(y=torque_l3_top, color='red', linestyle='-', linewidth=2, alpha=0.8,
                        label=f'Q2 Fixed ({torque_l3_top:.1f})')
                ax.axhline(y=torque_l3_bottom, color='red', linestyle='-', linewidth=2, alpha=0.8,
                        label=f'Q4 Fixed ({torque_l3_bottom:.1f})')
                
            elif joint_idx in [3, 9]:  # L4 joints
                ax.plot(theoretical_speeds, l4_theory_top, 'g-', linewidth=3, 
                    label='L4 Theory Upper', alpha=0.9)
                ax.plot(theoretical_speeds, l4_theory_bottom, 'g--', linewidth=3, 
                    label='L4 Theory Lower', alpha=0.9)
                
                # 添加象限分割线和标注
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
                ax.axvline(x=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
                
                # 添加象限标注
                ax.text(7.5, torque_l4_top*0.8, 'Q1\n(Drive)', fontsize=10, ha='center', 
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
                ax.text(-7.5, torque_l4_top*0.8, 'Q2\n(Brake)', fontsize=10, ha='center',
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
                ax.text(-7.5, torque_l4_bottom*0.8, 'Q3\n(Drive)', fontsize=10, ha='center',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
                ax.text(7.5, torque_l4_bottom*0.8, 'Q4\n(Brake)', fontsize=10, ha='center',
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
                
                # 修改：添加独立的象限阈值线
                ax.axvline(x=speed_threshold_l4_q1, color='orange', linestyle=':', linewidth=2, alpha=0.8, 
                        label=f'Q1 Threshold ({speed_threshold_l4_q1})')
                ax.axvline(x=-speed_threshold_l4_q3, color='purple', linestyle=':', linewidth=2, alpha=0.8,
                        label=f'Q3 Threshold ({speed_threshold_l4_q3})')
                ax.axvline(x=max_speed_l4, color='brown', linestyle=':', linewidth=2, alpha=0.8, 
                        label=f'Max Speed ({max_speed_l4})')
                ax.axvline(x=-max_speed_l4, color='brown', linestyle=':', linewidth=2, alpha=0.8)
                
                # 突出显示固定扭矩线（Q2、Q4象限）
                ax.axhline(y=torque_l4_top, color='red', linestyle='-', linewidth=2, alpha=0.8,
                        label=f'Q2 Fixed ({torque_l4_top:.1f})')
                ax.axhline(y=torque_l4_bottom, color='red', linestyle='-', linewidth=2, alpha=0.8,
                        label=f'Q4 Fixed ({torque_l4_bottom:.1f})')
            
            ax.set_xlabel('Joint Velocity [rad/s]')
            ax.set_ylabel('Joint Torque [Nm]')
            ax.set_title(f'{joint_labels[idx]} - Asymmetric Four Quadrant Model')
            
            # 设置图例（只显示重要的）
            handles, labels = ax.get_legend_handles_labels()
            
            # 选择重要的图例项
            important_keywords = ['Theory', 'Isaac', 'Q1 Threshold', 'Q3 Threshold', 'Max Speed', 'Fixed', 'Real Run 0', 'Real Run 1']
            important_indices = []
            for i, label in enumerate(labels):
                if any(keyword in label for keyword in important_keywords):
                    important_indices.append(i)
            
            if important_indices:
                selected_handles = [handles[i] for i in important_indices]
                selected_labels = [labels[i] for i in important_indices]
                ax.legend(selected_handles, selected_labels, fontsize=7, loc='best', ncol=2)
            
            ax.grid(True, alpha=0.3)
            
            # 设置合理的坐标轴范围
            ax.set_xlim(-15, 15)
            if joint_idx in [2, 8]:  # L3
                ax.set_ylim(-70, 70)
            else:  # L4
                ax.set_ylim(-200, 100)
            
            # 修改：添加非对称参数统计信息
            total_points = sum(len(data_info['data']) for data_info in all_real_data.values())
            if joint_idx in [2, 8]:  # L3
                stats_text = f'Total: {total_points} points\nRuns: {len(all_real_data)}\nQ1_th: {speed_threshold_l3_q1}, Q3_th: {speed_threshold_l3_q3}'
            else:  # L4
                stats_text = f'Total: {total_points} points\nRuns: {len(all_real_data)}\nQ1_th: {speed_threshold_l4_q1}, Q3_th: {speed_threshold_l4_q3}'
                
            if sim_data is not None:
                stats_text += f'\nSim: {len(sim_data)} points'
            
            ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, 
                verticalalignment='bottom', fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        plt.tight_layout()
        
        # 保存图片
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # 创建详细的四象限分析图
        self._plot_four_quadrant_detailed_analysis(all_real_data, sim_data, key_joints, joint_labels,
                                                theoretical_speeds, l3_theory_top, l3_theory_bottom,
                                                l4_theory_top, l4_theory_bottom, params)
        
        plt.close('all')
        print(f"Asymmetric four-quadrant torque-velocity curves saved: {filename}")
        return save_path

    def _plot_four_quadrant_detailed_analysis(self, all_real_data, sim_data, key_joints, joint_labels,
                                            theoretical_speeds, l3_theory_top, l3_theory_bottom,
                                            l4_theory_top, l4_theory_bottom, params):
        """绘制四象限详细分析图（不需要额外的Q2/Q4参数）"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Four Quadrant Torque Model - Detailed Analysis (Q2/Q4 use same limits as Q1/Q3)', fontsize=16)
        
        # 为每个象限定义不同的颜色和标记
        quadrant_colors = {
            'Q1': 'green',    # 驱动象限
            'Q2': 'red',      # 制动象限  
            'Q3': 'green',    # 驱动象限
            'Q4': 'red'       # 制动象限
        }
        
        for idx, joint_idx in enumerate(key_joints):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]
            
            # 分别绘制四个象限的数据点
            all_velocities = []
            all_torques = []
            
            # 收集所有数据
            for run_index, data_info in all_real_data.items():
                data = data_info['data']
                joint_vel = data[:, 12:24]
                joint_torque = data[:, -12:]
                
                vel_data = joint_vel[:, joint_idx]
                torque_data = joint_torque[:, joint_idx]
                
                all_velocities.extend(vel_data)
                all_torques.extend(torque_data)
                
                # 分象限绘制数据点
                q1_mask = (vel_data >= 0) & (torque_data >= 0)  # Q1: +vel, +torque
                q2_mask = (vel_data < 0) & (torque_data >= 0)   # Q2: -vel, +torque
                q3_mask = (vel_data < 0) & (torque_data < 0)    # Q3: -vel, -torque
                q4_mask = (vel_data >= 0) & (torque_data < 0)   # Q4: +vel, -torque
                
                if np.any(q1_mask):
                    ax.scatter(vel_data[q1_mask], torque_data[q1_mask], 
                            alpha=0.3, s=0.5, color=quadrant_colors['Q1'], label='Q1 (Drive)' if run_index == 0 else "")
                if np.any(q2_mask):
                    ax.scatter(vel_data[q2_mask], torque_data[q2_mask], 
                            alpha=0.3, s=0.5, color=quadrant_colors['Q2'], label='Q2 (Brake)' if run_index == 0 else "")
                if np.any(q3_mask):
                    ax.scatter(vel_data[q3_mask], torque_data[q3_mask], 
                            alpha=0.3, s=0.5, color=quadrant_colors['Q3'], label='Q3 (Drive)' if run_index == 0 else "")
                if np.any(q4_mask):
                    ax.scatter(vel_data[q4_mask], torque_data[q4_mask], 
                            alpha=0.3, s=0.5, color=quadrant_colors['Q4'], label='Q4 (Brake)' if run_index == 0 else "")
            
            # 绘制理论曲线
            if joint_idx in [2, 8]:  # L3 joints
                ax.plot(theoretical_speeds, l3_theory_top, 'black', linewidth=3, 
                    label='Theory Envelope', alpha=0.9)
                ax.plot(theoretical_speeds, l3_theory_bottom, 'black', linewidth=3, alpha=0.9)
                
                # 获取参数
                speed_threshold = params.get('speed_threshold_l3', 5.0)
                max_speed = params.get('max_speed_l3', 10.0)
                torque_top = params.get('torque_l3_top', 35.0)
                torque_bottom = params.get('torque_l3_bottom', -35.0)
                
            else:  # L4 joints
                ax.plot(theoretical_speeds, l4_theory_top, 'black', linewidth=3, 
                    label='Theory Envelope', alpha=0.9)
                ax.plot(theoretical_speeds, l4_theory_bottom, 'black', linewidth=3, alpha=0.9)
                
                speed_threshold = params.get('speed_threshold_l4', 7.0)
                max_speed = params.get('max_speed_l4', 12.0)
                torque_top = params.get('torque_l4_top', 150.0)
                torque_bottom = params.get('torque_l4_bottom', -150.0)
            
            # 添加象限标注和特征线
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
            
            # 添加特征线
            ax.axvline(x=speed_threshold, color='orange', linestyle='--', linewidth=2, alpha=0.8, 
                    label=f'Threshold ±{speed_threshold}')
            ax.axvline(x=-speed_threshold, color='orange', linestyle='--', linewidth=2, alpha=0.8)
            
            # 添加固定扭矩线（Q2、Q4象限使用相同的limits）
            ax.axhline(y=torque_top, color='red', linestyle='-', linewidth=2, alpha=0.8,
                    label=f'Q2/Q4 Fixed Limits ({torque_top:.1f}/{torque_bottom:.1f})')
            ax.axhline(y=torque_bottom, color='red', linestyle='-', linewidth=2, alpha=0.8)
            
            ax.set_xlabel('Joint Velocity [rad/s]')
            ax.set_ylabel('Joint Torque [Nm]')
            ax.set_title(f'{joint_labels[idx]} - Quadrant Analysis')
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.3)
            
            # 设置坐标轴范围
            ax.set_xlim(-15, 15)
            if joint_idx in [2, 8]:  # L3
                ax.set_ylim(-50, 50)
            else:  # L4
                ax.set_ylim(-200, 200)
            
            # 计算各象限的数据点数量
            q1_count = sum((np.array(all_velocities) >= 0) & (np.array(all_torques) >= 0))
            q2_count = sum((np.array(all_velocities) < 0) & (np.array(all_torques) >= 0))
            q3_count = sum((np.array(all_velocities) < 0) & (np.array(all_torques) < 0))
            q4_count = sum((np.array(all_velocities) >= 0) & (np.array(all_torques) < 0))
            
            stats_text = f'Data Distribution:\nQ1: {q1_count}\nQ2: {q2_count}\nQ3: {q3_count}\nQ4: {q4_count}'
            ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, 
                verticalalignment='bottom', horizontalalignment='right', fontsize=8, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        plt.tight_layout()
        
        # 保存四象限详细分析图
        detailed_filename = 'four_quadrant_detailed_analysis.png'
        save_path = os.path.join(self.save_dir, detailed_filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Four-quadrant detailed analysis saved: {detailed_filename}")

    
    def _plot_detailed_data_distribution(self, all_real_data, sim_data, key_joints, joint_labels):
        """绘制详细的数据分布图"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('Detailed Data Distribution Analysis - Velocity & Torque', fontsize=16)
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        for idx, joint_idx in enumerate(key_joints):
            # 速度分布
            ax_vel = axes[0, idx]
            # 扭矩分布  
            ax_torque = axes[1, idx]
            
            all_velocities = []
            all_torques = []
            
            # 收集所有数据
            for run_index, data_info in all_real_data.items():
                data = data_info['data']
                joint_vel = data[:, 12:24]
                joint_torque = data[:, -12:]
                
                vel_data = joint_vel[:, joint_idx]
                torque_data = joint_torque[:, joint_idx]
                
                all_velocities.extend(vel_data)
                all_torques.extend(torque_data)
                
                # 绘制每条数据的分布
                color = colors[run_index % len(colors)]
                ax_vel.hist(vel_data, bins=30, alpha=0.3, color=color, 
                        label=f'Run {run_index}', density=True)
                ax_torque.hist(torque_data, bins=30, alpha=0.3, color=color, 
                            density=True)
            
            # 绘制合并的分布
            ax_vel.hist(all_velocities, bins=50, alpha=0.7, color='black', 
                    histtype='step', linewidth=2, label='Combined Real', density=True)
            ax_torque.hist(all_torques, bins=50, alpha=0.7, color='black', 
                        histtype='step', linewidth=2, label='Combined Real', density=True)
            
            # 如果有仿真数据，也绘制
            if sim_data is not None:
                sim_joint_vel = sim_data[:, 12:24]
                sim_joint_torque = sim_data[:, -12:]
                
                sim_vel_data = sim_joint_vel[:, joint_idx]
                sim_torque_data = sim_joint_torque[:, joint_idx]
                
                ax_vel.hist(sim_vel_data, bins=30, alpha=0.6, color='red', 
                        histtype='step', linewidth=2, label='Isaac Sim', density=True)
                ax_torque.hist(sim_torque_data, bins=30, alpha=0.6, color='red', 
                            histtype='step', linewidth=2, label='Isaac Sim', density=True)
            
            # 设置标题和标签
            ax_vel.set_title(f'{joint_labels[idx]} - Velocity Distribution')
            ax_vel.set_xlabel('Velocity [rad/s]')
            ax_vel.set_ylabel('Density')
            ax_vel.legend(fontsize=8)
            ax_vel.grid(True, alpha=0.3)
            
            ax_torque.set_title(f'{joint_labels[idx]} - Torque Distribution')
            ax_torque.set_xlabel('Torque [Nm]')
            ax_torque.set_ylabel('Density')
            ax_torque.legend(fontsize=8)
            ax_torque.grid(True, alpha=0.3)
            
            # 添加统计信息
            vel_stats = f'μ={np.mean(all_velocities):.2f}\nσ={np.std(all_velocities):.2f}'
            torque_stats = f'μ={np.mean(all_torques):.2f}\nσ={np.std(all_torques):.2f}'
            
            ax_vel.text(0.02, 0.98, vel_stats, transform=ax_vel.transAxes, 
                    verticalalignment='top', fontsize=8, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
            ax_torque.text(0.02, 0.98, torque_stats, transform=ax_torque.transAxes, 
                        verticalalignment='top', fontsize=8, 
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        plt.tight_layout()
        
        # 保存分布图
        distribution_filename = 'all_real_data_distribution_analysis.png'
        save_path = os.path.join(self.save_dir, distribution_filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Data distribution analysis saved: {distribution_filename}")

     
    
    def update_velocity_limits(self, env, params):
        """更新环境的关节速度限制 - 改进的对称/非对称处理"""
        dof_props = env.gym.get_actor_dof_properties(env.envs[0], 0)
        
        # 检查是否需要非对称限制
        l3_is_symmetric = abs(abs(params['angle_vel_l3_bottom']) - abs(params['angle_vel_l3_top'])) < 1e-6
        l4_is_symmetric = abs(abs(params['angle_vel_l4_bottom']) - abs(params['angle_vel_l4_top'])) < 1e-6
        
        # L3关节速度限制处理
        if l3_is_symmetric:
            # 对称情况：使用任一值的绝对值
            limit_val_l3 = abs(params['angle_vel_l3_top'])
            print(f"L3使用对称速度限制: ±{limit_val_l3:.2f} rad/s")
        else:
            # 非对称情况：有两种策略可选
            
            # 策略1：保守策略 - 使用较小的绝对值，确保不违反任一方向的限制
            # limit_val_l3 = min(abs(params['angle_vel_l3_bottom']), abs(params['angle_vel_l3_top']))
            
            # 策略2：激进策略 - 使用较大的绝对值，允许物理引擎有更大范围
            limit_val_l3 = max(abs(params['angle_vel_l3_bottom']), abs(params['angle_vel_l3_top']))
            
            # 策略3：平均策略 - 使用两个限制的平均值
            # limit_val_l3 = (abs(params['angle_vel_l3_bottom']) + abs(params['angle_vel_l3_top'])) / 2
            
            print(f"L3使用非对称速度限制: [{params['angle_vel_l3_bottom']:.2f}, {params['angle_vel_l3_top']:.2f}] rad/s")
            print(f"物理引擎限制设为: ±{limit_val_l3:.2f} rad/s (使用较大值策略)")
        
        dof_props['velocity'][2] = limit_val_l3  # leg_l3
        dof_props['velocity'][8] = limit_val_l3  # leg_r3
        
        # L4关节速度限制处理（相同逻辑）
        if l4_is_symmetric:
            limit_val_l4 = abs(params['angle_vel_l4_top'])
            print(f"L4使用对称速度限制: ±{limit_val_l4:.2f} rad/s")
        else:
            # 使用相同的策略（这里用策略2）
            limit_val_l4 = max(abs(params['angle_vel_l4_bottom']), abs(params['angle_vel_l4_top']))
            print(f"L4使用非对称速度限制: [{params['angle_vel_l4_bottom']:.2f}, {params['angle_vel_l4_top']:.2f}] rad/s")
            print(f"物理引擎限制设为: ±{limit_val_l4:.2f} rad/s (使用较大值策略)")
        
        dof_props['velocity'][3] = limit_val_l4  # leg_l4
        dof_props['velocity'][9] = limit_val_l4  # leg_r4

        # 关键：确保控制层使用精确的非对称限制
        if hasattr(env, 'dof_vel_limits') and env.dof_vel_limits.shape[1] >= 2:
            # 设置精确的非对称控制限制
            env.dof_vel_limits[2, 0] = params['angle_vel_l3_bottom']  # L3下限
            env.dof_vel_limits[2, 1] = params['angle_vel_l3_top']     # L3上限
            env.dof_vel_limits[8, 0] = params['angle_vel_l3_bottom']  # R3下限 
            env.dof_vel_limits[8, 1] = params['angle_vel_l3_top']     # R3上限
            
            env.dof_vel_limits[3, 0] = params['angle_vel_l4_bottom']  # L4下限
            env.dof_vel_limits[3, 1] = params['angle_vel_l4_top']     # L4上限
            env.dof_vel_limits[9, 0] = params['angle_vel_l4_bottom']  # R4下限
            env.dof_vel_limits[9, 1] = params['angle_vel_l4_top']     # R4上限
            
            print(f"控制层已设置精确的非对称速度限制")
        else:
            print(f"Warning: 环境不支持非对称速度限制，仅使用物理引擎的对称限制")
        
        # 应用到所有环境
        for i in range(env.num_envs):
            env.gym.set_actor_dof_properties(env.envs[i], 0, dof_props)
        
        # 添加验证信息
        print(f"物理引擎速度限制已更新:")
        print(f"  L3/R3: ±{limit_val_l3:.2f} rad/s")
        print(f"  L4/R4: ±{limit_val_l4:.2f} rad/s")

    def collect_mujoco_data(self, steps=500, command=[2., 0., 0., 0.]):
        """使用Mujoco环境收集'真实'数据，并可视化采集过程"""
        cfg = Sim2simCfg()
        model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
        model.opt.timestep = cfg.sim_config.dt
        data = mujoco.MjData(model)
        
        # 修改：使用与sim2sim_kuavo一致的default_dof_pos
        default_dof_pos = np.array([0., 0., -0.47, 0.86, -0.44, 0., 
                                    0., 0., -0.47, 0.86, -0.44, 0., 
                                    0, 0, 0, 0, 0, 0])

        # 优化：显式设置初始状态，而不是使用keyframe
        mujoco.mj_resetData(model, data)
        # 假设qpos[7:]是18个关节
        if len(data.qpos) > 7 + len(default_dof_pos):
             data.qpos[7:7+len(default_dof_pos)] = default_dof_pos.copy()
        else:
             data.qpos[7:] = default_dof_pos[:len(data.qpos)-7].copy()
        mujoco.mj_step(model, data)
        # 优化：在开始前稳定机器人
        # 仅使用PD控制来维持初始姿势，让其稳定下来
        print("Stabilizing robot in Mujoco...")
        for _ in range(100):
             # 修复：直接使用actuator数据，以匹配PD控制器的输入维度(18)
            q = np.array(data.actuator_length)
            dq = np.array(data.actuator_velocity)
            tau = pd_control(np.zeros_like(default_dof_pos), default_dof_pos, q, cfg.robot_config.kps,
                             np.zeros_like(default_dof_pos), dq, cfg.robot_config.kds)
            tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)
            data.ctrl = tau
            mujoco.mj_step(model, data)
        # 加入viewer
        if mujoco_see:
            viewer = mujoco_viewer.MujocoViewer(model, data)

        target_q = np.zeros((cfg.env.num_actions+6), dtype=np.double)  # 修改：包含手臂关节
        action = np.zeros((cfg.env.num_actions), dtype=np.double)

      
        hist_obs = deque()
        #先构建一个包含正确命令的初始观测
        initial_obs = np.zeros([1, cfg.env.num_single_obs], dtype=np.double)
        initial_obs[0, 2] = command[0] * cfg.normalization.obs_scales.lin_vel  # cmd.vx
        initial_obs[0, 3] = command[1] * cfg.normalization.obs_scales.lin_vel  # cmd.vy
        initial_obs[0, 4] = command[2] * cfg.normalization.obs_scales.ang_vel  # cmd.dyaw
        initial_obs[0, 5] = command[3]  # cmd.stand

         # 修复：用包含命令的initial_obs填充历史缓冲区
        for _ in range(cfg.env.frame_stack):
            hist_obs.append(initial_obs.copy())  # 用initial_obs.copy()而不是全零数组

        count_lowlevel = 0
        data_collected = []

        
        # 添加：获取关节名顺序（与sim2sim_kuavo一致）
        joint_names = [
            'leg_l1_joint', 'leg_l2_joint', 'leg_l3_joint', 'leg_l4_joint', 'leg_l5_joint', 'leg_l6_joint',
            'leg_r1_joint', 'leg_r2_joint', 'leg_r3_joint', 'leg_r4_joint', 'leg_r5_joint', 'leg_r6_joint'
        ]

        for step in range(steps * cfg.sim_config.decimation):
            # 修改：使用与sim2sim_kuavo一致的get_obs函数
            q_, dq_, quat, v, omega, gvec = get_obs_mujoco(data)
            
            # 修改：使用actuator数据（与sim2sim_kuavo一致）
            q = np.array(data.actuator_length)
            dq = np.array(data.actuator_velocity)
            
            if count_lowlevel % cfg.sim_config.decimation == 0:
                # 修改：使用与sim2sim_kuavo完全一致的观测构建
                obs = np.zeros([1, cfg.env.num_single_obs], dtype=np.float32)
                eu_ang = quaternion_to_euler_array(quat)
                
                eu_ang[eu_ang > math.pi] -= 2 * math.pi
                
                obs[0, 0] = math.sin(2 * math.pi * count_lowlevel * cfg.sim_config.dt / cfg.rewards.cycle_time + math.pi*0.5)
                obs[0, 1] = math.cos(2 * math.pi * count_lowlevel * cfg.sim_config.dt / cfg.rewards.cycle_time + math.pi*0.5)
                obs[0, 2] = command[0] * cfg.normalization.obs_scales.lin_vel  # cmd.vx
                obs[0, 3] = command[1] * cfg.normalization.obs_scales.lin_vel  # cmd.vy
                obs[0, 4] = command[2] * cfg.normalization.obs_scales.ang_vel  # cmd.dyaw
                obs[0, 5] = command[3]  # cmd.stand
                obs[0, 6:18] = (q[:cfg.env.num_actions] - default_dof_pos[:cfg.env.num_actions]) * cfg.normalization.obs_scales.dof_pos
                obs[0, 18:30] = dq[:cfg.env.num_actions] * cfg.normalization.obs_scales.dof_vel
                obs[0, 30:42] = action
                obs[0, 42:45] = omega
                obs[0, 45:47] = eu_ang[0:2]

                obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)
                hist_obs.append(obs)
                hist_obs.popleft()

                policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
                for i in range(cfg.env.frame_stack):
                    policy_input[0, i * cfg.env.num_single_obs : (i + 1) * cfg.env.num_single_obs] = hist_obs[i][0, :]
                
                action[:] = self.jit_policy(torch.tensor(policy_input))[0].detach().numpy()
                action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)
                target_q[:cfg.env.num_actions] = action * cfg.control.action_scale

                # 添加：手臂控制（与sim2sim_kuavo一致）
                target_q[cfg.env.num_actions] = -math.sin(2 * math.pi * count_lowlevel * cfg.sim_config.dt / cfg.rewards.cycle_time + math.pi*0.5)*0.9
                target_q[cfg.env.num_actions+1] = 0.
                target_q[cfg.env.num_actions+2] = -110*np.pi/180.
                target_q[cfg.env.num_actions+3] = math.sin(2 * math.pi * count_lowlevel * cfg.sim_config.dt / cfg.rewards.cycle_time + math.pi*0.5)*0.9
                target_q[cfg.env.num_actions+4] = 0.
                target_q[cfg.env.num_actions+5] = -110*np.pi/180.

                # 修改：使用sensor数据收集（与sim2sim_kuavo一致）
                jointpos_sensor_names = [name.replace('_joint', '_pos') for name in joint_names]
                jointvel_sensor_names = [name.replace('_joint', '_vel') for name in joint_names]
                dof_pos = np.array([data.sensor(name).data.copy()[0] for name in jointpos_sensor_names])
                dof_vel = np.array([data.sensor(name).data.copy()[0] for name in jointvel_sensor_names])
                           
                # 修改：从已有的 'linear-velocity' 传感器获取世界坐标线速度
                world_lin_vel = data.sensor('linear-velocity').data.copy().astype(np.double)
                
                # 获取机器人基座速度
                base_lin_vel = world_lin_vel.copy() 
                # ====== 这里加打印 ======
                if count_lowlevel < 20:
                    print(f"Step {count_lowlevel}: command={command[0]}, policy_input_vx={policy_input[0,2]}, action={action[:4]}, vx={world_lin_vel[0]:.3f}")
               
                
                # =======================

                #print(f"command[0]={command[0]}, obs_scales.lin_vel={cfg.normalization.obs_scales.lin_vel}")
                # 收集数据: [joint_pos, joint_vel, action] - 只收集腿部关节
                actual_torques_mujoco = data.ctrl[:cfg.env.num_actions].copy()  # 只取前12个关节的扭矩
            
                sample = np.concatenate([dof_pos[:cfg.env.num_actions], 
                                    dof_vel[:cfg.env.num_actions], 
                                    action, 
                                    base_lin_vel,
                                    world_lin_vel,
                                    actual_torques_mujoco  # 新增：Mujoco实际扭矩数据
                                    ])
                data_collected.append(sample)

            count_lowlevel += 1
            target_dq = np.zeros((cfg.env.num_actions+6), dtype=np.double)

            # 修改：PD控制使用完整的target_q（包含手臂）
            tau = pd_control(target_q, default_dof_pos, q, cfg.robot_config.kps,
                            target_dq, dq, cfg.robot_config.kds)
            tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)
            data.ctrl = tau
            mujoco.mj_step(model, data)

            # 可视化
            if mujoco_see:
                viewer.render()
        if mujoco_see:
            viewer.close()
        self.mujoco_data = np.array(data_collected)
        return np.array(data_collected)
    
    def simulate_and_collect_isaac(self, params, steps=500, command=[1.8, 0., 0., 0.]):
        """使用Isaac Gym环境收集仿真数据"""
        
        self.env.commands[:, 0] = command[0]
        self.env.commands[:, 1] = command[1]
        self.env.commands[:, 2] = command[2]
        self.env.commands[:, 3] = command[3]
        set_global_seed(SEED)

        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        
        # 修复：使用正确的参数名称
        try:
            self.update_friction_coeffs(self.env, params)
        except Exception as e:
            print(f"Warning: Failed to update friction coeffs: {e}")
        
        # 检查是否使用动态扭矩限制
        use_dynamic_torque = params.get('use_dynamic_torque', False)
        
        if use_dynamic_torque:
            print("Using dynamic torque limits")
            # 使用动态扭矩限制（在仿真循环中实时更新）
        else:
            print("Using static torque limits")
            try:
                self.update_torque_limits(self.env, params)
            except Exception as e:
                print(f"Warning: Failed to update static torque limits: {e}")
        
        try:
            self.update_velocity_limits(self.env, params)
        except Exception as e:
            print(f"Warning: Failed to update velocity limits: {e}")
        
        data_collected = []
        num_envs = self.env.num_envs
        
        for step in tqdm(range(steps), desc="Isaac Gym simulation"):
            # 设置command
            self.env.commands[:, 0] = command[0]
            self.env.commands[:, 1] = command[1]
            self.env.commands[:, 2] = command[2]
            self.env.commands[:, 3] = command[3]


            # 在每个仿真步骤中应用动态扭矩限制（如果启用）
            if use_dynamic_torque:
                try:
                    self.update_dynamic_torque_limits(self.env, params)
                except Exception as e:
                    print(f"Warning: Failed to update dynamic torque limits at step {step}: {e}")

            with torch.no_grad():
                action = self.policy(obs.detach())
            if isinstance(action, tuple):
                action = action[0]
            
            # 执行动作
            step_result = self.env.step(action.detach())
            if isinstance(step_result, tuple):
                obs = step_result[0]
            else:
                obs = step_result
            
            # 收集数据
            joint_pos = self.env.dof_pos.cpu().numpy()
            joint_vel = self.env.dof_vel.cpu().numpy()
            action_np = action.cpu().numpy()

            #收集机器人基座速度数据
            base_lin_vel = self.env.base_lin_vel.cpu().numpy()

            #世界坐标线速度
            root_states = self.env.root_states.cpu().numpy()
            world_lin_vel = root_states[:, 7:10]

            # 收集实际扭矩数据
            if hasattr(self.env, 'torques'):
                actual_torques = self.env.torques.cpu().numpy()
            else:
                actual_torques = np.zeros_like(action_np)  # 如果没有扭矩数据，用零填充
            
            # 只使用第一个环境的数据，增加扭矩数据
            sample = np.concatenate([joint_pos[0], joint_vel[0], action_np[0],
                                    base_lin_vel[0],
                                    world_lin_vel[0],
                                    actual_torques[0]  # 新增：实际扭矩数据
                                    ])
            data_collected.append(sample)

        return np.array(data_collected)

    # def align_time_series(self, sim_data, real_data):
    #     """
    #     通过计算多维信号的互相关来对齐两个时间序列。
    #     只使用第2和3维度（索引1和2）来计算延迟，以提高对齐的针对性。
    #     通过裁剪数据段来实现对齐，而不是滚动和填充。
    #     """
    #     from scipy import signal
    #     from scipy.stats import mode

    #     # 只使用第2和3维度（索引1和2）来计算延迟
    #     target_dims = [2, 3, 8, 9]  # 第2和3维度（0-based索引）
    #     delays = []
        
    #     for i in target_dims:
    #         # 确保维度存在
    #         if i >= sim_data.shape[1] or i >= real_data.shape[1]:
    #             print(f"Warning: Dimension {i} not available in data (sim: {sim_data.shape[1]}, real: {real_data.shape[1]})")
    #             continue
                
    #         # 过滤掉方差很小的信号，避免无效的相关性计算
    #         if np.var(real_data[:, i]) < 1e-6 or np.var(sim_data[:, i]) < 1e-6:
    #             print(f"Warning: Dimension {i} has very low variance, skipping")
    #             continue
            
    #         correlation = signal.correlate(real_data[:, i], sim_data[:, i], mode='full')
    #         # 延迟 = real_data相对于sim_data的偏移量
    #         delay = correlation.argmax() - (len(sim_data) - 1)
    #         delays.append(delay)
    #         print(f"Dimension {i} delay: {delay}")

    #     if not delays:
    #         # 如果所有维度都是常量，则无法对齐
    #         print("Warning: No valid dimensions for alignment, using zero delay")
    #         best_delay = 0
    #     else:
    #         # 使用中位数来抵抗异常值
    #         best_delay = int(np.median(delays))
    #         print(f"Selected delays from dims {target_dims}: {delays}")
    #         print(f"Final delay (median): {best_delay}")

    #     # 根据延迟裁剪数据以实现对齐
    #     if best_delay > 0:
    #         # sim_data 滞后于 real_data，real_data 需要裁剪头部
    #         # real_data[delay:] 对齐 sim_data[:-delay]
    #         common_len = min(len(real_data) - best_delay, len(sim_data))
    #         if common_len <= 0: # 避免无效裁剪
    #             return sim_data, real_data
    #         aligned_real_data = real_data[best_delay : best_delay + common_len]
    #         aligned_sim_data = sim_data[:common_len]
    #     elif best_delay < 0:
    #         # sim_data 领先于 real_data，sim_data 需要裁剪头部
    #         # sim_data[-delay:] 对齐 real_data[:delay]
    #         delay_abs = abs(best_delay)
    #         common_len = min(len(sim_data) - delay_abs, len(real_data))
    #         if common_len <= 0: # 避免无效裁剪
    #             return sim_data, real_data
    #         aligned_sim_data = sim_data[delay_abs : delay_abs + common_len]
    #         aligned_real_data = real_data[:common_len]
    #     else:
    #         # 无延迟
    #         min_len = min(len(sim_data), len(real_data))
    #         aligned_sim_data = sim_data[:min_len]
    #         aligned_real_data = real_data[:min_len]

    #     # 保存对齐信息
    #     self.last_aligned_sim_data = aligned_sim_data
    #     self.last_aligned_real_data = aligned_real_data
    #     self.last_delay = best_delay

    #     print(f"Alignment completed: delay={best_delay}, aligned length={len(aligned_sim_data)}")
    #     return aligned_sim_data, aligned_real_data
    def align_time_series(self, sim_data, real_data, alignment_dims=None):
        """
        通过计算多维信号的互相关来对齐两个时间序列。
        对actions单独进行裁剪，其他数据一起裁剪。
        """
        from scipy import signal
        from scipy.stats import mode

        # 默认使用第2和3维度进行对齐
        if alignment_dims is None:
            alignment_dims = [2, 3]
        
        print(f"Using dimensions {alignment_dims} for time series alignment")
        
        # 数据结构：[joint_pos(12), joint_vel(12), action(12), base_vel(3), world_vel(3), actual_torques(12)]
        # 分离actions和其他数据
        actions_sim = sim_data[:, 24:36]  # actions部分 (索引24-35)
        other_sim = np.concatenate([sim_data[:, :24], sim_data[:, 36:]], axis=1)  # 其他数据
        
        actions_real = real_data[:, 24:36]  # actions部分
        other_real = np.concatenate([real_data[:, :24], real_data[:, 36:]], axis=1)  # 其他数据
        
        # 调整alignment_dims索引，因为我们分离了数据
        # 原始的索引2,3现在仍然是2,3（在joint_pos部分）
        # 但如果要对齐actions，需要使用0,1（在actions部分的第0,1维）
        
        delays_actions = []
        delays_others = []
        
        # 1. 计算actions的对齐延迟（使用actions数据的前几个维度）
        print("=== 计算actions对齐延迟 ===")
        actions_alignment_dims = [0, 1]  # actions的前2个维度
        
        for i in actions_alignment_dims:
            if i >= actions_sim.shape[1] or i >= actions_real.shape[1]:
                print(f"Warning: Actions dimension {i} not available")
                continue
                
            # 过滤掉方差很小的信号
            if np.var(actions_real[:, i]) < 1e-6 or np.var(actions_sim[:, i]) < 1e-6:
                print(f"Warning: Actions dimension {i} has very low variance, skipping")
                continue
            
            correlation = signal.correlate(actions_real[:, i], actions_sim[:, i], mode='full')
            delay = correlation.argmax() - (len(actions_sim) - 1)
            delays_actions.append(delay)
            print(f"Actions dimension {i} delay: {delay}")
        
        # 2. 计算其他数据的对齐延迟
        print("=== 计算其他数据对齐延迟 ===")
        for i in alignment_dims:
            if i >= other_sim.shape[1] or i >= other_real.shape[1]:
                print(f"Warning: Other data dimension {i} not available")
                continue
                
            # 过滤掉方差很小的信号
            if np.var(other_real[:, i]) < 1e-6 or np.var(other_sim[:, i]) < 1e-6:
                print(f"Warning: Other data dimension {i} has very low variance, skipping")
                continue
            
            correlation = signal.correlate(other_real[:, i], other_sim[:, i], mode='full')
            delay = correlation.argmax() - (len(other_sim) - 1)
            delays_others.append(delay)
            print(f"Other data dimension {i} delay: {delay}")

        # 3. 确定最终的延迟值
        if not delays_actions and not delays_others:
            print("Warning: No valid dimensions for alignment, using zero delay")
            actions_delay = 0
            others_delay = 0
        else:
            # 分别计算actions和其他数据的延迟
            if delays_actions:
                actions_delay = int(np.median(delays_actions))
                print(f"Actions delays: {delays_actions}, selected: {actions_delay}")
            else:
                actions_delay = 0
                print("No valid actions delays, using 0")
                
            if delays_others:
                others_delay = int(np.median(delays_others))
                print(f"Others delays: {delays_others}, selected: {others_delay}")
            else:
                others_delay = 0
                print("No valid others delays, using 0")

        # 4. 分别对actions和其他数据进行对齐裁剪
        def apply_alignment(data_sim, data_real, delay):
            """应用对齐裁剪"""
            if delay > 0:
                # sim_data 滞后于 real_data
                common_len = min(len(data_real) - delay, len(data_sim))
                if common_len <= 0:
                    return data_sim, data_real
                aligned_real = data_real[delay : delay + common_len]
                aligned_sim = data_sim[:common_len]
            elif delay < 0:
                # sim_data 领先于 real_data
                delay_abs = abs(delay)
                common_len = min(len(data_sim) - delay_abs, len(data_real))
                if common_len <= 0:
                    return data_sim, data_real
                aligned_sim = data_sim[delay_abs : delay_abs + common_len]
                aligned_real = data_real[:common_len]
            else:
                # 无延迟
                min_len = min(len(data_sim), len(data_real))
                aligned_sim = data_sim[:min_len]
                aligned_real = data_real[:min_len]
            
            return aligned_sim, aligned_real

        # 应用对齐
        actions_sim_aligned, actions_real_aligned = apply_alignment(actions_sim, actions_real, actions_delay)
        other_sim_aligned, other_real_aligned = apply_alignment(other_sim, other_real, others_delay)
        
        # 5. 确保所有部分长度一致（取最小长度）
        min_len = min(len(actions_sim_aligned), len(actions_real_aligned), 
                    len(other_sim_aligned), len(other_real_aligned))
        
        actions_sim_final = actions_sim_aligned[:min_len]
        actions_real_final = actions_real_aligned[:min_len]
        other_sim_final = other_sim_aligned[:min_len]
        other_real_final = other_real_aligned[:min_len]
        
        # 6. 重新组合数据
        # 重新组合：[joint_pos(12), joint_vel(12), action(12), base_vel(3), world_vel(3), actual_torques(12)]
        aligned_sim_data = np.concatenate([
            other_sim_final[:, :24],  # joint_pos + joint_vel
            actions_sim_final,        # actions
            other_sim_final[:, 24:]   # base_vel + world_vel + actual_torques
        ], axis=1)
        
        aligned_real_data = np.concatenate([
            other_real_final[:, :24], # joint_pos + joint_vel
            actions_real_final,       # actions
            other_real_final[:, 24:]  # base_vel + world_vel + actual_torques
        ], axis=1)

        # 保存对齐信息
        self.last_aligned_sim_data = aligned_sim_data
        self.last_aligned_real_data = aligned_real_data
        self.last_actions_delay = actions_delay
        self.last_others_delay = others_delay

        print(f"Alignment completed:")
        print(f"  Actions delay: {actions_delay}")
        print(f"  Others delay: {others_delay}")
        print(f"  Final aligned length: {len(aligned_sim_data)}")
        
        return aligned_sim_data, aligned_real_data

    def compute_distance(
        self,
        sim_data, 
        real_data, 
        dim_weights={'pos': 1.5, 'vel': 1.0, 'act': 0.8},
        mmd_weight=50.0,
        verbose=False
    ):
        """
        计算仿真数据和真实数据之间的加权组合距离。

        这个函数整合了数据预处理、加权的1D Wasserstein距离和多维MMD距离的计算。

        Args:
            sim_data (np.ndarray): 仿真数据，形状为 (n_steps, n_features).
            real_data (np.ndarray): 真实数据，形状为 (m_steps, n_features).
            dim_weights (dict, optional): 用于Wasserstein距离的维度权重. 
                                        键为'pos', 'vel', 'act'。默认为 {'pos': 1.5, 'vel': 1.0, 'act': 0.8}.
            mmd_weight (float, optional): MMD距离在总距离中的权重. 默认为 50.0.
            verbose (bool, optional): 是否打印详细的计算过程信息. 默认为 False.

        Returns:
            float: 计算出的最终组合距离.
        """
        from scipy.stats import wasserstein_distance
        from sklearn.metrics.pairwise import pairwise_distances, rbf_kernel
        import numpy as np
        
        # --- 1. 定义常量和辅助函数 ---
        START_INDEX = 200
        
        # 定义维度的物理意义，方便加权
        DIMS_POS = {2, 3, 8, 9}          # 关节位置
        DIMS_VEL = {14, 15, 20, 21}      # 关节速度
        DIMS_ACT = {26, 27, 32, 33}      # 关节动作
        DIMS_TO_CALCULATE = DIMS_POS | DIMS_VEL | DIMS_ACT

        def _mmd_rbf(X, Y, gamma=None):
            """
            [内部辅助函数] 高效计算RBF核的MMD距离。
            """
            # 基于中位数距离的启发式方法来自动选择gamma
            if gamma is None:
                dists_sq = pairwise_distances(X, Y, metric='sqeuclidean')
                median_dist = np.median(dists_sq)
                # 防止中位数为0导致除零错误
                gamma = 1.0 / (2 * median_dist) if median_dist > 1e-6 else 1.0
                if verbose:
                    print(f"  [MMD] Auto-adjusted gamma to: {gamma:.4f}")

            K_XX = rbf_kernel(X, X, gamma=gamma)
            K_YY = rbf_kernel(Y, Y, gamma=gamma)
            K_XY = rbf_kernel(X, Y, gamma=gamma)
            
            mmd_value = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
            
            if verbose:
                print(f"  [MMD] Components: K_XX={K_XX.mean():.6f}, K_YY={K_YY.mean():.6f}, K_XY={K_XY.mean():.6f}")
                
            return mmd_value

        # --- 2. 数据预处理 ---
        if verbose:
            print("--- Step 1: Preprocessing Data ---")

        sim_data = sim_data[START_INDEX:]
        real_data = real_data[START_INDEX:]
        
        min_len = min(len(sim_data), len(real_data))
        sim_data = sim_data[:min_len]
        real_data = real_data[:min_len]

        aligned_sim_data, aligned_real_data = self.align_time_series(sim_data, real_data)
        if verbose:
            print(f"Data truncated and aligned to length: {min_len}")

        # --- 3. 计算加权的Wasserstein距离 ---
        if verbose:
            print("\n--- Step 2: Calculating Weighted Wasserstein Distance ---")
        wd_values = []
        for i in range(aligned_sim_data.shape[1]):
            if i not in DIMS_TO_CALCULATE:
                continue
            
            # a. 计算原始距离
            raw_wd = wasserstein_distance(aligned_sim_data[:, i], aligned_real_data[:, i])
            
            # b. 按组合标准差归一化
            std_combined = np.sqrt(np.var(aligned_sim_data[:, i]) + np.var(aligned_real_data[:, i]))
            wd_normalized = raw_wd / std_combined if std_combined > 1e-6 else 0.0
            
            # c. 应用维度权重
            if i in DIMS_POS:
                weight = dim_weights.get('pos', 1.0)
                dim_type = 'pos'
            elif i in DIMS_VEL:
                weight = dim_weights.get('vel', 1.0)
                dim_type = 'vel'
            elif i in DIMS_ACT:
                weight = dim_weights.get('act', 1.0)
                dim_type = 'act'
            
            weighted_wd = wd_normalized * weight
            wd_values.append(weighted_wd)
            
            if verbose:
                print(f"  Dim {i:<2} ({dim_type:<3}): raw_wd={raw_wd:.4f}, norm_wd={wd_normalized:.4f}, weighted_wd={weighted_wd:.4f}")
        
        total_wd = sum(wd_values)
        if verbose:
            print(f"Total Weighted Wasserstein Distance: {total_wd:.4f}")

        # --- 4. 计算MMD距离 ---
        if verbose:
            print("\n--- Step 3: Calculating MMD ---")
            
        dims_to_keep = sorted(list(DIMS_TO_CALCULATE))
        sim_data_filtered = aligned_sim_data[:, dims_to_keep]
        real_data_filtered = aligned_real_data[:, dims_to_keep]

        mmd = _mmd_rbf(sim_data_filtered, real_data_filtered)
        if verbose:
            print(f"MMD value: {mmd:.6f}")

        # --- 5. 组合最终距离 ---
        if verbose:
            print("\n--- Step 4: Combining Final Distance ---")
            
        final_distance = total_wd + mmd * mmd_weight
        
        if verbose:
            print(f"Final Distance = {total_wd:.4f} (WD) + {mmd:.6f} * {mmd_weight} (MMD) = {final_distance:.4f}")

        return final_distance


    def sim2real_distance(self, params):
        """计算sim2real距离"""
        print(f"Testing params: {params}")
        
            # 将列表参数转换为字典格式
        if isinstance(params, (list, np.ndarray)):
            param_dict = {
                'joint_friction_l3': params[0],
                'torque_l3_top': params[1],
                'torque_l3_bottom': params[2], 
                'torque_l4_top': params[1],      # 使用相同的扭矩限制
                'torque_l4_bottom': params[2],   # 使用相同的扭矩限制
                'angle_vel_l3_top': params[3],
                'angle_vel_l3_bottom': params[4],
                'angle_vel_l4_top': params[3],    # 使用相同的速度限制
                'angle_vel_l4_bottom': params[4], # 使用相同的速度限制
                # 新增：动态扭矩参数（如果参数数组足够长）
                'speed_threshold_l3': params[5] if len(params) > 5 else 5.0,
                'speed_threshold_l4': params[6] if len(params) > 6 else 7.0,
                'max_speed_l3': params[7] if len(params) > 7 else 10.0,
                'max_speed_l4': params[8] if len(params) > 8 else 12.0,
                'use_dynamic_torque': True if len(params) > 5 else False,
        
            }
        else:
            param_dict = params
        
        # 收集Isaac Gym数据
        sim_data = self.simulate_and_collect_isaac(param_dict)
        
        # 计算距离
        distance = self.compute_distance(sim_data, self.real_data)
        self.last_distance_score = distance
        
        if distance < self.best_score:
            self.best_score = distance
            self.best_params = param_dict
            print(f"New best score: {distance}, params: {param_dict}")
            
            # 保存最佳参数时的数据对比
            self.save_and_visualize_data(self.real_data, sim_data, param_dict)
        
        return distance

    def optimize(self):
        """执行CMA-ES优化"""
        print("Starting CMA-ES optimization...")
        # 修复：根据cma库的要求，将独立的sigma列表通过 'CMA_stds' 选项传入
        # sigma0 参数本身需要是一个标量
        options = {
            'maxiter': self.max_iter,
            'CMA_stds': self.sigma0  # self.sigma0 是包含各个参数独立步长的列表
        }
        opt = cma.CMAEvolutionStrategy(self.initial_params, 1.0, options)
        opt.optimize(self.sim2real_distance)
        self.best_params = opt.result.xbest
        print("Best Sim2Real Params:", self.best_params)
        return self.best_params

        

    def sample_param(self, trial: optuna.Trial) -> Dict[str, Any]:
        # 现有的扭矩限制参数
        torque_l3_top = trial.suggest_float('torque_l3_top', 46, 55)
        torque_l3_bottom = trial.suggest_float('torque_l3_bottom', -45, -38)

        angle_vel_l3_top = trial.suggest_float("angle_vel_l3_top", 5, 10)
        angle_vel_l3_bottom = trial.suggest_float("angle_vel_l3_bottom", -10, -5)

        torque_l4_top = trial.suggest_float('torque_l4_top', 30, 60)
        torque_l4_bottom = trial.suggest_float('torque_l4_bottom', -165, -150)
        angle_vel_l4_top = trial.suggest_float("angle_vel_l4_top", 7, 13)
        angle_vel_l4_bottom = trial.suggest_float("angle_vel_l4_bottom", -13, -7)
        
        # 新增：动态扭矩限制参数
        #对应扭矩开始衰减的速度阈值
        # 修改：独立的象限速度阈值参数
        speed_threshold_l3_q1 = trial.suggest_float('speed_threshold_l3_q1', 2.0, 6.0)  # L3 Q1象限速度阈值
        speed_threshold_l3_q3 = trial.suggest_float('speed_threshold_l3_q3', 2.0, 6.0)  # L3 Q3象限速度阈值
        speed_threshold_l4_q1 = trial.suggest_float('speed_threshold_l4_q1', 2.0, 8.0)  # L4 Q1象限速度阈值
        speed_threshold_l4_q3 = trial.suggest_float('speed_threshold_l4_q3', 2.0, 8.0)  # L4 Q3象限速度阈值
        
        #对应扭矩衰减到0的速度
        # max_speed_l3 = trial.suggest_float('max_speed_l3', 8.0, 12.0)             # L3最大速度
        # max_speed_l4 = trial.suggest_float('max_speed_l4', 10.0, 15.0)            # L4最大速度
        # #较大范围是用来看效果
        # torque_l3_top = trial.suggest_float('torque_l3_top', 50, 100)  # 直接定义范围
        # torque_l3_bottom = trial.suggest_float('torque_l3_bottom', -80, -50)  # 直接定义范围
        # angle_vel_l3_top = trial.suggest_float("angle_vel_l3_top", 10, 20)
        # angle_vel_l3_bottom = trial.suggest_float("angle_vel_l3_bottom", -20, -10)
        # #较大范围是用来看效果
        # torque_l4_top = trial.suggest_float('torque_l4_top', 50, 825)  # 直接定义范围
        # torque_l4_bottom = trial.suggest_float('torque_l4_bottom', -200, -150)  # 直接定义范围
        # angle_vel_l4_top = trial.suggest_float("angle_vel_l4_top", 10, 20)
        # angle_vel_l4_bottom = trial.suggest_float("angle_vel_l4_bottom", -20, -10)

        # speed_threshold_l3 = trial.suggest_float('speed_threshold_l3', 5.0, 10.0)  # L3速度阈值
        # speed_threshold_l4 = trial.suggest_float('speed_threshold_l4', 8.0, 15.0)  # L4速度阈值
        # max_speed_l3 = trial.suggest_float('max_speed_l3', 8.0, 15.0)             # L3最大速度
        # max_speed_l4 = trial.suggest_float('max_speed_l4', 10.0, 20.0)            # L4最大速度
        
        # 是否使用动态扭矩限制的开关
        use_dynamic_torque = True
        
        
        # 现有的其他参数...
        
        
        
        joint_friction_l3 = trial.suggest_float("joint_friction_l3", 0.02, 0.06)
        joint_friction_l4 = trial.suggest_float("joint_friction_l4", 0.02, 0.06)
        joint_friction_r3 = trial.suggest_float("joint_friction_r3", 0.02, 0.06)
        joint_friction_r4 = trial.suggest_float("joint_friction_r4", 0.02, 0.06)

        return {
            "torque_l3_top": torque_l3_top,
            "torque_l3_bottom": torque_l3_bottom,
            "torque_l4_top": torque_l4_top,
            "torque_l4_bottom": torque_l4_bottom,
            
            # 新增动态扭矩参数
            "speed_threshold_l3_q1": speed_threshold_l3_q1,
            "speed_threshold_l3_q3": speed_threshold_l3_q3,
            "speed_threshold_l4_q1": speed_threshold_l4_q1,
            "speed_threshold_l4_q3": speed_threshold_l4_q3,
            # "max_speed_l3": max_speed_l3,
            # "max_speed_l4": max_speed_l4,
            "use_dynamic_torque": use_dynamic_torque,
            
            # 其他参数
            "angle_vel_l3_top": angle_vel_l3_top,
            "angle_vel_l3_bottom": angle_vel_l3_bottom,
            "angle_vel_l4_top": angle_vel_l4_top,
            "angle_vel_l4_bottom": angle_vel_l4_bottom,
            "joint_friction_l3": joint_friction_l3,
            "joint_friction_l4": joint_friction_l4,
            "joint_friction_r3": joint_friction_r3,
            "joint_friction_r4": joint_friction_r4,
        }

    def objective(self,trial: optuna.Trial) -> float:
        args = self.sample_param(trial)
        # 收集Isaac Gym数据
        sim_data = self.simulate_and_collect_isaac(args,steps=500, command=[1.8, 0., 0., 0.])
        distance = self.compute_distance(sim_data, self.real_data)
        #self.save_and_visualize_data(self.real_data, sim_data, args)            
        self.last_distance_score = distance
    
        # 添加最佳分数检查
        if distance < self.best_score:
            self.best_score = distance
            self.best_params = args
            print(f"New best score: {distance}, params: {args}aaaaaaaa")
            
            # 保存最佳参数时的数据对比
            self.save_and_visualize_data(self.real_data, sim_data, args)

            # 新增：如果有all_real_data，也生成对比图
            if hasattr(self, 'all_real_data') and self.all_real_data is not None:
                try:
                    print("生成新的最佳参数下的all_real_data对比图...")
                    self.plot_all_real_data_torque_velocity_curves(
                        self.all_real_data, 
                        sim_data=sim_data,
                        filename=f'all_real_data_vs_sim_trial_{trial.number}_score_{distance:.4f}.png'
                    )
                except Exception as e:
                    print(f"生成all_real_data对比图失败: {e}")
    
        
        return distance

def check_500_data_timestamps():
    """检查从real_data中提取的500个数据点的时间戳一致性"""
    
    # 使用与当前代码相同的参数
    real_data_file = "data/real_run_data/082929.npz"
    real_data_time_long = 5       # 秒 (预期500个数据点)
    real_data_run = [6178, 6497, 6845, 7193, 7523, 7853, 8210, 8550, 8875]
    real_data_start_offset = real_data_run[1] * 10 - 59000  # 选择第2次运行数据作为示例
    
    print("=== 检查500个数据点的时间戳一致性 ===")
    print(f"文件: {real_data_file}")
    print(f"开始偏移: {real_data_start_offset}秒")
    print(f"持续时间: {real_data_time_long}秒")
    print(f"使用运行数据: real_data_run[1] = {real_data_run[1]}")
    print()
    
    # 加载数据
    np_load_data = np.load(real_data_file)
    
    # 使用与当前代码相同的加载逻辑
    try:
        # 方案1：尝试使用标准字段名
        joint_pos = np_load_data["joint_pos"]
        joint_pos_ts = np_load_data["timestamps_joint_pos"]
        joint_vel = np_load_data["joint_vel"]
        joint_vel_ts = np_load_data["timestamps_joint_vel"]
        actions = np_load_data["actions"]
        actions_ts = np_load_data["timestamps_actions"]
        speeds = np_load_data["linear_vel"]
        speeds_ts = np_load_data["timestamps_linear_vel"]
        print("使用标准字段名加载成功")
    except KeyError as e:
        print(f"标准字段名失败: {e}")
        try:
            joint_pos = np_load_data["jointpos"]
            joint_pos_ts = np_load_data["jointpostimestamps"]
            joint_vel = np_load_data["jointvel"]
            joint_vel_ts = np_load_data["jointveltimestamps"]
            actions = np_load_data["actions"]
            actions_ts = np_load_data["actionstimestamps"]
            speeds = np_load_data["linear_velocity"]
            speeds_ts = np_load_data["timestamps_linear_velocity"]
            print("使用当前字段名加载成功")
        except KeyError as e2:
            print(f"当前字段名也失败: {e2}")
            return
    
    # 检查是否有扭矩数据
    has_torque_data = "motor_cur" in np_load_data.files
    if has_torque_data:
        torques = np_load_data["motor_cur"]
        torques_ts = np_load_data["timestamps_motor_cur"]
        if torques.shape[1] >= 12:
            torques = torques[:, :12]
    
    print("=== 数据基本信息 ===")
    print(f"joint_pos: {joint_pos.shape}, 时间戳数量: {len(joint_pos_ts)}")
    print(f"joint_vel: {joint_vel.shape}, 时间戳数量: {len(joint_vel_ts)}")
    print(f"actions: {actions.shape}, 时间戳数量: {len(actions_ts)}")
    print(f"speeds: {speeds.shape}, 时间戳数量: {len(speeds_ts)}")
    if has_torque_data:
        print(f"torques: {torques.shape}, 时间戳数量: {len(torques_ts)}")
    
    # 使用与当前代码完全相同的索引计算
    action_offset = 25  # actions比joint_pos早25个数据点
    torques_offset = 7901  # torques比joint_pos早7901个数据点
    
    start_idx = real_data_start_offset
    end_idx = real_data_start_offset + real_data_time_long * 100
    action_start_idx = real_data_start_offset + action_offset
    action_end_idx = real_data_start_offset + real_data_time_long * 100 + action_offset
    
    print(f"\n=== 索引计算 ===")
    print(f"action_offset: {action_offset}")
    if has_torque_data:
        print(f"torques_offset: {torques_offset}")
    print(f"joint_pos/joint_vel/speeds 索引: [{start_idx}:{end_idx}] (长度: {end_idx - start_idx})")
    print(f"actions 索引: [{action_start_idx}:{action_end_idx}] (长度: {action_end_idx - action_start_idx})")
    if has_torque_data:
        torque_start_idx = start_idx + torques_offset
        torque_end_idx = end_idx + torques_offset
        print(f"torques 索引: [{torque_start_idx}:{torque_end_idx}] (长度: {torque_end_idx - torque_start_idx})")
    
    # 提取对应的时间戳
    print(f"\n=== 时间戳提取检查 ===")
    
    # joint_pos 时间戳
    joint_pos_ts_slice = joint_pos_ts[start_idx:end_idx]
    print(f"joint_pos 时间戳:")
    print(f"  数量: {len(joint_pos_ts_slice)}")
    print(f"  时间范围: {joint_pos_ts_slice[0]:.10f} - {joint_pos_ts_slice[-1]:.10f}")
    print(f"  总时长: {joint_pos_ts_slice[-1] - joint_pos_ts_slice[0]:.6f}秒")
    
    # joint_vel 时间戳
    joint_vel_ts_slice = joint_vel_ts[start_idx:end_idx]
    print(f"joint_vel 时间戳:")
    print(f"  数量: {len(joint_vel_ts_slice)}")
    print(f"  时间范围: {joint_vel_ts_slice[0]:.10f} - {joint_vel_ts_slice[-1]:.10f}")
    print(f"  总时长: {joint_vel_ts_slice[-1] - joint_vel_ts_slice[0]:.6f}秒")
    
    # actions 时间戳
    actions_ts_slice = actions_ts[action_start_idx:action_end_idx]
    print(f"actions 时间戳:")
    print(f"  数量: {len(actions_ts_slice)}")
    print(f"  时间范围: {actions_ts_slice[0]:.10f} - {actions_ts_slice[-1]:.10f}")
    print(f"  总时长: {actions_ts_slice[-1] - actions_ts_slice[0]:.6f}秒")
    
    # speeds 时间戳
    speeds_ts_slice = speeds_ts[start_idx:end_idx]
    print(f"speeds 时间戳:")
    print(f"  数量: {len(speeds_ts_slice)}")
    print(f"  时间范围: {speeds_ts_slice[0]:.10f} - {speeds_ts_slice[-1]:.10f}")
    print(f"  总时长: {speeds_ts_slice[-1] - speeds_ts_slice[0]:.6f}秒")
    
    if has_torque_data:
        torques_ts_slice = torques_ts[torque_start_idx:torque_end_idx]
        print(f"torques 时间戳:")
        print(f"  数量: {len(torques_ts_slice)}")
        print(f"  时间范围: {torques_ts_slice[0]:.10f} - {torques_ts_slice[-1]:.10f}")
        print(f"  总时长: {torques_ts_slice[-1] - torques_ts_slice[0]:.6f}秒")
    
    # 检查时间间隔一致性
    print(f"\n=== 时间间隔一致性检查 ===")
    
    def analyze_intervals(timestamps, name):
        if len(timestamps) < 2:
            print(f"{name}: 数据点不足")
            return
        
        intervals = np.diff(timestamps)
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        min_interval = np.min(intervals)
        max_interval = np.max(intervals)
        cv = std_interval / mean_interval * 100  # 变异系数
        
        print(f"{name}:")
        print(f"  平均间隔: {mean_interval:.6f}秒")
        print(f"  标准差: {std_interval:.6f}秒")
        print(f"  变异系数: {cv:.4f}%")
        print(f"  间隔范围: [{min_interval:.6f}, {max_interval:.6f}]")
        print(f"  理论频率: {1/mean_interval:.2f} Hz")
        
        # 一致性评估
        if cv < 1.0:
            consistency = "非常一致"
        elif cv < 5.0:
            consistency = "基本一致"
        elif cv < 10.0:
            consistency = "中等一致"
        else:
            consistency = "不一致"
        print(f"  一致性评估: {consistency}")
        
        return intervals
    
    joint_pos_intervals = analyze_intervals(joint_pos_ts_slice, "joint_pos")
    joint_vel_intervals = analyze_intervals(joint_vel_ts_slice, "joint_vel")
    actions_intervals = analyze_intervals(actions_ts_slice, "actions")
    speeds_intervals = analyze_intervals(speeds_ts_slice, "speeds")
    
    if has_torque_data:
        torques_intervals = analyze_intervals(torques_ts_slice, "torques")
    
    # 检查数据源间的时间对齐
    print(f"\n=== 数据源间时间对齐检查 ===")
    
    # 比较开始时间
    print("开始时间对比:")
    print(f"  joint_pos: {joint_pos_ts_slice[0]:.10f}")
    print(f"  joint_vel: {joint_vel_ts_slice[0]:.10f}")
    print(f"  actions: {actions_ts_slice[0]:.10f}")
    print(f"  speeds: {speeds_ts_slice[0]:.10f}")
    if has_torque_data:
        print(f"  torques: {torques_ts_slice[0]:.10f}")
    
    # 比较结束时间
    print("\n结束时间对比:")
    print(f"  joint_pos: {joint_pos_ts_slice[-1]:.10f}")
    print(f"  joint_vel: {joint_vel_ts_slice[-1]:.10f}")
    print(f"  actions: {actions_ts_slice[-1]:.10f}")
    print(f"  speeds: {speeds_ts_slice[-1]:.10f}")
    if has_torque_data:
        print(f"  torques: {torques_ts_slice[-1]:.10f}")
    
    # 计算时间差
    print("\n与joint_pos的时间差:")
    ref_start = joint_pos_ts_slice[0]
    ref_end = joint_pos_ts_slice[-1]
    
    print(f"  joint_vel: 开始差{joint_vel_ts_slice[0] - ref_start:.6f}s, 结束差{joint_vel_ts_slice[-1] - ref_end:.6f}s")
    print(f"  actions: 开始差{actions_ts_slice[0] - ref_start:.6f}s, 结束差{actions_ts_slice[-1] - ref_end:.6f}s")
    print(f"  speeds: 开始差{speeds_ts_slice[0] - ref_start:.6f}s, 结束差{speeds_ts_slice[-1] - ref_end:.6f}s")
    if has_torque_data:
        print(f"  torques: 开始差{torques_ts_slice[0] - ref_start:.6f}s, 结束差{torques_ts_slice[-1] - ref_end:.6f}s")
    
    # 验证数据构建逻辑
    print(f"\n=== 模拟real_data构建过程 ===")
    
    # 按照当前代码的逻辑构建real_data
    if has_torque_data:
        real_data = np.concatenate([
            joint_pos[start_idx:end_idx, :12],  # 关节位置 (12)
            joint_vel[start_idx:end_idx, :12],  # 关节速度 (12)
            actions[action_start_idx:action_end_idx, :12],  # 动作 (12)
            speeds[start_idx:end_idx, :3],  # 基座速度占位符 (3)
            speeds[start_idx:end_idx, :3],  # 世界速度占位符 (3)
            torques[start_idx+torques_offset:end_idx+torques_offset, :12]  # 实际扭矩 (12)
        ], axis=1)
        print(f"构建的real_data形状: {real_data.shape}")
        print(f"包含扭矩数据")
    else:
        real_data = np.concatenate([
            joint_pos[start_idx:end_idx, :12],  # 关节位置 (12)
            joint_vel[start_idx:end_idx, :12],  # 关节速度 (12)
            actions[action_start_idx:action_end_idx, :12],  # 动作 (12)
            speeds[start_idx:end_idx, :3],  # 基座速度占位符 (3)
            speeds[start_idx:end_idx, :3],  # 世界速度占位符 (3)
            np.zeros((end_idx-start_idx, 12))   # 扭矩占位符 (12)
        ], axis=1)
        print(f"构建的real_data形状: {real_data.shape}")
        print(f"使用零填充扭矩数据")
    
    # 验证是否接近500个数据点和5秒
    print(f"\n=== 数据规模验证 ===")
    expected_points = 500
    expected_duration = 5.0
    
    print(f"预期数据点数: {expected_points}")
    print(f"实际数据点数: {len(joint_pos_ts_slice)}")
    print(f"预期持续时间: {expected_duration}秒")
    print(f"实际持续时间: {joint_pos_ts_slice[-1] - joint_pos_ts_slice[0]:.6f}秒")
    
    # 检查采样率
    actual_duration = joint_pos_ts_slice[-1] - joint_pos_ts_slice[0]
    actual_rate = len(joint_pos_ts_slice) / actual_duration
    expected_rate = expected_points / expected_duration
    
    print(f"预期采样率: {expected_rate:.2f} Hz")
    print(f"实际采样率: {actual_rate:.2f} Hz")
    print(f"采样率误差: {abs(actual_rate - expected_rate):.2f} Hz")
    
    # 检查final_torques部分（如果有扭矩数据）
    if has_torque_data:
        print(f"\n=== 扭矩数据详细检查 ===")
        final_torques = real_data[:, -12:]  # 最后12列应该是扭矩
        print(f"real_data中扭矩部分形状: {final_torques.shape}")
        print(f"real_data中扭矩部分范围: [{final_torques.min():.6f}, {final_torques.max():.6f}]")
        print(f"real_data中扭矩部分统计: mean={final_torques.mean():.6f}, std={final_torques.std():.6f}")
        
        print(f"\nreal_data中扭矩前3行:")
        for i in range(min(3, len(final_torques))):
            print(f"  [{i}]: {final_torques[i]}")
    
    print(f"\n=== 检查完成 ===")
    print(f"使用的参数:")
    print(f"  real_data_start_offset: {real_data_start_offset}")
    print(f"  real_data_time_long: {real_data_time_long}")
    print(f"  action_offset: {action_offset}")
    if has_torque_data:
        print(f"  torques_offset: {torques_offset}")

def check_all_data_timestamps(npz_file):
    """检查npz文件中所有数据的时间戳一致性"""
    
    print("=== 检查npz文件中所有数据的时间戳一致性 ===")
    print(f"文件: {npz_file}")
    print()
    
    # 加载数据
    np_load_data = np.load(npz_file)
    
    print("=== 可用的数据字段 ===")
    print("Available data fields:")
    for field in np_load_data.files:
        data_field = np_load_data[field]
        print(f"  {field}: shape={data_field.shape}, dtype={data_field.dtype}")
    print()
    
    # 识别数据和时间戳字段
    data_fields = {}
    timestamp_fields = {}
    
    for field in np_load_data.files:
        if 'timestamp' in field.lower():
            # 这是时间戳字段
            base_name = field.replace('timestamps_', '').replace('_timestamps', '').replace('timestamps', '')
            timestamp_fields[base_name] = field
        else:
            # 这是数据字段
            data_fields[field] = field
    
    print("=== 识别的数据和时间戳配对 ===")
    data_timestamp_pairs = {}
    
    # 尝试匹配数据和时间戳
    for data_name, data_field in data_fields.items():
        # 寻找对应的时间戳字段
        timestamp_field = None
        
        # 方法1：直接匹配
        if data_name in timestamp_fields:
            timestamp_field = timestamp_fields[data_name]
        
        # 方法2：模糊匹配
        if timestamp_field is None:
            for ts_base, ts_field in timestamp_fields.items():
                if data_name in ts_base or ts_base in data_name:
                    timestamp_field = ts_field
                    break
        
        # 方法3：特殊规则匹配
        if timestamp_field is None:
            for ts_base, ts_field in timestamp_fields.items():
                if (data_name == 'jointpos' and 'joint_pos' in ts_base) or \
                   (data_name == 'jointvel' and 'joint_vel' in ts_base) or \
                   (data_name == 'actions' and 'action' in ts_base) or \
                   (data_name == 'linear_velocity' and 'linear_vel' in ts_base):
                    timestamp_field = ts_field
                    break
        
        if timestamp_field:
            data_timestamp_pairs[data_name] = {
                'data_field': data_field,
                'timestamp_field': timestamp_field,
                'data': np_load_data[data_field],
                'timestamps': np_load_data[timestamp_field]
            }
            print(f"✓ {data_name}: {data_field} <-> {timestamp_field}")
        else:
            print(f"✗ {data_name}: {data_field} (无对应时间戳)")
    
    print(f"\n成功配对 {len(data_timestamp_pairs)} 个数据源")
    print()
    
    # 检查每个数据源的时间戳
    print("=== 各数据源时间戳基本信息 ===")
    all_analysis = {}
    
    for data_name, pair_info in data_timestamp_pairs.items():
        data = pair_info['data']
        timestamps = pair_info['timestamps']
        
        print(f"\n{data_name}:")
        print(f"  数据形状: {data.shape}")
        print(f"  时间戳数量: {len(timestamps)}")
        
        if len(timestamps) > 1:
            print(f"  时间范围: {timestamps[0]:.6f} - {timestamps[-1]:.6f}")
            print(f"  总时长: {timestamps[-1] - timestamps[0]:.6f}秒")
            
            # 计算时间间隔统计
            intervals = np.diff(timestamps)
            mean_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            min_interval = np.min(intervals)
            max_interval = np.max(intervals)
            cv = std_interval / mean_interval * 100 if mean_interval > 0 else 0
            
            print(f"  平均时间间隔: {mean_interval:.6f}秒")
            print(f"  时间间隔标准差: {std_interval:.6f}秒")
            print(f"  变异系数(CV): {cv:.4f}%")
            print(f"  间隔范围: [{min_interval:.6f}, {max_interval:.6f}]")
            print(f"  理论频率: {1/mean_interval:.2f} Hz")
            
            # 一致性评估
            if cv < 1.0:
                consistency = "非常一致"
            elif cv < 5.0:
                consistency = "基本一致"
            elif cv < 10.0:
                consistency = "中等一致"
            else:
                consistency = "不一致"
            print(f"  一致性评估: {consistency}")
            
            # 检查是否有异常间隔
            median_interval = np.median(intervals)
            outlier_threshold = 3 * std_interval
            outliers = intervals[np.abs(intervals - median_interval) > outlier_threshold]
            if len(outliers) > 0:
                print(f"  异常间隔数量: {len(outliers)} (超过3倍标准差)")
                print(f"  异常间隔范围: [{outliers.min():.6f}, {outliers.max():.6f}]")
            
            all_analysis[data_name] = {
                'timestamps': timestamps,
                'intervals': intervals,
                'mean_interval': mean_interval,
                'std_interval': std_interval,
                'cv': cv,
                'consistency': consistency,
                'outliers': len(outliers)
            }
        else:
            print(f"  数据点不足，无法分析时间间隔")
    
    # 数据源间时间对齐分析
    if len(all_analysis) > 1:
        print(f"\n=== 数据源间时间对齐分析 ===")
        
        source_names = list(all_analysis.keys())
        reference_source = source_names[0]
        ref_timestamps = all_analysis[reference_source]['timestamps']
        
        print(f"使用 '{reference_source}' 作为参考")
        print(f"\n开始时间对比:")
        print(f"  {reference_source}: {ref_timestamps[0]:.6f}")
        
        print(f"\n结束时间对比:")
        print(f"  {reference_source}: {ref_timestamps[-1]:.6f}")
        
        for source_name in source_names[1:]:
            comp_timestamps = all_analysis[source_name]['timestamps']
            
            print(f"  {source_name}: {comp_timestamps[0]:.6f}")
            
        print(f"\n与 '{reference_source}' 的时间差:")
        ref_start = ref_timestamps[0]
        ref_end = ref_timestamps[-1]
        
        for source_name in source_names[1:]:
            comp_timestamps = all_analysis[source_name]['timestamps']
            start_diff = comp_timestamps[0] - ref_start
            end_diff = comp_timestamps[-1] - ref_end
            print(f"  {source_name}: 开始差{start_diff:.6f}s, 结束差{end_diff:.6f}s")
        
        # 计算重叠时间段
        print(f"\n重叠时间段分析:")
        all_starts = [all_analysis[name]['timestamps'][0] for name in source_names]
        all_ends = [all_analysis[name]['timestamps'][-1] for name in source_names]
        
        overlap_start = max(all_starts)
        overlap_end = min(all_ends)
        overlap_duration = overlap_end - overlap_start
        
        print(f"  重叠开始时间: {overlap_start:.6f}")
        print(f"  重叠结束时间: {overlap_end:.6f}")
        print(f"  重叠时长: {overlap_duration:.6f}秒")
        
        if overlap_duration > 0:
            print(f"  ✓ 所有数据源都有重叠区间")
        else:
            print(f"  ✗ 数据源之间没有重叠区间")
    
    # 生成详细的频率分析
    print(f"\n=== 详细频率分析 ===")
    for data_name, analysis in all_analysis.items():
        print(f"\n{data_name}:")
        intervals = analysis['intervals']
        
        # 计算不同的频率统计
        frequencies = 1.0 / intervals
        mean_freq = np.mean(frequencies)
        std_freq = np.std(frequencies)
        median_freq = np.median(frequencies)
        
        print(f"  平均频率: {mean_freq:.2f} Hz")
        print(f"  中位数频率: {median_freq:.2f} Hz")
        print(f"  频率标准差: {std_freq:.2f} Hz")
        print(f"  频率范围: [{frequencies.min():.2f}, {frequencies.max():.2f}] Hz")
        
        # 频率分布分析
        freq_bins = [50, 90, 110, 150, 200]  # 常见的采样频率阈值
        print(f"  频率分布:")
        for i, threshold in enumerate(freq_bins):
            count = np.sum(frequencies >= threshold)
            percentage = count / len(frequencies) * 100
            print(f"    >={threshold}Hz: {count}个点 ({percentage:.1f}%)")
    
    # 生成时间间隔的分布图
    #create_timestamp_analysis_plots(all_analysis, npz_file)
    
    # 生成总结报告
    print(f"\n=== 总结报告 ===")
    print(f"检查的数据源数量: {len(all_analysis)}")
    
    if all_analysis:
        # 最一致的数据源
        most_consistent = min(all_analysis.items(), key=lambda x: x[1]['cv'])
        print(f"最一致的数据源: {most_consistent[0]} (CV: {most_consistent[1]['cv']:.4f}%)")
        
        # 最不一致的数据源
        least_consistent = max(all_analysis.items(), key=lambda x: x[1]['cv'])
        print(f"最不一致的数据源: {least_consistent[0]} (CV: {least_consistent[1]['cv']:.4f}%)")
        
        # 平均频率最高的数据源
        avg_freqs = {name: 1/analysis['mean_interval'] for name, analysis in all_analysis.items()}
        highest_freq = max(avg_freqs.items(), key=lambda x: x[1])
        print(f"最高频率数据源: {highest_freq[0]} ({highest_freq[1]:.2f} Hz)")
        
        # 有异常的数据源
        sources_with_outliers = [name for name, analysis in all_analysis.items() if analysis['outliers'] > 0]
        if sources_with_outliers:
            print(f"有时间间隔异常的数据源: {', '.join(sources_with_outliers)}")
        else:
            print(f"所有数据源的时间间隔都正常")
    
    print(f"\n检查完成！")

def extract_all_real_data_runs(real_data_file, real_data_time_long=5):
    """
    简化版：提取所有9条real_data的信息
    """
    print("=== 提取所有9条real_data运行信息 ===")
    
    # 所有9条数据的运行信息
    real_data_run = [6358, 6660, 6985, 7315, 7623, 7933, 8270, 8580, 8895]
    
    # 加载npz文件
    np_load_data = np.load(real_data_file)
    
    # 加载数据
    try:
        joint_pos = np_load_data["joint_pos"]
        joint_pos_ts = np_load_data["timestamps_joint_pos"]
        joint_vel = np_load_data["joint_vel"]
        joint_vel_ts = np_load_data["timestamps_joint_vel"]
        actions = np_load_data["actions"]
        actions_ts = np_load_data["timestamps_actions"]
        speeds = np_load_data["linear_vel"]
    except KeyError:
        joint_pos = np_load_data["jointpos"]
        joint_pos_ts = np_load_data["jointpostimestamps"]
        joint_vel = np_load_data["jointvel"]
        joint_vel_ts = np_load_data["jointveltimestamps"]
        actions = np_load_data["actions"]
        actions_ts = np_load_data["actionstimestamps"]
        speeds = np_load_data["linear_velocity"]
    
    # 检查扭矩数据
    has_torque_data = "motor_cur" in np_load_data.files
    if has_torque_data:
        torques = np_load_data["motor_cur"][:, :12]
        # 应用扭矩系数
        torque_coefficients = np.array([2.0, 1.2, 1.2, 4.1, 2.1, 2.1] * 2)
        torques = torques * torque_coefficients
    
    # 时间对齐参数（只保留action_offset）
    action_offset = -20
    
    all_real_data = {}
    
    for run_index, run_value in enumerate(real_data_run):
        print(f"提取第{run_index}条数据 (run_value: {run_value})")
        
        # 计算索引
        real_data_start_offset = run_value * 10 - 59000
        start_idx = real_data_start_offset
        end_idx = real_data_start_offset + real_data_time_long * 100
        action_start_idx = real_data_start_offset + action_offset
        action_end_idx = real_data_start_offset + real_data_time_long * 100 + action_offset
        
        # 检查索引范围
        if (start_idx < 0 or end_idx > len(joint_pos) or 
            action_start_idx < 0 or action_end_idx > len(actions)):
            print(f"  跳过：索引超出范围")
            continue
        
        if has_torque_data:
            # 直接使用相同的索引，不加offset
            if end_idx > len(torques):
                print(f"  ❌ 扭矩索引超出范围")
                print(f"    需要最大索引: {end_idx}")
                print(f"    实际最大索引: {len(torques) - 1}")
                continue
        
        try:
            # 提取数据片段
            joint_pos_segment = joint_pos[start_idx:end_idx, :12]
            joint_vel_segment = joint_vel[start_idx:end_idx, :12]
            actions_segment = actions[action_start_idx:action_end_idx, :12]
            speeds_segment = speeds[start_idx:end_idx, :3]
            
            if has_torque_data:
                # 直接使用相同的索引，不使用offset
                torques_segment = torques[start_idx:end_idx, :12]
                # 构建完整数据数组
                current_real_data = np.concatenate([
                    joint_pos_segment,   # 关节位置 (12)
                    joint_vel_segment,   # 关节速度 (12)
                    actions_segment,     # 动作 (12)
                    speeds_segment,      # 基座速度 (3)
                    speeds_segment,      # 世界速度 (3)
                    torques_segment      # 实际扭矩 (12)
                ], axis=1)
            else:
                # 用零填充扭矩
                zero_torques = np.zeros((end_idx-start_idx, 12))
                current_real_data = np.concatenate([
                    joint_pos_segment,   # 关节位置 (12)
                    joint_vel_segment,   # 关节速度 (12)
                    actions_segment,     # 动作 (12)
                    speeds_segment,      # 基座速度 (3)
                    speeds_segment,      # 世界速度 (3)
                    zero_torques         # 扭矩占位符 (12)
                ], axis=1)
            
            # 存储数据
            all_real_data[run_index] = {
                'data': current_real_data,
                'run_value': run_value,
                'shape': current_real_data.shape,
                'has_torque': has_torque_data
            }
            
            print(f"  ✅ 成功提取，形状: {current_real_data.shape}")
            
        except Exception as e:
            print(f"  ❌ 提取失败: {e}")
            continue
    
    print(f"=== 提取完成：成功 {len(all_real_data)}/{len(real_data_run)} 条 ===")
    return all_real_data

def test_all_real_data_quality(all_real_data):
    """
    简化版：测试所有提取数据的质量
    """
    print(f"=== 测试{len(all_real_data)}条数据质量 ===")
    
    for run_index, data_info in all_real_data.items():
        data = data_info['data']
        
        # 基本检查
        nan_count = np.isnan(data).sum()
        inf_count = np.isinf(data).sum()
        
        # 数据变化检查
        joint_vel = data[:, 12:24]
        actions = data[:, 24:36]
        
        print(f"Run {run_index}: 形状{data.shape}, "
              f"NaN={nan_count}, Inf={inf_count}, "
              f"vel_std={joint_vel.std():.4f}, "
              f"action_std={actions.std():.4f}")
        
        if nan_count > 0 or inf_count > 0:
            print(f"  ⚠️  数据包含异常值")
        elif joint_vel.std() < 1e-6 or actions.std() < 1e-6:
            print(f"  ⚠️  数据变化很小，可能有问题")
        else:
            print(f"  ✅ 数据质量正常")
if __name__ == "__main__":
    #check_500_data_timestamps()
    #check_all_data_timestamps("data/real_run_data/919191.npz")
    args = get_args()
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    env_cfg.rewards.cycle_time = 0.7
    
    train_cfg.seed = SEED
    set_global_seed(SEED)
    
    # 参考play.py设置环境参数
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.sim.max_gpu_contact_pairs = 2**10
    env_cfg.terrain.mesh_type = 'plane'
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False     
    env_cfg.terrain.max_init_terrain_level = 5
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.push_robots = False 
    env_cfg.domain_rand.joint_angle_noise = 0.
    env_cfg.noise.curriculum = False
    env_cfg.noise.noise_level = 0.5
    train_cfg.runner.resume = True

    # 创建Isaac Gym环境和policy
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # 加载JIT policy用于Mujoco
    jit_policy_path = f"../logs/kuavo_jog/exported/policies_test_1/policy_1.pt"
    jit_policy = torch.jit.load(jit_policy_path)

    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    # Do not prune before 1/3 of the max budget is used.
    pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=0)

    # 目标文件夹
    save_dir = "../logs/optuna_results"
    os.makedirs(save_dir, exist_ok=True)
    # 生成带时间戳的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    db_path = os.path.join(save_dir, f"optuna_{timestamp}.sqlite3")
    study = optuna.create_study(sampler=sampler, pruner=pruner, storage=f"sqlite:///{db_path}",study_name= "Gap-kill" )


    initial_params = [
        0.007,    # 初始关节摩擦系数
        35.0,     # 初始关节最大扭矩
        -35.0,    # 初始关节最小扭矩
        8.0,      # 初始关节最大速度 (rad/s)
        -8.0,     # 初始关节最小速度 (rad/s)
        # 新增：动态扭矩参数
        5.0,      # L3速度阈值
        7.0,      # L4速度阈值
        # 10.0,     # L3最大速度
        # 12.0,     # L4最大速度
    ]
    
    initial_sigmas = [
        0.05,     # 摩擦力的搜索步长
        10.0,     # 最大扭矩的搜索步长
        10.0,     # 最小扭矩的搜索步长
        1.0,      # 最大速度的搜索步长
        1.0,      # 最小速度的搜索步长
        # 新增：动态扭矩参数的搜索步长
        1.0,      # L3速度阈值搜索步长
        1.0,      # L4速度阈值搜索步长
        # 2.0,      # L3最大速度搜索步长
        # 2.0,      # L4最大速度搜索步长
    ]
    
    # ... 其余代码保持不变 ...
    # 创建优化器
    optimizer = Sim2RealCMAOptimizer(
        initial_params=initial_params,  # 初始化各个系数
        sigma0=initial_sigmas,           # 初始搜索步长
        real_data=None,        # 将在optimize中设置
        env=env,
        policy=policy,
        jit_policy=jit_policy,
        max_iter=10
    )
    
    optimizer.all_real_data = all_real_data if 'all_real_data' in locals() else None
    # 收集"真实"数据（来自Mujoco或实机）
    real_data_from = "real"   #"mujoco" or "real"
    real_data_file = "data/real_run_data/919191.npz"
    #real_data_start_offset = 199
    # real_data_start_offset = 199
    real_data_time_long = 5     #单位s

    # real_data_run = [6178,6497,6845,7193,7523,7853,8210,8550,8875]
    real_data_run = [6358, 6660,6985,7315,7623,7933,8270,8580,8895]

    real_data_start_offset = real_data_run[7] * 10 - 59000  # 选择第4次运行数据作为示例

    if real_data_from=="mujoco":
        print("Collecting 'real' data from Mujoco...")
        real_data = optimizer.collect_mujoco_data(steps=500, command=[1.8,0., 0., 0.])
        optimizer.real_data = real_data
        print(f"Collected real data shape: {real_data.shape}")
    # filepath: /home/wegg/humanoid-gym/humanoid/scripts/optuna_kill_gap.py
# 在数据加载前添加详细检查
    else:
        np_load_data = np.load(real_data_file)
        print("=== 检查 .npz 文件字段 ===")
        print("Available data fields in real data file:")
        print(np_load_data.files)  # 这会显示所有可用的数据字段
        
        # 打印每个字段的形状和类型
        for field in np_load_data.files:
            data_field = np_load_data[field]
            print(f"  {field}: shape={data_field.shape}, dtype={data_field.dtype}")
        
        print("\n" + "="*50)
        
        # 根据实际字段名加载数据 - 请根据上面的输出调整字段名
        try:
            # 方案1：尝试使用标准字段名
            joint_pos = np_load_data["joint_pos"]
            joint_pos_ts = np_load_data["timestamps_joint_pos"]
            joint_vel = np_load_data["joint_vel"]
            joint_vel_ts = np_load_data["timestamps_joint_vel"]
            actions = np_load_data["actions"]
            actions_ts = np_load_data["timestamps_actions"]
            speeds = np_load_data["linear_vel"]
            speeds_ts = np_load_data["timestamps_linear_vel"]
            print("使用标准字段名加载成功")
        except KeyError as e:
            print(f"标准字段名失败: {e}")
            # 方案2：尝试使用你当前的字段名
            try:
                joint_pos = np_load_data["jointpos"]
                joint_pos_ts = np_load_data["jointpostimestamps"]
                joint_vel = np_load_data["jointvel"]
                joint_vel_ts = np_load_data["jointveltimestamps"]
                actions = np_load_data["actions"]
                actions_ts = np_load_data["actionstimestamps"]
                speeds = np_load_data["linear_velocity"]
                speeds_ts = np_load_data["timestamps_linear_velocity"]
                print("使用当前字段名加载成功")
            except KeyError as e2:
                print(f"当前字段名也失败: {e2}")
                print("请检查 .npz 文件的实际字段名")
                exit(1)
        
       
        
        
                
        # 新增：检查并加载扭矩数据
        if "motor_cur" in np_load_data.files:
            torques = np_load_data["motor_cur"]
            torques_ts = np_load_data["timestamps_motor_cur"]
            print(f"Found torque data with shape: {torques.shape}")
            
            # 只取前12个关节（腿部关节）的扭矩数据
            if torques.shape[1] >= 12:
                torques = torques[:, :12]  # 只取前12列
                print(f"Using only first 12 joints (legs), new shape: {torques.shape}")
                
                # 定义12个腿部关节的扭矩系数
                torque_coefficients = np.array([
                    2.0, 1.2, 1.2, 4.1, 2.1, 2.1,  # 左腿关节系数
                    2.0, 1.2, 1.2, 4.1, 2.1, 2.1   # 右腿关节系数
                ])
                
                # 应用扭矩系数
                torques = torques * torque_coefficients
                print(f"Applied torque coefficients: {torque_coefficients}")
            else:
                print(f"Warning: Expected at least 12 joints but got {torques.shape[1]}")
                torques = torques * np.ones(torques.shape[1])  # 使用全1系数
            
            has_torque_data = True
        
        # 找到重叠时间段
        if has_torque_data:
            start_time = max(joint_pos_ts[0], joint_vel_ts[0], actions_ts[0], torques_ts[0])
            end_time = min(joint_pos_ts[-1], joint_vel_ts[-1], actions_ts[-1], torques_ts[-1])
        else:
            start_time = max(joint_pos_ts[0], joint_vel_ts[0], actions_ts[0])
            end_time = min(joint_pos_ts[-1], joint_vel_ts[-1], actions_ts[-1])
            
        print(f">>>实机数据总长 :{end_time - start_time} s")   
        
        print(f"{start_time=}, {end_time=}")
        print(f"Requested data segment: start_offset={real_data_start_offset}s, duration={real_data_time_long}s")
        # 计算时间对齐的偏移
        action_offset = -20 # actions比joint_pos早25个数据点0
        
        torques_offset = 7901 # torques比joint_pos早7902个数据点

        # print("=== actions_ts 前50个时间戳（高精度）===")
        # for i, ts in enumerate(torques_ts[7900:7930]):
        #     print(f"torques_ts[{i:2d}]: {ts:.10f}")
        
        # print("\n=== joint_pos_ts 前10个时间戳（高精度）===")
        # for i, ts in enumerate(joint_pos_ts[0:10]):
        #     print(f"joint_pos_ts[{i:2d}]: {ts:.10f}")
        

        # 构建数据数组
        start_idx = real_data_start_offset
        end_idx = real_data_start_offset+real_data_time_long*100
        action_start_idx = real_data_start_offset + action_offset   # actions时间对齐偏移
        action_end_idx = real_data_start_offset+real_data_time_long*100 + action_offset
        

        print(f"\n=== 数据切片索引检查 ===")
        print(f"speeds: start_idx={start_idx}, end_idx={end_idx}, shape={speeds.shape}")
        print(speeds_ts[start_idx], speeds_ts[end_idx-1])
        print(f"joint_pos: start_idx={start_idx}, end_idx={end_idx}, shape={joint_pos.shape}")
        print(joint_pos_ts[start_idx], joint_pos_ts[end_idx-1])
        print(f"torques: start_idx+torques_offset={start_idx+torques_offset}, end_idx+torques_offset={end_idx+torques_offset}, shape={torques.shape}")
        print(torques_ts[start_idx], torques_ts[end_idx-1])
        print(action_start_idx, action_end_idx, actions.shape)
        print(actions_ts[action_start_idx], actions_ts[action_end_idx-1])
        print(actions_ts[start_idx], actions_ts[end_idx-1])


        if has_torque_data:
            # 包含真实扭矩数据
            real_data = np.concatenate([
                joint_pos[start_idx:end_idx, :12],  # 关节位置 (12)
                joint_vel[start_idx:end_idx, :12],  # 关节速度 (12)
                actions[action_start_idx:action_end_idx, :12],  # 动作 (12)
                speeds[start_idx:end_idx, :3],  # 基座速度占位符 (3)
                speeds[start_idx:end_idx, :3],  # 世界速度占位符 (3)
                torques[start_idx:end_idx, :12]     # 实际扭矩 (12)
            ], axis=1)
            print(f"Real data with torques shape: {real_data.shape}")
        else:
            # 没有扭矩数据，用零填充
            real_data = np.concatenate([
                joint_pos[start_idx:end_idx, :12],  # 关节位置 (12)
                joint_vel[start_idx:end_idx, :12],  # 关节速度 (12)
                actions[action_start_idx:action_end_idx, :12],  # 动作 (12)
                speeds[start_idx:end_idx, :3],  # 基座速度占位符 (3)
                speeds[start_idx:end_idx, :3],  # 世界速度占位符 (3)
                np.zeros((end_idx-start_idx, 12))   # 扭矩占位符 (12)
            ], axis=1)
            print(f"Real data without torques (using zeros) shape: {real_data.shape}")
            print("Warning: Using zero torques for real data - torque comparison will not be meaningful")
        
        print("\n=== 最终real_data中的扭矩检查 ===")

        
        if has_torque_data:
            final_torques = real_data[:, -12:]  # 最后12列应该是扭矩
            # print(f"real_data中扭矩部分形状: {final_torques.shape}")
            # print(f"real_data中扭矩部分范围: [{final_torques.min():.6f}, {final_torques.max():.6f}]")
            # print(f"real_data中扭矩部分统计: mean={final_torques.mean():.6f}, std={final_torques.std():.6f}")
            
            # print(f"\nreal_data中扭矩前3行:")
            # print(final_torques[:3])
        optimizer.real_data = real_data
        print(f"Final real data size: {len(real_data)}")





    # else:
    #     np_load_data = np.load(real_data_file)
    #     print(np_load_data.files)
    #     joint_pos = np_load_data["jointpos"]
    #     joint_pos_ts = np_load_data["jointpostimestamps"]

    #     joint_vel = np_load_data["jointvel"]
    #     joint_vel_ts = np_load_data["jointveltimestamps"]

    #     actions = np_load_data["actions"]
    #     actions_ts = np_load_data["actionstimestamps"]
    #     # 找到重叠时间段
    #     start_time = max(joint_pos_ts[0], joint_vel_ts[0], actions_ts[0])
    #     end_time = min(joint_pos_ts[-1], joint_vel_ts[-1], actions_ts[-1])     
    #     print(f">>>实机数据总长 :{end_time - start_time} s")   
    #     # 计算时间对齐的偏移
    #     offset = np.abs(len(actions) - len(joint_pos))

    #     real_data = np.concatenate([
    #         joint_pos[real_data_start_offset*100+offset:real_data_start_offset*100+real_data_time_long*100+offset,:12],
    #         joint_vel[real_data_start_offset*100+offset:real_data_start_offset*100+real_data_time_long*100+offset,:12],
    #         actions[real_data_start_offset*100:real_data_start_offset*100+real_data_time_long*100,:12],
    #         np.random.rand(500,6)
    #     ],axis=1)
    #     optimizer.real_data = real_data
    #     print(f"size of realdata: {len(real_data)}")

    try:
        all_real_data = extract_all_real_data_runs(
            real_data_file=real_data_file, 
            real_data_time_long=real_data_time_long
        )
        
        # 简单的质量测试
        test_all_real_data_quality(all_real_data)
        
        print(f"✅ 成功提取{len(all_real_data)}条real_data")
        
        # 设置优化器的参数用于理论曲线绘制
        optimizer.best_params = {
            'torque_l3_top': 35.0,
            'torque_l3_bottom': -35.0,
            'torque_l4_top': 150.0,
            'torque_l4_bottom': -150.0,
            'speed_threshold_l3': 5.0,
            'speed_threshold_l4': 7.0,
            'max_speed_l3': 10.0,
            'max_speed_l4': 12.0,
            'use_dynamic_torque': True,
        }
        
        # 绘制所有real_data的扭矩-速度曲线（暂时不包含仿真对比）
        optimizer.plot_all_real_data_torque_velocity_curves(
            all_real_data, 
            sim_data=None,  # 先不包含仿真数据
            filename='all_real_data_torque_velocity_baseline.png'
        )
        
        print("✅ 所有real_data的基础扭矩-速度曲线绘制完成")
        
    except Exception as e:
        print(f"❌ 处理all_real_data失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 开始优化流程，在优化过程中收集仿真数据
    print("\n=== 开始优化流程 ===")
    
    try:
        study.optimize(optimizer.objective, n_trials=N_TRIALS)
    except KeyboardInterrupt:
        print("优化被用户中断")
    
    # 优化完成后，使用最佳参数生成最终对比图
    if hasattr(optimizer, 'best_params') and optimizer.best_params is not None:
        print(f"\n=== 优化完成，使用最佳参数生成最终对比 ===")
        print(f"最佳参数: {optimizer.best_params}")
        print(f"最佳分数: {optimizer.best_score}")
        
        # 使用最佳参数再次收集仿真数据（用于最终对比图）
        print("使用最佳参数收集最终仿真数据...")
        final_sim_data = optimizer.simulate_and_collect_isaac(
            optimizer.best_params, 
            steps=500, 
            command=[1.8, 0., 0., 0.]
        )
        
        # 更新优化器的最佳参数（确保理论曲线使用正确参数）
        optimizer.best_params.update(optimizer.best_params)
        
        # 绘制包含最佳仿真数据的完整对比图
        try:
            optimizer.plot_all_real_data_torque_velocity_curves(
                all_real_data, 
                sim_data=final_sim_data,
                filename='all_real_data_vs_optimized_sim_comparison.png'
            )
            print("✅ 最终对比图生成完成")
        except Exception as e:
            print(f"❌ 生成最终对比图失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("⚠️ 没有找到最佳参数，跳过最终对比图生成")
    
    print("\n=== 程序执行完成 ===")

    
    # 收集初始Isaac Gym数据进行对比
    # print("Collecting initial Isaac Gym data for comparison...")
    # initial_isaac_data = optimizer.simulate_and_collect_isaac(initial_params, steps=500, command=[1.7, 0., 0., 0.])
    
    # 生成初始对比
    # optimizer.save_and_visualize_data(real_data, initial_isaac_data, initial_params)
    
    # 执行优化
    # best_params = optimizer.optimize()
    # print(f"Optimization completed. Best parameters: {best_params}")
    
    # 最终对比
    # print("Generating final comparison with optimized parameters...")
    # final_isaac_data = optimizer.simulate_and_collect_isaac(best_params, steps=500, command=[1.7, 0., 0., 0.])
    
    # 创建最终对比目录
    # final_dir = f"final_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    # os.makedirs(final_dir, exist_ok=True)
    # optimizer.save_dir = final_dir
    # optimizer.save_and_visualize_data(real_data, final_isaac_data, best_params)



