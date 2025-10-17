#!/usr/bin/env python3
"""
悬挂机器人可视化脚本（带预设动作和碰撞检测）
显示悬挂的机器人，腿部执行预设的周期性动作，并检测两腿之间的碰撞

使用方法：
python view_suspended.py --task=kuavo_ppo --num_envs=1

按键说明：
- 鼠标操作查看器
- ESC 退出
"""


from humanoid.envs import *
from humanoid.utils import get_args, task_registry
import torch
import numpy as np
from isaacgym import gymapi
from isaacgym import gymtorch 
import argparse
import matplotlib
# 🔥 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from collections import defaultdict
import os
from pathlib import Path
from datetime import datetime

class JointDataCollector:
    """关节数据收集器"""
    
    def __init__(self, num_joints=12):
        self.num_joints = num_joints
        
        # 存储数据
        self.time_steps = []
        self.actions = defaultdict(list)  # 目标动作
        self.positions = defaultdict(list)  # 实际位置
        
        # 关节名称
        self.joint_names = [
            "左髋俯仰", "左髋外展", "左髋旋转",
            "左膝关节", "左踝俯仰", "左踝侧倾",
            "右髋俯仰", "右髋外展", "右髋旋转",
            "右膝关节", "右踝俯仰", "右踝侧倾"
        ]
    
    def collect(self, t, actions, dof_pos):
        """
        收集数据
        
        Args:
            t: 当前时间
            actions: 动作张量 [num_envs, num_actions]
            dof_pos: 关节位置张量 [num_envs, num_dofs]
        """
        self.time_steps.append(t)
        
        # 收集第一个环境的数据
        actions_np = actions[0].cpu().numpy()
        dof_pos_np = dof_pos[0].cpu().numpy()
        
        for i in range(self.num_joints):
            self.actions[i].append(actions_np[i])
            self.positions[i].append(dof_pos_np[i])
    
    def plot(self, save_path="joint_data.png"):
        """
        绘制所有关节的动作和位置曲线
        
        Args:
            save_path: 保存路径
        """
        if len(self.time_steps) == 0:
            print("⚠️  没有数据可以绘制")
            return
        
        time_array = np.array(self.time_steps)
        
        # ========== 绘制 Actions 图像 ==========
        fig_actions, axes_actions = plt.subplots(4, 3, figsize=(18, 16))
        fig_actions.suptitle('关节目标动作 (Actions)', fontsize=16, fontweight='bold')
        
        for idx in range(self.num_joints):
            row = idx // 3
            col = idx % 3
            ax = axes_actions[row, col]
            
            actions_array = np.array(self.actions[idx])
            
            # 绘制曲线
            ax.plot(time_array, actions_array, 'b-', linewidth=2, alpha=0.8)
            
            # 设置标题和标签
            ax.set_title(f'{self.joint_names[idx]} (关节 {idx})', fontsize=11, fontweight='bold')
            ax.set_xlabel('时间 (s)', fontsize=9)
            ax.set_ylabel('角度 (rad)', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # 计算统计信息
            mean_val = np.mean(actions_array)
            max_val = np.max(actions_array)
            min_val = np.min(actions_array)
            range_val = max_val - min_val
            
            # 在图上显示统计信息
            stats_text = f'平均: {mean_val:.4f}\n范围: {range_val:.4f}\n最大: {max_val:.4f}\n最小: {min_val:.4f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=7, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        plt.tight_layout()
        actions_path = save_path.replace('.png', '_actions.png')
        plt.savefig(actions_path, dpi=150, bbox_inches='tight')
        print(f"✅ Actions 图像已保存到: {actions_path}")
        plt.close(fig_actions)
        
        # ========== 绘制 Positions 图像 ==========
        fig_positions, axes_positions = plt.subplots(4, 3, figsize=(18, 16))
        fig_positions.suptitle('关节实际位置 (Positions)', fontsize=16, fontweight='bold')
        
        for idx in range(self.num_joints):
            row = idx // 3
            col = idx % 3
            ax = axes_positions[row, col]
            
            positions_array = np.array(self.positions[idx])
            
            # 绘制曲线
            ax.plot(time_array, positions_array, 'r-', linewidth=2, alpha=0.8)
            
            # 设置标题和标签
            ax.set_title(f'{self.joint_names[idx]} (关节 {idx})', fontsize=11, fontweight='bold')
            ax.set_xlabel('时间 (s)', fontsize=9)
            ax.set_ylabel('角度 (rad)', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # 计算统计信息
            mean_val = np.mean(positions_array)
            max_val = np.max(positions_array)
            min_val = np.min(positions_array)
            range_val = max_val - min_val
            
            # 在图上显示统计信息
            stats_text = f'平均: {mean_val:.4f}\n范围: {range_val:.4f}\n最大: {max_val:.4f}\n最小: {min_val:.4f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=7, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
        
        plt.tight_layout()
        positions_path = save_path.replace('.png', '_positions.png')
        plt.savefig(positions_path, dpi=150, bbox_inches='tight')
        print(f"✅ Positions 图像已保存到: {positions_path}")
        plt.close(fig_positions)
        
        # ========== 绘制对比图像 ==========
        fig_compare, axes_compare = plt.subplots(4, 3, figsize=(18, 16))
        fig_compare.suptitle('关节动作与位置对比', fontsize=16, fontweight='bold')
        default_joint_angles = {
            "leg_l1_joint": 0.0,
            "leg_l2_joint": 0.0,
            "leg_l3_joint": -0.47,
            "leg_l4_joint": 0.86,
            "leg_l5_joint": -0.44,
            "leg_l6_joint": 0.0,
            "leg_r1_joint": 0.0,
            "leg_r2_joint": 0.0,
            "leg_r3_joint": -0.47,
            "leg_r4_joint": 0.86,
            "leg_r5_joint": -0.44,
            "leg_r6_joint": 0.0,
        }
        for idx in range(self.num_joints):
            row = idx // 3
            col = idx % 3
            ax = axes_compare[row, col]
            
            actions_array = np.array(self.actions[idx])*0.25+default_joint_angles[list(default_joint_angles.keys())[idx]]
            positions_array = np.array(self.positions[idx])
            
            # 绘制曲线
            ax.plot(time_array, actions_array, 'b-', label='目标动作', linewidth=1.5, alpha=0.8)
            ax.plot(time_array, positions_array, 'r--', label='实际位置', linewidth=1.5, alpha=0.8)
            
            # 设置标题和标签
            ax.set_title(f'{self.joint_names[idx]} (关节 {idx})', fontsize=11, fontweight='bold')
            ax.set_xlabel('时间 (s)', fontsize=9)
            ax.set_ylabel('角度 (rad)', fontsize=9)
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3)
            
            # 计算误差统计
            error = np.abs(actions_array - positions_array)
            mean_error = np.mean(error)
            max_error = np.max(error)
            
            # 在图上显示统计信息
            stats_text = f'平均误差: {mean_error:.4f}\n最大误差: {max_error:.4f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=7, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        compare_path = save_path.replace('.png', '_compare.png')
        plt.savefig(compare_path, dpi=150, bbox_inches='tight')
        print(f"✅ 对比图像已保存到: {compare_path}")
        plt.close(fig_compare)
    
    def print_statistics(self):
        """打印统计信息"""
        if len(self.time_steps) == 0:
            print("⚠️  没有数据")
            return
        
        print("\n" + "="*70)
        print("📊 关节数据统计")
        print("="*70)
        print(f"采样点数: {len(self.time_steps)}")
        print(f"时间范围: {self.time_steps[0]:.2f}s - {self.time_steps[-1]:.2f}s")
        print(f"持续时间: {self.time_steps[-1] - self.time_steps[0]:.2f}s")
        print("\n各关节误差统计:")
        print("-"*70)
        print(f"{'关节名称':<12} | {'平均误差':>10} | {'最大误差':>10} | {'标准差':>10}")
        print("-"*70)
        
        for idx in range(self.num_joints):
            actions_array = np.array(self.actions[idx])
            positions_array = np.array(self.positions[idx])
            error = np.abs(actions_array - positions_array)
            
            mean_error = np.mean(error)
            max_error = np.max(error)
            std_error = np.std(error)
            
            print(f"{self.joint_names[idx]:<12} | {mean_error:>10.6f} | {max_error:>10.6f} | {std_error:>10.6f}")
        
        print("="*70 + "\n")
    
def generate_walking_motion(t, env):
    """
    生成行走动作
    
    Args:
        t: 当前时间
        env: 环境对象
    
    Returns:
        actions: 关节动作张量
    """
    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device, dtype=torch.float)
    
    # 参数设置
    freq = 1.0  # 频率 (Hz)
    
    # 假设关节顺序：
    # 0-2: 左髋 (俯仰, 外展, 旋转)
    # 3-5: 左膝, 左踝俯仰, 左踝侧倾
    # 6-8: 右髋 (俯仰, 外展, 旋转)
    # 9-11: 右膝, 右踝俯仰, 右踝侧倾
    
    # 行走模式：左右腿交替摆动
    phase_left = np.sin(2 * np.pi * freq * t)
    phase_right = np.sin(2 * np.pi * freq * t + np.pi)  # 相位差180度
    
    # 髋关节俯仰 (前后摆动)
    hip_amplitude = 1.4
    actions[:, 0] = hip_amplitude * phase_left   # 左髋
    actions[:, 6] = hip_amplitude * phase_right  # 右髋
    
    # 膝关节 (弯曲配合髋关节)
    knee_amplitude = 5.8
    actions[:, 3] = knee_amplitude * (0.5 + 0.5 * np.abs(phase_left))   # 左膝
    actions[:, 9] = knee_amplitude * (0.5 + 0.5 * np.abs(phase_right))  # 右膝
    
    # 踝关节 (轻微补偿)
    ankle_amplitude = 1.2
    actions[:, 4] = ankle_amplitude * phase_left   # 左踝
    actions[:, 10] = ankle_amplitude * phase_right  # 右踝
    
    return actions



def generate_single_sin_motion(t, env):
    """
    让每个关节依次做正弦运动，左右对称处理
    🔥 每个关节运行一个完整周期后切换
    
    Args:
        t: 当前时间
        env: 环境对象
    
    Returns:
        actions: 关节动作张量
        is_finished: 是否完成所有关节的运动
    """
    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device, dtype=torch.float)
    
    # 🔥 正弦波参数
    freq_base = 0.5  # 频率 (Hz)
    omega = 2 * np.pi * freq_base
    
    # 🔥 一个完整周期的时间
    period_duration = 1.0 / freq_base  # T = 1/f = 2.0秒
    
    # 关节对（左右对称）
    joint_pairs = [
        (0, 6, "髋俯仰"),
        (1, 7, "髋外展"),
        (2, 8, "髋旋转"),
        (3, 9, "膝关节"),
        (4, 10, "踝俯仰"),
        (5, 11, "踝侧倾"),
    ]
    
    # 🔥 计算当前激活的关节对（基于完整周期）
    total_cycle_time = period_duration * len(joint_pairs)
    current_joint_idx = int(t / period_duration)
    t_joint = t % period_duration  # 当前关节的局部时间
    
    # 🔥 检查是否完成所有关节
    is_finished = (t >= total_cycle_time)
    
    if current_joint_idx < len(joint_pairs) and not is_finished:
        left_joint, right_joint, joint_name = joint_pairs[current_joint_idx]
        
        # 🔥 计算正弦值 (从0开始: sin(0) = 0)
        phase = np.sin(omega * t_joint)
        
        # 根据不同关节设置不同的幅度
        if "髋俯仰" in joint_name:
            amplitude = 1.2
        elif "髋外展" in joint_name:
            amplitude = 1.5
        elif "髋旋转" in joint_name:
            amplitude = 2.0
        elif "膝关节" in joint_name:
            amplitude = 2.5
        elif "踝俯仰" in joint_name:
            amplitude = 1.4
        else:  # 踝侧倾
            amplitude = 1.2
        
        # 左右对称运动
        actions[:, left_joint] = amplitude * phase
        actions[:, right_joint] = amplitude * phase
        
        # 🔥 显示进度
        cycle_progress = (t_joint / period_duration) * 100
        if int(t * 10) % 25 == 0 and cycle_progress < 5:
            print(f"🔄 关节 {current_joint_idx + 1}/{len(joint_pairs)}: {joint_name} | "
                  f"进度: {cycle_progress:.1f}% | 剩余: {(total_cycle_time - t):.1f}s | "
                  f"幅值: {amplitude * phase:.3f}")
    
    return actions, is_finished



def generate_single_fourier_motion(t, env):
    """
    让每个关节依次运动，左右对称处理
    🔥 每个关节运行一个完整周期后切换
    
    Args:
        t: 当前时间
        env: 环境对象
    
    Returns:
        actions: 关节动作张量
        is_finished: 是否完成所有关节的运动
    """
    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device, dtype=torch.float)
    
    # 🔥 傅里叶级数参数
    freq_base = 0.2
    omega = 2 * np.pi * freq_base
    
    # 🔥 一个完整周期的时间
    period_duration = 1.0 / freq_base  # T = 1/f ≈ 6.67秒
    
    # 关节对（左右对称）
    joint_pairs = [
        (0, 6, "髋俯仰"),
        (1, 7, "髋外展"),
        (2, 8, "髋旋转"),
        (3, 9, "膝关节"),
        (4, 10, "踝俯仰"),
        (5, 11, "踝侧倾"),
    ]
    
    # 🔥 计算当前激活的关节对（基于完整周期）
    total_cycle_time = period_duration * len(joint_pairs)
    current_joint_idx = int(t / period_duration)
    t_joint = t % period_duration  # 当前关节的局部时间
    
    # 🔥 检查是否完成所有关节
    is_finished = (t >= total_cycle_time)
    
    if current_joint_idx < len(joint_pairs) and not is_finished:
        left_joint, right_joint, joint_name = joint_pairs[current_joint_idx]
        
        # 应用时间偏移，从零点开始
        t_joint -= 0.11864 / freq_base
        
        # 🔥 计算傅里叶级数值
        phase = ( 0.8*np.sin(omega * t_joint) + 0.7*np.cos(omega*t_joint)
                 -0.2 * np.sin(3 * omega * t_joint) +  0.3 * np.cos(3 * omega * t_joint)
                 -0.2 * np.sin(5 * omega * t_joint) +  -0.2 * np.cos(5 * omega * t_joint))
        
        # 根据不同关节设置不同的幅度
        if "髋俯仰" in joint_name:
            amplitude = 1.0
        elif "髋外展" in joint_name:
            amplitude = 1.0
        elif "髋旋转" in joint_name:
            amplitude = 2.3
        elif "膝关节" in joint_name:
            amplitude = 1.0
            phase = 1.5 * phase
        elif "踝俯仰" in joint_name:
            amplitude = 1.4
        else:  # 踝侧倾
            amplitude = 1.2
        
        # 左右对称运动
        actions[:, left_joint] = amplitude * phase
        actions[:, right_joint] = amplitude * phase
        
        # 🔥 显示进度
        cycle_progress = (t_joint / period_duration) * 100
        if int(t * 10) % 25 == 0 and cycle_progress < 5:
            print(f"🔄 关节 {current_joint_idx + 1}/{len(joint_pairs)}: {joint_name} | "
                  f"进度: {cycle_progress:.1f}% | 剩余: {(total_cycle_time - t):.1f}s")
    
    return actions, is_finished

def generate_zero_motion(t, env):
    """
    生成零动作（保持默认姿态）
    
    Args:
        t: 当前时间
        env: 环境对象
    
    Returns:
        actions: 关节动作张量
        is_finished: 是否完成（保持5秒后结束）
    """
    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device, dtype=torch.float)
    
    hold_time = 5.0  # 保持5秒
    is_finished = (t >= hold_time)
    
    return actions, is_finished

def generate_single_hip_motion(t, env):
    """
    让髋关节做3D圆周运动
    🔥 运行5个周期后结束
    
    Args:
        t: 当前时间
        env: 环境对象
    
    Returns:
        actions: 关节动作张量
        is_finished: 是否完成运动
    """
    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device, dtype=torch.float)
    
    # 圆周运动参数
    freq = 0.3
    num_cycles = 5
    total_time = num_cycles / freq
    
    # 检查是否完成
    is_finished = (t >= total_time)
    
    if not is_finished:
        radius_pitch = 1.8
        radius_abduct = 1.8
        radius_rotate = 1.0
        
        angle = 2 * np.pi * freq * t
        
        pitch_value = radius_pitch * np.sin(2 * angle)
        abduct_value = radius_abduct * np.sin(2 * angle)
        rotate_value = radius_rotate * np.sin(3 * angle)
        
        actions[:, 0] = pitch_value
        actions[:, 1] = abduct_value
        actions[:, 2] = rotate_value
        
        actions[:, 6] = pitch_value
        actions[:, 7] = abduct_value
        actions[:, 8] = rotate_value
    
    return actions, is_finished

def generate_single_ankle_motion(t, env):
    """
    让踝关节做3D圆周运动
    🔥 运行5个周期后结束
    
    Args:
        t: 当前时间
        env: 环境对象
    
    Returns:
        actions: 关节动作张量
        is_finished: 是否完成运动
    """
    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device, dtype=torch.float)
    
    # 圆周运动参数
    freq = 0.5
    num_cycles = 5
    total_time = num_cycles / freq
    
    # 检查是否完成
    is_finished = (t >= total_time)
    
    if not is_finished:
        radius_pitch = 1.2
        radius_roll = 1.2
        
        angle = 2 * np.pi * freq * t
        
        pitch_value = radius_pitch * np.cos(angle)
        roll_value = radius_roll * np.sin(angle)
        
        actions[:, 4] = pitch_value
        actions[:, 5] = roll_value
        
        actions[:, 10] = pitch_value
        actions[:, 11] = roll_value
    
    return actions, is_finished

def generate_leg_motion(t, env):
    """
    让所有腿部关节同时运动，形成球形轨迹
    🔥 运行10秒后结束
    
    Args:
        t: 当前时间
        env: 环境对象
    
    Returns:
        actions: 关节动作张量
        is_finished: 是否完成运动
    """
    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device, dtype=torch.float)
    
    total_time = 10.0  # 运行10秒
    is_finished = (t >= total_time)
    
    if not is_finished:
        omega_theta = 2.4
        omega_phi = 2.6
        
        theta_period = 2 * np.pi / omega_theta
        t_normalized = (t % theta_period) / theta_period
        
        if t_normalized < 0.5:
            theta = 2 * t_normalized * np.pi
        else:
            theta = 2 * (1 - t_normalized) * np.pi
        
        phi = (omega_phi * t) % (2 * np.pi)
        
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        
        hip_pitch_amp = 1.2
        hip_abduct_amp = 1.2
        hip_rotate_amp = 1.0
        knee_amp = 1.8
        ankle_pitch_amp = 1.5
        ankle_roll_amp = 1.5
        
        actions[:, 0] = hip_pitch_amp * z
        actions[:, 6] = hip_pitch_amp * z
        
        actions[:, 1] = hip_abduct_amp * y
        actions[:, 7] = hip_abduct_amp * y
        
        actions[:, 2] = hip_rotate_amp * x
        actions[:, 8] = hip_rotate_amp * x
        
        knee_normalized = theta / np.pi
        knee_value = knee_amp * (0.3 + 0.7 * knee_normalized)
        actions[:, 3] = knee_value
        actions[:, 9] = knee_value
        
        ankle_pitch_value = ankle_pitch_amp * (x * z)
        ankle_roll_value = ankle_roll_amp * (y * z)
        
        actions[:, 4] = ankle_pitch_value
        actions[:, 10] = ankle_pitch_value
        
        actions[:, 5] = ankle_roll_value
        actions[:, 11] = ankle_roll_value
        
        if int(t * 2) % 2 == 0 and (t * 2) - int(t * 2) < 0.05:
            theta_deg = np.degrees(theta)
            phi_deg = np.degrees(phi)
            print(f"🌐 球形轨迹 | t: {t:.2f}s | 剩余: {(total_time - t):.1f}s | θ: {theta_deg:.1f}° | φ: {phi_deg:.1f}°")
    
    return actions, is_finished

def generate_fourier_motion(t, env):
    """
    使用傅里叶级数生成复杂周期运动
    🔥 运行10秒后结束
    
    Args:
        t: 当前时间
        env: 环境对象
    
    Returns:
        actions: 关节动作张量
        is_finished: 是否完成运动
    """
    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device, dtype=torch.float)
    
    total_time = 10.0  # 运行10秒
    is_finished = (t >= total_time)
    
    if not is_finished:
        A1, f1, phi1 = 1.0, 0.5, 0.0
        A2, f2, phi2 = 0.3, 1.5, np.pi/4
        A3, f3, phi3 = 0.1, 2.5, np.pi/2
        
        fourier_value = (A1 * np.sin(2 * np.pi * f1 * t + phi1) +
                         A2 * np.sin(2 * np.pi * f2 * t + phi2) +
                         A3 * np.sin(2 * np.pi * f3 * t + phi3))
        
        for i in range(6):
            actions[:, i] = fourier_value
            actions[:, i + 6] = fourier_value
    
    return actions, is_finished


def print_joint_properties(env):
    """打印所有关节的物理属性"""
    print("\n" + "="*90)
    print("📋 关节物理属性检查")
    print("="*90)
    
    # 获取第一个环境的 actor
    env_ptr = env.envs[0]
    actor_handle = env.actor_handles[0]
    
    # 获取 DOF 属性
    dof_props = env.gym.get_actor_dof_properties(env_ptr, actor_handle)
    num_dofs = env.gym.get_actor_dof_count(env_ptr, actor_handle)
    
    print(f"{'关节索引':<10} {'关节名称':<25} {'Damping':<12} {'Friction':<12} {'Armature':<12} {'Lower':<10} {'Upper':<10}")
    print("-"*90)
    
    for i in range(num_dofs):
        
        damping = dof_props['damping'][i]
        friction = dof_props['friction'][i]
        armature = dof_props['armature'][i]
        lower = dof_props['lower'][i]
        upper = dof_props['upper'][i]
        
        # 如果参数不为0，用标记显示
        highlight = " ✓" if (damping > 0 or friction > 0 or armature > 0) else ""
        
        print(f"{i:<10}  {damping:<12.6f} {friction:<12.6f} {armature:<12.6f} {lower:<10.3f} {upper:<10.3f}{highlight}")
    
    print("="*90)
    
    # 统计信息
    non_zero_damping = sum(1 for i in range(num_dofs) if dof_props['damping'][i] > 0)
    non_zero_friction = sum(1 for i in range(num_dofs) if dof_props['friction'][i] > 0)
    non_zero_armature = sum(1 for i in range(num_dofs) if dof_props['armature'][i] > 0)
    
    print(f"\n📊 统计:")
    print(f"  总关节数: {num_dofs}")
    print(f"  有 damping 的关节: {non_zero_damping}/{num_dofs}")
    print(f"  有 friction 的关节: {non_zero_friction}/{num_dofs}")
    print(f"  有 armature 的关节: {non_zero_armature}/{num_dofs}")
    
    # 检查是否所有参数都为0
    if non_zero_damping == 0 and non_zero_friction == 0 and non_zero_armature == 0:
        print("\n⚠️  警告: 所有关节的 damping、friction、armature 都为 0!")
        print("   请检查 URDF 文件中是否添加了 <dynamics> 标签")
    else:
        print("\n✅ 已检测到非零参数")
    
    print("="*90 + "\n")



class LegCollisionDetector:
    """两腿之间的碰撞检测器"""
    
    def __init__(self, env):
        self.env = env
        self.gym = env.gym
        self.sim = env.sim
        
        # 存储左右腿的body索引
        self.left_leg_bodies = []
        self.right_leg_bodies = []
        
        # 碰撞统计
        self.collision_count = 0
        self.last_collision_step = -1
        
        # 🔥 获取 rigid body state tensor
        self.rb_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        # 将 tensor 包装为 PyTorch tensor
        self.rb_states = gymtorch.wrap_tensor(self.rb_state_tensor)
        
        print(f"✅ 成功获取 rigid_body_state_tensor, shape: {self.rb_states.shape}")
        
        # 初始化腿部body索引
        self._init_leg_bodies()
    
    def _init_leg_bodies(self):
        """初始化左右腿的body索引"""
        # 获取第一个环境的actor
        env_ptr = self.env.envs[0]
        actor_handle = self.env.actor_handles[0]
        
        # 获取body数量
        num_bodies = self.gym.get_actor_rigid_body_count(env_ptr, actor_handle)
        
        print(f"📝 检测到 {num_bodies} 个 rigid bodies")
        
        # 获取body名称字典
        body_dict = self.gym.get_actor_rigid_body_dict(env_ptr, actor_handle)
        
        # 按索引排序并显示
        body_list = sorted(body_dict.items(), key=lambda x: x[1])
        
        for body_name, body_idx in body_list:
            body_name_lower = body_name.lower()
            print(f"  Body {body_idx}: {body_name}")
            
            # 排除 base_link
            if 'base' in body_name_lower:
                continue
            
            # 针对 leg_lX 和 leg_rX 的命名格式进行精确匹配
            if 'leg_l' in body_name_lower:  # leg_l1, leg_l2, ...
                self.left_leg_bodies.append(body_idx)
                print(f"    ✓ 添加到左腿")
            elif 'leg_r' in body_name_lower:  # leg_r1, leg_r2, ...
                self.right_leg_bodies.append(body_idx)
                print(f"    ✓ 添加到右腿")
        
        print(f"\n✅ 左腿body索引: {self.left_leg_bodies}")
        print(f"✅ 右腿body索引: {self.right_leg_bodies}")
        
        if not self.left_leg_bodies or not self.right_leg_bodies:
            print("⚠️  警告：未能正确识别左右腿body，碰撞检测可能不准确")
        else:
            print(f"✅ 成功识别 {len(self.left_leg_bodies)} 个左腿body和 {len(self.right_leg_bodies)} 个右腿body")
    def check_collision(self, step):
        """
        使用tensor API检查两腿之间是否有碰撞
        
        Args:
            step: 当前步数
        
        Returns:
            has_collision: 是否有碰撞
            min_distance: 最小距离
            collision_pairs: 碰撞对列表
        """
        has_collision = False
        min_distance = float('inf')
        collision_pairs = []
        all_distances = []  # 存储所有距离用于调试
        
        # 🔥 刷新 rigid body state tensor
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        # 调试：检查 rigid_body_state 是否存在
        if step == 1:
            print(f"\n🔍 调试信息:")
            print(f"  rb_states shape: {self.rb_states.shape}")
            print(f"  rb_states device: {self.rb_states.device}")
        
        num_bodies_per_env = self.gym.get_actor_rigid_body_count(
            self.env.envs[0], self.env.actor_handles[0]
        )
        
        if step == 1:
            print(f"  num_bodies_per_env: {num_bodies_per_env}")
            print(f"  左腿bodies: {self.left_leg_bodies}")
            print(f"  右腿bodies: {self.right_leg_bodies}")
        
        # 检查左右腿是否为空
        if not self.left_leg_bodies or not self.right_leg_bodies:
            if step % 100 == 0:
                print(f"⚠️  警告：左腿或右腿body列表为空，无法检测碰撞")
            return has_collision, min_distance, collision_pairs
        
        for env_idx in range(self.env.num_envs):
            # 计算左右腿body之间的最小距离
            for left_body in self.left_leg_bodies:
                left_idx = env_idx * num_bodies_per_env + left_body
                # rigid body state 格式: [pos(3), quat(4), lin_vel(3), ang_vel(3)]
                left_pos = self.rb_states[left_idx, :3]
                
                for right_body in self.right_leg_bodies:
                    right_idx = env_idx * num_bodies_per_env + right_body
                    right_pos = self.rb_states[right_idx, :3]
                    
                    # 计算距离
                    distance = torch.norm(left_pos - right_pos).item()
                    all_distances.append({
                        'left_body': left_body,
                        'right_body': right_body,
                        'distance': distance,
                        'left_pos': left_pos.cpu().numpy(),
                        'right_pos': right_pos.cpu().numpy()
                    })
                    min_distance = min(min_distance, distance)
                    
                    # 如果距离小于阈值，认为有碰撞
                    collision_threshold = 0.05  # 5cm
                    if distance < collision_threshold:
                        has_collision = True
                        collision_pairs.append((left_body, right_body, distance))
                        if step != self.last_collision_step:
                            self.collision_count += 1
                            self.last_collision_step = step
                            print(f"⚠️  碰撞检测！左腿body {left_body} 和右腿body {right_body} 距离: {distance:.4f}m")
        
        # 每50步打印一次所有距离（调试用）
        if step % 50 == 0 and all_distances:
            print(f"\n🔍 第 {step} 步距离详情:")
            print(f"  左腿bodies: {self.left_leg_bodies}")
            print(f"  右腿bodies: {self.right_leg_bodies}")
            print(f"  总共计算了 {len(all_distances)} 对距离")
            # 按距离排序，显示前5个最近的
            sorted_distances = sorted(all_distances, key=lambda x: x['distance'])[:5]
            for i, d in enumerate(sorted_distances, 1):
                print(f"  Top {i}: 左body {d['left_body']} <-> 右body {d['right_body']}")
                print(f"         距离: {d['distance']:.4f}m")
                print(f"         左pos: [{d['left_pos'][0]:.3f}, {d['left_pos'][1]:.3f}, {d['left_pos'][2]:.3f}]")
                print(f"         右pos: [{d['right_pos'][0]:.3f}, {d['right_pos'][1]:.3f}, {d['right_pos'][2]:.3f}]")
        
        return has_collision, min_distance, collision_pairs
    
    
    
    def get_statistics(self):
        """获取碰撞统计信息"""
        return {
            'collision_count': self.collision_count,
            'last_collision_step': self.last_collision_step
        }


def view_suspended(args):
    """可视化悬挂的机器人"""

    # 🔥 创建带时间戳的保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(__file__).resolve().parent.parent.parent / "data_collection"
    save_dir = base_dir / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)

    
    # 获取配置
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    # 🔥 关键修改：实现悬挂效果
    env_cfg.asset.fix_base_link = True  # 固定base_link实现悬挂
    env_cfg.init_state.pos = [0.0, 0.0, 1.2]  # 悬挂高度
    

    # 简化环境设置
    env_cfg.env.num_envs = args.num_envs if hasattr(args, 'num_envs') and args.num_envs else 1
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.curriculum = False
    env_cfg.domain_rand.push_robots = False

    # 创建环境
    print("\n⏳ 正在创建环境...")
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    
    # 🔥 添加：打印关节属性
    print_joint_properties(env)
    
    motion_generators = {
        'walking': (generate_walking_motion, True),
        'hip_circle': (generate_single_hip_motion, True),
        'sin_single': (generate_single_sin_motion, True),
        'fourier_single': (generate_single_fourier_motion, True),
        'ankle_circle': (generate_single_ankle_motion, True),
        'leg_sphere': (generate_leg_motion, True),
        'fourier': (generate_fourier_motion, True),
        'zero': (generate_zero_motion, True),
    }
    # 选择动作模式（可以通过命令行参数选择）
    motion_mode = 'fourier_single' # walking, squat, kick, dance

    if motion_mode not in motion_generators:
        print(f"⚠️  未知的动作模式: {motion_mode}，使用默认的 walking 模式")
        motion_mode = 'hip_circle'
    
    motion_generator, auto_exit = motion_generators[motion_mode]

    
    print("\n" + "="*70)
    print("🎯 悬挂机器人可视化（预设动作 + 碰撞检测 + 数据收集）")
    print("="*70)
    print(f"✅ 悬挂高度: {env_cfg.init_state.pos[2]:.2f} 米")
    print(f"✅ Base Link: {'固定（悬挂）' if env_cfg.asset.fix_base_link else '自由'}")
    print(f"✅ 环境数量: {env_cfg.env.num_envs}")
    print(f"✅ 地形类型: {env_cfg.terrain.mesh_type}")
    print(f"✅ 动作模式: {motion_mode}")
    print(f"✅ 碰撞检测: 启用")
    print(f"✅ 数据收集: 启用")
    print("="*70)
    
    # 创建环境
    print("\n⏳ 正在创建环境...")
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    
    # 创建碰撞检测器
    print("\n⏳ 初始化碰撞检测器...")
    collision_detector = LegCollisionDetector(env)
    
    # 🔥 创建数据收集器
    print("\n⏳ 初始化数据收集器...")
    data_collector = JointDataCollector(num_joints=12)
    
    # 设置相机位置
    cam_pos = [2.5, 0, 3.2]
    cam_target = [0,0, 2.5]
    env.set_camera(cam_pos, cam_target)
    
    print("\n" + "="*70)
    print("✅ 环境创建成功！")
    print("\n🎬 动作模式说明：")
    print("  walking      - 行走动作（左右腿交替摆动）")
    print("  single       - 单关节测试（每个关节依次运动）")
    print("  hip_circle   - 髋关节3D圆周运动")
    print("  ankle_circle - 踝关节圆周运动")
    print("  leg_sphere   - 球形轨迹运动（所有关节协同）")
    print("\n操作说明：")
    print("  🖱️  鼠标左键拖动 - 旋转视角")
    print("  🖱️  鼠标滚轮     - 缩放")
    print("  🖱️  鼠标中键拖动 - 平移")
    print("  ⌨️  Ctrl+C       - 退出并保存数据")
    print("="*70 + "\n")
    
    # 获取初始观测
    obs = env.get_observations()
    is_finished = False
    
    # 主循环
    step = 0
    start_time = 0.0
    try:
        while not is_finished:
            # 当前时间
            t = step * env.dt
            
            # 生成预设动作
            if auto_exit:
                actions, is_finished = motion_generator(t, env)
            else:
                actions = motion_generator(t, env)
            # 执行环境步进
            step_results = env.step(actions)
            obs = step_results[0]
            
            #actions_env = env.action.cpu().numpy()
            # 🔥 收集关节数据
            data_collector.collect(t, actions, env.dof_pos)
            
            step += 1
            
            if step == 1:
                print("\n🔍 数据来源验证:")
                print(f"  actions 来源: motion_generator 生成")
                print(f"  actions shape: {actions.shape}")
                print(f"  dof_pos 来源: env.dof_pos (仿真器实际关节位置)")
                print(f"  dof_pos shape: {env.dof_pos.shape}")
                print(f"  第一个关节 - 目标: {actions[0, 0].item():.4f}, 实际: {env.dof_pos[0, 0].item():.4f}")
            # 每100步打印一次状态
            if step % 100 == 0:
                joint_pos = actions[0, :6].cpu().numpy()
                print(f"⏱️  时间: {t:6.2f}s | 步数: {step:6d} | "
                      f"前6个关节动作: [{', '.join([f'{x:+.3f}' for x in joint_pos])}]")
             # 🔥 检查是否完成
            if is_finished:
                print("\n" + "="*70)
                print("✅ 所有关节运动完成！")
                print("="*70)
                break
                    
    except KeyboardInterrupt:
        print("\n\n👋 用户中断")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 🔥 无论如何都保存数据
        print("\n📊 正在保存数据...")
        
        # 打印统计信息
        data_collector.print_statistics()
        
        # 绘制并保存图像
        print("\n📊 正在生成图像...")
        save_path = save_dir / f"joint_data_{motion_mode}.png"
        print(f"📁 保存目录: {save_dir}")
        data_collector.plot(save_path=str(save_path))
        
        # 碰撞统计
        stats = collision_detector.get_statistics()
        print(f"\n📊 碰撞统计:")
        print(f"  总碰撞次数: {stats['collision_count']}")
        print(f"  最后碰撞步数: {stats['last_collision_step']}")
        
        print("\n🔚 退出可视化\n")
                        
    


if __name__ == "__main__":
    args = get_args()
    view_suspended(args)