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
from RL_train.humanoid.scripts.SI.plotfun import TorqueVelocityPlotter
# 在文件开头添加导入
from RL_train.humanoid.scripts.SI.realworld_data import (
    RealDataProcessor, 
    load_real_data_single_run, 
    load_real_data_all_runs,
    check_real_data_timestamps,
    get_real_data_info
)

SEED = 42
mujoco_see = True
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
        # 🔥 修改：只包含12个腿部关节的参数
        kps = np.array([60, 60, 100, 150, 15, 15, 60, 60, 100, 150, 15, 15], dtype=np.double)
        kds = np.array([5.0, 5.0, 5.0, 6.0, 5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 5.0, 5.0], dtype=np.double)
        tau_limit = np.array([100., 25.5, 84., 250., 40., 30., 100., 25.5, 84., 250., 40., 30.])


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
  
   

    def save_and_visualize_data(self, real_data, sim_data, params=None):
        """修正函数参数名称和数据处理逻辑"""
        print("=== 函数入口数据检查 ===")

        torque_real_entry = real_data[:, -12:]
        torque_sim_entry = sim_data[:, -12:]
        
        # 切片数据（去掉前200步）
        real_data = real_data[200:]
        sim_data = sim_data[200:]
        
        # 切片后再次检查关节位置
        joint_pos_real_after_slice = real_data[:, :12]
        
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

        # 🔥 使用 TorqueVelocityPlotter 类来生成所有图表
        print("=== 开始使用 TorqueVelocityPlotter 生成图表 ===")
        
        # 创建 TorqueVelocityPlotter 实例，使用相同的保存目录
        plotter = TorqueVelocityPlotter(save_dir=self.save_dir)
        
        # 1. 生成主要的扭矩-速度曲线图（包含四象限理论曲线）
        print("生成扭矩-速度曲线图...")
        plotter.plot_torque_velocity_curves(real_data, sim_data, params=params, 
                                        filename='torque_velocity_curves.png')
        
        # 2. 生成Mujoco和Isaac Gym的扭矩对比图
        print("生成扭矩对比图...")
        plotter.plot_mujoco_isaac_torque_comparison(real_data, sim_data, 
                                                filename='mujoco_isaac_torque_comparison.png')
        
        # 3. 生成个别关节详细分析图
        print("生成个别关节详细分析图...")
        plotter.plot_individual_torque_velocity_analysis(real_data, sim_data, 
                                                        filename='detailed_torque_velocity_analysis.png')
        
        # 4. 如果有all_real_data，生成所有真实数据对比图
        # if hasattr(self, 'all_real_data') and self.all_real_data is not None:
        #     print("生成所有真实数据对比图...")
        #     try:
        #         plotter.plot_all_real_data_torque_velocity_curves(
        #             self.all_real_data, 
        #             sim_data=sim_data, 
        #             params=params,
        #             filename='all_real_data_torque_velocity_comparison.png'
        #         )
        #     except Exception as e:
        #         print(f"生成所有真实数据对比图失败: {e}")
        
        # 5. 调用原有的速度对比绘图函数
        print("生成速度对比图...")
        vel_names = ['vx', 'vy', 'vz']
        plotter._plot_velocity_comparison(base_lin_vel_real, base_lin_vel_sim, vel_names,
                                'Base Linear Velocity', 'base_linear_velocity_comparison.png')
       # plotter._plot_velocity_comparison(world_lin_vel_real, world_lin_vel_sim, vel_names,
                                #'World Linear Velocity', 'world_linear_velocity_comparison.png')
        
        # 6. 调用原有的关节对比绘图函数
        print("生成关节对比图...")
        plotter._plot_joint_comparison(joint_pos_real, joint_pos_sim, joint_names, 
                                'Joint Positions (Real vs Sim)', 'joint_positions_comparison.png')
        plotter._plot_joint_comparison(joint_vel_real, joint_vel_sim, joint_names,
                                'Joint Velocities (Real vs Sim)', 'joint_velocities_comparison.png')
        plotter._plot_joint_comparison(action_real, action_sim, joint_names,
                                'Actions (Real vs Sim)', 'actions_comparison.png')
        
        # 7. 生成分布对比图
        print("生成分布对比图...")
        plotter._plot_distribution_comparison(real_data, sim_data)
        
        # 8. 生成数据分析报告
        print("生成数据分析报告...")
        plotter._generate_data_report(real_data, sim_data, params)
        
        print(f"✅ 所有图表和数据已保存到: {self.save_dir}")

    def plot_torque_velocity_curves(self, real_data, sim_data, params=None):
        """使用 TorqueVelocityPlotter 绘制扭矩-速度曲线"""
        plotter = TorqueVelocityPlotter(save_dir=self.save_dir)
        plotter.plot_torque_velocity_curves(real_data, sim_data, params=params)

    def plot_individual_torque_velocity_analysis(self, real_data, sim_data):
        """使用 TorqueVelocityPlotter 绘制个别关节详细分析"""
        plotter = TorqueVelocityPlotter(save_dir=self.save_dir)
        plotter.plot_individual_torque_velocity_analysis(real_data, sim_data)

    def plot_mujoco_isaac_torque_comparison(self, real_data, sim_data):
        """使用 TorqueVelocityPlotter 绘制Mujoco和Isaac Gym扭矩对比"""
        plotter = TorqueVelocityPlotter(save_dir=self.save_dir)
        plotter.plot_mujoco_isaac_torque_comparison(real_data, sim_data)

    def plot_all_real_data_torque_velocity_curves(self, all_real_data, sim_data=None, params=None, filename='all_real_data_torque_velocity_curves.png'):
        """使用 TorqueVelocityPlotter 绘制所有真实数据的扭矩-速度曲线"""
        plotter = TorqueVelocityPlotter(save_dir=self.save_dir)
        return plotter.plot_all_real_data_torque_velocity_curves(all_real_data, sim_data=sim_data, params=params, filename=filename)
    ############### 环境参数更新函数 ################

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


    

    def update_dynamic_torque_limits(self, env, params):
        """
        将参数传递给kuavo_env中的动态扭矩限制功能
        """
        # 检查环境是否支持动态扭矩限制
        if not hasattr(env.cfg, 'dynamic_torque_params'):
            # 如果配置中没有这个属性，就创建一个
            env.cfg.dynamic_torque_params = {}
        
        # 将优化的参数传递给环境配置
        env.cfg.dynamic_torque_params.update({
            'torque_l3_top': params.get('torque_l3_top', 75.0),
            'torque_l3_bottom': params.get('torque_l3_bottom', -60.0),
            'torque_l4_top': params.get('torque_l4_top', 100.0),
            'torque_l4_bottom': params.get('torque_l4_bottom', -180.0),
            'speed_threshold_l3_q1': params.get('speed_threshold_l3_q1', 8.0),
            'speed_threshold_l3_q3': params.get('speed_threshold_l3_q3', 8.0),
            'speed_threshold_l4_q1': params.get('speed_threshold_l4_q1', 11.0),
            'speed_threshold_l4_q3': params.get('speed_threshold_l4_q3', 11.0),
            'angle_vel_l3_top': params.get('angle_vel_l3_top', 10.0),
            'angle_vel_l3_bottom': params.get('angle_vel_l3_bottom', -10.0),
            'angle_vel_l4_top': params.get('angle_vel_l4_top', 13.0),
            'angle_vel_l4_bottom': params.get('angle_vel_l4_bottom', -13.0),
        })
        
        
     
    
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

    def collect_mujoco_data(self, steps=500, command=[1.5, 0., 0., 0.]):
        """使用Mujoco环境收集'真实'数据，并可视化采集过程"""
        cfg = Sim2simCfg()
        model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
        model.opt.timestep = cfg.sim_config.dt
        data = mujoco.MjData(model)
        
        # 🔥 修改1：只使用12个腿部关节的默认位置
        default_dof_pos = np.array([0., 0., -0.47, 0.86, -0.44, 0., 
                                    0., 0., -0.47, 0.86, -0.44, 0.])  # 只有12个关节

        # 优化：显式设置初始状态，而不是使用keyframe
        mujoco.mj_resetData(model, data)
        
        # 🔥 修改2：设置关节位置（腿部 + 手臂）
        data.qpos[7:7+len(default_dof_pos)] = default_dof_pos.copy()
        
        # 🔥 新增：强制设置手臂为垂直向下的固定位置
        arm_start_qpos = 7 + len(default_dof_pos)  # 手臂在qpos中的起始索引
        if len(data.qpos) > arm_start_qpos:
            # 设置手臂完全垂直向下（所有关节角度为0）
            num_arm_joints = min(6, len(data.qpos) - arm_start_qpos)  # 最多6个手臂关节
            data.qpos[arm_start_qpos:arm_start_qpos + num_arm_joints] = 0.0
            
            # 设置手臂速度为0
            arm_start_qvel = 6 + len(default_dof_pos)  # 手臂在qvel中的起始索引
            if len(data.qvel) > arm_start_qvel:
                data.qvel[arm_start_qvel:arm_start_qvel + num_arm_joints] = 0.0
            
            print(f"强制设置 {num_arm_joints} 个手臂关节为垂直向下位置")
        
        mujoco.mj_step(model, data)
        
        # 🔥 新增：调试函数 - 检查关节索引和名称
        def debug_joint_info(model, data):
            print("=== 关节信息调试 ===")
            print(f"总关节数: {model.njnt}")
            print(f"总执行器数: {model.nu}")
            print(f"控制向量长度: {len(data.ctrl)}")
            print(f"qpos长度: {len(data.qpos)}")
            print(f"qvel长度: {len(data.qvel)}")
            
            print("\n关节名称和索引:")
            for i in range(model.njnt):
                try:
                    joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                    print(f"  Joint {i}: {joint_name}")
                except:
                    print(f"  Joint {i}: [无名称]")
            
            print("\n执行器名称和索引:")
            for i in range(model.nu):
                try:
                    actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                    print(f"  Actuator {i}: {actuator_name}")
                except:
                    print(f"  Actuator {i}: [无名称]")
        
        # 调用调试函数（只在第一次运行时）
        debug_joint_info(model, data)
        
        # 优化：在开始前稳定机器人
        print("Stabilizing robot in Mujoco...")
        for i in range(100):
            # 🔥 修改3：只使用前12个关节进行PD控制
            q = np.array(data.actuator_length[:cfg.env.num_actions])  # 只取前12个
            dq = np.array(data.actuator_velocity[:cfg.env.num_actions])  # 只取前12个
            tau = pd_control(np.zeros_like(default_dof_pos), default_dof_pos, q, 
                            cfg.robot_config.kps[:cfg.env.num_actions],  # 只用前12个增益
                            np.zeros_like(default_dof_pos), dq, 
                            cfg.robot_config.kds[:cfg.env.num_actions])  # 只用前12个增益
            tau = np.clip(tau, -cfg.robot_config.tau_limit[:cfg.env.num_actions], 
                        cfg.robot_config.tau_limit[:cfg.env.num_actions])  # 只用前12个限制
            
            # 🔥 修改4：只设置前12个关节的控制信号
            data.ctrl[:cfg.env.num_actions] = tau
            
            # 🔥 强制手臂关节控制为0（完全无力）
            if len(data.ctrl) > cfg.env.num_actions:
                data.ctrl[cfg.env.num_actions:] = 0.0
                
            # 🔥 关键：在每一步都强制重置手臂位置和速度
            arm_start_qpos = 7 + len(default_dof_pos)
            arm_start_qvel = 6 + len(default_dof_pos)
            
            if len(data.qpos) > arm_start_qpos:
                num_arm_joints = min(6, len(data.qpos) - arm_start_qpos)
                # 强制手臂位置保持为0（垂直向下）
                data.qpos[arm_start_qpos:arm_start_qpos + num_arm_joints] = 0.0
                
            if len(data.qvel) > arm_start_qvel:
                # 强制手臂速度保持为0
                data.qvel[arm_start_qvel:arm_start_qvel + num_arm_joints] = 0.0
            
            mujoco.mj_step(model, data)
            
            # 每20步打印一次手臂状态
            if i % 20 == 0:
                if len(data.qpos) > arm_start_qpos:
                    arm_positions = data.qpos[arm_start_qpos:arm_start_qpos + num_arm_joints]
                    arm_velocities = data.qvel[arm_start_qvel:arm_start_qvel + num_arm_joints]
                    arm_controls = data.ctrl[cfg.env.num_actions:] if len(data.ctrl) > cfg.env.num_actions else []
                    print(f"  Step {i}: 手臂位置={arm_positions}, 速度={arm_velocities}, 控制={arm_controls}")
        
        # 加入viewer
        if mujoco_see:
            viewer = mujoco_viewer.MujocoViewer(model, data)

        # 🔥 修改5：只需要腿部关节的目标位置
        target_q = np.zeros(cfg.env.num_actions, dtype=np.double)  # 只有12个关节
        action = np.zeros(cfg.env.num_actions, dtype=np.double)

        hist_obs = deque()
        # 先构建一个包含正确命令的初始观测
        initial_obs = np.zeros([1, cfg.env.num_single_obs], dtype=np.double)
        initial_obs[0, 2] = command[0] * cfg.normalization.obs_scales.lin_vel  # cmd.vx
        initial_obs[0, 3] = command[1] * cfg.normalization.obs_scales.lin_vel  # cmd.vy
        initial_obs[0, 4] = command[2] * cfg.normalization.obs_scales.ang_vel  # cmd.dyaw
        initial_obs[0, 5] = command[3]  # cmd.stand

        # 修复：用包含命令的initial_obs填充历史缓冲区
        for _ in range(cfg.env.frame_stack):
            hist_obs.append(initial_obs.copy())

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
            
            # 🔥 修改6：只使用前12个关节的数据
            q = np.array(data.actuator_length[:cfg.env.num_actions])
            dq = np.array(data.actuator_velocity[:cfg.env.num_actions])
            
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
            
            # 🔥 修改7：目标速度也只需要12个关节
            target_dq = np.zeros(cfg.env.num_actions, dtype=np.double)

            # 🔥 修改8：PD控制只使用腿部关节
            tau = pd_control(target_q, default_dof_pos, q, 
                            cfg.robot_config.kps[:cfg.env.num_actions],
                            target_dq, dq, 
                            cfg.robot_config.kds[:cfg.env.num_actions])
            tau = np.clip(tau, -cfg.robot_config.tau_limit[:cfg.env.num_actions], 
                        cfg.robot_config.tau_limit[:cfg.env.num_actions])
            
            # 🔥 修改9：只设置腿部关节控制，手臂保持静止
            data.ctrl[:cfg.env.num_actions] = tau
            
            # 🔥 强制手臂关节控制为0（完全无力）
            if len(data.ctrl) > cfg.env.num_actions:
                data.ctrl[cfg.env.num_actions:] = 0.0  # 手臂关节无控制力
            
            # 🔥 关键：在主循环的每一步都强制重置手臂位置和速度
            arm_start_qpos = 7 + len(default_dof_pos)
            arm_start_qvel = 6 + len(default_dof_pos)
            
            if len(data.qpos) > arm_start_qpos:
                num_arm_joints = min(6, len(data.qpos) - arm_start_qpos)
                # 强制手臂位置保持为0（垂直向下）
                data.qpos[arm_start_qpos:arm_start_qpos + num_arm_joints] = 0.0
                
            if len(data.qvel) > arm_start_qvel:
                # 强制手臂速度保持为0
                data.qvel[arm_start_qvel:arm_start_qvel + num_arm_joints] = 0.0
            
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
        
        # 🔥 关键修改：使用kuavo_env中的动态扭矩限制功能
        use_dynamic_torque = params.get('use_dynamic_torque', False)
        
        if use_dynamic_torque:
            print("✅ 使用kuavo_env中的动态扭矩限制功能")
            try:
                self.update_dynamic_torque_limits(self.env, params)
            except Exception as e:
                print(f"Warning: Failed to set dynamic torque params: {e}")
        else:
            print("使用静态扭矩限制")
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

            # 🔥 删除这部分，因为参数已经在环境配置中了，kuavo_env会自动使用
            # if use_dynamic_torque:
            #     try:
            #         self.update_dynamic_torque_limits(self.env, params)
            #     except Exception as e:
            #         print(f"Warning: Failed to update dynamic torque limits at step {step}: {e}")

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

  
    def align_time_series(self, sim_data, real_data, alignment_dims=None):
        """
        通过计算多维信号的互相关来对齐两个时间序列。
        对actions单独进行裁剪，其他数据一起裁剪。
        """
        from scipy import signal
        from scipy.stats import mode

        # 默认使用第2和3维度进行对齐
        if alignment_dims is None:
            alignment_dims = [2]
        
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
        actions_alignment_dims = [0]  # actions的前2个维度
        
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
        verbose=True
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
        # torque_l3_top = trial.suggest_float('torque_l3_top', 46, 55)
        # torque_l3_bottom = trial.suggest_float('torque_l3_bottom', -45, -38)

        # angle_vel_l3_top = trial.suggest_float("angle_vel_l3_top", 5, 10)
        # angle_vel_l3_bottom = trial.suggest_float("angle_vel_l3_bottom", -10, -5)

        # torque_l4_top = trial.suggest_float('torque_l4_top', 30, 60)
        # torque_l4_bottom = trial.suggest_float('torque_l4_bottom', -165, -150)
        # angle_vel_l4_top = trial.suggest_float("angle_vel_l4_top", 7, 13)
        # angle_vel_l4_bottom = trial.suggest_float("angle_vel_l4_bottom", -13, -7)
        torque_l3_top = 75  # 固定值
        torque_l3_bottom = -60  # 固定值
        angle_vel_l3_top = 10  # 固定值    
        angle_vel_l3_bottom = -10  # 固定值
        torque_l4_top = 100  # 固定值
        torque_l4_bottom = -180  # 固定值
        angle_vel_l4_top = 13  # 固定值
        angle_vel_l4_bottom = -13  # 固定值
        
        
        # 新增：动态扭矩限制参数
        #对应扭矩开始衰减的速度阈值
        # 修改：独立的象限速度阈值参数
        speed_threshold_l3_q1 = trial.suggest_float('speed_threshold_l3_q1', 1.0, 8.0)  # L3 Q1象限速度阈值
        speed_threshold_l3_q3 = trial.suggest_float('speed_threshold_l3_q3', 1.0, 8.0)  # L3 Q3象限速度阈值
        speed_threshold_l4_q1 = trial.suggest_float('speed_threshold_l4_q1', 1.0, 8.0)  # L4 Q1象限速度阈值
        speed_threshold_l4_q3 = trial.suggest_float('speed_threshold_l4_q3', 1.0, 11.0)  # L4 Q3象限速度阈值
        
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

            
        
        return distance


if __name__ == "__main__":
    # 获取命令行参数和配置
    args = get_args()
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # 设置环境参数
    #env_cfg.rewards.cycle_time = 0.7
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

    # 设置Optuna优化器
    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=0)

    # 创建保存目录和数据库
    save_dir = "../logs/optuna_results"
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    db_path = os.path.join(save_dir, f"optuna_{timestamp}.sqlite3")
    study = optuna.create_study(sampler=sampler, pruner=pruner, storage=f"sqlite:///{db_path}", study_name="Gap-kill")

    # 定义初始优化参数
    initial_params = [
        0.007,    # 初始关节摩擦系数
        35.0,     # 初始关节最大扭矩
        -35.0,    # 初始关节最小扭矩
        8.0,      # 初始关节最大速度 (rad/s)
        -8.0,     # 初始关节最小速度 (rad/s)
        5.0,      # L3速度阈值
        7.0,      # L4速度阈值
    ]
    
    initial_sigmas = [
        0.05,     # 摩擦力的搜索步长
        10.0,     # 最大扭矩的搜索步长
        10.0,     # 最小扭矩的搜索步长
        1.0,      # 最大速度的搜索步长
        1.0,      # 最小速度的搜索步长
        1.0,      # L3速度阈值搜索步长
        1.0,      # L4速度阈值搜索步长
    ]
    
    # 创建优化器实例
    optimizer = Sim2RealCMAOptimizer(
        initial_params=initial_params,
        sigma0=initial_sigmas,
        real_data=None,  # 将在下面设置
        env=env,
        policy=policy,
        jit_policy=jit_policy,
        max_iter=10
    )
    
    # 设置真实数据参数
    real_data_from = "mujoco"   # "mujoco" or "real"
    real_data_file = "data/real_run_data/919191.npz"
    real_data_time_long = 5     # 单位：秒
    real_data_run = [6358, 6660, 6985, 7315, 7623, 7933, 8270, 8580, 8895]
    real_data_start_offset = real_data_run[5] * 10 - 59000  # 选择第8次运行数据

    # 🔥 数据加载部分
    if real_data_from == "mujoco":
        print("Collecting 'real' data from Mujoco...")
        real_data = optimizer.collect_mujoco_data(steps=500, command=[1.2, 0., 0., 0.])
        optimizer.real_data = real_data
        print(f"Collected real data shape: {real_data.shape}")
        all_real_data = None
    else:
        print("=== 加载真实数据 ===")
        
        # 🔥 使用新的便捷函数加载数据
        try:
            # 加载单次运行数据
            single_run_data = load_real_data_single_run(
                data_file=real_data_file,
                run_value=real_data_run[7],  # 第8次运行
                time_duration=real_data_time_long
            )
            optimizer.real_data = single_run_data
            print(f"✅ 单次运行数据加载完成，形状: {single_run_data.shape}")
            
            # 加载所有运行数据
            all_real_data = load_real_data_all_runs(
                data_file=real_data_file,
                run_values=real_data_run,
                time_duration=real_data_time_long
            )
            print(f"✅ 所有运行数据加载完成，共 {len(all_real_data)} 条")
            
            # 可选：检查数据时间戳
            # check_real_data_timestamps(real_data_file, run_value=real_data_run[7])
            
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            # 降级到原始加载方法
            print("尝试使用原始方法加载数据...")
            
            np_load_data = np.load(real_data_file)
            print("=== 检查 .npz 文件字段 ===")
            print("Available data fields in real data file:")
            print(np_load_data.files)
            
            # 打印每个字段的形状和类型
            for field in np_load_data.files:
                data_field = np_load_data[field]
                print(f"  {field}: shape={data_field.shape}, dtype={data_field.dtype}")
            
            print("\n" + "="*50)
            
            # 根据实际字段名加载数据
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
                # 方案2：尝试使用当前字段名
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
            
            # 检查并加载扭矩数据
            has_torque_data = False
            if "motor_cur" in np_load_data.files:
                torques = np_load_data["motor_cur"]
                torques_ts = np_load_data["timestamps_motor_cur"]
                print(f"Found torque data with shape: {torques.shape}")
                
                # 只取前12个关节（腿部关节）的扭矩数据
                if torques.shape[1] >= 12:
                    torques = torques[:, :12]
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
                    torques = torques * np.ones(torques.shape[1])
                
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
            action_offset = -20  # actions比joint_pos早20个数据点
            torques_offset = 7901  # torques比joint_pos早7901个数据点

            # 构建数据数组
            start_idx = real_data_start_offset
            end_idx = real_data_start_offset + real_data_time_long * 100
            action_start_idx = real_data_start_offset + action_offset
            action_end_idx = real_data_start_offset + real_data_time_long * 100 + action_offset
            
            print(f"\n=== 数据切片索引检查 ===")
            print(f"speeds: start_idx={start_idx}, end_idx={end_idx}, shape={speeds.shape}")
            print(f"joint_pos: start_idx={start_idx}, end_idx={end_idx}, shape={joint_pos.shape}")
            if has_torque_data:
                print(f"torques: start_idx={start_idx}, end_idx={end_idx}, shape={torques.shape}")
            print(f"actions: start_idx={action_start_idx}, end_idx={action_end_idx}, shape={actions.shape}")

            if has_torque_data:
                # 包含真实扭矩数据
                real_data = np.concatenate([
                    joint_pos[start_idx:end_idx, :12],  # 关节位置 (12)
                    joint_vel[start_idx:end_idx, :12],  # 关节速度 (12)
                    actions[action_start_idx:action_end_idx, :12],  # 动作 (12)
                    speeds[start_idx:end_idx, :3],  # 基座速度 (3)
                    speeds[start_idx:end_idx, :3],  # 世界速度 (3)
                    torques[start_idx:end_idx, :12]     # 实际扭矩 (12)
                ], axis=1)
                print(f"Real data with torques shape: {real_data.shape}")
            else:
                # 没有扭矩数据，用零填充
                real_data = np.concatenate([
                    joint_pos[start_idx:end_idx, :12],  # 关节位置 (12)
                    joint_vel[start_idx:end_idx, :12],  # 关节速度 (12)
                    actions[action_start_idx:action_end_idx, :12],  # 动作 (12)
                    speeds[start_idx:end_idx, :3],  # 基座速度 (3)
                    speeds[start_idx:end_idx, :3],  # 世界速度 (3)
                    np.zeros((end_idx-start_idx, 12))   # 扭矩占位符 (12)
                ], axis=1)
                print(f"Real data without torques (using zeros) shape: {real_data.shape}")
                print("Warning: Using zero torques for real data - torque comparison will not be meaningful")
            
            optimizer.real_data = real_data
            print(f"Final real data size: {len(real_data)}")
            all_real_data = None

    # 设置优化器的all_real_data
    optimizer.all_real_data = all_real_data if 'all_real_data' in locals() else None
    
   
    # 🔥 开始优化流程
    print("\n=== 开始优化流程 ===")
    
    try:
        study.optimize(optimizer.objective, n_trials=N_TRIALS)
    except KeyboardInterrupt:
        print("优化被用户中断")
    
    # 🔥 优化完成后，使用最佳参数生成最终对比图
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
        
        # 绘制包含最佳仿真数据的完整对比图
        if all_real_data is not None:
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
        
        # 生成最终的单次运行对比
        optimizer.save_and_visualize_data(optimizer.real_data, final_sim_data, optimizer.best_params)
        
    else:
        print("⚠️ 没有找到最佳参数，跳过最终对比图生成")
    
    print("\n=== 程序执行完成 ===")
    print(f"优化结果已保存到: {save_dir}")
    print(f"数据库文件: {db_path}")
    
    # 输出最终结果摘要
    if hasattr(optimizer, 'best_params') and optimizer.best_params is not None:
        print(f"\n=== 最终结果摘要 ===")
        print(f"最佳距离分数: {optimizer.best_score:.6f}")
        print(f"最佳参数:")
        for key, value in optimizer.best_params.items():
            print(f"  {key}: {value}")
        print(f"优化完成的试验数: {len(study.trials)}")
        print(f"数据对比图已保存到: {optimizer.save_dir}")