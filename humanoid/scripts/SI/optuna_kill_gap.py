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
from optuna.samplers import CmaEsSampler 
from typing import Any
from typing import Dict
import seaborn as sns
from humanoid.scripts.SI.plotfun import TorqueVelocityPlotter
# 在文件开头添加导入
from humanoid.scripts.SI.realworld_data import (
    RealDataProcessor, 
    load_real_data_single_run, 
    load_real_data_all_runs,
    check_real_data_timestamps,
    get_real_data_info
)

SEED = 42
mujoco_see = False
N_TRIALS = 100  # 尝试100次不同的超参数组合
N_STARTUP_TRIALS = 5  # 前5次是随机采样，用于TPE算法“热身”
SLICE = 200  # 只比较前200秒的数据


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
                 real_data, 
                 env, 
                 policy, 
                 jit_policy
                 ):
        
        self.real_data = real_data
        self.env = env
        self.policy = policy
        self.jit_policy = jit_policy
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
  
   

    def save_and_visualize_data(self, real_data, sim_data, params=None,slice_time=0):
        """修正函数参数名称和数据处理逻辑"""
        print("=== 函数入口数据检查 ===")

        torque_real_entry = real_data[:, -12:]
        torque_sim_entry = sim_data[:, -12:]
        
        # 切片数据（去掉前200步）
        real_data = real_data[slice_time:]
        sim_data = sim_data[slice_time:]

        
        
        # 保存原始数据
        np.save(os.path.join(self.save_dir, 'real_data.npy'), real_data)
        np.save(os.path.join(self.save_dir, 'sim_data.npy'), sim_data)

        
        
            
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
        if hasattr(self, 'all_real_data') and self.all_real_data is not None:
            print("生成所有真实数据对比图...")
            try:
                plotter.plot_all_real_data_torque_velocity_curves(
                    self.all_real_data, 
                    sim_data=sim_data, 
                    params=params,
                    filename='all_real_data_torque_velocity_comparison.png'
                )
            except Exception as e:
                print(f"生成所有真实数据对比图失败: {e}")
        
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
        

        plotter.plot_joint_position_velocity_difference(real_data, sim_data, 
                                                   filename='joint_pos_vel_difference.png')
   
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
        """更新环境的关节摩擦系数 - 使用DOF属性方法"""
        
        # 🔥 修复：使用正确的DOF属性更新方法
        try:
            # 获取DOF属性
            dof_props = env.gym.get_actor_dof_properties(env.envs[0], 0)
            
            # 定义要更新的关节摩擦系数
            friction_dict = {
                2: params.get('joint_friction_l3', params.get('joint_friction', 0.04)),  # leg_l3
                3: params.get('joint_friction_l4', params.get('joint_friction', 0.04)),  # leg_l4
                8: params.get('joint_friction_r3', params.get('joint_friction', 0.04)),  # leg_r3
                9: params.get('joint_friction_r4', params.get('joint_friction', 0.04)),  # leg_r4
            }
            
            print(f"DOF属性数组长度: {len(dof_props)}")
            print(f"摩擦系数数组长度: {len(dof_props['friction'])}")
            
            # 🔥 使用您建议的方法更新摩擦系数
            for joint_idx, friction_value in friction_dict.items():
                if len(dof_props['friction']) > joint_idx:
                    dof_props["friction"][joint_idx] = friction_value
                    print(f"  关节 {joint_idx}: {friction_value:.4f}")
                else:
                    print(f"  ⚠️ 跳过关节 {joint_idx}: 索引超出范围")
            
            # 应用到所有环境
            for i in range(env.num_envs):
                env.gym.set_actor_dof_properties(env.envs[i], 0, dof_props)
            
            print("✅ 摩擦系数已通过DOF属性更新完成")
            
            # 🔥 可选：验证更新是否成功
            verify_props = env.gym.get_actor_dof_properties(env.envs[0], 0)
            print("验证更新结果:")
            for joint_idx in friction_dict.keys():
                if len(verify_props['friction']) > joint_idx:
                    print(f"  关节 {joint_idx}: {verify_props['friction'][joint_idx]:.4f}")
            
        except Exception as e:
            print(f"❌ 通过DOF属性更新摩擦系数失败: {e}")
        
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
        
        # 🔥 修改：从单一速度值生成对称的上下限
        angle_vel_l3 = abs(params.get('angle_vel_l3', 10.0))
        angle_vel_l4 = abs(params.get('angle_vel_l4', 13.0))
        
        # 将优化的参数传递给环境配置
        env.cfg.dynamic_torque_params.update({
            'torque_l3_top': params.get('torque_l3_top', 75.0),
            'torque_l3_bottom': params.get('torque_l3_bottom', -60.0),
            'torque_l4_top': params.get('torque_l4_top', 100.0),
            'torque_l4_bottom': params.get('torque_l4_bottom', -180.0),
            'speed_threshold_l3_q1': params.get('speed_threshold_l3_q1', 5.0),
            'speed_threshold_l3_q3': params.get('speed_threshold_l3_q3', 5.0),
            'speed_threshold_l4_q1': params.get('speed_threshold_l4_q1', 7.0),
            'speed_threshold_l4_q3': params.get('speed_threshold_l4_q3', 7.0),
            
            # 🔥 修改：使用对称的速度限制
            'angle_vel_l3_top': angle_vel_l3,
            'angle_vel_l3_bottom': -angle_vel_l3,
            'angle_vel_l4_top': angle_vel_l4,
            'angle_vel_l4_bottom': -angle_vel_l4,
        })
        
        print(f"✅ 动态扭矩参数已传递给环境:")
        print(f"  L3: 扭矩[{params.get('torque_l3_bottom', -60.0):.1f}, {params.get('torque_l3_top', 75.0):.1f}]")
        print(f"      速度阈值 Q1:{params.get('speed_threshold_l3_q1', 5.0):.1f}, Q3:{params.get('speed_threshold_l3_q3', 5.0):.1f}")
        print(f"      速度限制: ±{angle_vel_l3:.1f} rad/s")
        print(f"  L4: 扭矩[{params.get('torque_l4_bottom', -180.0):.1f}, {params.get('torque_l4_top', 100.0):.1f}]")
        print(f"      速度阈值 Q1:{params.get('speed_threshold_l4_q1', 7.0):.1f}, Q3:{params.get('speed_threshold_l4_q3', 7.0):.1f}")
        print(f"      速度限制: ±{angle_vel_l4:.1f} rad/s")

     
    
   
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
        简化版时间序列对齐：通过互相关对齐两个时间序列
        """
        from scipy import signal
        
        if alignment_dims is None:
            alignment_dims = [2, 3]  # 默认使用关节位置的前两维
        
        print(f"使用维度 {alignment_dims} 进行时间序列对齐")
        
        # 🔥 简化：只计算一个统一的延迟，不分别处理actions和other data
        delays = []
        
        for dim in alignment_dims:
            if dim >= min(sim_data.shape[1], real_data.shape[1]):
                continue
                
            # 检查信号方差，跳过无效维度
            if np.var(real_data[:, dim]) < 1e-6 or np.var(sim_data[:, dim]) < 1e-6:
                continue
            
            # 计算互相关
            correlation = signal.correlate(real_data[:, dim], sim_data[:, dim], mode='full')
            delay = correlation.argmax() - (len(sim_data) - 1)
            delays.append(delay)
            print(f"维度 {dim} 延迟: {delay}")
        
        # 🔥 简化：使用中位数延迟
        if delays:
            final_delay = int(np.median(delays))
            print(f"最终选择延迟: {final_delay} (来自延迟列表: {delays})")
        else:
            final_delay = 0
            print("未找到有效延迟，使用 0")
        
        # 🔥 简化：统一应用延迟到整个数据
        if final_delay > 0:
            # sim_data 滞后于 real_data
            common_len = min(len(real_data) - final_delay, len(sim_data))
            if common_len <= 0:
                print("警告：延迟太大，无法对齐数据")
                return sim_data, real_data
            aligned_real = real_data[final_delay : final_delay + common_len]
            aligned_sim = sim_data[:common_len]
        elif final_delay < 0:
            # sim_data 领先于 real_data
            delay_abs = abs(final_delay)
            common_len = min(len(sim_data) - delay_abs, len(real_data))
            if common_len <= 0:
                print("警告：延迟太大，无法对齐数据")
                return sim_data, real_data
            aligned_sim = sim_data[delay_abs : delay_abs + common_len]
            aligned_real = real_data[:common_len]
        else:
            # 无延迟，只保证长度一致
            min_len = min(len(sim_data), len(real_data))
            aligned_sim = sim_data[:min_len]
            aligned_real = real_data[:min_len]
        
        # 保存对齐结果供其他函数使用
        self.last_aligned_sim_data = aligned_sim
        self.last_aligned_real_data = aligned_real
        
        print(f"对齐完成:")
        print(f"  应用延迟: {final_delay}")
        print(f"  对齐后长度: {len(aligned_sim)}")
        print(f"  原始长度: sim={len(sim_data)}, real={len(real_data)}")
        
        return aligned_sim, aligned_real

    def compute_distance(self, sim_data, real_data, slice=0):
        """
        简化版距离计算函数
        """
        from scipy.stats import wasserstein_distance
        from sklearn.metrics.pairwise import rbf_kernel
        
        # 数据预处理
        sim_data = sim_data[slice:]
        real_data = real_data[slice:]
        
        # 时间序列对齐
        aligned_sim_data, aligned_real_data = self.align_time_series(sim_data, real_data)
        
        # 🔥 简化：只关注关键关节 (L3, L4, R3, R4)
        key_joints = [2, 3, 8, 9]  # L3, L4, R3, R4 关节索引
        key_dims = []
        
        # 位置、速度、动作维度
        for joint in key_joints:
            key_dims.extend([joint, joint+12, joint+24])  # pos, vel, action
        
        # 计算Wasserstein距离
        wd_total = 0.0
        for dim in key_dims:
            if dim < aligned_sim_data.shape[1] and dim < aligned_real_data.shape[1]:
                wd = wasserstein_distance(aligned_sim_data[:, dim], aligned_real_data[:, dim])
                
                # 简单归一化
                std_combined = np.sqrt(np.var(aligned_sim_data[:, dim]) + np.var(aligned_real_data[:, dim]))
                wd_normalized = wd / std_combined if std_combined > 1e-6 else 0.0
                wd_total += wd_normalized
        
        # 计算MMD距离
        sim_filtered = aligned_sim_data[:, key_dims]
        real_filtered = aligned_real_data[:, key_dims]
        
        # 简化的MMD计算
        gamma = 1.0 / sim_filtered.shape[1]  # 简单的gamma选择
        K_XX = rbf_kernel(sim_filtered, sim_filtered, gamma=gamma).mean()
        K_YY = rbf_kernel(real_filtered, real_filtered, gamma=gamma).mean()
        K_XY = rbf_kernel(sim_filtered, real_filtered, gamma=gamma).mean()
        mmd = K_XX + K_YY - 2 * K_XY
        
        # 组合最终距离
        final_distance = wd_total + mmd * 50.0
        
        print(f"Distance: WD={wd_total:.4f}, MMD={mmd:.6f}, Total={final_distance:.4f}")
        
        return final_distance


    def compute_distance_pos(self, sim_data, real_data,slice=0):
        """
        计算基于关节位置的简单欧几里得距离: ||p_real - p_sim||^2
        
        Args:
            sim_data (np.ndarray): 仿真数据，形状为 (n_steps, n_features)
            real_data (np.ndarray): 真实数据，形状为 (m_steps, n_features)
        
        Returns:
            float: 位置距离的平方和
        """
        
        # --- 1. 数据预处理 ---
        START_INDEX = slice
        
        sim_data = sim_data[START_INDEX:]
        real_data = real_data[START_INDEX:]
        
        joint_list = [3,4,7,8]  # 只使用 l3 和 l4 关节

        pos_sim = sim_data[:, joint_list]  
        pos_real = real_data[:, joint_list]  

        print(f"关节位置数据形状: sim={pos_sim.shape}, real={pos_real.shape}")
        
        # --- 4. 计算欧几里得距离的平方 ---
        # 计算每个时间步的位置差异
        pos_diff = pos_real - pos_sim  # 形状: (n_steps, 12)
        
        # 计算每个时间步的距离平方: ||p_real - p_sim||^2
        distance_squared_per_step = np.sum(pos_diff**2, axis=1)  # 形状: (n_steps,)
        
        # 计算总的平均距离平方
        total_distance_squared = np.mean(distance_squared_per_step)
        
        # --- 5. 输出详细信息 ---
        print(f"位置差异统计:")
        print(f"  每个关节的平均绝对误差: {np.mean(np.abs(pos_diff), axis=0)}")
        print(f"  每个关节的RMS误差: {np.sqrt(np.mean(pos_diff**2, axis=0))}")
        print(f"  每个时间步的平均距离: {np.sqrt(np.mean(distance_squared_per_step)):.6f}")
        print(f"  总距离平方: {total_distance_squared:.6f}")
        
       
        return total_distance_squared

    # def sim2real_distance(self, params, slice=SLICE):
    #     """计算sim2real距离"""
    #     print(f"Testing params: {params}")
        
    #         # 将列表参数转换为字典格式
    #     if isinstance(params, (list, np.ndarray)):
    #         param_dict = {
    #             'joint_friction_l3': params[0],
    #             'torque_l3_top': params[1],
    #             'torque_l3_bottom': params[2], 
    #             'torque_l4_top': params[1],      # 使用相同的扭矩限制
    #             'torque_l4_bottom': params[2],   # 使用相同的扭矩限制
    #             'angle_vel_l3': abs(params[3]) if len(params) > 3 else 10.0,
    #             'angle_vel_l4': abs(params[4]) if len(params) > 4 else 13.0,
            
    #             # 新增：动态扭矩参数（如果参数数组足够长）
    #             'speed_threshold_l3': params[5] if len(params) > 5 else 5.0,
    #             'speed_threshold_l4': params[6] if len(params) > 6 else 7.0,
    #             'max_speed_l3': params[7] if len(params) > 7 else 10.0,
    #             'max_speed_l4': params[8] if len(params) > 8 else 12.0,
    #             'use_dynamic_torque': True if len(params) > 5 else False,
        
    #         }
    #     else:
    #         param_dict = params
        
    #     # 收集Isaac Gym数据
    #     sim_data = self.simulate_and_collect_isaac(param_dict)

    #     sim_data = sim_data[slice:]
    #     real_data = self.real_data[slice:]
        
    #     aligned_sim_data, aligned_real_data = self.align_time_series(sim_data, real_data)
    #     # 计算距离
    #     distance = self.compute_distance(aligned_sim_data, aligned_real_data)
    #     self.last_distance_score = distance
        
    #     if distance < self.best_score:
    #         self.best_score = distance
    #         self.best_params = param_dict
    #         print(f"New best score: {distance}, params: {param_dict}")
    #         self.align_real_data = aligned_real_data  # 保存对齐后的真实数据
    #         self.align_sim_data = aligned_sim_data    # 保存对齐后的仿真数据
    #         # 保存最佳参数时的数据对比
    #         self.save_and_visualize_data(self.align_real_data, self.align_sim_data, param_dict)

    #     return distance
        

    def sample_param(self, trial: optuna.Trial) -> Dict[str, Any]:
        # 🔥 修改：固定扭矩限制值
        torque_l3_top = 75  # 固定值
        torque_l3_bottom = -60  # 固定值
        torque_l4_top = 100  # 固定值
        torque_l4_bottom = -180  # 固定值
        
        # 🔥 修改：使用单一对称速度限制值
        angle_vel_l3 = trial.suggest_float("angle_vel_l3", 8.0, 15.0)  # L3对称速度限制
        angle_vel_l4 = trial.suggest_float("angle_vel_l4", 10.0, 18.0)  # L4对称速度限制
        
        # 新增：动态扭矩限制参数
        # 对应扭矩开始衰减的速度阈值
        speed_threshold_l3_q1 = trial.suggest_float('speed_threshold_l3_q1', 1.0, 8.0)  # L3 Q1象限速度阈值
        speed_threshold_l3_q3 = trial.suggest_float('speed_threshold_l3_q3', 1.0, 8.0)  # L3 Q3象限速度阈值
        speed_threshold_l4_q1 = trial.suggest_float('speed_threshold_l4_q1', 1.0, 8.0)  # L4 Q1象限速度阈值
        speed_threshold_l4_q3 = trial.suggest_float('speed_threshold_l4_q3', 1.0, 11.0)  # L4 Q3象限速度阈值
        
        # 是否使用动态扭矩限制的开关
        use_dynamic_torque = True
        
        # 关节摩擦系数
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
            "use_dynamic_torque": use_dynamic_torque,
            
            # 🔥 修改：使用单一对称速度限制
            "angle_vel_l3": angle_vel_l3,
            "angle_vel_l4": angle_vel_l4,
            
            # 关节摩擦系数
            "joint_friction_l3": joint_friction_l3,
            "joint_friction_l4": joint_friction_l4,
            "joint_friction_r3": joint_friction_r3,
            "joint_friction_r4": joint_friction_r4,
        }

    def objective(self, trial: optuna.Trial) -> float:
        """Optuna优化的目标函数"""
        args = self.sample_param(trial)
        
        # 🔥 整合sim2real_distance的功能
        print(f"Testing params: {args}")
        
        # 收集Isaac Gym数据
        sim_data = self.simulate_and_collect_isaac(args, steps=500, command=[1.8, 0., 0., 0.])
        
        # 应用切片和对齐
        sim_data_sliced = sim_data[SLICE:]
        real_data_sliced = self.real_data[SLICE:]
        aligned_sim_data, aligned_real_data = self.align_time_series(sim_data_sliced, real_data_sliced)
        
        # 计算距离
        distance = self.compute_distance(aligned_sim_data, aligned_real_data)
        self.last_distance_score = distance
        
        # 检查是否为最佳分数
        if distance < self.best_score:
            self.best_score = distance
            self.best_params = args
            print(f"New best score: {distance}, params: {args}")
            
            # 保存对齐后的数据
            self.align_real_data = aligned_real_data
            self.align_sim_data = aligned_sim_data
            
            # 保存最佳参数时的数据对比
            self.save_and_visualize_data(self.align_real_data, self.align_sim_data, args)
        
        return distance

            
        
        return distance


if __name__ == "__main__":
    # 获取命令行参数和配置
    args = get_args()
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # 设置环境参数
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
    jit_policy_path = f"../logs/kuavo_jog/exported/policies_Oct_test1/policy_1.pt"
    jit_policy = torch.jit.load(jit_policy_path)

    # 设置Optuna优化器
    sampler = CmaEsSampler(
        n_startup_trials=N_STARTUP_TRIALS,  # 前5次随机采样
        seed=SEED,  # 设置随机种子
        # 可选参数：
        # sigma0=0.2,  # 初始标准差
        # warn_independent_sampling=True  # 是否警告独立采样
    )
    pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=0)

    # 创建保存目录和数据库
    save_dir = "../logs/optuna_results"
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    db_path = os.path.join(save_dir, f"optuna_{timestamp}.sqlite3")
    study = optuna.create_study(sampler=sampler, pruner=pruner, storage=f"sqlite:///{db_path}", study_name="Gap-kill")

    
    # 创建优化器实例
    optimizer = Sim2RealCMAOptimizer(        
        real_data=None,  # 将在下面设置
        env=env,
        policy=policy,
        jit_policy=jit_policy,
            )
    
    # 设置真实数据参数
    real_data_from = "real"   # "mujoco" or "real"
    real_data_file = "data/real_run_data/octnew.npz"
    real_data_time_long = 5     # 单位：秒
    real_data_start = 810

    # 🔥 数据加载部分
    if real_data_from == "mujoco":
        print("Collecting 'real' data from Mujoco...")
        real_data = optimizer.collect_mujoco_data(steps=500, command=[1.8, 0., 0., 0.])
        optimizer.real_data = real_data
        print(f"Collected real data shape: {real_data.shape}")
        all_real_data = None
    else:
        print("=== 加载真实数据 ===")
        
        # 🔥 使用新的便捷函数加载数据
        try:
            check_real_data_timestamps(real_data_file)
            # 加载单次运行数据
            single_run_data = load_real_data_single_run(
                data_file=real_data_file,
                run_value=real_data_start,  # 第8次运行
                time_duration=real_data_time_long
            )
            optimizer.real_data = single_run_data
            print(f"✅ 单次运行数据加载完成，形状: {single_run_data.shape}")
            
            # 加载所有运行数据
            all_real_data = load_real_data_single_run(
                data_file=real_data_file,
                run_value=real_data_start,  # 第8次运行
                time_duration = 10
            )
            print(f"✅ 所有运行数据加载完成，共 {len(all_real_data)} 条")
            
            # 可选：检查数据时间戳
            # check_real_data_timestamps(real_data_file, run_value=real_data_run[7])
            pass
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            
    # 设置优化器的all_real_data
   
    # 🔥 开始优化流程
    print("\n=== 开始优化流程 ===")
    
    try:
        study.optimize(optimizer.objective, n_trials=N_TRIALS)
    except KeyboardInterrupt:
        print("优化被用户中断")
    
    
    
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