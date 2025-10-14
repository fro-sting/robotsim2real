import numpy as np
from humanoid.envs import *
from humanoid.utils import get_args, task_registry
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from datetime import datetime
import random

SEED = 42

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class SimpleModelValidator:
    def __init__(self, env, policy):
        self.env = env
        self.policy = policy
        
        # 目标关节：r3, r4, l3, l4 对应的索引
        self.target_indices = [2, 3, 8, 9]  # leg_l3, leg_l4, leg_r3, leg_r4
        self.joint_names = ['leg_l3', 'leg_l4', 'leg_r3', 'leg_r4']
        
        # 创建保存目录
        self.save_dir = f"test_model/simple_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"Results will be saved to: {self.save_dir}")
        
        # 存储所有运行的数据
        self.all_runs_data = []

    def run_test(self, run_id, steps=500):
        """运行单次测试"""
        print(f"Running test {run_id + 1}/10...")
        
        # 设置命令
        self.env.commands[:, 0] = 2.0  # vx
        self.env.commands[:, 1] = 0.0  # vy
        self.env.commands[:, 2] = 0.0  # vyaw
        self.env.commands[:, 3] = 0.0  # stand
        
        # 重置环境
        set_global_seed(SEED + run_id)
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        
        velocities = []
        torques = []
        
        for step in range(steps):
            # 设置命令
            self.env.commands[:, 0] = 2.0
            self.env.commands[:, 1] = 0.0
            self.env.commands[:, 2] = 0.0
            self.env.commands[:, 3] = 0.0
            
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
            
            # 收集目标关节的数据
            joint_vel = self.env.dof_vel[0].cpu().numpy()  # 第一个环境的关节速度
            joint_torques = self.env.torques[0].cpu().numpy()  # 第一个环境的关节扭矩
            
            # 只保存目标关节的数据
            target_velocities = [joint_vel[i] for i in self.target_indices]
            target_torques = [joint_torques[i] for i in self.target_indices]
            
            velocities.append(target_velocities)
            torques.append(target_torques)
        
        run_data = {
            'velocities': np.array(velocities),
            'torques': np.array(torques)
        }
        
        self.all_runs_data.append(run_data)
        print(f"Test {run_id + 1} completed. Data shape: {run_data['velocities'].shape}")
        
        return run_data

    def plot_results(self):
        """绘制结果"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Joint Torque vs Velocity - Model Validation (10 runs)', fontsize=16)
        
        colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
        
        # 理论限制
        torque_limits = [84.0, 250.0, 84.0, 250.0]  # l3, l4, r3, r4
        velocity_limits = [10.0, 12.0, 10.0, 12.0]  # l3, l4, r3, r4
        
        # 自定义约束参数
        params = {
            'torque_l3_top': 75,
            'torque_l3_bottom': -60,
            'torque_l4_top': 100,
            'torque_l4_bottom': -180,
            'speed_threshold_l3_q1': 4.905901356333346,
            'speed_threshold_l3_q3': 3.8994731833906897,
            'speed_threshold_l4_q1': 3.879016896783313,
            'speed_threshold_l4_q3': 1.0258867744652869,
            'angle_vel_l3_top': 10,
            'angle_vel_l3_bottom': -10,
            'angle_vel_l4_top': 13,
            'angle_vel_l4_bottom': -13
        }
        # params = {
        #     'torque_l3_top': 43.693,
        #     'torque_l3_bottom': -38.823,
        #     'torque_l4_top': 40.626,
        #     'torque_l4_bottom': -156.410,
        #     'speed_threshold_l3_q1': 4.383,
        #     'speed_threshold_l3_q3': 4.062,
        #     'speed_threshold_l4_q1': 6.860,
        #     'speed_threshold_l4_q3': 6.689,
        #     'angle_vel_l3_top': 5.1087,
        #     'angle_vel_l3_bottom': -9.2690,
        #     'angle_vel_l4_top': 7.9569,
        #     'angle_vel_l4_bottom': -7.3234,
        # }
        
        
        def draw_custom_constraint(ax, joint_name, vel_limit):
            """绘制自定义约束边界"""
            if joint_name == 'leg_l3':
                torque_top = params['torque_l3_top']
                torque_bottom = params['torque_l3_bottom']
                speed_threshold_q1 = params['speed_threshold_l3_q1']
                speed_threshold_q3 = params['speed_threshold_l3_q3']
                angle_vel_top = params['angle_vel_l3_top']
                angle_vel_bottom = params['angle_vel_l3_bottom']
            elif joint_name == 'leg_l4':
                torque_top = params['torque_l4_top']
                torque_bottom = params['torque_l4_bottom']
                speed_threshold_q1 = params['speed_threshold_l4_q1']
                speed_threshold_q3 = params['speed_threshold_l4_q3']
                angle_vel_top = params['angle_vel_l4_top']
                angle_vel_bottom = params['angle_vel_l4_bottom']
            else:
                # 对于 r3, r4 使用相同的 l3, l4 参数
                if 'r3' in joint_name:
                    torque_top = params['torque_l3_top']
                    torque_bottom = params['torque_l3_bottom']
                    speed_threshold_q1 = params['speed_threshold_l3_q1']
                    speed_threshold_q3 = params['speed_threshold_l3_q3']
                    angle_vel_top = params['angle_vel_l3_top']
                    angle_vel_bottom = params['angle_vel_l3_bottom']
                else:  # r4
                    torque_top = params['torque_l4_top']
                    torque_bottom = params['torque_l4_bottom']
                    speed_threshold_q1 = params['speed_threshold_l4_q1']
                    speed_threshold_q3 = params['speed_threshold_l4_q3']
                    angle_vel_top = params['angle_vel_l4_top']
                    angle_vel_bottom = params['angle_vel_l4_bottom']
            
            # 创建速度数组
            vel_range = np.linspace(-vel_limit, vel_limit, 1000)
            
            # 上边界（正扭矩）
            upper_boundary = []
            for v in vel_range:
                if v >= 0:  # 第一象限
                    if v <= speed_threshold_q1:
                        # 0到拐点：水平线
                        torque = torque_top
                    else:
                        # 拐点到速度限制：线性下降到0
                        if v <= vel_limit:
                            # 从(speed_threshold_q1, torque_top)到(angle_vel_top, 0)的直线
                            slope = (0 - torque_top) / (angle_vel_top - speed_threshold_q1)
                            torque = torque_top + slope * (v - speed_threshold_q1)
                            torque = max(torque, 0)  # 确保不小于0
                        else:
                            torque = 0
                else:  # 第二象限：直线
                    torque = torque_top
                
                upper_boundary.append(torque)
            
            # 下边界（负扭矩）
            lower_boundary = []
            for v in vel_range:
                if v <= 0:  # 第三象限
                    if abs(v) <= speed_threshold_q3:
                        # 0到拐点：水平线
                        torque = torque_bottom
                    else:
                        # 拐点到速度限制：线性上升到0
                        if abs(v) <= abs(angle_vel_bottom):
                            # 从(-speed_threshold_q3, torque_bottom)到(angle_vel_bottom, 0)的直线
                            slope = (0 - torque_bottom) / (angle_vel_bottom - (-speed_threshold_q3))
                            torque = torque_bottom + slope * (v - (-speed_threshold_q3))
                            torque = min(torque, 0)  # 确保不大于0
                        else:
                            torque = 0
                else:  # 第四象限：直线
                    torque = torque_bottom
                
                lower_boundary.append(torque)
            
            # 绘制边界
            ax.plot(vel_range, upper_boundary, 'r-', linewidth=2, alpha=0.8, label='Custom Upper Limit')
            ax.plot(vel_range, lower_boundary, 'r-', linewidth=2, alpha=0.8, label='Custom Lower Limit')
            
            # 标记拐点
            ax.plot(speed_threshold_q1, torque_top, 'ro', markersize=6, label=f'Q1 Point ({speed_threshold_q1:.1f}, {torque_top:.1f})')
            ax.plot(-speed_threshold_q3, torque_bottom, 'ro', markersize=6, label=f'Q3 Point ({-speed_threshold_q3:.1f}, {torque_bottom:.1f})')
            ax.plot(angle_vel_top, 0, 'go', markersize=6, label=f'Q1 End ({angle_vel_top:.1f}, 0)')
            ax.plot(angle_vel_bottom, 0, 'go', markersize=6, label=f'Q3 End ({angle_vel_bottom:.1f}, 0)')
        
        for joint_idx in range(4):  # 4个目标关节
            row = joint_idx // 2
            col = joint_idx % 2
            ax = axes[row, col]
            
            # 绘制所有运行的数据
            for run_id, run_data in enumerate(self.all_runs_data):
                # 🔥 只使用150步之后的数据
                start_idx = 200
                if len(run_data['velocities']) > start_idx:
                    velocities = run_data['velocities'][start_idx:, joint_idx]
                    torques = run_data['torques'][start_idx:, joint_idx]
                else:
                    # 如果数据不够150步，使用所有数据
                    velocities = run_data['velocities'][:, joint_idx]
                    torques = run_data['torques'][:, joint_idx]
                
                ax.scatter(velocities, torques, alpha=0.6, s=1, 
                        color=colors[0], label=f'Run {run_id + 1}'
                        )
                
            # 添加理论限制线（虚线）
            vel_limit = velocity_limits[joint_idx]
            torque_limit = torque_limits[joint_idx]
            
            # 速度限制线
            ax.axvline(x=vel_limit, color='gray', linestyle='--', alpha=0.5, linewidth=1, label='Vel Limit')
            ax.axvline(x=-vel_limit, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            
            # 扭矩限制线
            ax.axhline(y=torque_limit, color='gray', linestyle='--', alpha=0.5, linewidth=1, label='Torque Limit')
            ax.axhline(y=-torque_limit, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            
            # 绘制自定义约束边界
            draw_custom_constraint(ax, self.joint_names[joint_idx], vel_limit)
            
            # 添加零线
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
            
            ax.set_xlabel('Joint Velocity [rad/s]')
            ax.set_ylabel('Joint Torque [Nm]')
            ax.set_title(f'{self.joint_names[joint_idx]}')
            ax.grid(True, alpha=0.3)
            
            # 计算并显示统计信息
            all_velocities = []
            all_torques = []
            for run_data in self.all_runs_data:
                all_velocities.extend(run_data['velocities'][:, joint_idx])
                all_torques.extend(run_data['torques'][:, joint_idx])
            
            all_velocities = np.array(all_velocities)
            all_torques = np.array(all_torques)
            
            # 计算违反比例（基于原始限制）
            vel_violations = np.sum((np.abs(all_velocities) > vel_limit))
            torque_violations = np.sum((np.abs(all_torques) > torque_limit))
            total_points = len(all_velocities)
            
            vel_ratio = vel_violations / total_points
            torque_ratio = torque_violations / total_points
            
            # 显示统计信息
            stats_text = f'Vel violations: {vel_ratio:.2%}\nTorque violations: {torque_ratio:.2%}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        # 只在第一个子图显示图例
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
        
        plt.tight_layout()
        
        # 保存图像
        filename = os.path.join(self.save_dir, 'torque_velocity_validation.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Results saved to: {filename}")

    def analyze_violations(self):
        """分析约束违反"""
        print("\\n=== Constraint Violation Analysis ===")
        
        torque_limits = [84.0, 250.0, 84.0, 250.0]
        velocity_limits = [10.0, 12.0, 10.0, 12.0]
        
        for joint_idx in range(4):
            joint_name = self.joint_names[joint_idx]
            vel_limit = velocity_limits[joint_idx]
            torque_limit = torque_limits[joint_idx]
            
            print(f"\\n{joint_name}:")
            print(f"  Velocity limit: ±{vel_limit} rad/s")
            print(f"  Torque limit: ±{torque_limit} Nm")
            
            all_velocities = []
            all_torques = []
            for run_data in self.all_runs_data:
                all_velocities.extend(run_data['velocities'][:, joint_idx])
                all_torques.extend(run_data['torques'][:, joint_idx])
            
            all_velocities = np.array(all_velocities)
            all_torques = np.array(all_torques)
            
            # 违反统计
            vel_violations = np.sum((np.abs(all_velocities) > vel_limit))
            torque_violations = np.sum((np.abs(all_torques) > torque_limit))
            total_points = len(all_velocities)
            
            print(f"  Velocity violations: {vel_violations}/{total_points} ({vel_violations/total_points:.2%})")
            print(f"  Torque violations: {torque_violations}/{total_points} ({torque_violations/total_points:.2%})")
            print(f"  Max velocity: {np.max(np.abs(all_velocities)):.3f} rad/s")
            print(f"  Max torque: {np.max(np.abs(all_torques)):.3f} Nm")

    def run_validation(self, num_runs=10, steps_per_run=500):
        """运行完整验证"""
        print(f"Starting validation: {num_runs} runs × {steps_per_run} steps")
        
        # 运行测试
        for run_id in range(num_runs):
            self.run_test(run_id, steps_per_run)
        
        # 分析和绘图
        self.analyze_violations()
        self.plot_results()
        
        print(f"\\nValidation complete! Results saved to: {self.save_dir}")


if __name__ == "__main__":
    # 设置种子
    set_global_seed(SEED)
    
    # 获取参数
    args = get_args()
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    # 配置环境
    train_cfg.seed = SEED
    env_cfg.env.num_envs = 1  # 只用一个环境
    env_cfg.terrain.mesh_type = 'plane'
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False  # 关闭噪声以获得更清晰的结果
    env_cfg.domain_rand.push_robots = False
    train_cfg.runner.resume = True
    
    # 创建环境和策略
    print("Creating Isaac Gym environment...")
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    
    # 尝试加载策略
    try:
        policy = ppo_runner.get_inference_policy(device=env.device)
        print("✓ Policy loaded successfully from training runner")
    except Exception as e:
        print(f"⚠️ Failed to load policy from runner: {e}")
        print("Trying to load JIT policy directly...")
        
        # 尝试直接加载JIT模型
        jit_policy_path = "/home/wegg/kuavo_rl_asap-main111/RL_train/logs/kuavo_jog/Sep17_18-20-38_v1/model_60000.pt"
        if os.path.exists(jit_policy_path):
            policy = torch.jit.load(jit_policy_path)
            print(f"✓ JIT policy loaded from: {jit_policy_path}")
        else:
            print(f"❌ JIT policy not found at: {jit_policy_path}")
            print("Please check the model path or train a model first")
            
            # 列出可能的模型路径
            logs_dir = "../logs"
            if os.path.exists(logs_dir):
                print(f"\nAvailable log directories in {logs_dir}:")
                for item in os.listdir(logs_dir):
                    item_path = os.path.join(logs_dir, item)
                    if os.path.isdir(item_path):
                        print(f"  - {item}")
                        # 检查是否有exported模型
                        exported_path = os.path.join(item_path, "exported")
                        if os.path.exists(exported_path):
                            print(f"    └── has exported models")
            exit(1)
    
    print("Environment and policy loaded successfully!")
    
    # 创建简单验证器
    validator = SimpleModelValidator(env=env, policy=policy)
    
    # 运行验证
    validator.run_validation(num_runs=5, steps_per_run=500)
    
    print("\\n=== Validation Summary ===")
    print("✓ 10 test runs completed")
    print("✓ Torque-velocity curves generated")
    print("✓ Constraint violation analysis completed")
    print("✓ Check the output directory for detailed results")
