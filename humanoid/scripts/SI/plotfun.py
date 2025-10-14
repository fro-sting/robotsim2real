"""
扭矩-速度曲线绘制工具类 (TorqueVelocityPlotter)

本模块提供了一套完整的扭矩-速度关系曲线绘制和分析工具，主要用于机器人关节特性分析和仿真验证。

主要功能：
1. 扭矩-速度曲线绘制 (包含四象限特性)
2. 真实数据与仿真数据对比分析
3. 关节位置-速度差异分析
4. 数据分布统计分析
5. 多数据集综合分析

核心类：
- TorqueVelocityPlotter: 主要绘图工具类

主要方法及用法：

1. plot_torque_velocity_curves(mujoco_data, isaac_data, params=None, filename='torque_velocity_curves.png')
   作用：绘制12个关节的扭矩-速度关系曲线，包含非对称四象限扭矩限制理论曲线
   用法：
   ```python
   plotter = TorqueVelocityPlotter(save_dir='output')
   plotter.plot_torque_velocity_curves(real_data, sim_data, params={'torque_l3_top': 35.0})
   ```

2. plot_mujoco_isaac_torque_comparison(mujoco_data, isaac_data, filename='mujoco_isaac_torque_comparison.png')
   作用：绘制Mujoco和Isaac Gym的扭矩时间序列对比图
   用法：
   ```python
   plotter.plot_mujoco_isaac_torque_comparison(real_data, sim_data)
   ```

3. plot_individual_torque_velocity_analysis(mujoco_data, isaac_data, filename='detailed_torque_velocity_analysis.png')
   作用：详细分析关键关节(L3,L4,R3,R4)的力矩-速度特性，包含速度区间平均力矩计算
   用法：
   ```python
   plotter.plot_individual_torque_velocity_analysis(real_data, sim_data)
   ```

4. plot_all_real_data_torque_velocity_curves(all_real_data, sim_data=None, params=None, filename='all_real_data_torque_velocity_curves.png')
   作用：绘制所有真实数据运行的L3、L4、R3、R4关节扭矩-速度曲线综合分析图
   用法：
   ```python
   all_data = {0: {'data': data1}, 1: {'data': data2}}
   plotter.plot_all_real_data_torque_velocity_curves(all_data, sim_data, params)
   ```

5. plot_joint_position_velocity_difference(real_data, sim_data, filename='joint_pos_vel_difference.png')
   作用：绘制真实数据与仿真数据的关节位置和速度差异散点图，X轴为位置差异，Y轴为速度差异
   用法：
   ```python
   plotter.plot_joint_position_velocity_difference(real_data, sim_data)
   ```

6. calculate_asymmetric_four_quadrant_torque_curve(speeds, torque_top, torque_bottom, threshold_q1, threshold_q3, max_speed)
   作用：计算支持不同象限独立阈值的四象限动态扭矩曲线
   参数：
   - speeds: 速度数组
   - torque_top/bottom: 最大/最小扭矩值
   - threshold_q1/q3: Q1和Q3象限的速度阈值
   - max_speed: 最大速度
   用法：
   ```python
   speeds = np.linspace(-15, 15, 300)
   top_curve, bottom_curve = plotter.calculate_asymmetric_four_quadrant_torque_curve(
       speeds, 35.0, -35.0, 5.0, 5.0, 10.0
   )
   ```

7. _plot_velocity_comparison(muj_data, isaac_data, vel_names, title, filename='velocity_comparison.png', command=None)
   作用：绘制速度对比图，包含真实数据、仿真数据和指令速度的对比，提供详细的速度统计信息
   用法：
   ```python
   plotter._plot_velocity_comparison(real_vel, sim_vel, ['vx', 'vy', 'vz'], 'Linear Velocity', command=cmd_data)
   ```

8. _plot_joint_comparison(muj_data, isaac_data, joint_names, title, filename='joint_comparison.png')
   作用：绘制关节数据时间序列对比图（位置、速度或扭矩）
   用法：
   ```python
   joint_names = ['leg_l1', 'leg_l2', ..., 'leg_r6']
   plotter._plot_joint_comparison(real_joints, sim_joints, joint_names, 'Joint Position')
   ```

9. _plot_distribution_comparison(muj_data, isaac_data)
   作用：绘制数据分布直方图对比，分析真实数据与仿真数据的统计分布差异
   用法：
   ```python
   plotter._plot_distribution_comparison(real_data, sim_data)
   ```

10. _generate_data_report(muj_data, isaac_data, params)
    作用：生成详细的数据分析报告，包含MSE、MAE、相关性等统计指标
    用法：
    ```python
    plotter._generate_data_report(real_data, sim_data, optimization_params)
    ```

便捷函数：

1. plot_torque_velocity_curves(mujoco_data, isaac_data, params=None, save_dir=None, filename='torque_velocity_curves.png')
   作用：快速绘制扭矩-速度曲线的便捷函数
   用法：
   ```python
   save_dir = plot_torque_velocity_curves(real_data, sim_data, params={'torque_l3_top': 35.0})
   ```

2. plot_all_real_data_torque_velocity_curves(all_real_data, sim_data=None, params=None, save_dir=None, filename='all_real_data_torque_velocity_curves.png')
   作用：快速绘制所有真实数据扭矩-速度曲线的便捷函数
   用法：
   ```python
   plot_path = plot_all_real_data_torque_velocity_curves(all_data, sim_data, params)
   ```

数据格式要求：
- 输入数据格式：[joint_pos(12), joint_vel(12), action(12), base_vel(3), world_vel(3), actual_torques(12)]
- 关节索引：0-11 对应 leg_l1~leg_l6, leg_r1~leg_r6
- 关键关节：[2,3,8,9] 对应 leg_l3,leg_l4,leg_r3,leg_r4

使用示例：
```python
from plotfun import TorqueVelocityPlotter

# 创建绘图器
plotter = TorqueVelocityPlotter(save_dir='analysis_output')

# 绘制基本扭矩-速度曲线
params = {
    'torque_l3_top': 35.0, 'torque_l3_bottom': -35.0,
    'torque_l4_top': 150.0, 'torque_l4_bottom': -150.0,
    'speed_threshold_l3_q1': 5.0, 'speed_threshold_l3_q3': 5.0,
    'speed_threshold_l4_q1': 7.0, 'speed_threshold_l4_q3': 7.0
}
plotter.plot_torque_velocity_curves(real_data, sim_data, params)

# 绘制位置-速度差异图
plotter.plot_joint_position_velocity_difference(real_data, sim_data)

# 生成综合分析报告
plotter._generate_data_report(real_data, sim_data, params)
```

注意事项：
1. 确保输入数据格式正确，特别是关节顺序
2. 四象限模型参数需要根据实际机器人特性调整
3. 图片保存需要足够的磁盘空间
4. 建议在非GUI环境下使用matplotlib的'Agg'后端
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import seaborn as sns

# ...existing code...

import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import seaborn as sns






class TorqueVelocityPlotter:
    """扭矩-速度曲线绘制工具类"""
    
    def __init__(self, save_dir=None):
        """
        初始化绘图工具
        
        Args:
            save_dir (str): 保存图片的目录，如果为None则创建时间戳目录
        """
        if save_dir is None:
            self.save_dir = f"plot_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 启用matplotlib的非交互模式
        plt.ioff()
    
    def plot_torque_velocity_curves(self, mujoco_data, isaac_data, params=None, filename='torque_velocity_curves.png'):
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
        
        # 获取优化参数（用于绘制理论曲线）
        if params is None:
            params = {}
        
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
                ax.set_ylim(-90, 90)
            else:  # L4
                ax.set_ylim(-200, 120)
            
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
    
    def plot_individual_torque_velocity_analysis(self, mujoco_data, isaac_data, filename='detailed_torque_velocity_analysis.png'):
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
        plt.savefig(os.path.join(self.save_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_all_real_data_torque_velocity_curves(self, all_real_data, sim_data=None, params=None, filename='all_real_data_torque_velocity_curves.png'):
        """
        绘制所有9条real_data的L3、L4、R3、R4关节扭矩-速度曲线
        包含非对称四象限扭矩限制理论曲线
        """
        # 关节索引和名称
        key_joints = [2, 3, 8, 9]  # leg_l3, leg_l4, leg_r3, leg_r4
        joint_labels = ['Left Hip Pitch (L3)', 'Left Knee (L4)', 'Right Hip Pitch (R3)', 'Right Knee (R4)']
        
        # 获取优化参数（用于绘制理论曲线）
        if params is None:
            params = {}
        
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
                ax.set_ylim(-90, 90)
            else:  # L4
                ax.set_ylim(-200, 150)
            
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
        
        plt.close('all')
        print(f"Asymmetric four-quadrant torque-velocity curves saved: {filename}")
        return save_path
    
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
        
        return np.array(torque_top_curve), np.array(torque_bottom_curve)
    
    # def _plot_velocity_comparison(self, muj_data, isaac_data, vel_names, title, filename='velocity_comparison.png', command=None):
    #     """绘制速度数据对比图，包含指令速度"""
    #     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    #     fig.suptitle(f'{title} Comparison: Mujoco vs Isaac Gym vs Command', fontsize=16)
        
    #     time_steps_muj = np.arange(len(muj_data))
    #     time_steps_isaac = np.arange(len(isaac_data))
        
    #     for i in range(3):
    #         ax = axes[i]
            
    #         # 绘制真实数据和仿真数据
    #         ax.plot(time_steps_muj, muj_data[:, i], 'b-', label='Real', alpha=0.7, linewidth=1.5)
    #         ax.plot(time_steps_isaac, isaac_data[:, i], 'r--', label='Simdata in Isaac', alpha=0.7, linewidth=1.5)
            
    #         # 🔥 修改：处理渐进command序列
    #         if command is not None:
    #             # 检查command是否是时间序列（二维数组）
    #             if isinstance(command, np.ndarray) and command.ndim == 2:
    #                 # command是时间序列数据 [时间步, 4个command值]
    #                 if i < command.shape[1]:
    #                     # 创建与数据长度匹配的时间轴
    #                     command_length = len(command)
    #                     data_length = len(time_steps_muj)
                        
    #                     if command_length == data_length:
    #                         # 长度匹配，直接绘制
    #                         command_time = time_steps_muj
    #                         command_values = command[:, i]
    #                     else:
    #                         # 长度不匹配，进行插值对齐
    #                         command_time = np.linspace(0, len(time_steps_muj)-1, command_length)
    #                         command_values = command[:, i]
                        
    #                     ax.plot(command_time, command_values, 'g:', 
    #                         label=f'Command (Dynamic)', alpha=0.9, linewidth=2)
                        
    #                     # 🔥 计算指令跟踪误差（使用时间序列command）
    #                     if command_length == data_length:
    #                         real_cmd_error = np.mean(np.abs(muj_data[:, i] - command_values))
    #                         sim_cmd_error = np.mean(np.abs(isaac_data[:, i] - command_values))
    #                     else:
    #                         # 如果长度不匹配，插值计算误差
    #                         from scipy.interpolate import interp1d
    #                         interp_func = interp1d(command_time, command_values, 
    #                                             kind='linear', fill_value='extrapolate')
    #                         command_interp = interp_func(time_steps_muj)
    #                         real_cmd_error = np.mean(np.abs(muj_data[:, i] - command_interp))
    #                         sim_cmd_error = np.mean(np.abs(isaac_data[:, i] - command_interp))
                        
    #                 else:
    #                     # 该维度没有对应的command值，使用0
    #                     command_values = np.zeros(len(time_steps_muj))
    #                     ax.axhline(y=0, color='green', linestyle=':', 
    #                             label='Command (0)', alpha=0.9, linewidth=2)
    #                     real_cmd_error = np.mean(np.abs(muj_data[:, i]))
    #                     sim_cmd_error = np.mean(np.abs(isaac_data[:, i]))
                        
    #             else:
    #                 # command是固定值，绘制水平线（原来的逻辑）
    #                 if hasattr(command, '__len__') and len(command) > i:
    #                     command_value = command[i]
    #                 elif hasattr(command, '__len__') and len(command) > 0:
    #                     command_value = command[0] if i == 0 else 0  # vx用command[0]，vy,vz用0
    #                 else:
    #                     command_value = command if i == 0 else 0  # 如果command是标量
                    
    #                 ax.axhline(y=command_value, color='green', linestyle=':', 
    #                         label=f'Command ({command_value:.2f})', alpha=0.9, linewidth=2)
                    
    #                 # 计算与固定command的误差
    #                 real_cmd_error = np.mean(np.abs(muj_data[:, i] - command_value))
    #                 sim_cmd_error = np.mean(np.abs(isaac_data[:, i] - command_value))
    #         else:
    #             # 没有command数据
    #             real_cmd_error = None
    #             sim_cmd_error = None
            
    #         ax.set_title(f'{vel_names[i]}', fontsize=12)
    #         ax.set_xlabel('Time Steps')
    #         ax.set_ylabel('Velocity (m/s)' if 'Linear' in title else 'Angular Velocity (rad/s)')
    #         ax.grid(True, alpha=0.3)
    #         ax.legend()
            
    #         # 计算相关性（真实数据vs仿真数据）
    #         min_len = min(len(muj_data), len(isaac_data))
    #         correlation = np.corrcoef(muj_data[:min_len, i], isaac_data[:min_len, i])[0, 1]
            
    #         # 🔥 显示统计信息
    #         stats_text = f'Real vs Sim\nCorr: {correlation:.3f}'
            
    #         if real_cmd_error is not None and sim_cmd_error is not None:
    #             stats_text += f'\nCmd Track Error:\nReal: {real_cmd_error:.3f}\nSim: {sim_cmd_error:.3f}'
            
    #         ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
    #             verticalalignment='top', fontsize=9, 
    #             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(self.save_dir, filename), dpi=300, bbox_inches='tight')
    #     plt.close()
    def _plot_velocity_comparison(self, muj_data, isaac_data, vel_names, title, filename='velocity_comparison.png', command=None):
        """
        绘制速度对比图，包含速度统计信息
        
        Args:
            muj_data: 真实数据（可以为None或空数组）
            isaac_data: 仿真数据
            vel_names: 速度分量名称
            title: 图表标题
            filename: 保存文件名
            command: command序列（可选）
        """
        plt.figure(figsize=(15, 10))  # 增加高度以容纳统计信息
        
        # 🔥 修复：处理muj_data为None或空数组的情况
        if muj_data is not None and len(muj_data) > 0:
            time_steps_muj = np.arange(len(muj_data))
            has_real_data = True
        else:
            has_real_data = False
            time_steps_muj = None
        
        time_steps_isaac = np.arange(len(isaac_data))
        
        # 🔥 新增：计算速度统计信息
        def calculate_velocity_stats(data, label_prefix):
            stats = {}
            for i in range(min(2, len(vel_names))):  # 只计算vx和vy
                vel_name = vel_names[i]
                vel_data = data[:, i]
                
                stats[f'{vel_name}_mean'] = np.mean(vel_data)
                stats[f'{vel_name}_std'] = np.std(vel_data)
                stats[f'{vel_name}_max'] = np.max(vel_data)
                stats[f'{vel_name}_min'] = np.min(vel_data)
                stats[f'{vel_name}_rms'] = np.sqrt(np.mean(vel_data**2))
                
            return stats
        
        # 计算统计信息
        isaac_stats = calculate_velocity_stats(isaac_data, 'Sim')
        if has_real_data:
            real_stats = calculate_velocity_stats(muj_data, 'Real')
        
        # 🔥 修改主标题，包含速度统计摘要
        main_title = f'{title} Comparison'
        if has_real_data:
            main_title += f'\nReal: vx_μ={real_stats["vx_mean"]:.3f}±{real_stats["vx_std"]:.3f}, vy_μ={real_stats["vy_mean"]:.3f}±{real_stats["vy_std"]:.3f}'
        main_title += f'\nSim: vx_μ={isaac_stats["vx_mean"]:.3f}±{isaac_stats["vx_std"]:.3f}, vy_μ={isaac_stats["vy_mean"]:.3f}±{isaac_stats["vy_std"]:.3f}'
        
        plt.suptitle(main_title, fontsize=14)
        
        for i in range(2):  # 只处理vx和vy
            vel_name = vel_names[i]
            plt.subplot(2, 1, i+1)  # 🔥 改为2行1列布局
            
            # 🔥 只在有真实数据时绘制
            if has_real_data:
                plt.plot(time_steps_muj, muj_data[:, i], 'b-', label=f'Real {vel_name}', linewidth=2)
            
            # 绘制仿真数据
            plt.plot(time_steps_isaac, isaac_data[:, i], 'r--', label=f'Sim {vel_name}', linewidth=2)
            
            # 🔥 绘制command曲线（如果提供）
            if command is not None:
                if i == 0:  # 只为vx绘制command
                    if isinstance(command, np.ndarray) and len(command.shape) > 1:
                        # 渐进command序列
                        command_time_steps = np.arange(len(command))
                        plt.plot(command_time_steps, command[:, 0], 'g:', 
                                label='Command (Dynamic)', linewidth=2, alpha=0.8)
                        
                        # 🔥 修复：只在command不为空时计算误差
                        if len(command) > 0:
                            # 对command进行插值以匹配isaac_data的长度
                            if len(command) != len(isaac_data):
                                command_interp = np.interp(
                                    np.linspace(0, len(command)-1, len(isaac_data)),
                                    np.arange(len(command)),
                                    command[:, 0]
                                )
                            else:
                                command_interp = command[:, 0]
                            
                            # 计算与command的误差（只使用仿真数据）
                            sim_cmd_error = np.mean(np.abs(isaac_data[:, i] - command_interp))
                            
                            # 🔥 修复：只在有真实数据时计算真实数据与command的误差
                            if has_real_data and len(command) > 0:
                                # 为真实数据也插值command
                                if len(command) != len(muj_data):
                                    real_command_interp = np.interp(
                                        np.linspace(0, len(command)-1, len(muj_data)),
                                        np.arange(len(command)),
                                        command[:, 0]
                                    )
                                else:
                                    real_command_interp = command[:, 0]
                                
                                real_cmd_error = np.mean(np.abs(muj_data[:, i] - real_command_interp))
                    
                    else:
                        # 固定command值
                        if isinstance(command, (list, np.ndarray)):
                            command_val = command[0]
                        else:
                            command_val = command
                        plt.axhline(y=command_val, color='g', linestyle=':', 
                                label=f'Command ({command_val})', linewidth=2, alpha=0.8)
                        
                        # 计算误差
                        sim_cmd_error = np.mean(np.abs(isaac_data[:, i] - command_val))
                        
                        if has_real_data:
                            real_cmd_error = np.mean(np.abs(muj_data[:, i] - command_val))
            
            plt.xlabel('Time Steps')
            plt.ylabel(f'{vel_name} (m/s)')
            plt.title(f'{title} - {vel_name}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 🔥 新增：在每个子图上显示详细的速度统计信息
            stats_text = f'=== {vel_name.upper()} Statistics ===\n'
            
            # 仿真数据统计
            sim_vel_data = isaac_data[:, i]
            stats_text += f'Sim: μ={isaac_stats[f"{vel_name}_mean"]:.4f}, σ={isaac_stats[f"{vel_name}_std"]:.4f}\n'
            stats_text += f'     range=[{isaac_stats[f"{vel_name}_min"]:.4f}, {isaac_stats[f"{vel_name}_max"]:.4f}]\n'
            stats_text += f'     RMS={isaac_stats[f"{vel_name}_rms"]:.4f}\n'
            
            # 真实数据统计（如果有）
            if has_real_data:
                real_vel_data = muj_data[:, i]
                stats_text += f'Real: μ={real_stats[f"{vel_name}_mean"]:.4f}, σ={real_stats[f"{vel_name}_std"]:.4f}\n'
                stats_text += f'      range=[{real_stats[f"{vel_name}_min"]:.4f}, {real_stats[f"{vel_name}_max"]:.4f}]\n'
                stats_text += f'      RMS={real_stats[f"{vel_name}_rms"]:.4f}\n'
                
                # 计算差异统计
                min_len = min(len(real_vel_data), len(sim_vel_data))
                vel_diff = real_vel_data[:min_len] - sim_vel_data[:min_len]
                diff_mean = np.mean(vel_diff)
                diff_std = np.std(vel_diff)
                diff_mae = np.mean(np.abs(vel_diff))
                diff_rmse = np.sqrt(np.mean(vel_diff**2))
                
                stats_text += f'Diff (R-S): μ={diff_mean:.4f}, σ={diff_std:.4f}\n'
                stats_text += f'            MAE={diff_mae:.4f}, RMSE={diff_rmse:.4f}'
                
                # 计算相关性
                correlation = np.corrcoef(real_vel_data[:min_len], sim_vel_data[:min_len])[0, 1]
                stats_text += f'\nCorrelation: {correlation:.4f}'
            
            # 🔥 Command跟踪误差（如果有command）
            if command is not None and 'sim_cmd_error' in locals():
                stats_text += f'\n=== Command Tracking ===\n'
                stats_text += f'Sim-Cmd Error: {sim_cmd_error:.4f}'
                if has_real_data and 'real_cmd_error' in locals():
                    stats_text += f'\nReal-Cmd Error: {real_cmd_error:.4f}'
            
            # 在图的右侧显示统计信息
            plt.text(1.02, 0.5, stats_text, transform=plt.gca().transAxes, 
                    verticalalignment='center', fontsize=9, 
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8),
                    fontfamily='monospace')  # 使用等宽字体对齐数字
        
        # 🔥 新增：在图底部添加速度幅值统计
        if has_real_data:
            # 计算速度幅值 (speed = sqrt(vx^2 + vy^2))
            real_speed = np.sqrt(muj_data[:, 0]**2 + muj_data[:, 1]**2)
            real_speed_stats = f'Real Speed: μ={np.mean(real_speed):.4f}, σ={np.std(real_speed):.4f}, max={np.max(real_speed):.4f}'
        
        isaac_speed = np.sqrt(isaac_data[:, 0]**2 + isaac_data[:, 1]**2)
        isaac_speed_stats = f'Sim Speed: μ={np.mean(isaac_speed):.4f}, σ={np.std(isaac_speed):.4f}, max={np.max(isaac_speed):.4f}'
        
        speed_summary = isaac_speed_stats
        if has_real_data:
            speed_summary = real_speed_stats + '\n' + isaac_speed_stats
        
        # 在图的底部添加速度幅值统计
        plt.figtext(0.1, 0.02, f'Speed Magnitude Statistics:\n{speed_summary}', 
                    fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15, right=0.8)  # 为统计信息留出空间
        
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 🔥 新增：在控制台打印速度统计摘要
        print(f"速度对比图已保存: {save_path}")
        print(f"\n=== 速度统计摘要 ===")
        if has_real_data:
            print(f"真实数据:")
            print(f"  vx: μ={real_stats['vx_mean']:.4f} ± {real_stats['vx_std']:.4f} (range: [{real_stats['vx_min']:.4f}, {real_stats['vx_max']:.4f}])")
            print(f"  vy: μ={real_stats['vy_mean']:.4f} ± {real_stats['vy_std']:.4f} (range: [{real_stats['vy_min']:.4f}, {real_stats['vy_max']:.4f}])")
            print(f"  speed: μ={np.mean(real_speed):.4f} ± {np.std(real_speed):.4f} (max: {np.max(real_speed):.4f})")
        
        print(f"仿真数据:")
        print(f"  vx: μ={isaac_stats['vx_mean']:.4f} ± {isaac_stats['vx_std']:.4f} (range: [{isaac_stats['vx_min']:.4f}, {isaac_stats['vx_max']:.4f}])")
        print(f"  vy: μ={isaac_stats['vy_mean']:.4f} ± {isaac_stats['vy_std']:.4f} (range: [{isaac_stats['vy_min']:.4f}, {isaac_stats['vy_max']:.4f}])")
        print(f"  speed: μ={np.mean(isaac_speed):.4f} ± {np.std(isaac_speed):.4f} (max: {np.max(isaac_speed):.4f})")

    def _plot_joint_comparison(self, muj_data, isaac_data, joint_names, title, filename='joint_comparison.png'):
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
        dimensions = [2, 3, 14 ,15, 26, 27]  # 每种数据类型选2个关节
        dim_names = ['Joint Pos (leg_l3)', 'Joint Pos (leg_l4)',
                     'Joint Vel (leg_l3)', 'Joint Vel (leg_l4)',
                     'Action (leg_l3)', 'Action (leg_l4)']
        
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

    # def plot_joint_position_velocity_difference(self, real_data, sim_data, filename='joint_pos_vel_difference.png'):
    #     """
    #     绘制真实数据与仿真数据的关节位置和速度差异图（只显示L3、L4、R3、R4四个关节）
    #     x轴：真实关节位置 - 模拟关节位置
    #     y轴：真实关节速度 - 模拟关节速度
        
    #     Args:
    #         real_data: 真实数据 (Mujoco数据)
    #         sim_data: 仿真数据 (Isaac Gym数据)
    #         filename: 保存的文件名
    #     """
    #     # 数据结构：[joint_pos(12), joint_vel(12), action(12), base_vel(3), world_vel(3), actual_torques(12)]
    #     num_joints = 12
        
    #     # 提取关节位置和速度
    #     real_joint_pos = real_data[:, :num_joints]           # 0:12
    #     real_joint_vel = real_data[:, num_joints:2*num_joints]  # 12:24
        
    #     sim_joint_pos = sim_data[:, :num_joints]             # 0:12
    #     sim_joint_vel = sim_data[:, num_joints:2*num_joints]    # 12:24
        
    #     # 确保数据长度一致
    #     min_len = min(len(real_data), len(sim_data))
    #     real_joint_pos = real_joint_pos[:min_len]
    #     real_joint_vel = real_joint_vel[:min_len]
    #     sim_joint_pos = sim_joint_pos[:min_len]
    #     sim_joint_vel = sim_joint_vel[:min_len]
        
    #     # 计算差异
    #     pos_diff = real_joint_pos - sim_joint_pos  # x轴：位置差异
    #     vel_diff = real_joint_vel - sim_joint_vel  # y轴：速度差异
        
    #     # 🔥 只选择关键关节：L3、L4、R3、R4
    #     key_joints = [2, 3, 8, 9]  # leg_l3, leg_l4, leg_r3, leg_r4
    #     joint_labels = ['Left Hip Pitch (L3)', 'Left Knee (L4)', 'Right Hip Pitch (R3)', 'Right Knee (R4)']
        
    #     # 创建2x2的子图
    #     fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    #     fig.suptitle('Joint Position vs Velocity Difference: Real - Simulation\n(X: Position Difference, Y: Velocity Difference)', fontsize=16)
        
    #     # 为不同时间段设置不同颜色
    #     #time_steps = np.arange(min_len)
        
    #     for idx, joint_idx in enumerate(key_joints):
    #         row = idx // 2
    #         col = idx % 2
    #         ax = axes[row, col]
            
    #         # 绘制散点图，颜色表示时间进程
    #         scatter = ax.scatter(pos_diff[:, joint_idx], vel_diff[:, joint_idx], 
    #                         color='blue', cmap='viridis', 
    #                         alpha=0.6, s=2)
            
    #         # 添加原点参考线
    #         ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=1)
    #         ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=1)
            
    #         # 添加原点标记
    #         ax.scatter([0], [0], color='red', s=50, marker='x', linewidth=3, label='Perfect Match')
            
    #         ax.set_xlabel('Position Difference (Real - Sim) [rad]')
    #         ax.set_ylabel('Velocity Difference (Real - Sim) [rad/s]')
    #         ax.set_title(f'{joint_labels[idx]}')
    #         ax.grid(True, alpha=0.3)
    #         ax.legend(fontsize=8)
            
    #         # 计算并显示统计信息
    #         pos_diff_mean = np.mean(pos_diff[:, joint_idx])
    #         pos_diff_std = np.std(pos_diff[:, joint_idx])
    #         vel_diff_mean = np.mean(vel_diff[:, joint_idx])
    #         vel_diff_std = np.std(vel_diff[:, joint_idx])
            
    #         # 计算距离原点的平均距离（综合误差指标）
    #         distance_from_origin = np.sqrt(pos_diff[:, joint_idx]**2 + vel_diff[:, joint_idx]**2)
    #         mean_distance = np.mean(distance_from_origin)
            
    #         stats_text = f'Pos: μ={pos_diff_mean:.4f}, σ={pos_diff_std:.4f}\n'
    #         stats_text += f'Vel: μ={vel_diff_mean:.4f}, σ={vel_diff_std:.4f}\n'
    #         stats_text += f'Dist: {mean_distance:.4f}'
            
    #         ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
    #                 verticalalignment='top', fontsize=8, 
    #                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
            
    #         # 设置对称的坐标轴范围
    #         pos_range = max(abs(np.min(pos_diff[:, joint_idx])), abs(np.max(pos_diff[:, joint_idx]))) * 1.1
    #         vel_range = max(abs(np.min(vel_diff[:, joint_idx])), abs(np.max(vel_diff[:, joint_idx]))) * 1.1
            
    #         if pos_range > 0:
    #             ax.set_xlim(-pos_range, pos_range)
    #         if vel_range > 0:
    #             ax.set_ylim(-vel_range, vel_range)
        
       
        
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(self.save_dir, filename), dpi=300, bbox_inches='tight')
    #     plt.close()
        
    #     print(f"Joint position-velocity difference plot saved: {filename}")

    def plot_joint_position_velocity_difference(self, real_data, sim_data, filename='joint_pos_vel_difference.png'):
        """
        绘制真实数据与仿真数据的关节位置和速度差异图（只显示L3、L4、R3、R4四个关节）
        x轴：真实关节位置 - 模拟关节位置
        y轴：真实关节速度 - 模拟关节速度
        
        Args:
            real_data: 真实数据 (Mujoco数据)
            sim_data: 仿真数据 (Isaac Gym数据)
            filename: 保存的文件名
        """
        # 数据结构：[joint_pos(12), joint_vel(12), action(12), base_vel(3), world_vel(3), actual_torques(12)]
        num_joints = 12
        
        # 提取关节位置和速度
        real_joint_pos = real_data[:, :num_joints]           # 0:12
        real_joint_vel = real_data[:, num_joints:2*num_joints]  # 12:24
        
        sim_joint_pos = sim_data[:, :num_joints]             # 0:12
        sim_joint_vel = sim_data[:, num_joints:2*num_joints]    # 12:24
        
        # 🔥 处理不同长度的数据 - 使用时间归一化对齐
        real_len = len(real_data)
        sim_len = len(sim_data)
        
        # 选择更短的长度作为目标长度，或者使用固定长度
        target_len = min(real_len, sim_len, 1000)  # 最多使用500个点进行比较
        
        # 时间归一化重采样
        if real_len != target_len:
            real_indices = np.linspace(0, real_len-1, target_len).astype(int)
            real_joint_pos = real_joint_pos[real_indices]
            real_joint_vel = real_joint_vel[real_indices]
        else:
            real_joint_pos = real_joint_pos[:target_len]
            real_joint_vel = real_joint_vel[:target_len]
        
        if sim_len != target_len:
            sim_indices = np.linspace(0, sim_len-1, target_len).astype(int)
            sim_joint_pos = sim_joint_pos[sim_indices]
            sim_joint_vel = sim_joint_vel[sim_indices]
        else:
            sim_joint_pos = sim_joint_pos[:target_len]
            sim_joint_vel = sim_joint_vel[:target_len]
        
        # 计算差异
        pos_diff = real_joint_pos - sim_joint_pos  # x轴：位置差异
        vel_diff = real_joint_vel - sim_joint_vel  # y轴：速度差异
        
        # 🔥 只选择关键关节：L3、L4、R3、R4
        key_joints = [2, 3, 8, 9]  # leg_l3, leg_l4, leg_r3, leg_r4
        joint_labels = ['Left Hip Pitch (L3)', 'Left Knee (L4)', 'Right Hip Pitch (R3)', 'Right Knee (R4)']
        
        # 🔥 定义固定的坐标轴范围
        axis_ranges = {
            2: {'x': 0.5, 'y': 15},    # L3: x=±0.3, y=±6
            3: {'x': 0.5, 'y': 15},   # L4: x=±0.4, y=±15
            8: {'x': 0.5, 'y': 15},    # R3: x=±0.3, y=±6
            9: {'x': 0.5, 'y': 15}    # R4: x=±0.4, y=±15
        }
        
        # 创建2x2的子图
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Joint Position vs Velocity Difference: Real - Simulation\n(X: Position Difference, Y: Velocity Difference)\nData Length: Real={real_len}, Sim={sim_len}, Compared={target_len}', fontsize=14)
        
        # 🔥 计算综合误差度量指标
        joint_error_metrics = {}
        
        for idx, joint_idx in enumerate(key_joints):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]

            # 🔥 新增：绘制连接线显示时间轨迹
            pos_diff_data = pos_diff[:, joint_idx]
            vel_diff_data = vel_diff[:, joint_idx]
            
            # 绘制轨迹线（相邻时间点连接）
            ax.plot(pos_diff_data, vel_diff_data, 
                    color='blue', alpha=0.7, linewidth=1, 
                    label='Trajectory', zorder=1)
            
            # 🔥 新增：添加时间颜色渐变的散点图
            # 创建颜色映射，从开始（绿色）到结束（红色）
            time_colors = plt.cm.viridis(np.linspace(0, 1, len(pos_diff_data)))
            
            # 绘制散点图，颜色表示时间进程
            scatter = ax.scatter(pos_diff_data, vel_diff_data, 
                            c=time_colors, s=8, alpha=0.8, 
                            label='Time Progress', zorder=2)
            
            # 🔥 新增：标记起点和终点
            ax.scatter(pos_diff_data[0], vel_diff_data[0], 
                    color='green', s=50, marker='o', 
                    label='Start', zorder=3, edgecolor='black', linewidth=1)
            ax.scatter(pos_diff_data[-1], vel_diff_data[-1], 
                    color='red', s=50, marker='s', 
                    label='End', zorder=3, edgecolor='black', linewidth=1)
            
            # 添加原点参考线
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7, linewidth=1)
            ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7, linewidth=1)
            
            
            # 绘制散点图
            ax.scatter(pos_diff[:, joint_idx], vel_diff[:, joint_idx], 
                    color='blue', alpha=0.6, s=2)
            
            # 添加原点参考线
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=1)
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=1)
            
            # 添加原点标记
            ax.scatter([0], [0], color='red', s=50, marker='x', linewidth=3, label='Perfect Match')
            
            # 🔥 改进的统计指标计算
            pos_diff_data = pos_diff[:, joint_idx]
            vel_diff_data = vel_diff[:, joint_idx]
            
            # 基本统计
            pos_mae = np.mean(np.abs(pos_diff_data))
            vel_mae = np.mean(np.abs(vel_diff_data))
            pos_rmse = np.sqrt(np.mean(pos_diff_data**2))
            vel_rmse = np.sqrt(np.mean(vel_diff_data**2))
            
            # 🔥 综合误差指标
            # 1. 欧氏距离的均值（综合位置-速度误差）
            euclidean_distances = np.sqrt(pos_diff_data**2 + vel_diff_data**2)
            mean_euclidean_distance = np.mean(euclidean_distances)
            max_euclidean_distance = np.max(euclidean_distances)
            
            # 2. 加权综合误差（位置和速度的加权RMSE）
            # 可以根据实际需求调整权重
            # pos_weight = 1.0  # 位置权重
            # vel_weight = 0.5  # 速度权重（通常速度的量级更大）
            # weighted_error = np.sqrt(pos_weight * pos_rmse**2 + vel_weight * vel_rmse**2)
            
            # 3. 归一化误差（相对于数据范围）
            pos_range = np.max(real_joint_pos[:, joint_idx]) - np.min(real_joint_pos[:, joint_idx])
            vel_range = np.max(real_joint_vel[:, joint_idx]) - np.min(real_joint_vel[:, joint_idx])
            
            normalized_pos_rmse = pos_rmse / max(pos_range, 1e-6) * 100  # 百分比
            normalized_vel_rmse = vel_rmse / max(vel_range, 1e-6) * 100  # 百分比
            
            # 4. 95%分位数误差（排除异常值）
            pos_95th = np.percentile(np.abs(pos_diff_data), 95)
            vel_95th = np.percentile(np.abs(vel_diff_data), 95)
            euclidean_95th = np.percentile(euclidean_distances, 95)
            
            # 5. 一致性指标（数据的标准差）
            pos_consistency = np.std(pos_diff_data)
            vel_consistency = np.std(vel_diff_data)
            
            # 保存指标
            joint_error_metrics[joint_labels[idx]] = {
                'pos_mae': pos_mae,
                'vel_mae': vel_mae,
                'pos_rmse': pos_rmse,
                'vel_rmse': vel_rmse,
                'mean_euclidean': mean_euclidean_distance,
                'max_euclidean': max_euclidean_distance,
                #'weighted_error': weighted_error,
                'normalized_pos_rmse': normalized_pos_rmse,
                'normalized_vel_rmse': normalized_vel_rmse,
                'pos_95th': pos_95th,
                'vel_95th': vel_95th,
                'euclidean_95th': euclidean_95th,
                'pos_consistency': pos_consistency,
                'vel_consistency': vel_consistency
            }
            
            ax.set_xlabel('Position Difference (Real - Sim) [rad]')
            ax.set_ylabel('Velocity Difference (Real - Sim) [rad/s]')
            ax.set_title(f'{joint_labels[idx]}')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            
            # 🔥 显示改进的统计信息
            stats_text = f'MAE: P={pos_mae:.4f}, V={vel_mae:.4f}\n'
            stats_text += f'RMSE: P={pos_rmse:.4f}, V={vel_rmse:.4f}\n'
            stats_text += f'Euclidean: μ={mean_euclidean_distance:.4f}\n'
            # stats_text += f'Weighted: {weighted_error:.4f}\n'
            stats_text += f'95th%: P={pos_95th:.4f}, V={vel_95th:.4f}, Euc={euclidean_95th:.4f}\n'
            stats_text += f'Norm%: P={normalized_pos_rmse:.1f}, V={normalized_vel_rmse:.1f}'
            stats_text += f'\nStd Dev: P={pos_consistency:.4f}, V={vel_consistency:.4f}'
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                    verticalalignment='top', fontsize=10, 
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
            
            # 🔥 添加误差椭圆（表示数据分布范围）
            from matplotlib.patches import Ellipse
            
            # 计算误差椭圆参数
            pos_std = np.std(pos_diff_data)
            vel_std = np.std(vel_diff_data)
            pos_mean = np.mean(pos_diff_data)
            vel_mean = np.mean(vel_diff_data)
            
            # 绘制1倍和2倍标准差椭圆
            ellipse_1sigma = Ellipse((pos_mean, vel_mean), 
                                    width=2*pos_std, height=2*vel_std,
                                    facecolor='yellow', alpha=0.2, 
                                    edgecolor='orange', linewidth=1,
                                    label='1σ Range')
            ellipse_2sigma = Ellipse((pos_mean, vel_mean), 
                                    width=4*pos_std, height=4*vel_std,
                                    facecolor='orange', alpha=0.1, 
                                    edgecolor='red', linewidth=1,
                                    label='2σ Range')
            
            ax.add_patch(ellipse_2sigma)
            ax.add_patch(ellipse_1sigma)
            
            # 🔥 设置固定的坐标轴范围
            x_range = axis_ranges[joint_idx]['x']
            y_range = axis_ranges[joint_idx]['y']
            
            ax.set_xlim(-x_range, x_range)
            ax.set_ylim(-y_range, y_range)
            
            # 🔥 添加坐标轴范围信息到标题
            ax.set_title(f'{joint_labels[idx]}\n(Range: x=±{x_range}, y=±{y_range})')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Joint position-velocity difference plot saved: {filename}")
        print(f"Fixed axis ranges applied:")
        print(f"  L3/R3: x=±0.3, y=±6")
        print(f"  L4/R4: x=±0.4, y=±15")
            

# 便捷函数，用于快速使用
def plot_torque_velocity_curves(mujoco_data, isaac_data, params=None, save_dir=None, filename='torque_velocity_curves.png'):
    """
    便捷函数：快速绘制扭矩-速度曲线
    
    Args:
        mujoco_data: Mujoco数据
        isaac_data: Isaac Gym数据
        params: 参数字典
        save_dir: 保存目录
        filename: 文件名
    """
    plotter = TorqueVelocityPlotter(save_dir=save_dir)
    plotter.plot_torque_velocity_curves(mujoco_data, isaac_data, params=params, filename=filename)
    return plotter.save_dir

def plot_all_real_data_torque_velocity_curves(all_real_data, sim_data=None, params=None, save_dir=None, filename='all_real_data_torque_velocity_curves.png'):
    """
    便捷函数：快速绘制所有真实数据的扭矩-速度曲线
    
    Args:
        all_real_data: 所有真实数据字典
        sim_data: 仿真数据（可选）
        params: 参数字典
        save_dir: 保存目录
        filename: 文件名
    """
    plotter = TorqueVelocityPlotter(save_dir=save_dir)
    return plotter.plot_all_real_data_torque_velocity_curves(all_real_data, sim_data=sim_data, params=params, filename=filename)