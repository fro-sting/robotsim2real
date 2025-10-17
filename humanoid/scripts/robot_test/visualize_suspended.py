#!/usr/bin/env python3
"""
可视化悬挂机器人 - 不进行训练，只显示机器人悬挂在空中的状态
使用方法: python visualize_suspended.py
按 'V' 键切换相机视角
按 'ESC' 退出
"""

import isaacgym
from isaacgym import gymapi
from isaacgym import gymutil
import torch
import numpy as np


def create_suspended_robot():
    """创建并可视化悬挂的机器人"""
    
    # 初始化gym
    gym = gymapi.acquire_gym()
    
    # 解析参数
    args = gymutil.parse_arguments(
        description="Kuavo Suspended Robot Visualization",
        custom_parameters=[
            {"name": "--num_envs", "type": int, "default": 1, "help": "Number of environments"},
            {"name": "--suspend_height", "type": float, "default": 1.2, "help": "Suspension height in meters"},
        ])
    
    # 配置sim参数
    sim_params = gymapi.SimParams()
    sim_params.dt = 0.01
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    
    # PhysX参数
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.contact_offset = 0.01
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.use_gpu = args.use_gpu
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline
    
    # 创建sim
    device = args.sim_device if args.use_gpu_pipeline else 'cpu'
    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, 
                         gymapi.SIM_PHYSX, sim_params)
    
    if sim is None:
        print("❌ 创建模拟失败")
        quit()
    
    # 创建地面
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane_params)
    
    # 加载机器人资产
    asset_root = "/home/wegg/kuavo_rl_asap-main/RL_train/resources/robots/biped_s44/urdf"
    asset_file = "biped_s44.urdf"
    
    asset_options = gymapi.AssetOptions()
    asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
    asset_options.collapse_fixed_joints = False
    asset_options.replace_cylinder_with_capsule = False
    asset_options.flip_visual_attachments = False
    asset_options.fix_base_link = True  # 🔥 固定base_link，实现悬挂
    asset_options.density = 0.001
    asset_options.angular_damping = 0.0
    asset_options.linear_damping = 0.0
    asset_options.max_angular_velocity = 1000.0
    asset_options.max_linear_velocity = 1000.0
    asset_options.armature = 0.0
    asset_options.thickness = 0.01
    asset_options.disable_gravity = False
    
    print("🤖 正在加载机器人模型...")
    robot_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
    
    # 配置环境
    num_envs = args.num_envs
    spacing = 2.0
    env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)
    
    print(f"🌍 创建 {num_envs} 个环境...")
    
    envs = []
    actor_handles = []
    
    # 初始姿态（悬挂高度）
    suspend_height = args.suspend_height
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, suspend_height)
    pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    
    # 默认关节角度（站立姿态）
    default_dof_pos = np.array([
        0.0,    # leg_l1_joint
        0.0,    # leg_l2_joint
        -0.47,  # leg_l3_joint
        0.86,   # leg_l4_joint
        -0.44,  # leg_l5_joint
        0.0,    # leg_l6_joint
        0.0,    # leg_r1_joint
        0.0,    # leg_r2_joint
        -0.47,  # leg_r3_joint
        0.86,   # leg_r4_joint
        -0.44,  # leg_r5_joint
        0.0,    # leg_r6_joint
    ])
    
    # 创建环境
    for i in range(num_envs):
        # 创建环境
        env = gym.create_env(sim, env_lower, env_upper, int(np.sqrt(num_envs)))
        envs.append(env)
        
        # 创建actor
        actor_handle = gym.create_actor(env, robot_asset, pose, f"robot_{i}", i, 0)
        actor_handles.append(actor_handle)
        
        # 设置关节属性
        dof_props = gym.get_actor_dof_properties(env, actor_handle)
        dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
        dof_props["stiffness"].fill(100.0)
        dof_props["damping"].fill(5.0)
        gym.set_actor_dof_properties(env, actor_handle, dof_props)
        
        # 设置初始关节位置
        gym.set_actor_dof_states(env, actor_handle, 
                                np.zeros_like(default_dof_pos, dtype=np.float32), 
                                gymapi.STATE_ALL)
        gym.set_actor_dof_position_targets(env, actor_handle, default_dof_pos)
    
    # 创建查看器
    cam_props = gymapi.CameraProperties()
    viewer = gym.create_viewer(sim, cam_props)
    
    if viewer is None:
        print("❌ 创建查看器失败")
        quit()
    
    # 设置相机位置
    cam_pos = gymapi.Vec3(3.0, 3.0, 2.0)
    cam_target = gymapi.Vec3(0.0, 0.0, suspend_height)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)
    
    print("\n" + "="*60)
    print("🎉 悬挂机器人可视化已启动！")
    print(f"   - 悬挂高度: {suspend_height:.2f} 米")
    print(f"   - 环境数量: {num_envs}")
    print(f"   - Base Link: 固定（悬挂状态）")
    print("\n操作说明：")
    print("   - 鼠标左键拖动：旋转视角")
    print("   - 鼠标滚轮：缩放")
    print("   - 鼠标中键拖动：平移")
    print("   - V 键：切换相机视角")
    print("   - ESC：退出")
    print("="*60 + "\n")
    
    # 主循环
    frame_count = 0
    while not gym.query_viewer_has_closed(viewer):
        # 处理事件
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        
        # 更新查看器
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)
        
        frame_count += 1
        
        # 每隔一段时间可以让机器人做一些简单的腿部动作（可选）
        if frame_count % 500 == 0:
            print(f"⏱️  运行中... (Frame: {frame_count})")
    
    print("\n👋 退出可视化")
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == "__main__":
    create_suspended_robot()
