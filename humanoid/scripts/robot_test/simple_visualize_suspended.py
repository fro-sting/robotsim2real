#!/usr/bin/env python3
"""
简单可视化悬挂机器人 - 使用现有环境框架，不进行训练
使用方法: python simple_visualize_suspended.py
"""

import isaacgym
from humanoid.envs import *
from humanoid.utils import get_args, task_registry
import torch
import time


def visualize_suspended():
    """创建环境并可视化悬挂的机器人"""
    
    # 获取参数
    args = get_args()
    
    # 强制启用图形界面
    args.headless = False
    
    # 设置较少的环境数量以便观察
    args.num_envs = 4
    
    # 使用kuavo配置
    env_cfg, train_cfg = task_registry.get_cfgs(name="kuavo_ppo")
    
    # 🔥 修改配置以实现悬挂
    env_cfg.asset.fix_base_link = True  # 固定base_link
    env_cfg.init_state.pos = [0.0, 0.0, 1.2]  # 悬挂高度1.2米
    env_cfg.terrain.mesh_type = "plane"  # 使用平面地形
    env_cfg.terrain.curriculum = False
    env_cfg.env.num_envs = args.num_envs
    
    # 禁用一些不需要的功能
    env_cfg.domain_rand.push_robots = False
    if hasattr(env_cfg.domain_rand, 'disturbance'):
        env_cfg.domain_rand.disturbance = False
    
    print("\n" + "="*60)
    print("🤖 正在创建悬挂机器人环境...")
    print(f"   - 悬挂高度: {env_cfg.init_state.pos[2]:.2f} 米")
    print(f"   - 环境数量: {env_cfg.env.num_envs}")
    print(f"   - Base Link: 固定（悬挂状态）")
    print("="*60 + "\n")
    
    # 创建环境
    env, env_cfg = task_registry.make_env(name="kuavo_ppo", args=args, env_cfg=env_cfg)
    
    print("\n" + "="*60)
    print("✅ 环境创建成功！")
    print("\n操作说明：")
    print("   - 鼠标左键拖动：旋转视角")
    print("   - 鼠标滚轮：缩放")
    print("   - 鼠标中键拖动：平移")
    print("   - V 键：切换相机视角")
    print("   - ESC：退出")
    print("\n⏸️  机器人将保持悬挂状态，执行默认动作...")
    print("="*60 + "\n")
    
    # 初始化观测
    obs = env.get_observations()
    
    # 主循环 - 不训练，只可视化
    step_count = 0
    try:
        while True:
            # 使用零动作或默认动作
            actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
            
            # 也可以使用一些简单的周期性动作让腿动起来（可选）
            if step_count % 100 == 0:
                # 每100步随机一些小动作
                actions = torch.randn_like(actions) * 0.1
            
            # 执行动作
            obs, privileged_obs, rewards, dones, infos = env.step(actions)
            
            step_count += 1
            
            # 每1000步打印一次信息
            if step_count % 1000 == 0:
                print(f"⏱️  运行中... (步数: {step_count})")
            
            # 添加小延迟使可视化更平滑
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\n👋 用户中断，退出可视化")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
    finally:
        print("清理环境...")


if __name__ == "__main__":
    visualize_suspended()
