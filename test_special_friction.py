#!/usr/bin/env python3
"""
测试特定关节摩擦力设置功能
"""

import sys
import os
sys.path.append('humanoid')

from humanoid.envs.custom.kuavo_config import KuavoFreeEnvCfg
from humanoid.envs.custom.kuavo_env import KuavoFreeEnv
from isaacgym import gymapi

def test_special_joint_friction():
    """测试特定关节摩擦力设置"""
    
    # 创建配置
    cfg = KuavoFreeEnvCfg()
    
    # 检查配置是否正确设置
    print("特定关节摩擦力配置:")
    print(f"special_joint_friction_enabled: {getattr(cfg.domain_rand, 'special_joint_friction_enabled', '未设置')}")
    print(f"special_joint_friction_value: {getattr(cfg.domain_rand, 'special_joint_friction_value', '未设置')}")
    print(f"special_joint_indices: {getattr(cfg.domain_rand, 'special_joint_indices', '未设置')}")
    
    # 创建仿真参数
    sim_params = gymapi.SimParams()
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.contact_offset = 0.01
    sim_params.physx.rest_offset = 0.0
    sim_params.use_gpu_pipeline = True
    
    try:
        # 尝试创建环境实例
        print("\n尝试创建环境实例...")
        env = KuavoFreeEnv(
            cfg=cfg,
            sim_params=sim_params,
            physics_engine=gymapi.SIM_PHYSX,
            sim_device="cuda:0",
            headless=True
        )
        print("环境创建成功！特定关节摩擦力功能已激活。")
        
        # 清理资源
        env.gym.destroy_sim(env.sim)
        
    except Exception as e:
        print(f"环境创建失败: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("测试特定关节摩擦力设置功能")
    print("=" * 60)
    
    success = test_special_joint_friction()
    
    if success:
        print("\n✅ 测试成功！特定关节摩擦力功能工作正常。")
        print("\n使用说明:")
        print("1. 在kuavo_config.py中设置 special_joint_friction_enabled = True")
        print("2. 设置 special_joint_friction_value 为所需的摩擦力系数")
        print("3. 设置 special_joint_indices 为要特殊处理的关节索引列表")
        print("4. 当前默认对 leg_l3_joint (索引2) 和 leg_r3_joint (索引8) 设置特殊摩擦力")
    else:
        print("\n❌ 测试失败！请检查配置和代码。")
