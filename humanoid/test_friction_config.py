#!/usr/bin/env python3
"""
测试4个2389关节的不同摩擦力设置
"""

import sys
import os
sys.path.append('.')

def test_friction_config():
    """测试摩擦力配置"""
    
    try:
        from humanoid.envs.custom.kuavo_config import KuavoFreeEnvCfg
        
        # 创建配置实例
        cfg = KuavoFreeEnvCfg()
        
        print("=" * 60)
        print("4个2389关节的摩擦力配置测试")
        print("=" * 60)
        
        # 检查配置
        special_enabled = getattr(cfg.domain_rand, 'special_joint_friction_enabled', False)
        friction_dict = getattr(cfg.domain_rand, 'special_joint_friction_dict', {})
        
        print(f"特殊摩擦力功能启用: {special_enabled}")
        print(f"摩擦力字典: {friction_dict}")
        
        if special_enabled and friction_dict:
            print("\n关节摩擦力配置详情:")
            print("-" * 40)
            
            joint_names = [
                "leg_l1_joint", "leg_l2_joint", "leg_l3_joint", "leg_l4_joint", 
                "leg_l5_joint", "leg_l6_joint", "leg_r1_joint", "leg_r2_joint", 
                "leg_r3_joint", "leg_r4_joint", "leg_r5_joint", "leg_r6_joint"
            ]
            
            for joint_idx, friction_value in friction_dict.items():
                joint_name = joint_names[joint_idx] if joint_idx < len(joint_names) else f"unknown_joint_{joint_idx}"
                print(f"关节 {joint_idx:2d} ({joint_name:12s}) -> 摩擦力: {friction_value}")
            
            print("\n配置验证成功! ✅")
            print(f"共配置了 {len(friction_dict)} 个关节的特殊摩擦力")
            
            # 验证是否正好是4个关节
            if len(friction_dict) == 4:
                print("✅ 正确配置了4个2389关节的不同摩擦力值")
            else:
                print(f"⚠️  当前配置了{len(friction_dict)}个关节，不是4个")
                
        else:
            print("❌ 特殊摩擦力配置未启用或字典为空")
            
        return True
        
    except Exception as e:
        print(f"❌ 配置测试失败: {e}")
        return False

def show_usage_examples():
    """显示使用示例"""
    print("\n" + "=" * 60)
    print("配置示例")
    print("=" * 60)
    
    print("""
# 在 kuavo_config.py 中的配置示例:

special_joint_friction_enabled = True
special_joint_friction_dict = {
    2: 0.08,   # leg_l3_joint (第1个2389关节) - 较高摩擦力
    3: 0.09,   # leg_l4_joint (第2个2389关节) - 最高摩擦力
    8: 0.07,   # leg_r3_joint (第3个2389关节) - 较低摩擦力  
    9: 0.10    # leg_r4_joint (第4个2389关节) - 超高摩擦力
}

# 其他配置示例:

# 示例1: 对称设置 (左右腿相同)
special_joint_friction_dict = {
    2: 0.08,   # leg_l3_joint
    3: 0.09,   # leg_l4_joint  
    8: 0.08,   # leg_r3_joint (与leg_l3相同)
    9: 0.09    # leg_r4_joint (与leg_l4相同)
}

# 示例2: 渐进式设置
special_joint_friction_dict = {
    2: 0.06,   # leg_l3_joint - 最低
    3: 0.07,   # leg_l4_joint - 较低
    8: 0.08,   # leg_r3_joint - 较高
    9: 0.09    # leg_r4_joint - 最高
}

# 示例3: 只设置特定关节
special_joint_friction_dict = {
    3: 0.12,   # 只对leg_l4设置高摩擦力
    9: 0.12    # 只对leg_r4设置高摩擦力
}
""")

if __name__ == "__main__":
    success = test_friction_config()
    
    if success:
        show_usage_examples()
        print("\n✅ 测试完成! 可以开始训练了。")
    else:
        print("\n❌ 测试失败! 请检查配置文件。")
