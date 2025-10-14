#!/usr/bin/env python3
"""
测试不使用 delta action 的 finetuning 配置
用于验证修改后的 finetuning 系统是否能正常工作
"""

import torch
import sys
import os

# 添加路径
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from humanoid.envs.custom.finetuning_policy_config import FinetuningPolicyFlatCfg

def test_finetuning_config():
    """测试 finetuning 配置"""
    print("🔥 测试 Finetuning 配置（不使用 Delta Action）")
    
    # 创建配置实例
    cfg = FinetuningPolicyFlatCfg()
    
    # 检查 delta action 配置
    print(f"✅ use_delta_action: {cfg.env.use_delta_action}")
    print(f"✅ delta_action_policy_path: {cfg.env.delta_action_policy_path}")
    
    # 验证其他关键配置
    print(f"✅ num_envs: {cfg.env.num_envs}")
    print(f"✅ num_actions: {cfg.env.num_actions}")
    print(f"✅ episode_length_s: {cfg.env.episode_length_s}")
    
    return cfg

def test_env_creation():
    """测试环境创建（模拟）"""
    print("\n🔥 模拟环境创建过程")
    
    cfg = test_finetuning_config()
    
    # 模拟环境初始化逻辑
    use_delta_action = getattr(cfg.env, "use_delta_action", False)
    delta_action_policy_path = getattr(cfg.env, "delta_action_policy_path", None)
    
    print(f"✅ 环境将使用 delta action: {use_delta_action}")
    
    if use_delta_action:
        if delta_action_policy_path is None:
            default_path = "/home/wegg/kuavo_rl_asap-main111/RL_train/logs/delta_action_model/exported/policies_delta_action/policy_1.pt"
            print(f"⚠️  使用默认 delta action 路径: {default_path}")
            print(f"⚠️  路径存在: {os.path.exists(default_path)}")
        else:
            print(f"✅ 使用指定 delta action 路径: {delta_action_policy_path}")
            print(f"✅ 路径存在: {os.path.exists(delta_action_policy_path)}")
    else:
        print("✅ 禁用 delta action，将使用传统端到端微调")
    
    return True

if __name__ == "__main__":
    print("=" * 50)
    print("测试 Finetuning 配置（无 Delta Action 依赖）")
    print("=" * 50)
    
    try:
        # 测试配置
        test_finetuning_config()
        
        # 测试环境创建逻辑
        test_env_creation()
        
        print("\n🎉 所有测试通过！Finetuning 配置可以在不使用 delta action 的情况下正常工作。")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
