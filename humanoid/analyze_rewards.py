#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import os

def analyze_wandb_rewards(wandb_dir):
    """分析 wandb 运行中的奖励函数"""
    
    # 读取 summary 文件
    summary_file = os.path.join(wandb_dir, "files", "wandb-summary.json")
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            summary = json.load(f)
    else:
        print(f"未找到 summary 文件: {summary_file}")
        return
    
    # 提取奖励相关的数据
    reward_data = {}
    episode_data = {}
    
    for key, value in summary.items():
        if key.startswith("Episode/rew_"):
            reward_name = key.replace("Episode/rew_", "")
            reward_data[reward_name] = value
        elif key.startswith("Episode/"):
            episode_data[key] = value
        elif key.startswith("Train/"):
            episode_data[key] = value
    
    # 打印奖励分析
    print("=" * 60)
    print(f"奖励函数分析 - 运行: {os.path.basename(wandb_dir)}")
    print("=" * 60)
    
    print(f"\n📊 总体训练指标:")
    print(f"  • 平均总奖励: {episode_data.get('Train/mean_reward', 'N/A'):.2f}")
    print(f"  • 平均episode长度: {episode_data.get('Train/mean_episode_length', 'N/A'):.0f}")
    print(f"  • 当前迭代: {summary.get('iteration', 'N/A')}")
    print(f"  • 全局步数: {summary.get('global_step', 'N/A')}")
    
    print(f"\n🎯 奖励组件分析:")
    
    # 按类别分组奖励
    tracking_rewards = {}
    imitation_rewards = {}
    regularization_rewards = {}
    other_rewards = {}
    
    for name, value in reward_data.items():
        if name.startswith('tracking'):
            tracking_rewards[name] = value
        elif name.startswith('imition'):
            imitation_rewards[name] = value
        elif name in ['dof_vel', 'dof_acc', 'action_smoothness', 'torques', 'energy']:
            regularization_rewards[name] = value
        else:
            other_rewards[name] = value
    
    # 打印各类奖励
    if tracking_rewards:
        print(f"\n  🎯 跟踪奖励 (Tracking Rewards):")
        for name, value in tracking_rewards.items():
            print(f"    • {name}: {value:.4f}")
    
    if imitation_rewards:
        print(f"\n  🤖 模仿奖励 (Imitation Rewards):")
        for name, value in imitation_rewards.items():
            clean_name = name.replace('imition_', '').replace('_', ' ')
            print(f"    • {clean_name}: {value:.4f}")
    
    if regularization_rewards:
        print(f"\n  ⚡ 正则化奖励 (Regularization Rewards):")
        for name, value in regularization_rewards.items():
            print(f"    • {name}: {value:.4f}")
    
    if other_rewards:
        print(f"\n  🔧 其他奖励 (Other Rewards):")
        for name, value in other_rewards.items():
            print(f"    • {name}: {value:.4f}")
    
    # 计算奖励贡献度
    positive_rewards = {k: v for k, v in reward_data.items() if v > 0}
    negative_rewards = {k: v for k, v in reward_data.items() if v < 0}
    
    total_positive = sum(positive_rewards.values())
    total_negative = sum(negative_rewards.values())
    
    print(f"\n📈 奖励贡献分析:")
    print(f"  • 正奖励总和: {total_positive:.4f}")
    print(f"  • 负奖励总和: {total_negative:.4f}")
    print(f"  • 净奖励: {total_positive + total_negative:.4f}")
    
    print(f"\n🏆 主要正奖励贡献:")
    sorted_positive = sorted(positive_rewards.items(), key=lambda x: x[1], reverse=True)
    for name, value in sorted_positive[:5]:
        percentage = (value / total_positive * 100) if total_positive > 0 else 0
        print(f"    • {name}: {value:.4f} ({percentage:.1f}%)")
    
    print(f"\n⚠️  主要负奖励:")
    sorted_negative = sorted(negative_rewards.items(), key=lambda x: x[1])
    for name, value in sorted_negative[:5]:
        percentage = (abs(value) / abs(total_negative) * 100) if total_negative < 0 else 0
        print(f"    • {name}: {value:.4f} ({percentage:.1f}%)")
    
    return reward_data, episode_data

def plot_recent_training_log(wandb_dir, lines=1000):
    """分析最近的训练日志"""
    output_file = os.path.join(wandb_dir, "files", "output.log")
    
    if not os.path.exists(output_file):
        print(f"未找到输出日志: {output_file}")
        return
    
    print(f"\n📝 最近训练日志分析 (最后 {lines} 行):")
    print("-" * 50)
    
    # 读取最后几行
    import subprocess
    try:
        result = subprocess.run(['tail', f'-{lines}', output_file], 
                              capture_output=True, text=True)
        log_content = result.stdout
        
        # 提取平均奖励
        reward_pattern = r'Mean reward:\s+([\d.]+)'
        rewards = re.findall(reward_pattern, log_content)
        
        if rewards:
            recent_rewards = [float(r) for r in rewards[-10:]]  # 最近10个记录
            print(f"  • 最近平均奖励趋势: {recent_rewards}")
            if len(recent_rewards) > 1:
                trend = "📈 上升" if recent_rewards[-1] > recent_rewards[0] else "📉 下降"
                print(f"  • 趋势: {trend}")
                print(f"  • 最新奖励: {recent_rewards[-1]:.2f}")
                print(f"  • 奖励变化: {recent_rewards[-1] - recent_rewards[0]:+.2f}")
        
        # 查找其他关键信息
        episode_pattern = r'Episode\s+(\d+)'
        episodes = re.findall(episode_pattern, log_content)
        if episodes:
            print(f"  • 当前Episode: {episodes[-1]}")
            
    except Exception as e:
        print(f"读取日志失败: {e}")

def find_max_reward_in_training(wandb_dir):
    """从训练日志中找到最大奖励值"""
    output_file = os.path.join(wandb_dir, "files", "output.log")
    
    if not os.path.exists(output_file):
        print(f"未找到输出日志: {output_file}")
        return None
    
    print(f"🔍 分析训练日志中的奖励变化...")
    print(f"日志文件: {output_file}")
    print("-" * 60)
    
    # 读取所有奖励记录
    rewards = []
    iterations = []
    
    try:
        with open(output_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                # 查找 Mean reward 记录
                if 'Mean reward:' in line:
                    try:
                        # 提取奖励值
                        reward_match = re.search(r'Mean reward:\s+([\d.-]+)', line)
                        if reward_match:
                            reward = float(reward_match.group(1))
                            rewards.append(reward)
                            
                            # 尝试提取迭代次数
                            iter_match = re.search(r'Iteration\s+(\d+)', line)
                            if iter_match:
                                iteration = int(iter_match.group(1))
                                iterations.append(iteration)
                            else:
                                iterations.append(len(rewards))  # 使用记录数作为迭代
                                
                    except ValueError:
                        continue
    
    except Exception as e:
        print(f"读取日志失败: {e}")
        return None
    
    if not rewards:
        print("未找到任何奖励记录")
        return None
    
    # 分析奖励数据
    max_reward = max(rewards)
    min_reward = min(rewards)
    avg_reward = sum(rewards) / len(rewards)
    
    # 找到最大奖励的位置
    max_reward_idx = rewards.index(max_reward)
    max_reward_iteration = iterations[max_reward_idx] if max_reward_idx < len(iterations) else max_reward_idx
    
    # 找到最小奖励的位置
    min_reward_idx = rewards.index(min_reward)
    min_reward_iteration = iterations[min_reward_idx] if min_reward_idx < len(iterations) else min_reward_idx
    
    print(f"📊 训练奖励统计:")
    print(f"  • 总记录数: {len(rewards)}")
    print(f"  • 🏆 最大奖励: {max_reward:.4f} (第 {max_reward_iteration} 次迭代)")
    print(f"  • 📉 最小奖励: {min_reward:.4f} (第 {min_reward_iteration} 次迭代)")
    print(f"  • 📈 平均奖励: {avg_reward:.4f}")
    print(f"  • 📏 奖励范围: {max_reward - min_reward:.4f}")
    
    # 最近的奖励趋势
    recent_rewards = rewards[-10:] if len(rewards) >= 10 else rewards
    print(f"\n📝 最近 {len(recent_rewards)} 次奖励:")
    for i, reward in enumerate(recent_rewards):
        idx = len(rewards) - len(recent_rewards) + i + 1
        print(f"  {idx:3d}. {reward:.2f}")
    
    print(f"\n🎯 当前奖励: {rewards[-1]:.4f}")
    print(f"🔥 距离最佳奖励差距: {max_reward - rewards[-1]:.4f}")
    
    # 计算奖励改进
    if len(rewards) > 1:
        initial_reward = rewards[0]
        final_reward = rewards[-1]
        improvement = final_reward - initial_reward
        print(f"📈 整体改进: {improvement:+.4f} (从 {initial_reward:.2f} 到 {final_reward:.2f})")
    
    return {
        'max_reward': max_reward,
        'max_reward_iteration': max_reward_iteration,
        'min_reward': min_reward,
        'avg_reward': avg_reward,
        'current_reward': rewards[-1],
        'total_records': len(rewards),
        'rewards': rewards,
        'iterations': iterations
    }

if __name__ == "__main__":
    # 分析指定运行的最大奖励
    target_run_dir = "/home/wegg/kuavo_rl_asap-main/RL_train/humanoid/wandb/run-20251004_164354-1rdeicoi"
    
    print("🎯 分析训练运行: run-20251004_164354-1rdeicoi")
    print("=" * 80)
    
    if os.path.exists(target_run_dir):
        reward_stats = find_max_reward_in_training(target_run_dir)
        
        if reward_stats:
            print(f"\n🏆 最大奖励详情:")
            print(f"  • 最高奖励值: {reward_stats['max_reward']:.4f}")
            print(f"  • 出现在第: {reward_stats['max_reward_iteration']} 次记录")
            print(f"  • 当前奖励: {reward_stats['current_reward']:.4f}")
            print(f"  • 总共记录: {reward_stats['total_records']} 次")
    else:
        print(f"未找到目标运行目录: {target_run_dir}")
        
    print("\n" + "=" * 80)
    print("分析完成!")
