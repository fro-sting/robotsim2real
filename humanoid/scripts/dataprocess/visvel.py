import numpy as np
import matplotlib.pyplot as plt
import rosbag
import sys
sys.path.insert(0, '/home/wegg/anaconda3/envs/rosbag_env/lib/python3.8/site-packages')


def visualize_linear_vel_from_npz(npz_file, start_idx=None, end_idx=None, start_time=None, end_time=None):
    """
    从npz文件中可视化linear_vel数据 - 分别显示xyz三个方向的速度
    横坐标使用数据点索引而非时间戳
    
    Args:
        npz_file: npz文件路径
        start_idx: 起始数据点索引（与start_time互斥）
        end_idx: 结束数据点索引（与end_time互斥）
        start_time: 起始时间（秒，与start_idx互斥）
        end_time: 结束时间（秒，与end_idx互斥）
    
    Example:
        # 显示全部数据
        visualize_linear_vel_from_npz("data.npz")
        
        # 显示数据点 100-500
        visualize_linear_vel_from_npz("data.npz", start_idx=100, end_idx=500)
        
        # 显示时间 10-30秒的数据
        visualize_linear_vel_from_npz("data.npz", start_time=10, end_time=30)
    """
    print(f"正在加载 .npz 文件: {npz_file}")
    
    try:
        data = np.load(npz_file)
        linear_vel = data['linear_vel']
        timestamps = data['timestamps_linear_vel']
        
        print(f"原始数据形状: {linear_vel.shape}")
        print(f"原始数据长度: {len(linear_vel)}")
        
        # 检查是否有足够的数据
        if len(linear_vel) == 0:
            print("没有线速度数据！")
            return
        
        # 🔥 新增：数据范围选择逻辑
        original_length = len(linear_vel)
        
        if start_time is not None or end_time is not None:
            # 基于时间范围选择
            if len(timestamps) == 0:
                print("❌ 错误：没有时间戳数据，无法使用时间范围选择")
                return
            
            print(f"=== 基于时间范围选择数据 ===")
            print(f"时间戳范围: {timestamps[0]:.3f} - {timestamps[-1]:.3f} 秒")
            
            # 确定时间范围
            if start_time is None:
                start_time = timestamps[0]
            if end_time is None:
                end_time = timestamps[-1]
            
            print(f"选择时间范围: {start_time:.3f} - {end_time:.3f} 秒")
            
            # 找到对应的索引
            start_idx = np.argmin(np.abs(timestamps - start_time))
            end_idx = np.argmin(np.abs(timestamps - end_time))
            
            # 确保索引顺序正确
            if start_idx > end_idx:
                start_idx, end_idx = end_idx, start_idx
            
            print(f"对应的数据点索引: {start_idx} - {end_idx}")
            
        elif start_idx is not None or end_idx is not None:
            # 基于索引范围选择
            print(f"=== 基于索引范围选择数据 ===")
            
            # 确定索引范围
            if start_idx is None:
                start_idx = 0
            if end_idx is None:
                end_idx = len(linear_vel) - 1
            
            # 检查索引有效性
            start_idx = max(0, min(start_idx, len(linear_vel) - 1))
            end_idx = max(start_idx, min(end_idx, len(linear_vel) - 1))
            
            print(f"选择索引范围: {start_idx} - {end_idx}")
            
            if len(timestamps) > 0:
                print(f"对应时间范围: {timestamps[start_idx]:.3f} - {timestamps[end_idx]:.3f} 秒")
        
        else:
            # 使用全部数据
            start_idx = 0
            end_idx = len(linear_vel) - 1
            print(f"=== 显示全部数据 ===")
        
        # 🔥 应用数据范围选择
        linear_vel_selected = linear_vel[start_idx:end_idx+1]
        timestamps_selected = timestamps[start_idx:end_idx+1] if len(timestamps) > 0 else []
        
        # 创建对应的数据点索引（相对于选择的范围）
        data_indices = np.arange(len(linear_vel_selected))
        
        print(f"选择后的数据长度: {len(linear_vel_selected)} (原始: {original_length})")
        print(f"数据选择比例: {len(linear_vel_selected)/original_length*100:.1f}%")
        
        # 速度方向标签
        vel_labels = ['X速度 (前后)', 'Y速度 (左右)', 'Z速度 (上下)']
        colors = ['red', 'green', 'blue']
        
        # 创建可视化图表 - 3个子图分别显示xyz速度
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # 🔥 修改：标题包含范围信息
        if start_idx != 0 or end_idx != original_length - 1:
            range_info = f"[索引 {start_idx}-{end_idx}]"
            if len(timestamps_selected) > 0:
                range_info += f" [时间 {timestamps_selected[0]:.2f}-{timestamps_selected[-1]:.2f}s]"
            fig.suptitle(f'线速度数据可视化 {range_info}', fontsize=14)
        else:
            fig.suptitle('线速度数据可视化 [全部数据]', fontsize=14)
        
        # 为每个速度分量创建单独的子图
        for i in range(min(3, linear_vel_selected.shape[1])):
            # 使用data_indices作为横坐标
            axes[i].plot(data_indices, linear_vel_selected[:, i], 
                        linewidth=1.5, color=colors[i], alpha=0.8)
            axes[i].set_title(f'{vel_labels[i]} 随数据点变化')
            axes[i].set_xlabel('数据点索引 (相对于选择范围)')
            axes[i].set_ylabel('速度 (m/s)')
            axes[i].grid(True, alpha=0.3)
            
            # 显示当前速度分量的统计信息
            vel_data = linear_vel_selected[:, i]
            
            # 🔥 更新：添加范围选择信息
            stats_text = f'选择范围: 索引 [{start_idx}, {end_idx}]\n'
            stats_text += f'数据点数: {len(linear_vel_selected)} / {original_length}\n'
            stats_text += f'速度范围: [{np.min(vel_data):.3f}, {np.max(vel_data):.3f}] m/s\n'
            stats_text += f'均值: {np.mean(vel_data):.3f} m/s\n'
            stats_text += f'绝对值均值: {np.mean(np.abs(vel_data)):.3f} m/s\n'
            stats_text += f'标准差: {np.std(vel_data):.3f} m/s'
            
            # 如果有时间戳，显示时间信息
            if len(timestamps_selected) > 0:
                total_time = timestamps_selected[-1] - timestamps_selected[0]
                sampling_rate = len(linear_vel_selected) / total_time if total_time > 0 else 0
                stats_text += f'\n时间跨度: {total_time:.2f}秒\n采样率: {sampling_rate:.1f} Hz'
            
            axes[i].text(0.02, 0.98, stats_text,
                        transform=axes[i].transAxes,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                        fontsize=9)
            
            # 🔥 新增：添加全局索引信息（显示在图上的实际索引）
            if start_idx != 0 or end_idx != original_length - 1:
                # 在x轴上方添加全局索引信息
                ax2 = axes[i].twiny()
                ax2.set_xlim(axes[i].get_xlim())
                
                # 设置全局索引刻度
                if len(linear_vel_selected) > 100:
                    global_tick_interval = len(linear_vel_selected) // 5
                    local_ticks = np.arange(0, len(linear_vel_selected), global_tick_interval)
                    global_ticks = local_ticks + start_idx
                    ax2.set_xticks(local_ticks)
                    ax2.set_xticklabels([f'{int(t)}' for t in global_ticks])
                    ax2.set_xlabel('全局索引', fontsize=10, color='gray')
                    ax2.tick_params(axis='x', colors='gray', labelsize=8)
        
        # 设置横坐标显示格式
        for ax in axes[:3]:  # 只处理前3个主轴
            # 设置横坐标刻度间隔
            if len(linear_vel_selected) > 1000:
                tick_interval = len(linear_vel_selected) // 10
                ax.set_xticks(np.arange(0, len(linear_vel_selected), tick_interval))
            
            # 设置横坐标范围
            ax.set_xlim(0, len(linear_vel_selected)-1)
        
        plt.tight_layout()
        plt.show()
        
        # 🔥 更新：打印详细信息
        print(f"\n=== 数据概况 ===")
        print(f"原始数据长度: {original_length}")
        print(f"选择数据长度: {len(linear_vel_selected)}")
        print(f"选择比例: {len(linear_vel_selected)/original_length*100:.1f}%")
        
        if len(timestamps_selected) > 0:
            print(f"选择时间跨度: {timestamps_selected[-1] - timestamps_selected[0]:.2f} 秒")
            print(f"平均采样率: {len(linear_vel_selected) / (timestamps_selected[-1] - timestamps_selected[0]):.1f} Hz")
        
        print(f"\n=== 各方向速度统计 (选择范围) ===")
        for i in range(min(3, linear_vel_selected.shape[1])):
            vel_data = linear_vel_selected[:, i]
            print(f"{vel_labels[i]}:")
            print(f"  范围: [{np.min(vel_data):.4f}, {np.max(vel_data):.4f}] m/s")
            print(f"  均值: {np.mean(vel_data):.4f} m/s")
            print(f"  标准差: {np.std(vel_data):.4f} m/s")
        
    except Exception as e:
        print(f"读取NPZ文件时出错: {e}")


def visualize_interactive_range_selection(npz_file):
    """
    交互式范围选择可视化
    """
    print(f"=== 交互式范围选择 ===")
    
    try:
        data = np.load(npz_file)
        linear_vel = data['linear_vel']
        timestamps = data['timestamps_linear_vel']
        
        print(f"数据总长度: {len(linear_vel)}")
        if len(timestamps) > 0:
            print(f"时间范围: {timestamps[0]:.3f} - {timestamps[-1]:.3f} 秒")
            print(f"总时长: {timestamps[-1] - timestamps[0]:.2f} 秒")
        
        while True:
            print(f"\n=== 选择可视化范围 ===")
            print("1. 基于数据点索引")
            print("2. 基于时间范围")
            print("3. 显示全部数据")
            print("4. 退出")
            
            choice = input("请选择 (1-4): ").strip()
            
            if choice == '1':
                print(f"数据点索引范围: [0, {len(linear_vel)-1}]")
                try:
                    start_str = input(f"起始索引 (默认 0): ").strip()
                    start_idx = int(start_str) if start_str else 0
                    
                    end_str = input(f"结束索引 (默认 {len(linear_vel)-1}): ").strip()
                    end_idx = int(end_str) if end_str else len(linear_vel)-1
                    
                    visualize_linear_vel_from_npz(npz_file, start_idx=start_idx, end_idx=end_idx)
                except ValueError:
                    print("❌ 输入的索引不是有效数字")
                    
            elif choice == '2':
                if len(timestamps) == 0:
                    print("❌ 没有时间戳数据，无法使用时间范围")
                    continue
                    
                print(f"时间范围: [{timestamps[0]:.3f}, {timestamps[-1]:.3f}] 秒")
                try:
                    start_str = input(f"起始时间 (秒, 默认 {timestamps[0]:.2f}): ").strip()
                    start_time = float(start_str) if start_str else timestamps[0]
                    
                    end_str = input(f"结束时间 (秒, 默认 {timestamps[-1]:.2f}): ").strip()
                    end_time = float(end_str) if end_str else timestamps[-1]
                    
                    visualize_linear_vel_from_npz(npz_file, start_time=start_time, end_time=end_time)
                except ValueError:
                    print("❌ 输入的时间不是有效数字")
                    
            elif choice == '3':
                visualize_linear_vel_from_npz(npz_file)
                
            elif choice == '4':
                print("退出")
                break
                
            else:
                print("❌ 无效选择，请输入 1-4")
    
    except Exception as e:
        print(f"加载数据时出错: {e}")


if __name__ == "__main__":
    npz_file = "data/real_run_data/919191.npz"
    
    # 🔥 使用示例
    print("=== 线速度数据可视化工具 ===")
    print("使用方式:")
    print("1. 直接运行 - 交互式选择范围")
    print("2. 修改代码中的参数 - 直接指定范围")
    
    # 选择使用方式
    mode = input("选择模式 (1=交互式, 2=直接运行全部数据, 3=示例范围): ").strip()
    
    if mode == '1':
        # 交互式模式
        visualize_interactive_range_selection(npz_file)
    elif mode == '3':
        # 示例：显示数据点 1000-3000
        print("示例：显示数据点 1000-3000")
        visualize_linear_vel_from_npz(npz_file, start_idx=6000, end_idx=9000)
    else:
        # 默认：显示全部数据
        visualize_linear_vel_from_npz(npz_file)