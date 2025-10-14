import rosbag
import numpy as np
import sys
sys.path.insert(0, '/home/wegg/anaconda3/envs/rosbag_env/lib/python3.8/site-packages')


def downsample_by_timestamp(data, timestamps, time_interval=0.01):
    """通过固定时间间隔对数据进行采样"""
    if len(data) == 0 or len(timestamps) == 0:
        return data, timestamps
    
    # 获取时间范围
    start_time = timestamps[0]
    end_time = timestamps[-1]
    
    # 生成目标时间点序列
    target_times = np.arange(start_time, end_time + time_interval, time_interval)
    
    # 为每个目标时间点找到最接近的数据点
    sampled_indices = []
    sampled_timestamps = []
    
    for target_time in target_times:
        # 找到最接近目标时间的索引
        closest_idx = np.argmin(np.abs(timestamps - target_time))
        sampled_indices.append(closest_idx)
        sampled_timestamps.append(timestamps[closest_idx])
    
    # 去除重复索引，保持顺序
    unique_indices = []
    unique_timestamps = []
    prev_idx = -1
    
    for idx, ts in zip(sampled_indices, sampled_timestamps):
        if idx != prev_idx:
            unique_indices.append(idx)
            unique_timestamps.append(ts)
            prev_idx = idx
    
    return data[unique_indices], np.array(unique_timestamps)

def align_all_data_by_timestamps(data_dict, time_interval=0.01):
    """
    根据时间戳对齐所有数据
    
    Args:
        data_dict: 字典，格式为 {'data_name': (data_array, timestamps_array)}
        time_interval: 采样时间间隔（秒）
    
    Returns:
        aligned_data_dict: 对齐后的数据字典
    """
    print("\n=== 执行时间戳对齐 ===")
    
    # 1. 找出所有数据的时间范围
    all_start_times = []
    all_end_times = []
    
    for data_name, (data, timestamps) in data_dict.items():
        if len(timestamps) > 0:
            all_start_times.append(timestamps[0])
            all_end_times.append(timestamps[-1])
            print(f"{data_name}: 起始时间={timestamps[0]:.3f}, 结束时间={timestamps[-1]:.3f}, 长度={len(data)}")
    
    # 2. 确定公共时间范围：最大的起始时间 到 最小的结束时间
    common_start_time = max(all_start_times)
    common_end_time = min(all_end_times)
    
    print(f"\n公共时间范围: {common_start_time:.3f} 到 {common_end_time:.3f}")
    print(f"公共时间长度: {common_end_time - common_start_time:.3f} 秒")
    
    # 检查公共时间范围是否有效
    if common_end_time <= common_start_time:
        print("❌ 错误: 没有公共时间范围!")
        return {}
    
    # 3. 生成统一的目标时间序列
    target_times = np.arange(common_start_time, common_end_time + time_interval, time_interval)
    expected_length = len(target_times)
    print(f"目标时间点数量: {expected_length}")
    
    # 4. 对每个数据集进行对齐采样
    aligned_data_dict = {}
    
    for data_name, (data, timestamps) in data_dict.items():
        print(f"\n处理 {data_name}...")
        
        # 找到在公共时间范围内的数据索引
        valid_mask = (timestamps >= common_start_time) & (timestamps <= common_end_time)
        valid_data = data[valid_mask]
        valid_timestamps = timestamps[valid_mask]
        
        if len(valid_data) == 0:
            print(f"⚠️ 警告: {data_name} 在公共时间范围内没有数据")
            # 创建空数组，但保持正确的形状
            if len(data.shape) == 1:
                aligned_data = np.full(expected_length, np.nan)
            else:
                aligned_data = np.full((expected_length, data.shape[1]), np.nan)
            aligned_timestamps = target_times.copy()
        else:
            # 为每个目标时间点找到最接近的数据点
            aligned_data = []
            aligned_timestamps = []
            
            for target_time in target_times:
                # 找到最接近目标时间的索引
                time_diffs = np.abs(valid_timestamps - target_time)
                closest_idx = np.argmin(time_diffs)
                
                # 检查时间差是否在合理范围内
                time_diff = time_diffs[closest_idx]
                if time_diff <= time_interval * 2:  # 允许最大2倍时间间隔的误差
                    aligned_data.append(valid_data[closest_idx])
                    aligned_timestamps.append(valid_timestamps[closest_idx])
                else:
                    # 如果时间差太大，使用插值或填充NaN
                    if len(valid_data.shape) == 1:
                        aligned_data.append(np.nan)
                    else:
                        aligned_data.append(np.full(valid_data.shape[1], np.nan))
                    aligned_timestamps.append(target_time)
            
            aligned_data = np.array(aligned_data)
            aligned_timestamps = np.array(aligned_timestamps)
        
        aligned_data_dict[data_name] = (aligned_data, aligned_timestamps)
        
        # 统计信息
        valid_count = np.sum(~np.isnan(aligned_data.flatten())) if aligned_data.ndim > 1 else np.sum(~np.isnan(aligned_data))
        total_count = aligned_data.size
        print(f"  原始长度: {len(data)} -> 对齐后长度: {len(aligned_data)}")
        print(f"  有效数据比例: {valid_count/total_count*100:.1f}% ({valid_count}/{total_count})")
    
    return aligned_data_dict

# 修改主函数中的调用部分
def bag_to_npz(bag_file, output_file):
    """
    将 .bag 文件中的数据提取并保存为 .npz 文件
    """
    print(f"正在加载 .bag 文件: {bag_file}")
    bag = rosbag.Bag(bag_file)

    # 初始化数据容器
    joint_pos = []
    joint_vel = []
    linear_vel = []
    motor_cur = []
    actions = []
    timestamps_joint_pos = []
    timestamps_joint_vel = []
    timestamps_linear_vel = []
    timestamps_motor_cur = []
    timestamps_actions = []

    # 解析特定话题的数据
    for topic, msg, t in bag.read_messages(topics=[
        '/state_estimate/measuredRbdState/joint_pos',
        '/state_estimate/measuredRbdState/joint_vel',
        '/sensor_data_motor/motor_cur',
        '/rl_controller/actions',
        '/state_estimate/measuredRbdState/linear_vel_xyz'
    ]):
        if topic == '/state_estimate/measuredRbdState/joint_pos':
            joint_pos.append(msg.data)
            timestamps_joint_pos.append(t.to_sec())
        elif topic == '/state_estimate/measuredRbdState/joint_vel':
            joint_vel.append(msg.data)
            timestamps_joint_vel.append(t.to_sec())
        elif topic == '/state_estimate/measuredRbdState/linear_vel_xyz':
            linear_vel.append(msg.data)
            timestamps_linear_vel.append(t.to_sec())
        elif topic == '/sensor_data_motor/motor_cur':
            motor_cur.append(msg.data)
            timestamps_motor_cur.append(t.to_sec())
        elif topic == '/rl_controller/actions':
            actions.append(msg.data)
            timestamps_actions.append(t.to_sec())

    # 转换为 NumPy 数组
    joint_pos = np.array(joint_pos)
    joint_vel = np.array(joint_vel)
    linear_vel = np.array(linear_vel)
    motor_cur = np.array(motor_cur)
    actions = np.array(actions)
    timestamps_joint_pos = np.array(timestamps_joint_pos)
    timestamps_joint_vel = np.array(timestamps_joint_vel)
    timestamps_linear_vel = np.array(timestamps_linear_vel)
    timestamps_motor_cur = np.array(timestamps_motor_cur)
    timestamps_actions = np.array(timestamps_actions)

    # 打印原始数据长度
    print("=== 原始数据长度 ===")
    print(f"joint_pos: {len(joint_pos)}")
    print(f"joint_vel: {len(joint_vel)}")
    print(f"linear_vel: {len(linear_vel)}")
    print(f"motor_cur: {len(motor_cur)}")
    print(f"actions: {len(actions)}")

    # 裁剪actions数据（如果需要）
    actions = actions[32:]
    timestamps_actions = timestamps_actions[32:]

    # 🔥 新的对齐方法：使用统一的时间戳对齐
    # 准备数据字典
    data_dict = {
        'joint_pos': (joint_pos, timestamps_joint_pos),
        'joint_vel': (joint_vel, timestamps_joint_vel),
        'linear_vel': (linear_vel, timestamps_linear_vel),
        'motor_cur': (motor_cur, timestamps_motor_cur),
        'actions': (actions, timestamps_actions)
    }
    
    # 执行时间戳对齐
    aligned_data_dict = align_all_data_by_timestamps(data_dict, time_interval=0.01)
    
    # 提取对齐后的数据
    joint_pos_aligned, timestamps_joint_pos_aligned = aligned_data_dict['joint_pos']
    joint_vel_aligned, timestamps_joint_vel_aligned = aligned_data_dict['joint_vel']
    linear_vel_aligned, timestamps_linear_vel_aligned = aligned_data_dict['linear_vel']
    motor_cur_aligned, timestamps_motor_cur_aligned = aligned_data_dict['motor_cur']
    actions_aligned, timestamps_actions_aligned = aligned_data_dict['actions']
    
    # 验证所有数据长度一致
    print("\n=== 对齐后数据长度验证 ===")
    print(f"joint_pos_aligned: {len(joint_pos_aligned)}")
    print(f"joint_vel_aligned: {len(joint_vel_aligned)}")
    print(f"linear_vel_aligned: {len(linear_vel_aligned)}")
    print(f"motor_cur_aligned: {len(motor_cur_aligned)}")
    print(f"actions_aligned: {len(actions_aligned)}")
    
    # 检查是否所有数据长度相同
    lengths = [len(joint_pos_aligned), len(joint_vel_aligned), len(linear_vel_aligned), 
              len(motor_cur_aligned), len(actions_aligned)]
    if len(set(lengths)) == 1:
        print(f"✅ 所有数据长度一致: {lengths[0]}")
    else:
        print(f"❌ 数据长度不一致: {lengths}")
    
    # 检查时间戳是否对齐
    print("\n=== 时间戳对齐验证 ===")
    print(f"joint_pos时间范围: {timestamps_joint_pos_aligned[0]:.3f} - {timestamps_joint_pos_aligned[-1]:.3f}")
    print(f"actions时间范围: {timestamps_actions_aligned[0]:.3f} - {timestamps_actions_aligned[-1]:.3f}")
    
    # 保存为 .npz 文件（使用对齐后的数据）
    np.savez(output_file,
             joint_pos=joint_pos_aligned, 
             joint_vel=joint_vel_aligned, 
             linear_vel=linear_vel_aligned, 
             motor_cur=motor_cur_aligned, 
             actions=actions_aligned,
             timestamps_joint_pos=timestamps_joint_pos_aligned, 
             timestamps_joint_vel=timestamps_joint_vel_aligned,
             timestamps_linear_vel=timestamps_linear_vel_aligned, 
             timestamps_motor_cur=timestamps_motor_cur_aligned,
             timestamps_actions=timestamps_actions_aligned)
    
    print(f"\n✅ 数据已保存为 {output_file}")
    print("所有数据已通过时间戳对齐到相同的时间范围和长度")

# 主函数
if __name__ == "__main__":
    bag_file = "data/2025-10-10-15-07-13_2.bag"
    output_file = "data/real_run_data/octnew.npz"
    bag_to_npz(bag_file, output_file)