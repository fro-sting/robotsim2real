import rosbag
import numpy as np
import sys
sys.path.insert(0, '/home/wegg/anaconda3/envs/rosbag_env/lib/python3.8/site-packages')

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
        '/rl_controller/actions',# 添加 actions 话题
        '/state_estimate/measuredRbdState/linear_vel_xyz'
        #'/rl_controller/cmd',
        #'/rl_controller/torque'
    ]):
        if topic == '/state_estimate/measuredRbdState/joint_pos':
            joint_pos.append(msg.data)
            timestamps_joint_pos.append(t.to_sec())
        elif topic == '/state_estimate/measuredRbdState/joint_vel':
            joint_vel.append(msg.data)
            timestamps_joint_vel.append(t.to_sec())
        elif topic == '/state_estimate/measuredRbdState/linear_vel_xyz':
        #elif topic == '/rl_controller/cmd':
            linear_vel.append(msg.data)
            timestamps_linear_vel.append(t.to_sec())
        elif topic == '/sensor_data_motor/motor_cur':
        #elif topic == '/rl_controller/torque':
            motor_cur.append(msg.data)
            timestamps_motor_cur.append(t.to_sec())
        elif topic == '/rl_controller/actions':  # 处理 actions 数据
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

    # 目标长度（actions的长度）
    target_length = len(actions)
    print(f"\n目标长度: {target_length}")


    
    actions = actions[32:]
    timestamps_actions = timestamps_actions[32:]

    

    # 定义间隔采样函数
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

    # 对所有数据进行基于时间戳的采样
    print("\n=== 执行基于时间戳的采样 (0.01s间隔) ===")
    
    joint_pos_sampled, timestamps_joint_pos_sampled = downsample_by_timestamp(
        joint_pos, timestamps_joint_pos, 0.01)
    print(f"joint_pos: {len(joint_pos)} -> {len(joint_pos_sampled)}")
    
    joint_vel_sampled, timestamps_joint_vel_sampled = downsample_by_timestamp(
        joint_vel, timestamps_joint_vel, 0.01)
    print(f"joint_vel: {len(joint_vel)} -> {len(joint_vel_sampled)}")
    
    linear_vel_sampled, timestamps_linear_vel_sampled = downsample_by_timestamp(
        linear_vel, timestamps_linear_vel, 0.01)
    print(f"linear_vel: {len(linear_vel)} -> {len(linear_vel_sampled)}")
    
    motor_cur_sampled, timestamps_motor_cur_sampled = downsample_by_timestamp(
        motor_cur_trimmed, timestamps_motor_cur_trimmed, 0.01)
    print(f"motor_cur: {len(motor_cur_trimmed)} -> {len(motor_cur_sampled)}")

    actions_sampled, timestamps_actions_sampled = downsample_by_timestamp(
        actions, timestamps_actions, 0.01)
    print(f"actions: {len(actions)} -> {len(actions_sampled)}")

    
    # 验证所有数据长度一致
    print("\n=== 采样后数据长度验证 ===")
    print(f"joint_pos_sampled: {len(joint_pos_sampled)}")
    print(f"joint_vel_sampled: {len(joint_vel_sampled)}")
    print(f"linear_vel_sampled: {len(linear_vel_sampled)}")
    print(f"motor_cur_sampled: {len(motor_cur_sampled)}")
    print(f"actions_sampled: {len(actions_sampled)}")

    # 保存为 .npz 文件（使用采样后的数据）
    np.savez(output_file,
             joint_pos=joint_pos_sampled, 
             joint_vel=joint_vel_sampled, 
             linear_vel=linear_vel_sampled, 
             motor_cur=motor_cur_sampled, 
             actions=actions_sampled,
             timestamps_joint_pos=timestamps_joint_pos_sampled, 
             timestamps_joint_vel=timestamps_joint_vel_sampled,
             timestamps_linear_vel=timestamps_linear_vel_sampled, 
             timestamps_motor_cur=timestamps_motor_cur_sampled,
             timestamps_actions=timestamps_actions_sampled)
    print(f"\n数据已保存为 {output_file}")
    print("所有数据（除actions外）已通过间隔采样调整到相同长度")

# 主函数
if __name__ == "__main__":
    bag_file = "data/2025-10-10-15-10-09_0.bag"
    output_file = "data/real_run_data/octold.npz"
    bag_to_npz(bag_file, output_file)