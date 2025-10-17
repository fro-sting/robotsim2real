import torch
import numpy as np

def generate_walking_motion(t, env):
    """
    生成行走动作
    
    Args:
        t: 当前时间
        env: 环境对象
    
    Returns:
        actions: 关节动作张量
    """
    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device, dtype=torch.float)
    
    # 参数设置
    freq = 1.0  # 频率 (Hz)
    
    # 假设关节顺序：
    # 0-2: 左髋 (俯仰, 外展, 旋转)
    # 3-5: 左膝, 左踝俯仰, 左踝侧倾
    # 6-8: 右髋 (俯仰, 外展, 旋转)
    # 9-11: 右膝, 右踝俯仰, 右踝侧倾
    
    # 行走模式：左右腿交替摆动
    phase_left = np.sin(2 * np.pi * freq * t)
    phase_right = np.sin(2 * np.pi * freq * t + np.pi)  # 相位差180度
    
    # 髋关节俯仰 (前后摆动)
    hip_amplitude = 1.4
    actions[:, 0] = hip_amplitude * phase_left   # 左髋
    actions[:, 6] = hip_amplitude * phase_right  # 右髋
    
    # 膝关节 (弯曲配合髋关节)
    knee_amplitude = 5.8
    actions[:, 3] = knee_amplitude * (0.5 + 0.5 * np.abs(phase_left))   # 左膝
    actions[:, 9] = knee_amplitude * (0.5 + 0.5 * np.abs(phase_right))  # 右膝
    
    # 踝关节 (轻微补偿)
    ankle_amplitude = 1.2
    actions[:, 4] = ankle_amplitude * phase_left   # 左踝
    actions[:, 10] = ankle_amplitude * phase_right  # 右踝
    
    return actions


def generate_single_sin_motion(t, env):
    """
    让每个关节依次做正弦运动，左右对称处理
    🔥 每个关节运行一个完整周期后切换
    
    Args:
        t: 当前时间
        env: 环境对象
    
    Returns:
        actions: 关节动作张量
        is_finished: 是否完成所有关节的运动
    """
    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device, dtype=torch.float)
    
    # 🔥 正弦波参数
    freq_base = 0.5  # 频率 (Hz)
    omega = 2 * np.pi * freq_base
    
    # 🔥 一个完整周期的时间
    period_duration = 1.0 / freq_base  # T = 1/f = 2.0秒
    
    # 关节对（左右对称）
    joint_pairs = [
        (0, 6, "髋俯仰"),
        (1, 7, "髋外展"),
        (2, 8, "髋旋转"),
        (3, 9, "膝关节"),
        (4, 10, "踝俯仰"),
        (5, 11, "踝侧倾"),
    ]
    
    # 🔥 计算当前激活的关节对（基于完整周期）
    total_cycle_time = period_duration * len(joint_pairs)
    current_joint_idx = int(t / period_duration)
    t_joint = t % period_duration  # 当前关节的局部时间
    
    # 🔥 检查是否完成所有关节
    is_finished = (t >= total_cycle_time)
    
    if current_joint_idx < len(joint_pairs) and not is_finished:
        left_joint, right_joint, joint_name = joint_pairs[current_joint_idx]
        
        # 🔥 计算正弦值 (从0开始: sin(0) = 0)
        phase = np.sin(omega * t_joint)
        
        # 根据不同关节设置不同的幅度
        if "髋俯仰" in joint_name:
            amplitude = 1.2
        elif "髋外展" in joint_name:
            amplitude = 1.5
        elif "髋旋转" in joint_name:
            amplitude = 2.0
        elif "膝关节" in joint_name:
            amplitude = 2.5
        elif "踝俯仰" in joint_name:
            amplitude = 1.4
        else:  # 踝侧倾
            amplitude = 1.2
        
        # 左右对称运动
        actions[:, left_joint] = amplitude * phase
        actions[:, right_joint] = amplitude * phase
        
        # 🔥 显示进度
        cycle_progress = (t_joint / period_duration) * 100
        if int(t * 10) % 25 == 0 and cycle_progress < 5:
            print(f"🔄 关节 {current_joint_idx + 1}/{len(joint_pairs)}: {joint_name} | "
                  f"进度: {cycle_progress:.1f}% | 剩余: {(total_cycle_time - t):.1f}s | "
                  f"幅值: {amplitude * phase:.3f}")
    
    return actions, is_finished


def generate_single_fourier_motion(t, env):
    """
    让每个关节依次运动，左右对称处理
    🔥 每个关节运行一个完整周期后切换
    
    Args:
        t: 当前时间
        env: 环境对象
    
    Returns:
        actions: 关节动作张量
        is_finished: 是否完成所有关节的运动
    """
    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device, dtype=torch.float)
    
    # 🔥 傅里叶级数参数
    freq_base = 0.2
    omega = 2 * np.pi * freq_base
    
    # 🔥 一个完整周期的时间
    period_duration = 1.0 / freq_base  # T = 1/f ≈ 6.67秒
    
    # 关节对（左右对称）
    joint_pairs = [
        (0, 6, "髋俯仰"),
        (1, 7, "髋外展"),
        (2, 8, "髋旋转"),
        (3, 9, "膝关节"),
        (4, 10, "踝俯仰"),
        (5, 11, "踝侧倾"),
    ]
    
    # 🔥 计算当前激活的关节对（基于完整周期）
    total_cycle_time = period_duration * len(joint_pairs)
    current_joint_idx = int(t / period_duration)
    t_joint = t % period_duration  # 当前关节的局部时间
    
    # 🔥 检查是否完成所有关节
    is_finished = (t >= total_cycle_time)
    
    if current_joint_idx < len(joint_pairs) and not is_finished:
        left_joint, right_joint, joint_name = joint_pairs[current_joint_idx]
        
        # 应用时间偏移，从零点开始
        t_joint -= 0.11864 / freq_base
        
        # 🔥 计算傅里叶级数值
        phase = ( 0.8*np.sin(omega * t_joint) + 0.7*np.cos(omega*t_joint)
                 -0.2 * np.sin(3 * omega * t_joint) +  0.3 * np.cos(3 * omega * t_joint)
                 -0.2 * np.sin(5 * omega * t_joint) +  -0.2 * np.cos(5 * omega * t_joint))
        
        # 根据不同关节设置不同的幅度
        if "髋俯仰" in joint_name:
            amplitude = 1.0
        elif "髋外展" in joint_name:
            amplitude = 1.0
        elif "髋旋转" in joint_name:
            amplitude = 2.3
        elif "膝关节" in joint_name:
            amplitude = 1.0
            phase = 1.5 * phase
        elif "踝俯仰" in joint_name:
            amplitude = 1.4
        else:  # 踝侧倾
            amplitude = 1.2
        
        # 左右对称运动
        actions[:, left_joint] = amplitude * phase
        actions[:, right_joint] = amplitude * phase
        
        # 🔥 显示进度
        cycle_progress = (t_joint / period_duration) * 100
        if int(t * 10) % 25 == 0 and cycle_progress < 5:
            print(f"🔄 关节 {current_joint_idx + 1}/{len(joint_pairs)}: {joint_name} | "
                  f"进度: {cycle_progress:.1f}% | 剩余: {(total_cycle_time - t):.1f}s")
    
    return actions, is_finished


def generate_zero_motion(t, env):
    """
    生成零动作（保持默认姿态）
    
    Args:
        t: 当前时间
        env: 环境对象
    
    Returns:
        actions: 关节动作张量
        is_finished: 是否完成（保持5秒后结束）
    """
    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device, dtype=torch.float)
    
    hold_time = 5.0  # 保持5秒
    is_finished = (t >= hold_time)
    
    return actions, is_finished


def generate_single_hip_motion(t, env):
    """
    让髋关节做3D圆周运动
    🔥 运行5个周期后结束
    
    Args:
        t: 当前时间
        env: 环境对象
    
    Returns:
        actions: 关节动作张量
        is_finished: 是否完成运动
    """
    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device, dtype=torch.float)
    
    # 圆周运动参数
    freq = 0.3
    num_cycles = 5
    total_time = num_cycles / freq
    
    # 检查是否完成
    is_finished = (t >= total_time)
    
    if not is_finished:
        radius_pitch = 1.8
        radius_abduct = 1.8
        radius_rotate = 1.0
        
        angle = 2 * np.pi * freq * t
        
        pitch_value = radius_pitch * np.sin(2 * angle)
        abduct_value = radius_abduct * np.sin(2 * angle)
        rotate_value = radius_rotate * np.sin(3 * angle)
        
        actions[:, 0] = pitch_value
        actions[:, 1] = abduct_value
        actions[:, 2] = rotate_value
        
        actions[:, 6] = pitch_value
        actions[:, 7] = abduct_value
        actions[:, 8] = rotate_value
    
    return actions, is_finished


def generate_single_ankle_motion(t, env):
    """
    让踝关节做3D圆周运动
    🔥 运行5个周期后结束
    
    Args:
        t: 当前时间
        env: 环境对象
    
    Returns:
        actions: 关节动作张量
        is_finished: 是否完成运动
    """
    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device, dtype=torch.float)
    
    # 圆周运动参数
    freq = 0.5
    num_cycles = 5
    total_time = num_cycles / freq
    
    # 检查是否完成
    is_finished = (t >= total_time)
    
    if not is_finished:
        radius_pitch = 1.2
        radius_roll = 1.2
        
        angle = 2 * np.pi * freq * t
        
        pitch_value = radius_pitch * np.cos(angle)
        roll_value = radius_roll * np.sin(angle)
        
        actions[:, 4] = pitch_value
        actions[:, 5] = roll_value
        
        actions[:, 10] = pitch_value
        actions[:, 11] = roll_value
    
    return actions, is_finished


def generate_leg_motion(t, env):
    """
    让所有腿部关节同时运动，形成球形轨迹
    🔥 运行10秒后结束
    
    Args:
        t: 当前时间
        env: 环境对象
    
    Returns:
        actions: 关节动作张量
        is_finished: 是否完成运动
    """
    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device, dtype=torch.float)
    
    total_time = 10.0  # 运行10秒
    is_finished = (t >= total_time)
    
    if not is_finished:
        omega_theta = 2.4
        omega_phi = 2.6
        
        theta_period = 2 * np.pi / omega_theta
        t_normalized = (t % theta_period) / theta_period
        
        if t_normalized < 0.5:
            theta = 2 * t_normalized * np.pi
        else:
            theta = 2 * (1 - t_normalized) * np.pi
        
        phi = (omega_phi * t) % (2 * np.pi)
        
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        
        hip_pitch_amp = 1.2
        hip_abduct_amp = 1.2
        hip_rotate_amp = 1.0
        knee_amp = 1.8
        ankle_pitch_amp = 1.5
        ankle_roll_amp = 1.5
        
        actions[:, 0] = hip_pitch_amp * z
        actions[:, 6] = hip_pitch_amp * z
        
        actions[:, 1] = hip_abduct_amp * y
        actions[:, 7] = hip_abduct_amp * y
        
        actions[:, 2] = hip_rotate_amp * x
        actions[:, 8] = hip_rotate_amp * x
        
        knee_normalized = theta / np.pi
        knee_value = knee_amp * (0.3 + 0.7 * knee_normalized)
        actions[:, 3] = knee_value
        actions[:, 9] = knee_value
        
        ankle_pitch_value = ankle_pitch_amp * (x * z)
        ankle_roll_value = ankle_roll_amp * (y * z)
        
        actions[:, 4] = ankle_pitch_value
        actions[:, 10] = ankle_pitch_value
        
        actions[:, 5] = ankle_roll_value
        actions[:, 11] = ankle_roll_value
        
        if int(t * 2) % 2 == 0 and (t * 2) - int(t * 2) < 0.05:
            theta_deg = np.degrees(theta)
            phi_deg = np.degrees(phi)
            print(f"🌐 球形轨迹 | t: {t:.2f}s | 剩余: {(total_time - t):.1f}s | θ: {theta_deg:.1f}° | φ: {phi_deg:.1f}°")
    
    return actions, is_finished


def generate_fourier_motion(t, env):
    """
    使用傅里叶级数生成复杂周期运动
    🔥 运行10秒后结束
    
    Args:
        t: 当前时间
        env: 环境对象
    
    Returns:
        actions: 关节动作张量
        is_finished: 是否完成运动
    """
    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device, dtype=torch.float)
    
    total_time = 10.0  # 运行10秒
    is_finished = (t >= total_time)
    
    if not is_finished:
        A1, f1, phi1 = 1.0, 0.5, 0.0
        A2, f2, phi2 = 0.3, 1.5, np.pi/4
        A3, f3, phi3 = 0.1, 2.5, np.pi/2
        
        fourier_value = (A1 * np.sin(2 * np.pi * f1 * t + phi1) +
                         A2 * np.sin(2 * np.pi * f2 * t + phi2) +
                         A3 * np.sin(2 * np.pi * f3 * t + phi3))
        
        for i in range(6):
            actions[:, i] = fourier_value
            actions[:, i + 6] = fourier_value
    
    return actions, is_finished