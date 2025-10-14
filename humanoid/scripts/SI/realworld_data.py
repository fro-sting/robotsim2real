import numpy as np
import os
from typing import Dict, Any, Optional, Tuple


class RealDataProcessor:
    """真实数据处理工具类"""
    
    def __init__(self, data_file: str):
        """
        初始化真实数据处理器
        
        Args:
            data_file (str): 真实数据文件路径 (.npz格式)
        """
        self.data_file = data_file
        self.np_load_data = None
        self.has_torque_data = False
        self._load_data()
    
    def _load_data(self):
        """加载npz数据文件"""
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"Data file not found: {self.data_file}")
        
        self.np_load_data = np.load(self.data_file)
        print(f"Loaded data file: {self.data_file}")
        print(f"Available fields: {list(self.np_load_data.files)}")
        
        # 检查是否有扭矩数据
        self.has_torque_data = "motor_cur" in self.np_load_data.files
        if self.has_torque_data:
            print("Found torque data (motor_cur)")
        else:
            print("No torque data found, will use zeros")
    
    def get_data_fields(self):
        """获取数据字段信息"""
        if self.np_load_data is None:
            return {}
        
        fields_info = {}
        for field in self.np_load_data.files:
            data_field = self.np_load_data[field]
            fields_info[field] = {
                'shape': data_field.shape,
                'dtype': data_field.dtype
            }
        return fields_info
    
    def _load_arrays(self):
        """加载数据数组，支持不同的字段命名方式"""
        try:
            # 方案1：标准字段名
            joint_pos = self.np_load_data["joint_pos"]
            joint_pos_ts = self.np_load_data["timestamps_joint_pos"]
            joint_vel = self.np_load_data["joint_vel"]
            joint_vel_ts = self.np_load_data["timestamps_joint_vel"]
            actions = self.np_load_data["actions"]
            actions_ts = self.np_load_data["timestamps_actions"]
            speeds = self.np_load_data["linear_vel"]
            speeds_ts = self.np_load_data["timestamps_linear_vel"]
            print("Using standard field names")
        except KeyError:
            try:
                # 方案2：当前字段名
                joint_pos = self.np_load_data["jointpos"]
                joint_pos_ts = self.np_load_data["jointpostimestamps"]
                joint_vel = self.np_load_data["jointvel"]
                joint_vel_ts = self.np_load_data["jointveltimestamps"]
                actions = self.np_load_data["actions"]
                actions_ts = self.np_load_data["actionstimestamps"]
                speeds = self.np_load_data["linear_velocity"]
                speeds_ts = self.np_load_data["timestamps_linear_velocity"]
                print("Using current field names")
            except KeyError as e:
                raise KeyError(f"Unable to find required data fields. Available fields: {list(self.np_load_data.files)}")
        
        # 加载扭矩数据（如果存在）
        torques = None
        torques_ts = None
        if self.has_torque_data:
            torques = self.np_load_data["motor_cur"]
            torques_ts = self.np_load_data["timestamps_motor_cur"]
            
            # 只取前12个关节（腿部关节）
            if torques.shape[1] >= 12:
                torques = torques[:, :12]
                print(f"Using first 12 joints for torques, shape: {torques.shape}")
                
                # 应用扭矩系数
                torque_coefficients = np.array([
                    2.0, 1.2, 1.2, 4.1, 2.1, 2.1,  # 左腿关节系数
                    2.0, 1.2, 1.2, 4.1, 2.1, 2.1   # 右腿关节系数
                ])
                torques = torques * torque_coefficients
                print(f"Applied torque coefficients: {torque_coefficients}")
            else:
                print(f"Warning: Expected at least 12 joints but got {torques.shape[1]}")
                torques = torques * np.ones(torques.shape[1])
        
        return {
            'joint_pos': joint_pos,
            'joint_pos_ts': joint_pos_ts,
            'joint_vel': joint_vel,
            'joint_vel_ts': joint_vel_ts,
            'actions': actions,
            'actions_ts': actions_ts,
            'speeds': speeds,
            'speeds_ts': speeds_ts,
            'torques': torques,
            'torques_ts': torques_ts
        }
    
    def extract_single_run_data(self, 
                               run_value: int, 
                               time_duration: float = 5.0,
                               action_offset: int = -20,
                               torques_offset: int = 7901) -> np.ndarray:
        """
        提取单次运行的数据
        
        Args:
            run_value (int): 运行数据的值
            time_duration (float): 数据持续时间（秒）
            action_offset (int): actions相对于其他数据的时间偏移
            torques_offset (int): torques相对于其他数据的时间偏移
            
        Returns:
            np.ndarray: 形状为 (n_steps, 42) 的数据数组
                       [joint_pos(12), joint_vel(12), actions(12), base_vel(3), world_vel(3), torques(12)]
        """
        arrays = self._load_arrays()
        
        # 计算索引
        real_data_start_offset = run_value 
        start_idx = real_data_start_offset
        end_idx = real_data_start_offset + int(time_duration * 100)
        action_start_idx = real_data_start_offset 
        action_end_idx = real_data_start_offset + int(time_duration * 100) 
        
        # 检查索引范围
        if (start_idx < 0 or end_idx > len(arrays['joint_pos']) or 
            action_start_idx < 0 or action_end_idx > len(arrays['actions'])):
            raise IndexError(f"Index out of range for run_value {run_value}")
        
        if self.has_torque_data and end_idx > len(arrays['torques']):
            raise IndexError(f"Torque index out of range for run_value {run_value}")
        
        # 提取数据片段
        joint_pos_segment = arrays['joint_pos'][start_idx:end_idx, :12]
        joint_vel_segment = arrays['joint_vel'][start_idx:end_idx, :12]
        actions_segment = arrays['actions'][action_start_idx:action_end_idx, :12]
        speeds_segment = arrays['speeds'][start_idx:end_idx, :3]
        
        if self.has_torque_data:
            torques_segment = arrays['torques'][start_idx:end_idx, :12]
            real_data = np.concatenate([
                joint_pos_segment,   # 关节位置 (12)
                joint_vel_segment,   # 关节速度 (12)
                actions_segment,     # 动作 (12)
                speeds_segment,      # 基座速度 (3)
                speeds_segment,      # 世界速度 (3)
                torques_segment      # 实际扭矩 (12)
            ], axis=1)
        else:
            zero_torques = np.zeros((end_idx-start_idx, 12))
            real_data = np.concatenate([
                joint_pos_segment,   # 关节位置 (12)
                joint_vel_segment,   # 关节速度 (12)
                actions_segment,     # 动作 (12)
                speeds_segment,      # 基座速度 (3)
                speeds_segment,      # 世界速度 (3)
                zero_torques         # 扭矩占位符 (12)
            ], axis=1)
        
        return real_data
    
    def extract_all_runs_data(self, 
                             run_values: list = None, 
                             time_duration: float = 5.0,
                             action_offset: int = -20,
                             torques_offset: int = 7901) -> Dict[int, Dict[str, Any]]:
        """
        提取所有运行数据
        
        Args:
            run_values (list): 运行数据值列表，如果为None则使用默认值
            time_duration (float): 每次运行的数据持续时间（秒）
            action_offset (int): actions相对于其他数据的时间偏移
            torques_offset (int): torques相对于其他数据的时间偏移
            
        Returns:
            Dict[int, Dict[str, Any]]: 所有运行数据的字典
        """
        if run_values is None:
            run_values = [6358, 6660, 6985, 7315, 7623, 7933, 8270, 8580, 8895]
        
        all_real_data = {}
        
        print(f"=== 提取 {len(run_values)} 条运行数据 ===")
        
        for run_index, run_value in enumerate(run_values):
            try:
                print(f"提取第{run_index}条数据 (run_value: {run_value})")
                
                real_data = self.extract_single_run_data(
                    run_value=run_value,
                    time_duration=time_duration,
                    action_offset=action_offset,
                    torques_offset=torques_offset
                )
                
                all_real_data[run_index] = {
                    'data': real_data,
                    'run_value': run_value,
                    'shape': real_data.shape,
                    'has_torque': self.has_torque_data
                }
                
                print(f"  ✅ 成功提取，形状: {real_data.shape}")
                
            except Exception as e:
                print(f"  ❌ 提取失败: {e}")
                continue
        
        print(f"=== 提取完成：成功 {len(all_real_data)}/{len(run_values)} 条 ===")
        return all_real_data
    
    def test_data_quality(self, all_real_data: Dict[int, Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """
        测试数据质量
        
        Args:
            all_real_data: 所有运行数据
            
        Returns:
            Dict[int, Dict[str, Any]]: 数据质量报告
        """
        print(f"=== 测试{len(all_real_data)}条数据质量 ===")
        
        quality_reports = {}
        
        for run_index, data_info in all_real_data.items():
            data = data_info['data']
            
            # 基本检查
            nan_count = np.isnan(data).sum()
            inf_count = np.isinf(data).sum()
            
            # 数据变化检查
            joint_vel = data[:, 12:24]
            actions = data[:, 24:36]
            
            vel_std = joint_vel.std()
            action_std = actions.std()
            
            # 生成质量报告
            quality_report = {
                'shape': data.shape,
                'nan_count': nan_count,
                'inf_count': inf_count,
                'vel_std': vel_std,
                'action_std': action_std,
                'is_valid': True,
                'issues': []
            }
            
            # 检查问题
            if nan_count > 0:
                quality_report['issues'].append(f"Contains {nan_count} NaN values")
                quality_report['is_valid'] = False
            
            if inf_count > 0:
                quality_report['issues'].append(f"Contains {inf_count} infinite values")
                quality_report['is_valid'] = False
            
            if vel_std < 1e-6:
                quality_report['issues'].append("Very low velocity variation")
                quality_report['is_valid'] = False
            
            if action_std < 1e-6:
                quality_report['issues'].append("Very low action variation")
                quality_report['is_valid'] = False
            
            quality_reports[run_index] = quality_report
            
            # 打印报告
            status = "✅ 正常" if quality_report['is_valid'] else "⚠️  有问题"
            print(f"Run {run_index}: 形状{data.shape}, "
                  f"NaN={nan_count}, Inf={inf_count}, "
                  f"vel_std={vel_std:.4f}, "
                  f"action_std={action_std:.4f} - {status}")
            
            if quality_report['issues']:
                for issue in quality_report['issues']:
                    print(f"  - {issue}")
        
        return quality_reports
    
    def check_data_timestamps(self, run_value: int = None):
        """
        检查数据时间戳一致性
        
        Args:
            run_value (int): 如果指定，检查特定运行的数据；如果为None，检查全部数据
        """
        arrays = self._load_arrays()
        
        if run_value is not None:
            print(f"=== 检查运行 {run_value} 的时间戳一致性 ===")
            # 实现特定运行的时间戳检查
            self._check_single_run_timestamps(arrays, run_value)
        else:
            print("=== 检查所有数据的时间戳一致性 ===")
            self._check_all_timestamps(arrays)
    
    def _check_single_run_timestamps(self, arrays: dict, run_value: int, time_duration: float = 5.0):
        """检查单次运行的时间戳"""
        # 计算索引
        real_data_start_offset = run_value * 10 - 59000
        start_idx = real_data_start_offset
        end_idx = real_data_start_offset + int(time_duration * 100)
        action_offset = -20
        action_start_idx = real_data_start_offset + action_offset
        action_end_idx = real_data_start_offset + int(time_duration * 100) + action_offset
        
        print(f"检查索引范围: [{start_idx}:{end_idx}], 动作: [{action_start_idx}:{action_end_idx}]")
        
        # 提取时间戳
        joint_pos_ts_slice = arrays['joint_pos_ts'][start_idx:end_idx]
        joint_vel_ts_slice = arrays['joint_vel_ts'][start_idx:end_idx]
        actions_ts_slice = arrays['actions_ts'][action_start_idx:action_end_idx]
        speeds_ts_slice = arrays['speeds_ts'][start_idx:end_idx]
        
        # 分析时间间隔
        def analyze_intervals(timestamps, name):
            if len(timestamps) < 2:
                print(f"{name}: 数据点不足")
                return
            
            intervals = np.diff(timestamps)
            mean_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            cv = std_interval / mean_interval * 100 if mean_interval > 0 else 0
            
            print(f"{name}:")
            print(f"  数量: {len(timestamps)}")
            print(f"  时间范围: {timestamps[0]:.6f} - {timestamps[-1]:.6f}")
            print(f"  总时长: {timestamps[-1] - timestamps[0]:.6f}秒")
            print(f"  平均间隔: {mean_interval:.6f}秒")
            print(f"  变异系数: {cv:.4f}%")
            print(f"  理论频率: {1/mean_interval:.2f} Hz")
        
        analyze_intervals(joint_pos_ts_slice, "joint_pos")
        analyze_intervals(joint_vel_ts_slice, "joint_vel")
        analyze_intervals(actions_ts_slice, "actions")
        analyze_intervals(speeds_ts_slice, "speeds")
        
        if self.has_torque_data:
            torques_ts_slice = arrays['torques_ts'][start_idx:end_idx]
            analyze_intervals(torques_ts_slice, "torques")
    
    def _check_all_timestamps(self, arrays: dict):
        """检查所有数据的时间戳"""
        print("数据源时间戳基本信息:")
        
        data_sources = ['joint_pos', 'joint_vel', 'actions', 'speeds']
        if self.has_torque_data:
            data_sources.append('torques')
        
        for source in data_sources:
            ts_key = f'{source}_ts'
            if ts_key in arrays and arrays[ts_key] is not None:
                timestamps = arrays[ts_key]
                print(f"\n{source}:")
                print(f"  时间戳数量: {len(timestamps)}")
                print(f"  时间范围: {timestamps[0]:.6f} - {timestamps[-1]:.6f}")
                print(f"  总时长: {timestamps[-1] - timestamps[0]:.6f}秒")
                
                if len(timestamps) > 1:
                    intervals = np.diff(timestamps)
                    mean_interval = np.mean(intervals)
                    std_interval = np.std(intervals)
                    cv = std_interval / mean_interval * 100 if mean_interval > 0 else 0
                    
                    print(f"  平均间隔: {mean_interval:.6f}秒")
                    print(f"  变异系数: {cv:.4f}%")
                    print(f"  理论频率: {1/mean_interval:.2f} Hz")


# 便捷函数
def load_real_data_single_run(data_file: str, 
                             run_value: int, 
                             time_duration: float = 5.0,
                             action_offset: int = -20,
                             torques_offset: int = 7901) -> np.ndarray:
    """
    便捷函数：加载单次运行的真实数据
    
    Args:
        data_file (str): 数据文件路径
        run_value (int): 运行数据值
        time_duration (float): 数据持续时间（秒）
        action_offset (int): actions时间偏移
        torques_offset (int): torques时间偏移
        
    Returns:
        np.ndarray: 真实数据数组
    """
    processor = RealDataProcessor(data_file)
    return processor.extract_single_run_data(
        run_value=run_value,
        time_duration=time_duration,
        action_offset=action_offset,
        torques_offset=torques_offset
    )


def load_real_data_all_runs(data_file: str, 
                           run_values: list = None, 
                           time_duration: float = 5.0,
                           action_offset: int = -20,
                           torques_offset: int = 7901) -> Dict[int, Dict[str, Any]]:
    """
    便捷函数：加载所有运行的真实数据
    
    Args:
        data_file (str): 数据文件路径
        run_values (list): 运行数据值列表
        time_duration (float): 每次运行的数据持续时间（秒）
        action_offset (int): actions时间偏移
        torques_offset (int): torques时间偏移
        
    Returns:
        Dict[int, Dict[str, Any]]: 所有运行数据字典
    """
    processor = RealDataProcessor(data_file)
    all_data = processor.extract_all_runs_data(
        run_values=run_values,
        time_duration=time_duration,
        action_offset=action_offset,
        torques_offset=torques_offset
    )
    
    # 自动进行质量检查
    processor.test_data_quality(all_data)
    
    return all_data


def check_real_data_timestamps(data_file: str, run_value: int = None):
    """
    便捷函数：检查真实数据时间戳
    
    Args:
        data_file (str): 数据文件路径
        run_value (int): 如果指定，检查特定运行；如果为None，检查全部
    """
    processor = RealDataProcessor(data_file)
    processor.check_data_timestamps(run_value)


def get_real_data_info(data_file: str) -> Dict[str, Any]:
    """
    便捷函数：获取真实数据文件信息
    
    Args:
        data_file (str): 数据文件路径
        
    Returns:
        Dict[str, Any]: 数据文件信息
    """
    processor = RealDataProcessor(data_file)
    return {
        'fields': processor.get_data_fields(),
        'has_torque_data': processor.has_torque_data,
        'file_path': data_file
    }