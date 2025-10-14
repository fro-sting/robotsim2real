import numpy as np
import matplotlib.pyplot as plt
import rosbag
import sys
sys.path.insert(0, '/home/wegg/anaconda3/envs/rosbag_env/lib/python3.8/site-packages')


def visualize_motor_cur_from_npz(npz_file):
    """
    从npz文件中可视化motor_cur数据 - 每个电机单独显示
    """
    print(f"正在加载 .npz 文件: {npz_file}")
    
    try:
        data = np.load(npz_file)
        motor_cur = data['motor_cur']
        timestamps = data['timestamps_motor_cur']
        
        print(f"motor_cur数据形状: {motor_cur.shape}")
        print(f"数据长度: {len(motor_cur)}")
        print(f"电机数量: {motor_cur.shape[1]}")
        
        # 计算子图布局
        num_motors = motor_cur.shape[1]
        cols = 4  # 每行显示4个电机
        rows = (num_motors + cols - 1) // cols  # 向上取整
        
        # 创建可视化图表
        plt.figure(figsize=(20, 4 * rows))
        
        # 为每个电机创建单独的子图
        for i in range(num_motors):
            plt.subplot(rows, cols, i + 1)
            plt.plot(timestamps, motor_cur[:, i], linewidth=1.5, color=f'C{i % 10}')
            plt.title(f'电机 {i} 电流变化')
            plt.xlabel('时间 (秒)')
            plt.ylabel('电流')
            plt.grid(True, alpha=0.3)
            
            # 显示当前电机的统计信息
            motor_data = motor_cur[:, i]
            plt.text(0.02, 0.98, 
                    f'范围: [{np.min(motor_data):.2f}, {np.max(motor_data):.2f}]\n'
                    f'均值: {np.mean(motor_data):.2f}\n'
                    f'绝对值均值: {np.mean(np.abs(motor_data)):.2f}',
                    transform=plt.gca().transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    fontsize=8)
        
        plt.tight_layout()
        plt.show()
        
        # 打印总体统计信息
        print("\n=== 各电机电流统计信息 ===")
        for i in range(num_motors):
            motor_data = motor_cur[:, i]
            print(f"电机 {i:2d}: 范围[{np.min(motor_data):6.2f}, {np.max(motor_data):6.2f}] "
                  f"均值{np.mean(motor_data):6.2f} 绝对值均值{np.mean(np.abs(motor_data)):6.2f}")
        
    except Exception as e:
        print(f"读取NPZ文件时出错: {e}")

if __name__ == "__main__":
    
   #npz_file = "data/real_run_data/octold.npz"
    npz_file = "data/real_run_data/octnew.npz"
    visualize_motor_cur_from_npz(npz_file)