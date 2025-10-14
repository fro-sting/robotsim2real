import os
import rospy
from tf2_msgs.msg import TFMessage
import numpy as np
import csv
import math
import time  
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from tf.transformations import euler_from_quaternion
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
# from kuavo_msgs.msg import *
from nav_msgs.msg import Odometry
import mujoco
import mujoco_viewer  
from tqdm import tqdm  
from scipy.signal import argrelextrema

#                                                     $$$定义全局变量和缓冲区$$$
# 用于record
dof_pos_buffer = []
dof_vel_buffer = []
dof_torque_buffer = []
foot_force_buffer = []
foot_height_buffer = []
root_pos_buffer = []
root_eu_ang_buffer = []
root_ang_vel_buffer = []
root_lin_vel_buffer = []
phase_count_buffer = []

is_recoding = False
is_finish_recoding = False
count_sensorsData = 0
count_Odometry = 0
phase_count = 0
last_foot_force = 0

# 用于npz_show
ref_phase_dof_pos = []
ref_phase_dof_vel = []
ref_phase_dof_torque = []
ref_phase_foot_force = []
ref_phase_foot_height = []
ref_phase_root_pos = []
ref_phase_root_eu_ang = []
ref_phase_root_ang_vel = []
ref_phase_root_lin_vel = []
energy_dof= []
#定义关节名称列表
dof_name = ['leg_l1_joint', 'leg_l2_joint', 'leg_l3_joint', 'leg_l4_joint', 'leg_l5_joint', 'leg_l6_joint',
            'leg_r1_joint', 'leg_r2_joint', 'leg_r3_joint', 'leg_r4_joint', 'leg_r5_joint', 'leg_r6_joint',
            'zarm_l1_joint', 'zarm_l2_joint', 'zarm_l3_joint', 'zarm_l4_joint', 'zarm_l5_joint', 'zarm_l6_joint','zarm_l7_joint',
            'zarm_r1_joint', 'zarm_r2_joint', 'zarm_r3_joint', 'zarm_r4_joint', 'zarm_r5_joint', 'zarm_r6_joint', 'zarm_r7_joint']

# 定义mujoco模型
model = mujoco.MjModel.from_xml_path("/home/zl/Downloads/kuavo-rl-train-run/RL_train/resources/robots/biped_s44/xml/scene.xml")
model.opt.timestep = 0.001
model.opt.gravity[2] = -9.81  # 设置重力加速度
mj_data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, mj_data, 0)

#                                                           $$$函数定义$$$
def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat
    
    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    
    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z])

# 处理'/ground_truth/state'数据
def make_callback_Odometry(record_gap):
    def callback_Odometry(msg):
        # 处理 sensorsData 消息

        root_pos = np.zeros(3)
        root_eu_ang = np.zeros(7)
        root_ang_vel = np.zeros(3)
        root_lin_vel = np.zeros(3)

        global root_pos_buffer, root_eu_ang_buffer, root_ang_vel_buffer, root_lin_vel_buffer, is_recoding, count_Odometry
        
        # pose
        #     pose: 
        #         position: 
        #         x: -0.0006555448377156192
        #         y: 0.00019336976052814007
        #         z: 0.7976796332587985
        #         orientation: 
        #         x: -0.00020997930664468938
        #         y: 0.049364636101203396
        #         z: 0.00024142644554695297
        #         w: 0.9987807719037036
        #     covariance: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # twist: 
        #     twist: 
        #         linear: 
        #         x: 0.00031419748473544216
        #         y: -1.5017139517204781e-05
        #         z: 0.00013361427259204407
        #         angular: 
        #         x: -0.0002326707500198907
        #         y: -0.018256887306800498
        #         z: -0.000187934671419839
        #     covariance: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        if is_recoding and count_Odometry % record_gap == 0:
            root_pos[0] = msg.pose.pose.position.x
            root_pos[1] = msg.pose.pose.position.y
            root_pos[2] = msg.pose.pose.position.z

            quat = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
            eu_ang = quaternion_to_euler_array(quat)
            eu_ang[eu_ang > math.pi] -= 2 * math.pi

            root_eu_ang[0] = msg.pose.pose.orientation.x
            root_eu_ang[1] = msg.pose.pose.orientation.y
            root_eu_ang[2] = msg.pose.pose.orientation.z
            root_eu_ang[3] = msg.pose.pose.orientation.w
            root_eu_ang[4] = eu_ang[0]
            root_eu_ang[5] = eu_ang[1]
            root_eu_ang[6] = eu_ang[2]

            root_ang_vel[0] = msg.twist.twist.angular.x
            root_ang_vel[1] = msg.twist.twist.angular.y
            root_ang_vel[2] = msg.twist.twist.angular.z

            root_lin_vel[0] = msg.twist.twist.linear.x
            root_lin_vel[1] = msg.twist.twist.linear.y
            root_lin_vel[2] = msg.twist.twist.linear.z

            root_pos_buffer.append(root_pos)
            root_eu_ang_buffer.append(root_eu_ang)
            root_ang_vel_buffer.append(root_ang_vel)
            root_lin_vel_buffer.append(root_lin_vel)

        count_Odometry += 1
    return callback_Odometry

# 处理'/sensors_data_raw'的数据
def make_callback_sensorsData(file_name,final_length,record_gap):
    def callback_sensorsData(msg):        
        # 处理 sensorsData 消息
        dof_pos = np.zeros(26)
        dof_vel = np.zeros(26)
        dof_torque = np.zeros(26)
        foot_force = np.zeros(2)

        global dof_pos_buffer, dof_vel_buffer, dof_torque_buffer, foot_force_buffer, root_pos_buffer, root_eu_ang_buffer, root_ang_vel_buffer, root_lin_vel_buffer, is_recoding, count_sensorsData, is_finish_recoding, foot_height_buffer
        global phase_count_buffer, phase_count, last_foot_force

        # print(msg)
        # imu_data: 
        # gyro: 
        #     x: -0.00010841381027228585
        #     y: -0.003010133646546128
        #     z: 2.6656089522703985e-05
        # acc: 
        #     x: -0.9703350888558406
        #     y: -0.0036034753159534375
        #     z: 9.756156935470594
        # free_acc: 
        #     x: -0.00901864393967744
        #     y: 0.0002536216694569676
        #     z: 0.01849392961869789
        # quat: 
        #     x: -0.0002096841770109712
        #     y: 0.04918141035418275
        #     z: 0.0002508688647794875
        #     w: 0.9987898087049805
        # end_effector_data: 
        # name: []
        # position: []
        # velocity: []
        # effort: []
        # FTsensor_data: 
        # Fx: [-0.010834558199955073, -0.0050383034956931125]
        # Fy: [0.00047062202338536566, 0.0003167948922147912]
        # Fz: [-220.99668292632037, -220.28198509013956]
        # Mx: [-0.008996271944749652, -0.009082150300226209]
        # My: [6.672405277202348, 6.65065461926784]
        # Mz: [0.0005900234526897251, 0.0006895398859773281]
        # 497
        # header: 
        # seq: 1092226
        # stamp: 
        #     secs: 1725864038
        #     nsecs: 172729397
        # frame_id: "world"
        # sensor_time: 
        # secs: 1725863969
        # nsecs: 989451043
        # joint_data: 
        # joint_q: [-0.018041611945046835, -0.0015649649592276947, -0.48940821728026085, 0.7056553912982584, -0.3147454905221503, 0.018483749844499762, 0.01830524654672373, 0.00205020233260016, -0.48831146091253175, 0.7048997890730003, -0.31507791066242424, -0.01804243674911146, -0.00014447698488667448, -6.667482911693844e-05, -2.9131160578780012e-05, -7.568980899415256e-06, 0.00043893658340695964, -0.0009975549249409013, 6.458939845743316e-05, -0.0001352889817023931, 2.156652357110073e-05, 1.1036884353520287e-05, -1.25450933072564e-05, -0.000295506013554012, -0.0008689882961332215, -5.7074461848337565e-05]
        # joint_v: [4.014555791212237e-05, 2.6527913292775515e-05, -0.0017060612730691797, 0.0032974235178204014, 0.000928380763296117, 9.242810586123944e-05, 6.362192237588996e-05, -6.907200518157262e-05, -0.001562437298165153, 0.0030666330285543815, 0.001017152633134555, 7.514262723898338e-05, -1.053661424457608e-05, -6.634639741883697e-07, 6.9588573611855496e-06, -5.180756125992519e-05, 3.370809401550792e-06, -5.010744630114503e-05, 2.169938453872417e-05, 8.70176147005139e-06, 4.187026577130743e-07, -9.48118513569022e-06, -8.359327512641286e-05, -3.4886607904494823e-06, -5.257556251534501e-05, -2.2703076298824886e-05]
        # joint_vd: [-0.001907147950310679, 0.0012517620039214966, -0.2088588839211644, 0.03442320048185469, 0.11418812478409315, -0.004054032781341353, -0.0019100648447631916, 0.004195555049145565, -0.21637975464124548, 0.046078934775327506, 0.11016502390949195, -0.003910297822835942, -0.003577576431771738, -0.00019207557543223907, 0.0007134424325796955, 0.004076560836942373, -0.0009464230816781049, -0.0030323766292805533, 0.0009203505387100368, -0.0036212222563867017, 8.173441935915733e-05, -0.000193354863041016, 0.00497128397229797, 0.000999206861205899, -0.003128126929174597, -0.0001540169751674724]
        # joint_current: [-1.8658265623640173, -0.16850166828386437, 3.5587729882845167, -16.350420794205455, 6.6713623417302905, -0.00897576593444413, 1.898918527216798, 0.17528533563951793, 3.4658188619290495, -16.32124970785916, 6.649712384136552, -0.009065006083624675, 0.5231182608742044, 0.22073059031166328, 0.018889276405855773, 0.24783066733679168, 0.009792150782449485, -0.01369204073182342, 0.04430216853695169, 0.536912102829146, -0.2226472472653781, -0.01856066265931144, 0.2598127351959165, -0.009401618738964078, -0.003295829732657207, -0.04055643953781794]

        if is_recoding:
            if not root_pos_buffer:
                return
            if count_sensorsData % record_gap == 0:
                for i in range(26):
                    dof_pos[i] = msg.joint_data.joint_q[i]
                    dof_vel[i] = msg.joint_data.joint_v[i]
                    dof_torque[i] = msg.joint_data.joint_current[i]

                foot_force[0] = msg.FTsensor_data.Fz[0]
                foot_force[1] = msg.FTsensor_data.Fz[1]

                dof_pos_buffer.append(dof_pos)
                dof_vel_buffer.append(dof_vel)
                dof_torque_buffer.append(dof_torque)
                foot_force_buffer.append(foot_force)



                if abs(last_foot_force) < 50 and abs(msg.FTsensor_data.Fz[1]) > 50:
                    phase_count = 0
                else:
                    phase_count += 1

                phase_count_buffer.append(phase_count)

                # 通过mujoco计算foot_height
                mj_data.qpos[0:3] = root_pos_buffer[-1]
                mj_data.qpos[3:7] = root_eu_ang_buffer[-1][3:7]
                mj_data.qpos[7:7+26] = dof_pos_buffer[-1]
                mj_data.qvel[0:3] = root_lin_vel_buffer[-1]
                mj_data.qvel[3:6] = root_ang_vel_buffer[-1]
                mj_data.qvel[6:6+26] = dof_vel_buffer[-1]
                mujoco.mj_forward(model, mj_data)
                prit_body_list=[9,18]
                # for i in prit_body_list:
                #     print(f"当前body{i}: {mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,i)}")  # 打印当前body名称'')
                #     print(f"当前body_pose{i}:{mj_data.xpos[i]}")  # 打印当前body位置   
                foot_height_buffer.append([mj_data.xpos[9][2], mj_data.xpos[18][2]])
                if len(dof_pos_buffer)%500 == 0:
                    print(f"Recording data: {len(dof_pos_buffer)} frames")                
                        
        else:
            # if abs(msg.FTsensor_data.Fz[0] - msg.FTsensor_data.Fz[1]) < 10. and is_finish_recoding is False:
            if  abs(msg.joint_data.joint_q[3]  - msg.joint_data.joint_q[9]) < 0.001 and is_finish_recoding is False:
                is_recoding = True
                print(msg.FTsensor_data.Fz)
        
        count_sensorsData += 1
        last_foot_force = msg.FTsensor_data.Fz[1]

        if len(dof_pos_buffer) == final_length:
            is_recoding = False
            is_finish_recoding = True

            dof_pos_buffer = np.array(dof_pos_buffer[:final_length])
            dof_vel_buffer = np.array(dof_vel_buffer[:final_length])
            dof_torque_buffer = np.array(dof_torque_buffer[:final_length])
            foot_force_buffer = np.array(foot_force_buffer[:final_length])
            foot_height_buffer = np.array(foot_height_buffer[:final_length])

            root_pos_buffer = np.array(root_pos_buffer[:final_length])
            root_eu_ang_buffer = np.array(root_eu_ang_buffer[:final_length])
            root_ang_vel_buffer = np.array(root_ang_vel_buffer[:final_length])
            root_lin_vel_buffer = np.array(root_lin_vel_buffer[:final_length])
            phase_count_buffer = np.array(phase_count_buffer[:final_length])

            print(dof_pos_buffer.shape, dof_vel_buffer.shape, foot_force_buffer.shape, root_pos_buffer.shape, root_eu_ang_buffer.shape, root_ang_vel_buffer.shape, root_lin_vel_buffer.shape)

            ref_np = {"dof_pos":dof_pos_buffer,
                        "dof_vel":dof_vel_buffer,
                        "dof_torque":dof_torque_buffer,
                        "foot_force":foot_force_buffer,
                        "foot_height":foot_height_buffer,

                        "root_pos":root_pos_buffer,
                        "root_eu_ang":root_eu_ang_buffer,
                        "root_ang_vel":root_ang_vel_buffer,
                        "root_lin_vel":root_lin_vel_buffer}

            np.savez_compressed(f'npz_record/{file_name}.npz', **ref_np)
            print("data save finished")
            
            with open(f'npz_record/{file_name}.csv', mode='w', newline='') as csvfile:
                writer = csv.writer(csvfile)

                row_name = [name_ + 'pos' for name_ in dof_name] + \
                            [name_ + 'vel' for name_ in dof_name] + \
                            [name_ + 'torque' for name_ in dof_name] + \
                            ['foot_force_l', 'foot_force_r'] + \
                            ['foot_height_l', 'foot_height_r'] + \
                            ['root_pos_0', 'root_pos_1', 'root_pos_2'] + \
                            ['root_eu_quat_0', 'root_eu_quat_1', 'root_eu_quat_2', 'root_eu_quat_3', 'root_eu_ang_0', 'root_eu_ang_1', 'root_eu_ang_2'] + \
                            ['root_ang_vel_0', 'root_ang_vel_1', 'root_ang_vel_2'] + \
                            ['root_lin_vel_0', 'root_lin_vel_1', 'root_lin_vel_2', 'phase_count']
                            
                writer.writerow(row_name)

                for i in range(dof_pos_buffer.shape[0]):
                    writer.writerow(list(dof_pos_buffer[i]) + \
                                    list(dof_vel_buffer[i]) + \
                                    list(dof_torque_buffer[i]) + \
                                    list(foot_force_buffer[i]) + \
                                    list(foot_height_buffer[i]) + \
                                    list(root_pos_buffer[i]) + \
                                    list(root_eu_ang_buffer[i]) + \
                                    list(root_ang_vel_buffer[i]) + \
                                    list(root_lin_vel_buffer[i]) + \
                                    [phase_count_buffer[i]])
            
            rospy.signal_shutdown("csv save finished")    
    return callback_sensorsData                    


def smoothly_tile_clip(clip: np.ndarray, repeat_times: int) -> np.ndarray:
    """
    将 clip 沿时间维度重复拼接，但每次新段 clip 会被平移，使其首帧对齐上一段的末帧。
    
    :param clip: np.ndarray, shape (T, D)
    :param repeat_times: int, 重复次数
    :return: 拼接后 np.ndarray, shape (T * repeat_times, D)
    """
    clips = [clip]
    for i in range(1, repeat_times):
        prev_end = clips[-1][-1]         # 上一段的最后一帧
        next_start = clip[0]             # 新 clip 的第一帧
        offset = prev_end - next_start   # 计算所需平移量

        adjusted = clip + offset         # 对下一段整体做平移
        clips.append(adjusted)

    return np.vstack(clips)

def remove_spikes_and_interpolate(data, threshold=3.0):
    data = np.array(data)
    mean, std = np.mean(data), np.std(data)
    mask = np.abs(data - mean) < threshold * std
    valid_idx = np.where(mask)[0]
    interp = interp1d(valid_idx, data[valid_idx], kind='linear', fill_value="extrapolate")
    return interp(np.arange(len(data)))

def scaled_smooth_transition(data, scale=1.0, smooth=0.3):
    """
    对一维数据进行归一化 + 中间值缩放 + 平滑 + 反归一化。
    
    参数：
        data: 一维数组或列表
        scale: 中间值缩放因子（>1放大，<1压缩）
        smooth: 平滑程度（0~1，两端影响程度）
    
    返回：
        缩放和平滑处理后的一维数组，长度不变
    """
    data = np.array(data, dtype=np.float32)
    N = len(data)
    if N < 3:
        return data.copy()

    # Step 1: 归一化（到 [0, 1]）
    d_min_l = data[0]
    d_min_r = data[-1]
    d_max = np.max(data)
    max_index = np.argmax(data)
    if d_max - d_min_l < 1e-6:
        return data.copy()  # 避免除零
    if d_max - d_min_r < 1e-6:
        return data.copy()  # 避免除零

    data_norm_l = (data[0:max_index+1] - d_min_l) / (d_max - d_min_l)
    data_norm_r = (data[max_index:] - d_min_r) / (d_max - d_min_r)

    # Step 3: 计算缩放后值
    data_scaled_l = data_norm_l * scale
    data_scaled_r = data_norm_r * scale

    # Step 4: 反归一化
    data_result_l = data_scaled_l * (d_max - d_min_l) + d_min_l
    data_result_r = data_scaled_r * (d_max - d_min_r) + d_min_r
    mid_result = np.array([(data_result_l[-1]+data_result_r[0])/2])
    data_result = np.concatenate((data_result_l[:-1],mid_result,data_result_r[1:]))
    data_result = smooth_nd_trajectory(data_result,0,5)

    return data_result

# 开启录制 可传入文件名参数
def topic_record(file_name='record_1',final_length=3000,record_gap=10):
    # 确保目录存在
    os.makedirs('npz_record', exist_ok=True)      
    # 初始化 ROS 节点
    rospy.init_node('tf_listener', anonymous=True)
    print("Listening to /sensors_data_raw and /ground_truth/state topics...")     
    # 订阅 TFMessage 类型的 topic
    rospy.Subscriber('/ground_truth/state', Odometry, make_callback_Odometry(record_gap))    
    time.sleep(0.1)
    rospy.Subscriber('/sensors_data_raw', sensorsData, make_callback_sensorsData(file_name,final_length,record_gap))
    # 保持节点运行
    rospy.spin()

# 使用 mujoco 播放录制的数据
# play_fiile_name: npz文件名
def mujoco_play(play_fiile_name):  
    viewer = mujoco_viewer.MujocoViewer(model, mj_data)         
    # 加载MPC数据  
    print(f"Loading data from {play_fiile_name}...")  
    data = dict(np.load(play_fiile_name, allow_pickle=True))  
    print(f"Data keys: {list(data.keys())}")
    
    # 方法1：直接修改  
    # mask = data['dof_pos'][:, [3, 9]] < 0.7  
    # data['dof_pos'][:, [3, 9]] = np.where(mask,   
    #                                     data['dof_pos'][:, [3, 9]] -0.1,   
    #                                     data['dof_pos'][:, [3, 9]]) 
  
    # 检查数据格式  
    required_keys = ['dof_pos', 'root_pos', 'root_eu_ang']  
    for key in required_keys:  
        if key not in data:  
            print(f"Warning: {key} not found in data file")  
       
    # 获取数据长度  
    data_length = len(data['dof_pos']) if 'dof_pos' in data else 1000  
    print(f"Data contains {data_length} frames")  

    # 设置摄像头注视点为机器人初始位置
    if 'root_pos' in data and len(data['root_pos']) > 0:
        viewer.cam.lookat[:] = data['root_pos'][0]  # 注视点为第一帧的根部位置
    else:
        viewer.cam.lookat[:] = [0, 0, 1.0]  # 默认注视点    

    try:  
        # 播放数据  
        for i in tqdm(range(data_length), desc="Playing motion data"):  
              
            # 设置关节位置 (假设前7个qpos是根部状态，后面是关节)  
            if 'dof_pos' in data:  
                # 根据您的机器人模型调整索引，这里假设有12个关节  
                joint_start_idx = 7  # 跳过根部的7个自由度 (3位置 + 4四元数)  
                joint_end_idx = joint_start_idx + len(data['dof_pos'][i])  
                mj_data.qpos[joint_start_idx:joint_end_idx] = data['dof_pos'][i]  
              
            # 设置根部位置  
            if 'root_pos' in data:  
                mj_data.qpos[0:3] = data['root_pos'][i]  
              
            # 设置根部姿态 - 确保四元数格式正确  
            if 'root_eu_ang' in data:  
                if len(data['root_eu_ang'][i]) >= 7:  
                    # 使用四元数部分 [x,y,z,w]  
                    mj_data.qpos[3] = data['root_eu_ang'][i][3]  # qx  
                    mj_data.qpos[4] = data['root_eu_ang'][i][0]  # qy    
                    mj_data.qpos[5] = data['root_eu_ang'][i][1]  # qz  
                    mj_data.qpos[6] = data['root_eu_ang'][i][2]  # qw  
              
            # 设置关节速度  
            if 'dof_vel' in data:  
                joint_vel_start = 6  # 跳过根部的6个速度自由度  
                joint_vel_end = joint_vel_start + len(data['dof_vel'][i])  
                mj_data.qvel[joint_vel_start:joint_vel_end] = data['dof_vel'][i]  
              
            # 设置根部线速度  
            if 'root_lin_vel' in data:  
                mj_data.qvel[0:3] = data['root_lin_vel'][i]  
              
            # 设置根部角速度  
            if 'root_ang_vel' in data:  
                mj_data.qvel[3:6] = data['root_ang_vel'][i]  
              
            # 前向运动学计算  
            mujoco.mj_forward(model, mj_data)  
              
            # 渲染  
            viewer.render()  
            prit_body_list=[9,18]
            # for i in prit_body_list:
            #     print(f"当前body{i}: {mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,i)}")  # 打印当前body名称'')
            #     print(f"当前body_pose{i}:{mj_data.xpos[i]}")  # 打印当前body位置            
              
            # 控制播放速度 (10ms间隔)  
            time.sleep(0.01)  
              
            # 检查是否需要退出  
            if viewer.is_alive == False:  
                break  
                  
    except KeyboardInterrupt:  
        print("\nPlayback interrupted by user")  
      
    finally:  
        viewer.close()  
        print("Playback finished")      

# 显示 npz 数据
def npz_show(npz_file_names,show_dof_list,diff_r_l,plot_start,plot_end,plot_type=2):
    data = []
    for npz_file_name in npz_file_names:
        print(f"载入数据: {npz_file_name}")
        data_temp = dict(np.load(npz_file_name, allow_pickle=True))
        print(f"数据键: {list(data_temp.keys())}")
        data.append(data_temp)        
      
    global ref_phase_dof_pos, ref_phase_dof_vel, ref_phase_dof_torque, ref_phase_foot_force, ref_phase_foot_height, ref_phase_root_pos, ref_phase_root_eu_ang, ref_phase_root_ang_vel, ref_phase_root_lin_vel,energy_dof
    ref_phase_dof_pos = []
    ref_phase_dof_vel = []
    ref_phase_dof_torque = []
    ref_phase_foot_force = []
    ref_phase_foot_height = []
    ref_phase_root_pos = []
    ref_phase_root_eu_ang = []
    ref_phase_root_ang_vel = []
    ref_phase_root_lin_vel = []
    energy_dof = []    
    for data_np in data:
        ref_phase_dof_pos.append(data_np["dof_pos"][:, :])
        ref_phase_dof_vel.append(data_np["dof_vel"][:, :])
        # ref_phase_dof_torque.append(data_np["dof_torque"][:, :])
        ref_phase_foot_force.append(data_np["foot_force"][:, :])
        ref_phase_foot_height.append(data_np["foot_height"][:, :])
        ref_phase_root_pos.append(data_np["root_pos"][:, :])
        ref_phase_root_eu_ang.append(data_np["root_eu_ang"][:, :])
        ref_phase_root_ang_vel.append(data_np["root_ang_vel"][:, :])
        ref_phase_root_lin_vel.append(data_np["root_lin_vel"][:, :])

    print(f"show数据长度=>\n ref_phase_dof_pos:{np.shape(ref_phase_dof_pos)},\n ref_phase_dof_vel:{np.shape(ref_phase_dof_vel)},\n ref_phase_foot_force:{np.shape(ref_phase_foot_force)},\n ref_phase_foot_height:{np.shape(ref_phase_foot_height)},\n ref_phase_root_pos:{np.shape(ref_phase_root_pos)},\n ref_phase_root_eu_ang:{np.shape(ref_phase_root_eu_ang)},\n ref_phase_root_ang_vel:{np.shape(ref_phase_root_ang_vel)},\n ref_phase_root_lin_vel:{np.shape(ref_phase_root_lin_vel)}")
    axs_row = 4
    axs_col = 6

    if plot_type==0 or plot_type==2:
    # 使用matplotlib绘图        
        fig, axs = plt.subplots(axs_row, axs_col, figsize=(25, 10))
        fig.canvas.manager.set_window_title("Figure1 - 关节dof_pos")
        for j in range(len(data)):
            for i in range(axs_col):
                axs[0, i].plot(ref_phase_dof_pos[j][plot_start:plot_end, i], label=f'{j}_dof_l_leg_{i}')
                if diff_r_l:
                    axs[0, i].plot(ref_phase_dof_pos[j][plot_start:plot_end, 6+i], label=f'{j}_dof_r_leg_{i}')                
                axs[0, i].legend()
                axs[0, i].grid(True) 
            for i in range(axs_col):
                axs[1, i].plot(ref_phase_dof_pos[j][plot_start:plot_end, 6+i], label=f'{j}_dof_r_leg_{i}')
                axs[1, i].legend()
                axs[1, i].grid(True) 
            for i in range(axs_col):
                axs[2, i].plot(ref_phase_dof_pos[j][plot_start:plot_end, 12+i], label=f'{j}_dof_l_arm_{i}')
                if diff_r_l:
                    axs[2, i].plot(ref_phase_dof_pos[j][plot_start:plot_end, 19+i], label=f'{j}_dof_r_arm_{i}')                
                axs[2, i].legend()
                axs[2, i].grid(True) 
            for i in range(axs_col):
                axs[3, i].plot(ref_phase_dof_pos[j][plot_start:plot_end, 19+i], label=f'{j}_dof_r_arm_{i}')
                axs[3, i].legend()
                axs[3, i].grid(True) 
        plt.tight_layout()
        fig, axs = plt.subplots(axs_row, axs_col, figsize=(25, 10))
        fig.canvas.manager.set_window_title("Figure2 - 关节dof_vel")
        for j in range(len(data)):
            for i in range(axs_col):
                axs[0, i].plot(ref_phase_dof_vel[j][plot_start:plot_end, i], label=f'{j}_dof_l_leg_{i}')
                if diff_r_l:
                    axs[0, i].plot(ref_phase_dof_vel[j][plot_start:plot_end, 6+i], label=f'{j}_dof_r_leg_{i}')                
                axs[0, i].legend()
                axs[0, i].grid(True) 
            for i in range(axs_col):
                axs[1, i].plot(ref_phase_dof_vel[j][plot_start:plot_end, 6+i], label=f'{j}_dof_r_leg_{i}')
                axs[1, i].legend()
                axs[1, i].grid(True) 
            for i in range(axs_col):
                axs[2, i].plot(ref_phase_dof_vel[j][plot_start:plot_end, 12+i], label=f'{j}_dof_l_arm_{i}')
                if diff_r_l:
                    axs[2, i].plot(ref_phase_dof_vel[j][plot_start:plot_end, 19+i], label=f'{j}_dof_r_arm_{i}')                   
                axs[2, i].legend()
                axs[2, i].grid(True) 
            for i in range(axs_col):
                axs[3, i].plot(ref_phase_dof_vel[j][plot_start:plot_end, 19+i], label=f'{j}_dof_r_arm_{i}')
                axs[3, i].legend()
                axs[3, i].grid(True) 
        plt.tight_layout()    
        fig, axs = plt.subplots(axs_row, axs_col, figsize=(25, 10))
        fig.canvas.manager.set_window_title("Figure3 - 其余显示")
        for j in range(len(data)):
            for i in range(2):
                axs[0, i].plot(ref_phase_foot_force[j][plot_start:plot_end, i], label=f'{j}_foot_force_{i}')
                if diff_r_l and i==0:
                    axs[0, 0].plot(ref_phase_foot_force[j][plot_start:plot_end, 1], label=f'{j}_foot_force_{i}')                  
                axs[0, i].legend()
                axs[0, i].grid(True) 
            for i in range(2):
                axs[0, 2+i].plot(ref_phase_foot_height[j][plot_start:plot_end, i], label=f'{j}_foot_height_{i}')
                if diff_r_l and i==0:
                    axs[0, 2+0].plot(ref_phase_foot_height[j][plot_start:plot_end, 1], label=f'{j}_foot_height_{i}')
                axs[0, 2+i].legend()
                axs[0, 2+i].grid(True) 
            for i in range(3):
                axs[1, i].plot(ref_phase_root_pos[j][plot_start:plot_end, i], label=f'{j}_root_pos_{i}')
                axs[1, i].legend()
                axs[1, i].grid(True) 
            for i in range(3):
                axs[2, i].plot(ref_phase_root_eu_ang[j][plot_start:plot_end, 4+i], label=f'{j}_root_eu_ang_{i}')
                axs[2, i].legend()
                axs[2, i].grid(True) 
            for i in range(3):
                axs[3, i].plot(ref_phase_root_ang_vel[j][plot_start:plot_end, i], label=f'{j}_root_ang_vel_{i}')
                axs[3, i].legend()
                axs[3, i].grid(True) 
            for i in range(3):
                axs[3, 3+i].plot(ref_phase_root_lin_vel[j][plot_start:plot_end, i], label=f'{j}_root_lin_vel_{i}')
                axs[3, 3+i].legend()
                axs[3, 3+i].grid(True)             
        plt.tight_layout()        
        plt.show()

    if plot_type==1 or plot_type==2:
        # 使用plotly绘图，只绘制传入的第一个npz数据
        x_data = np.arange(plot_end-plot_start)  # 索引为 x 轴
        # 创建图形
        fig = make_subplots(rows=4, cols=6,vertical_spacing=0.05,horizontal_spacing=0.03,
            subplot_titles=['pos_x', 'pos_y', 'pos_z','vel_x','vel_y','vel_z',
                            'ang_x','ang_y','ang_z','ang_vel_x','ang_vel_y','ang_vel_z',
                            'dof_pos_rl', 'dof_pos_rl','dof_pos_rl','dof_pos_rl','dof_pos_rl', 'foot_height',
                            'dof_vel_rl','dof_vel_rl','dof_vel_rl','dof_vel_rl','dof_vel_rl',
        ])
        for j in range(len(data)):
            # 添加折线图
            for i in range(3):            
                fig.add_trace(go.Scatter(
                    x=x_data, 
                    y=ref_phase_root_pos[j][plot_start:plot_end, i],
                    mode='lines',
                    name=f'pos_{i}',
                ),row=1, col=1+i) 
            for i in range(3):            
                fig.add_trace(go.Scatter(
                    x=x_data, 
                    y=ref_phase_root_lin_vel[j][plot_start:plot_end, i],
                    mode='lines',
                    name=f'vel_{i}',
                ),row=1, col=4+i) 
            # 添加折线图
            for i in range(3):            
                fig.add_trace(go.Scatter(
                    x=x_data, 
                    y=ref_phase_root_eu_ang[j][plot_start:plot_end, i],
                    mode='lines',
                    name=f'ang_{i}',
                ),row=2, col=1+i) 
            for i in range(3):            
                fig.add_trace(go.Scatter(
                    x=x_data, 
                    y=ref_phase_root_ang_vel[j][plot_start:plot_end, i],
                    mode='lines',
                    name=f'ang_vel_{i}',
                ),row=2, col=4+i)                      
            
            for index,dof_ids in enumerate(show_dof_list):
                for dof_id in dof_ids:
                    for data_key, row in [('dof_pos', 3), ('dof_vel', 4)]:
                        data = ref_phase_dof_pos if data_key == 'dof_pos' else ref_phase_dof_vel
                        y = data[j][plot_start:plot_end, dof_id]
                        max_idx = argrelextrema(y, np.greater)[0]
                        min_idx = argrelextrema(y, np.less)[0]

                        # 主线条
                        fig.add_trace(go.Scatter(
                            x=x_data,
                            y=y,
                            mode='lines',
                            name=f'{data_key}_{dof_id}',
                        ), row=row, col=index+1)

                        # 极大值点（红色）
                        fig.add_trace(go.Scatter(
                            x=x_data[max_idx],
                            y=y[max_idx],
                            mode='markers',
                            marker=dict(color='red', size=8, symbol='diamond'),
                            name='max'
                        ), row=row, col=index+1)

                        # 极小值点（绿色）
                        fig.add_trace(go.Scatter(
                            x=x_data[min_idx],
                            y=y[min_idx],
                            mode='markers',
                            marker=dict(color='green', size=8, symbol='diamond'),
                            name='min'
                        ), row=row, col=index+1)
            
        # 单独绘制dof_pos_rl_5


        # 绘制height_rl
        for side, label in enumerate(['foot_height_l', 'foot_height_r']):
            y = ref_phase_foot_height[j][plot_start:plot_end, side]
            fig.add_trace(go.Scatter(
                x=x_data,
                y=y,
                mode='lines',
                name=label,
            ), row=4, col=6)

            max_idx = argrelextrema(y, np.greater)[0]
            min_idx = argrelextrema(y, np.less)[0]

            fig.add_trace(go.Scatter(
                x=x_data[max_idx],
                y=y[max_idx],
                mode='markers',
                marker=dict(color='red', size=8, symbol='diamond'),
                name='max'
            ), row=4, col=6)

            fig.add_trace(go.Scatter(
                x=x_data[min_idx],
                y=y[min_idx],
                mode='markers',
                marker=dict(color='green', size=8, symbol='diamond'),
                name='min'
            ), row=4, col=6)


        # 设置图表标题和轴标签
        fig.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),  # 左右上下边距
            hovermode='x unified',
        
        )    
        pio.renderers.default = 'browser'
        fig.show()           

def smooth_nd_trajectory(points: np.ndarray, alpha: float = 0.5, repeat: int = 5):
    """
    对 nD 轨迹进行平滑处理，适用于帧数少的情况。
    :param points: np.ndarray, shape (T, D)
    :param alpha: 平滑权重，0-1，越大越靠近原始数据
    :param repeat: 平滑次数
    :return: np.ndarray, shape (T, D)
    """
    points = points.copy()
    smoothed = points.copy()

    for _ in range(repeat):
        for t in range(1, len(points) - 1):
            smoothed[t] = (
                alpha * points[t] +
                (1 - alpha) * 0.5 * (smoothed[t - 1] + smoothed[t + 1])
            )
    return smoothed

def npz_clip(clip_start,clip_end,is_to_train,is_mirror,final_length,rebuild_vel,save_csv,npz_clip_origin,npz_clip_file_name='ref_clip_date'):
    """
    从全局ref_phase_*变量中裁剪clip_start:clip_end区间，并重复拼接到final_length，保存为npz
    """
    data = []
    for npz_file_name in npz_clip_origin:
        print(f"载入数据: {npz_file_name}")
        data_temp = dict(np.load(npz_file_name, allow_pickle=True))
        print(f"数据键: {list(data_temp.keys())}")
        data.append(data_temp)        
      
    global ref_phase_dof_pos, ref_phase_dof_vel, ref_phase_dof_torque, ref_phase_foot_force, ref_phase_foot_height, ref_phase_root_pos, ref_phase_root_eu_ang, ref_phase_root_ang_vel, ref_phase_root_lin_vel,energy_dof
    for data_np in data:
        ref_phase_dof_pos.append(data_np["dof_pos"][:, :])
        ref_phase_dof_vel.append(data_np["dof_vel"][:, :])
        # ref_phase_dof_torque.append(data_np["dof_torque"][:, :])
        ref_phase_foot_force.append(data_np["foot_force"][:, :])
        ref_phase_foot_height.append(data_np["foot_height"][:, :])
        ref_phase_root_pos.append(data_np["root_pos"][:, :])
        ref_phase_root_eu_ang.append(data_np["root_eu_ang"][:, :])
        ref_phase_root_ang_vel.append(data_np["root_ang_vel"][:, :])
        ref_phase_root_lin_vel.append(data_np["root_lin_vel"][:, :])
                             
    # 只处理第一个文件（ref_phase_*[0]），如需批量处理可循环
    data_dict = {}
    keys = [
        ("dof_pos", ref_phase_dof_pos),
        ("dof_vel", ref_phase_dof_vel),
        # ("dof_torque", ref_phase_dof_torque),
        ("foot_force", ref_phase_foot_force),
        ("foot_height", ref_phase_foot_height),
        ("root_pos", ref_phase_root_pos),
        ("root_eu_ang", ref_phase_root_eu_ang),
        ("root_ang_vel", ref_phase_root_ang_vel),
        ("root_lin_vel", ref_phase_root_lin_vel),
    ]
    for name, arr_list in keys:
        if len(arr_list) == 0:
            continue
        print(f"正在处理: {name}，数据长度: {len(arr_list[0])}")
        arr = arr_list[0]  # 只取第一个npz文件
        clip = arr[clip_start:clip_end]

        if name not in ['root_pos']:
            # 加入轨迹平滑处理
            clip_smooth = np.roll(clip,shift=-5,axis=0)  # 向前平移5帧
            clip_smooth[-6:-1] = smooth_nd_trajectory(clip_smooth[-6:-1], alpha=0, repeat=10)
            clip_smooth = np.roll(clip_smooth,shift=5,axis=0)  # 向后平移5帧回原位置
            clip = clip_smooth        
        # 加入左右镜像操作
        if is_mirror:
            mirror_step = int((clip_end-clip_start)/2)  # 镜像步长
            if name == 'dof_pos':
                # 镜像之前，做数据微调，改一半就好了
                # clip[0:10,3] = scaled_smooth_transition(clip[0:10,3], scale=0.8, smooth=0) #降低膝盖关节角度
                # clip[10:57,3] = scaled_smooth_transition(clip[10:57,3], scale=0.6, smooth=0) #降低膝盖关节角度
                # clip[:,1] = 0 #将髋关节的yaw置为0
                # clip[0:81,0] = smooth_nd_trajectory(clip[0:81,0], 0, 5)
                # clip[110:,4] = smooth_nd_trajectory(clip[110:,4], 0, 5)
                # clip[15:30,5] = smooth_nd_trajectory(clip[15:30,5], 0, 5)
                # 微调结束
                clip[:,6:12] = np.roll(clip[:,0:6], shift=mirror_step, axis=0)  # 镜像右腿
                clip[:,[6,7,11]] = -clip[:,[6,7,11]]  # 镜像右脚
                clip[:,19:25] = np.roll(clip[:,12:18], shift=mirror_step, axis=0)  # 镜像右臂
                clip[:,[20,21,23]] = -clip[:,[20,21,23]]  # 镜像右手
            elif name == 'dof_vel':
                clip[:,6:12] = np.roll(clip[:,0:6], shift=mirror_step, axis=0)  # 镜像右腿
                clip[:,[6,7,11]] = -clip[:,[6,7,11]]  # 镜像右脚
                clip[:,19:25] = np.roll(clip[:,12:18], shift=mirror_step, axis=0)  # 镜像右臂
                clip[:,[20,21,23]] = -clip[:,[20,21,23]]  # 镜像右手
            elif name == 'foot_force':
                clip[:,1] = np.roll(clip[:,0], shift=mirror_step, axis=0)  # 镜像右脚
            elif name == 'foot_height':
                clip[:,1] = np.roll(clip[:,0], shift=mirror_step, axis=0)  # 镜像右脚
            elif name == 'root_pos':
                clip[mirror_step:mirror_step*2,2] = clip[0:mirror_step,2]
            # elif name == 'root_eu_ang':
            #     clip[mirror_step:mirror_step*2] = clip[0:mirror_step]
            # elif name == 'root_ang_vel':
            #     clip[mirror_step:mirror_step*2,:] = clip[0:mirror_step,:]
            elif name == 'root_lin_vel':    
                clip[mirror_step:mirror_step*2,2] = clip[0:mirror_step,2]


        clip_len = clip.shape[0]
        repeat_times = int(np.ceil((final_length + clip_start) / clip_len))  # 注意要加上右移量
        if name in ['root_pos']:
            xy = smoothly_tile_clip(clip[:,:2],repeat_times)
            z = np.tile(clip[:, 2:3], (repeat_times, 1))        # (repeat_times*clip_len, 1)
            tiled = np.concatenate([xy, z], axis=1) 
        else:
            tiled = np.tile(clip, (repeat_times, 1))
        
        # 🔁 右移 clip_start 个位置，使 clip 对齐回原位置
        # shifted = np.roll(tiled, shift=clip_start, axis=0)
        
        # ✂️ 截取最终长度
        new_arr = tiled[:final_length]         
        print(f"{name} 裁剪并拼接后的数据长度: {new_arr.shape}")
        data_dict[name] = new_arr

    # 对数据进行微调修改    
    if 'foot_height' in data_dict:
        data_dict['foot_height'][data_dict['foot_height'] < 0.01] = 0.01

    # if 'root_pos' in data_dict:
    #     data_dict['root_pos'][:,2]+=0.855

    if is_to_train:
        data_dict['root_pos'][:,2] -= data_dict['root_pos'][:, 2].min()

    if rebuild_vel:
        # 重构速度信息
        qpos_list = []
        qvel_list = []
        dt = 0.01   #采样时间    
        for i in range(len(data_dict['dof_pos'])):
            root_pos = data_dict['root_pos'][i]        # shape=(3,)
            quat = data_dict['root_eu_ang'][i,:4]           # shape=(4,)
            dof_pos = data_dict['dof_pos'][i]          # shape=(n_dof,)
            qpos = np.concatenate([root_pos, quat, dof_pos])
            qpos_list.append(qpos)
        for i in range(len(qpos_list) - 1):
            q0 = qpos_list[i]
            q1 = qpos_list[i+1]

            base_lin_vel = (q1[0:3] - q0[0:3]) / dt
            base_ang_vel = quat_to_angvel(q0[3:7], q1[3:7], dt)
            dof_vel = (q1[7:] - q0[7:]) / dt
            qvel = np.concatenate([base_lin_vel, base_ang_vel, dof_vel])
            if (i+1)%(clip_end-clip_start)==0 and i!=0 :
                qvel_list.append(qvel_list[i-1])
            else:
                qvel_list.append(qvel)         
        # 复制最后一帧，更新速度
        qvel_list.append(qvel_list[-1])
        qvel_list = np.array(qvel_list)
        # data_dict['root_lin_vel'] = qvel_list[:,0:3]
        data_dict['root_lin_vel'] = savgol_filter(qvel_list[:,0:3],window_length=15, polyorder=2, axis=0)
        data_dict['root_ang_vel'] = savgol_filter(qvel_list[:,3:6],window_length=15, polyorder=2, axis=0)
        data_dict['dof_vel'] = savgol_filter(qvel_list[:,6:],window_length=9, polyorder=2, axis=0) 

    save_path = f'npz_record/{npz_clip_file_name}.npz'
    # 如果文件已存在，先删除
    if os.path.exists(save_path):
        os.remove(save_path)
    # 保存数据
    np.savez_compressed(save_path, **data_dict)    
    print(f"已保存裁剪并拼接后的npz文件: {npz_clip_file_name}.npz")
    # 保存为csv
    if save_csv:
        csv_file_name = f'npz_record/{npz_clip_file_name}.csv'
        row_name = [name_ + 'pos' for name_ in dof_name] + \
                [name_ + 'vel' for name_ in dof_name] + \
                ['foot_force_l', 'foot_force_r'] + \
                ['foot_height_l', 'foot_height_r'] + \
                ['root_pos_0', 'root_pos_1', 'root_pos_2'] + \
                ['root_eu_quat_0', 'root_eu_quat_1', 'root_eu_quat_2', 'root_eu_quat_3', 'root_eu_ang_0', 'root_eu_ang_1', 'root_eu_ang_2'] + \
                ['root_ang_vel_0', 'root_ang_vel_1', 'root_ang_vel_2'] + \
                ['root_lin_vel_0', 'root_lin_vel_1', 'root_lin_vel_2']

        with open(csv_file_name, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row_name)
            for i in range(final_length):
                row = []
                row += list(data_dict['dof_pos'][i]) if 'dof_pos' in data_dict else []
                row += list(data_dict['dof_vel'][i]) if 'dof_vel' in data_dict else []
                row += list(data_dict['foot_force'][i]) if 'foot_force' in data_dict else []
                row += list(data_dict['foot_height'][i]) if 'foot_height' in data_dict else []
                row += list(data_dict['root_pos'][i]) if 'root_pos' in data_dict else []
                row += list(data_dict['root_eu_ang'][i]) if 'root_eu_ang' in data_dict else []
                row += list(data_dict['root_ang_vel'][i]) if 'root_ang_vel' in data_dict else []
                row += list(data_dict['root_lin_vel'][i]) if 'root_lin_vel' in data_dict else []
                writer.writerow(row)
        print(f"已保存csv文件: {csv_file_name}")    
  
def quat_to_angvel(q1, q2, dt):
    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)
    delta_r = r2 * r1.inv()
    rotvec = delta_r.as_rotvec()
    return rotvec / dt  # approximate angular velocity in body frame

def main():
    # 检查文件路径
    # current_path = os.getcwd()    #/kuavo-RL
    # print("current_path:", current_path)    
    # 定义变量
    record_file_name = 'ref_clip_250722_1'  # 可以根据需要修改文件名
    play_file_name = 'humanoid/mpc_pose/play_cmu_0401_b.npz'  # 可以根据需要修改文件名   play_walk_0507_06ms_b
    npz_show_names = ['npz_record/ref_clip_250717_06m.npz'] #可以放多个文件做对比   
    npz_clip_origin = ['npz_record/ref_clip_250722_1.npz'] # clip原始npz文件，只能一个
    npz_clip_save = 'ref_clip_250717_08m'  # 可以根据需要修改文件名
    # 启动 ROS 监听器   
    # topic_record(record_file_name,final_length=2000,record_gap=12)  # 启动数据录制，默认长度7200，默认间隔是10
    # 裁减并扩充数据
    # is_mirror=0: 不进行左右镜像处理，is_mirror=1: 进行左右镜像处理(将左脚数据复制到右脚)
    # is_to _train = 1,处理成用作训练的数据，将pos_z减去pos_z_min
    # npz_clip(1440,1540,is_to_train=0,is_mirror=1,final_length=7200,rebuild_vel=True,save_csv=False,npz_clip_origin=npz_clip_origin,npz_clip_file_name=npz_clip_save)
    # 绘制数据
    # plot_type=0: matplotlib, plot_type=1: plotly, plot_type=2: both
    # diff_r_l=0: 不对比左右脚区别，diff_r_l=1: 将右脚数据绘制在左脚数据plot上，仅matplotlib绘图
    show_dof_list=[[0,6],[1,7],[2,8],[3,9],[4,10],[5,11]]
    # npz_show(npz_show_names,show_dof_list,diff_r_l=0,plot_start=0,plot_end=300,plot_type=1)
    # 使用mujoco播放录制数据
    mujoco_play(play_file_name)


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        print("ROS Interrupt Exception")