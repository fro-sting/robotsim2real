# 机器人悬挂训练模式

## 概述
这个模式将机器人上半身（base_link）固定在空中，只允许下半身（腿部）活动，用于专门训练腿部控制策略。

## 实现方案

### 方案一：直接修改kuavo_config.py（已完成）
最简单的方法，直接设置`fix_base_link = True`：
- ✅ 已修改：`kuavo_config.py` 中 `asset.fix_base_link = True`
- 适用场景：快速测试悬挂效果

### 方案二：使用专用悬挂配置（推荐）
创建了独立的配置文件`kuavo_suspended_config.py`：
- ✅ 固定base_link在1.2米高度
- ✅ 禁用base运动相关的奖励
- ✅ 禁用推力和扰动
- ✅ 使用平面地形
- ✅ 增强关节跟踪奖励

## 使用方法

### 1. 使用方案一（修改后的原始配置）
```bash
cd /home/wegg/kuavo_rl_asap-main/RL_train
python train.py --task=kuavo_ppo
```

### 2. 使用方案二（推荐 - 悬挂专用配置）
```bash
cd /home/wegg/kuavo_rl_asap-main/RL_train
python train_suspended.py
```

或者使用标准训练脚本：
```bash
python train.py --task=kuavo_suspended
```

## 配置说明

### 关键参数

1. **悬挂高度** (`init_state.pos[2]`)
   - 默认：1.2米
   - 可调整范围：0.8-2.0米
   - 确保腿部可以自由活动而不触地

2. **固定base_link** (`asset.fix_base_link`)
   - 设置为`True`即可固定上半身

3. **命令范围** (`commands.ranges`)
   - 悬挂模式下全部设为0（因为base不能移动）

4. **奖励权重调整** (`rewards.scales`)
   - `tracking_lin_vel = 0.0`：不追踪线速度
   - `tracking_ang_vel = 0.0`：不追踪角速度
   - `imition_Leg_joint_positions = 8.0`：增强腿部跟踪

## 训练效果

### 优点
- ✅ 专注训练腿部协调性
- ✅ 不受平衡问题影响
- ✅ 可以更快学习步态
- ✅ 便于调试和可视化

### 适用场景
- 🎯 初期腿部动作学习
- 🎯 步态模式研究
- 🎯 关节控制调试
- 🎯 快速验证奖励函数

## 恢复正常训练

如果要恢复正常的全身训练：

### 方案一：修改回原配置
编辑 `kuavo_config.py`：
```python
fix_base_link = False  # 改回False
```

### 方案二：使用原始任务
```bash
python train.py --task=kuavo_ppo --load_run=your_suspended_model --checkpoint=xxx
```

## 可视化

运行训练时会自动显示Isaac Gym窗口，你可以看到：
- 机器人上半身固定在空中
- 腿部自由活动
- 可以用鼠标旋转视角观察

## 高级调整

### 1. 调整悬挂高度
编辑 `kuavo_suspended_config.py`：
```python
class init_state(KuavoCfg.init_state):
    pos = [0.0, 0.0, 1.5]  # 改变高度值
```

### 2. 添加轻微扰动（可选）
```python
class domain_rand(KuavoCfg.domain_rand):
    disturbance = True
    disturbance_range = [-100.0, 100.0]  # 小扰动
```

### 3. 部分固定关节
如果只想固定某些自由度，可以在环境中修改：
```python
# 在kuavo_env.py的create_sim中添加
# 只固定xyz位置，但允许旋转
```

## 故障排除

### 问题1：腿部触地
- 增加 `init_state.pos[2]` 的值

### 问题2：训练不收敛
- 检查奖励权重是否合理
- 确认观测空间是否包含必要信息

### 问题3：导入错误
- 运行 `python setup.py develop` 重新安装包

## 文件列表

创建/修改的文件：
- ✅ `kuavo_config.py` - 修改了fix_base_link
- ✅ `kuavo_suspended_config.py` - 新增悬挂专用配置
- ✅ `train_suspended.py` - 新增训练脚本
- ✅ `humanoid/envs/__init__.py` - 注册新任务
- ✅ `SUSPENDED_TRAINING.md` - 本文档
