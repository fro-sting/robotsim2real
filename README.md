# Humanoid-Gym: Reinforcement Learning for Humanoid Robot with Zero-Shot Sim2Real Transfer

**[[Project Page]](https://sites.google.com/view/humanoid-gym/)**

![Demo](./images/demo.gif)

Welcome to our Humanoid-Gym!

Humanoid-Gym is an easy-to-use reinforcement learning (RL) framework based on Nvidia Isaac Gym, designed to train locomotion skills for humanoid robots, emphasizing zero-shot transfer from simulation to the real-world environment. Humanoid-Gym also integrates a sim-to-sim framework from Isaac Gym to Mujoco that allows users to verify the trained policies in different physical simulations to ensure the robustness and generalization of the policies.

This codebase is verified by RobotEra's XBot-S (1.2 meter tall humanoid robot) and XBot-L (1.65 meter tall humanoid robot) in real-world environment with zero-shot sim-to-real transfer.

## Features

### 1. Humanoid Robot Training
This repository offers comprehensive guidance and scripts for the training of humanoid robots. Humanoid-Gym features specialized rewards for humanoid robots, simplifying the difficulty of sim-to-real transfer. In this repository, we use RobotEra's XBot-L as a primary example. It can also be used for other robots with minimal adjustments. Our resources cover setup, configuration, and execution. Our goal is to fully prepare the robot for real-world locomotion by providing in-depth training and optimization.


- **Comprehensive Training Guidelines**: We offer thorough walkthroughs for each stage of the training process.
- **Step-by-Step Configuration Instructions**: Our guidance is clear and succinct, ensuring an efficient setup process.
- **Execution Scripts for Easy Deployment**: Utilize our pre-prepared scripts to streamline the training workflow.

### 2. Sim2Sim Support
We also share our sim2sim pipeline, which allows you to transfer trained policies to highly accurate and carefully designed simulated environments. Once you acquire the robot, you can confidently deploy the RL-trained policies in real-world settings.

Our simulator settings, particularly with Mujoco, are finely tuned to closely mimic real-world scenarios. This careful calibration ensures that the performances in both simulated and real-world environments are closely aligned. This improvement makes our simulations more trustworthy and enhances our confidence in their applicability to real-world scenarios.


### 3. Denoising World Model Learning (Coming Soon!)
Denoising World Model Learning(DWL) presents an advanced sim-to-real framework that integrates state estimation and system identification. This dual-method approach ensures the robot's learning and adaptation are both practical and effective in real-world contexts.

- **Enhanced Sim-to-real Adaptability**: Techniques to optimize the robot's transition from simulated to real environments.
- **Improved State Estimation Capabilities**: Advanced tools for precise and reliable state analysis.

### Dexterous Hand Manipulation (Coming Soon!)

## Installation

1. Generate a new Python virtual environment with Python 3.8 using `conda create -n myenv python=3.8`.
2. For the best performance, we recommend using NVIDIA driver version 525 `sudo apt install nvidia-driver-525`. The minimal driver version supported is 515. If you're unable to install version 525, ensure that your system has at least version 515 to maintain basic functionality.
3. Install PyTorch 1.13 with Cuda-11.7:
   - `conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia`
4. Install numpy-1.23 with `conda install numpy=1.23`.
5. Install Isaac Gym:
   - Download and install Isaac Gym Preview 4 from https://developer.nvidia.com/isaac-gym.
   - `cd isaacgym/python && pip install -e .`
   - Run an example with `cd examples && python 1080_balls_of_solitude.py`.
   - Consult `isaacgym/docs/index.html` for troubleshooting.
6. Install humanoid-gym:
   - Clone this repository.
   - `cd humanoid_gym && pip install -e .`



## Usage Guide

#### Examples

```bash
# Launching PPO Policy Training for 'v1' Across 8192 Environments
# This command initiates the PPO algorithm-based training for the humanoid task.
python scripts/train.py --task=kuavo_ppo --run_name v1 --headless --num_envs 8192

# Evaluating the Trained PPO Policy 'v1'
# This command loads the 'v1' policy for performance assessment in its environment.
# Additionally, it automatically exports a JIT model, suitable for deployment purposes.
python scripts/play.py --task=kuavo_ppo --run_name v1 --load_run Apr07_09-25-44

# Implementing Simulation-to-Simulation Model Transformation for mujoco
# This command facilitates a sim-to-sim transformation using exported 'v1' policy.
python scripts/sim2sim_kuavo.py --load_model ../logs/kuavo_jog/exported/policies/policy_1.pt

```

#### 1. Default Tasks


- **humanoid_ppo**
   - Purpose: Baseline, PPO policy, Multi-frame low-level control
   - Observation Space: Variable $(47 \times H)$ dimensions, where $H$ is the number of frames
   - $[O_{t-H} ... O_t]$
   - Privileged Information: $73$ dimensions

- **humanoid_dwl (coming soon)**

#### 2. PPO Policy
- **Training Command**: For training the PPO policy, execute:
  ```
  python humanoid/scripts/train.py --task=humanoid_ppo --load_run log_file_path --name run_name
  ```
- **Running a Trained Policy**: To deploy a trained PPO policy, use:
  ```
  python humanoid/scripts/play.py --task=humanoid_ppo --load_run log_file_path --name run_name

  ```
- By default, the latest model of the last run from the experiment folder is loaded. However, other run iterations/models can be selected by adjusting `load_run` and `checkpoint` in the training config.

#### 3. Sim-to-sim

- **Mujoco-based Sim2Sim Deployment**: Utilize Mujoco for executing simulation-to-simulation (sim2sim) deployments with the command below:
  ```
  python scripts/sim2sim.py --load_model /path/to/export/model.pt
  ```

#### 4. Parameters
- **CPU and GPU Usage**: To run simulations on the CPU, set both `--sim_device=cpu` and `--rl_device=cpu`. For GPU operations, specify `--sim_device=cuda:{0,1,2...}` and `--rl_device={0,1,2...}` accordingly. Please note that `CUDA_VISIBLE_DEVICES` is not applicable, and it's essential to match the `--sim_device` and `--rl_device` settings.
- **Headless Operation**: Include `--headless` for operations without rendering.
- **Rendering Control**: Press 'v' to toggle rendering during training.
- **Policy Location**: Trained policies are saved in `humanoid/logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`.

#### 5. Command-Line Arguments
For RL training, please refer to `humanoid/utils/helpers.py#L161`.
For the sim-to-sim process, please refer to `humanoid/scripts/sim2sim.py#L169`.

## 新增：ASAP

以下为ASAP相关常用脚本命令及用途说明，便于快速查阅和复现实验流程：

### 1. 回放预训练策略
```bash
python humanoid/scripts/play.py --task=kuavo_ppo --load_run Apr07_09-25-44
```

### 2. sim2sim 预训练策略验证
```bash
python humanoid/scripts/sim2sim_kuavo.py --load_model logs/kuavo_jog/exported/policies_pretraining/policy_1.pt
```

### 3. 训练delta action model
```bash
python humanoid/scripts/train.py --task=delta_action_model_ppo --num_envs 128
```

### 4. 导出delta action策略
```bash
python humanoid/scripts/play.py --task=delta_action_model_ppo --load_run Jun14_20-01-23_delta_action
```

### 5. 预训练策略输出action直接+delta action的sim2sim
```bash
python humanoid/scripts/sim2sim_delta_action.py \
  --pretrained_model logs/kuavo_jog/exported/policies_pretraining/policy_1.pt \
  --delta_action_model logs/delta_action_model/exported/policies_delta_action/policy_1.pt
```

### 6. 用delta action finetune pretrianed depolicy
```bash
python humanoid/scripts/train.py --task=finetuning_policy_ppo --num_envs 128
# 训练finetuning policy
```

### 7. 导出finetuning policy
```bash
python humanoid/scripts/play.py --task=finetuning_policy_ppo --load_run Jun06_19-39-42_finetuning
# 回放finetuning policy
```

### 8. sim2sim refined policy
```bash
python humanoid/scripts/sim2sim_kuavo.py --load_model logs/kuavo_jog/exported/policies_finetuning/policy_1.pt
```
```bash
python scripts/testmodel.py --task=kuavo_ppo --run_name test1 --load_run Sep17_18-20-38_v1 --checkpoint 60000
```

```bash
python scripts/testmodel_mujoco.py --task=kuavo_ppo --run_name test1 --load_run Sep17_18-20-38_v1 --checkpoint 60000
```

```bash
python scripts/testmodel_mujoco.py --task=kuavo_clean_ppo --run_name test1 --load_run Apr07_09-25-44 --checkpoint 3300
```