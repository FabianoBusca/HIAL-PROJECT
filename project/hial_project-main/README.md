# HIAL-2025 Course Project

## Introduction
In this group project, you will utilize preference-based learning and imitation learning methods to train control policies for a robot manipulation task. 

## Task Environment
The target task is a robot food-serving task built base on [panda-gym](https://github.com/qgallouedec/panda-gym/tree/v2.0.2), where a robot arm needs to learn a control policy to pick up a banana that is randomly placed in a food preparation area, and place it on the plate for food serving. 
For more high-level details regarding the code structure, refer to Section 2 and Section 3 in [this paper](https://arxiv.org/pdf/2106.13687).

<img src="./doc/env_screenshot.png" alt="Task Env" width="60%">

### State Space
The state of the task is defined as a dictionary with keys `"observation"`, `"achieved_goal"`, and `"desired_goal"`.
Each value is a 1D `np.ndarray`.

The task observation is the concatenation of:
- **Robot observation** (UR5):
  - end-effector position (3): `ee_pos_x, ee_pos_y, ee_pos_z`
  - end-effector linear velocity (3): `ee_vel_x, ee_vel_y, ee_vel_z`
  - gripper opening width (1): `finger_width`
- **Object observation** (banana):
  - object position (3): `obj_pos_x, obj_pos_y, obj_pos_z`
  - object rotation quaternion (4): `obj_quat_x, obj_quat_y, obj_quat_z, obj_quat_w`
  - object linear velocity (3): `obj_vel_x, obj_vel_y, obj_vel_z`
  - object angular velocity (3): `obj_ang_vel_x, obj_ang_vel_y, obj_ang_vel_z`

```python
state["observation"] = np.array([
    # robot (7)
    ee_pos_x, ee_pos_y, ee_pos_z,
    ee_vel_x, ee_vel_y, ee_vel_z,
    finger_width,
    # object (13)
    obj_pos_x, obj_pos_y, obj_pos_z,
    obj_quat_x, obj_quat_y, obj_quat_z, obj_quat_w,
    obj_vel_x, obj_vel_y, obj_vel_z,
    obj_ang_vel_x, obj_ang_vel_y, obj_ang_vel_z
])  # total: 20 dims
```

The goal representation follows the standard goal-conditioned interface:

```python
state["achieved_goal"] = np.array([obj_pos_x, obj_pos_y, obj_pos_z])              # 3 dims
state["desired_goal"]  = np.array([target_pos_x, target_pos_y, target_pos_z])    # 3 dims
```

For this robot food-serving task, the desired goal position is fixed and given, defined as the center of the red plate.
The values of the observation and the desired goal will change at each time step in accordance with the physical interaction between the robot and the banana object.
More details can be found in [pick_and_place.py](./envs/tasks/pick_and_place.py).

**Tips for training:** For many RL algorithms it is convenient to flatten the dict observation into a single vector. A common option is to concatenate `state["observation"]` and `state["desired_goal"]`, via `reconstruct_state()` in [env_wrappers](./utils/env_wrappers.py).

### Action space
The action is a 1D np.ndarray in [-1, 1] with:
- 3 values controlling the UR5 arm
- 1 value controlling the gripper (if block_gripper=False)

```python
action = np.array([dx, dy, dz, gripper_cmd])
```

(dx, dy, dz) is an end-effector displacement command (delta in Cartesian space):
- `target_ee_position` = `current_ee_position + 0.05 * [dx, dy, dz]`
- the controller then uses inverse kinematics to compute the 6 UR5 joint targets.

gripper_cmd controls the gripper opening:
- `gripper_cmd` > 0 closes the gripper
- `gripper_cmd` <= 0 opens the gripper

More details can be found in [ur_robot.py](./envs/tasks/ur_robot.py).

### Reward
The task is designed with a sparse reward function. The robot will get a positive reward of +1000.0 if it manages to place the banana at the plate.
Otherwise, the robot will only get a reward of -1.0 at each step. 
More details can be found in [task_envs.py](./envs/task_envs.py).

### Termination 
An episode will terminate immediately if the banana is placed on the plate, 
or will terminate if the episode reaches its maximum length of 150 steps.
More details can be found in [task_envs.py](./envs/task_envs.py).

### Example on creating the task
It is very easy to create the task environment. You can do it by:
```python
env = PnPNewRobotEnv(render=True)
env = ResetWrapper(env)                 # enable reset to a given object position via options
env = ActionNormalizer(env)             # normalize actions in [-1, 1]
env = TimeLimitWrapper(env, max_steps=150)

obs, info = env.reset(seed=0)

terminated = truncated = False
while not (terminated or truncated):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
```
Refer to [test_env.py](./utils/test_env.py) for how to roll out in the environment episodically.
More details about ActionNormalizer, ResetWrapper, and TimeLimitWrapper can be found in [env_wrappers.py](./utils/env_wrappers.py). 


## Installation

1. Create a conda environment under python 3.8 and activate it in your terminal. Make sure you already downloaded and installed [Anaconda](https://www.anaconda.com/download/success) on your laptop before running the following commands in the terminal:
    ```bash
    $ conda create --name hial_project_env python=3.8
    $ conda activate hial_project_env
    ```
2. Install the [required packages](./requirements.txt):
   ```bash
   $ pip install -r requirements.txt
    ```
3. Install the library of APReL following the instructions [here](https://github.com/Stanford-ILIAD/APReL).

   Notice: If you are a Windows user, or a MacOS user but run into troubles with ffmpeg, check this [post](https://stackoverflow.com/questions/48486281/ffmpeg-osx-error-while-writing-file-unrecognized-option-preset-error-splitt) and try to install it directly via conda:
   ```bash
   $ conda install -c conda-forge ffmpeg=4.2.2
    ```
4. After installing the APReL library, remember to go back to the parent directory (e.g., if you installed APReL at "/Desktop/APReL/", then go back to the directory of "/Desktop/"), and clone the hial_project repository under this parent directory:
   ```bash
   $ git clone https://github.com/MH-Hou/hial_project.git
    ```
   Notice: If you haven't installed git on your laptop, follow the instructions [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) and install git before you use the commend of "git clone" in your terminal.
5. Try to run the [test script](./scripts/test_env.py)
    ```bash
    $ python -m scripts.test_env # from the root directory
    ```
## Examples
1. We have prepared 20 human-expert demos for you under the folder of /demo_data/PickAndPlace/, including episodic trajectories of states, actions, rewards, next states, and dones (i.e., whether an episode is terminated or not).
    
    If you want to check and load these demos run the script [load_demos.py](./scripts/load_demos.py)
   ```bash
   $ python -m scripts.load_demos # from the root directory
   ```
   
    You should be able to see demos replaying in the simulator environment like below:

    <br>
    <img src="./doc/demo_replay.gif" alt="Demo replay" width="60%">

## Project Assignment
### 1. Recover reward function via preference-based learning
* **1.1** complete the file named banana.py under the directory of [alg](./alg/).
  * design your own features of the recovered reward and implement the feature function given an episodic trajectory
  * complete the code to generate 30 video clips of episodic trajectories, including 20 clips of given expert demos and 10 clips of trajectories with random actions.

* **1.2** complete the file named pref_learn.py, write your code to recover the reward function for the robot food-serving task using the [APReL](https://github.com/Stanford-ILIAD/APReL) library:
  * The human teacher will provide feedback whenever given a comparison between two     trajectories chosen from these 30 trajectories. 
    And the way they provide feedback is by watching the clips of the chosen trajectories and give their preference using keyboard.
  * utilize APReL to write your code to recover the reward function based on your designed feature function and generated clips.
    More specifically, the human teacher is allowed to give feedback on 10 comparison queries in total. More code-wise procedures can be found in simple.py of APReL.
  * save the learned weights for your designed reward feature as a csv file named "feature_weights" under the directory of [saved](./saved/).

### 2. Policy training via RL from Demonstrations with the learned reward function
* create a file named policy_learn.py under the directory of [alg](./alg/).
* in this policy_learn.py, write your code to learn robot control policy using RL from Demonstrations with your learned reward function:
  * choose your favorite RLfD method (e.g., AWAC) as the underlying policy learning algorithm
  * load the given 20 expert demonstrations into the demo replay buffer of RLfD before policy training starts
  * train your policy episodically with your learned reward function. More specifically, write your code to realize a training loop where
    the robot first roll-outs its current policy for one episode, recovers the rewards for this episode given this roll-out trajectory, saves the trajectory into the
    replay buffer, then updates the control policy
  * train your policy for a maximum of 500k environment steps, and save your policy models every 1k steps during the training process
  * to report your results, plot the policy learning curve during the training process every 1k steps, with x-axis as the number of environment steps and
    y-axis as the **average success rate** of your trained policy at the current step. 
    
    For every data point, you should get the value of it by calculating the average success rate across 10 test runs. 
    And for each test run, you should:
    * randomly initialize the task environment (i.e., env.reset())
    * roll out your saved policy model of the current training step
    * check whether the robot succeeded or not for the current test run

### Deliverable checklist
* the script of banana.py for recording the clips
* the script of pref_learn.py for reward learning
* the script of policy_learn.py for policy learning
* the `./saved/` folder
* **(important)** another script named policy_test.py under the directory of [alg](./alg/):
  * to facilitate TAs to check the performance of your final trained policy, we kindly ask you to write two simple functions named load_final_policy() and get_policy_action() inside the script of policy_test.py:
  * for the function of load_final_policy(), it should be in the format of:
    ```
    def load_final_policy(path_to_saved_policy):
      """
      load your final trained policy
  
      Args:
          path_to_saved_policy (str): the path to your saved policy model
  
      Returns:
          your saved policy model under the corresponding path
      """

      # your code here
    
    ```
  * for the function of get_policy_action(), it should be in the format of:
    ```
    def get_policy_action(state, saved_policy_model):
      """
      get the action that the policy decides to take for the given environment state
  
      Args:
          state (dict): the state of the environment returned by the env.step() or env.reset(), which is a dictionary including keys of "observation", "achieved_goal", and "desired_goal"
          saved_policy_model: a saved model in the same format as the one returned by your load_final_policy() function
  
      Returns:
          action (np.array): the action that the saved policy model decides to take under the given state
      """

      # your code here
    
      ```

### (Optional) Extra credits:
* Realize online policy training with RLfD and preference-based learning, and report your results. In other words, there will be no two separate phases for training. Instead, during every episode of the learning loop, the robot will:
  * ask for human feedback on a certain amount of trajectory-pair comparisons
  * recover a new reward function for the current training episode
  * roll out the current policy and recover the rewards based on just learned reward function
  * update the policy
  * add the rolled out trajectory to the trajectory set of the preference-based learning, and generate the video clip of it for future comparison
  

## Useful Resources
1. (Highly recommended) You can find a clean implementation and the original paper of the Advantage-Weighted Actor-Critic (AWAC) algorithm [here](https://github.com/hari-sikchi/AWAC)
2. You can also find the implementations of a set of other IL algorithms [here](https://github.com/HumanCompatibleAI/imitation)
