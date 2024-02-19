import numpy as np
from collections import Counter, deque
import mujoco
import os

from gym import Env
from gym.spaces import Box, Dict

import yaml
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter

from huc.utils.rendering import Camera, Context


class LocomotionControl(Env):

    def __init__(self):
        """
        Model the locomotion control in the lane switch task - Low level motor control task
        - switch the lane or not based on the current - make the decision as soon as possible
        The moment the agent receives the encoded environmental information instruction, then tries to react to it

        The agent learns: with the given instructions received from the middle-level attending environment task,
        whether to switch lanes or not
        """
        # Get directory of this file
        directory = os.path.dirname(os.path.realpath(__file__))

        # Read the configurations from the YAML file.
        root_dir = os.path.dirname(os.path.dirname(directory))
        with open(os.path.join(root_dir, "config.yaml")) as f:
            self._config = yaml.load(f, Loader=yaml.FullLoader)

        # Load the MuJoCo model - Remember to change the xml accordingly
        self._model = mujoco.MjModel.from_xml_path(os.path.join(directory, "12cells-lane-switch-v2.xml"))
        self._data = mujoco.MjData(self._model)
        mujoco.mj_forward(self._model, self._data)

        # Define how often policy is queried
        self.action_sample_freq = 20
        self._frame_skip = int((1 / self.action_sample_freq) / self._model.opt.timestep)

        # Get the joints idx in MuJoCo
        # Agent's locomotion joints
        self._agent_joint_y_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "agent-joint-y")
        self._agent_joint_x_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "agent-joint-x")

        # Get the motors idx in MuJoCo
        self._agent_y_motor_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, "agent-y-motor")
        self._agent_y_translation_range = self._model.actuator_ctrlrange[self._agent_y_motor_mjidx]
        self._agent_x_motor_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, "agent-x-motor")
        self._agent_x_translation_range = self._model.actuator_ctrlrange[self._agent_x_motor_mjidx]

        # Initialize the locomotion/translation parameters
        translation_speed = 2     # 2m/s for normal walking - 15 m/s will give you a fast view for demonstration
        self._step_wise_translation_speed = translation_speed / self.action_sample_freq

        # Determine the thresholds for right and left lanes justification
        self._left_lane_threshold = -0.5
        self._right_lane_threshold = 0.5

        # Initialize lane switch related parameters
        self._instructed_lane = None
        self._agent_lane = None
        self._dwell_time = 1   # 1 second
        self._dwell_timesteps = int(self._dwell_time * self.action_sample_freq)
        self._agent_on_lane_timesteps = None

        # Initialise RL framework related thresholds and counters
        self._steps = None
        self.ep_len = 50
        self._in_hrl = False

        # Define the observation space
        self._num_stk_frm = 1
        self._qpos_frames = None
        self._num_stateful_info = 3
        self.observation_space = Dict({
            "proprioception": Box(low=-1, high=1, shape=(self._num_stk_frm * 1 + 1,)),
            "stateful information": Box(low=-1, high=1, shape=(self._num_stateful_info,)),
        })

        # Define the action space
        self.action_space = Box(low=-1, high=1, shape=(1,))   # 1 dimension: change the lane or keep walking straight

        # Initialize the context and camera
        context = Context(self._model, max_resolution=[1280, 960])
        self._env_cam = Camera(context, self._model, self._data, camera_id="env", maxgeom=100,
                               dt=1 / self.action_sample_freq)

    def reset(self, grid_search_params=None, load_model_params=None):

        # Reset MuJoCo sim
        mujoco.mj_resetData(self._model, self._data)

        # Initiate the stacked frames
        self._qpos_frames = deque(maxlen=self._num_stk_frm)

        # Reset the variables, flags, and counters
        self._steps = 0
        self._agent_on_lane_timesteps = 0

        # Reset the training/testing/using mode: whether in HRL
        if load_model_params is None:
            self._in_hrl = False
        else:
            self._in_hrl = True

        # The low level locomotion control related training, testing (including the grid search), and debugging
        #  separately, out of HRL
        if self._in_hrl == False:
            # Initialize starting position of the agent
            self._data.qpos[self._agent_joint_y_mjidx] = 0

            # Initialize the lane the agent is on
            self._agent_lane = np.random.choice([0, 1])

            # Sample an instructed lane
            self._instructed_lane = np.random.choice([0, 1])

        # Using the loaded model in HRL - the model has to be pre-trained before loading
        else:
            self._agent_lane = load_model_params["agent_lane"]
            self._instructed_lane = load_model_params["instructed_lane"]

        # Reset the agent's initial position with the given initial lane
        if self._agent_lane == 0:
            self._data.qpos[self._agent_joint_x_mjidx] = self._left_lane_threshold
            self._data.ctrl[self._agent_x_motor_mjidx] = self._left_lane_threshold
        elif self._agent_lane == 1:
            self._data.qpos[self._agent_joint_x_mjidx] = self._right_lane_threshold
            self._data.ctrl[self._agent_x_motor_mjidx] = self._right_lane_threshold
        else:
            raise ValueError(f"The agent's initial lane is not valid! Should be either 0 or 1. Now get {self._agent_lane}")

        # Set up the whole scene by confirming the initializations
        mujoco.mj_forward(self._model, self._data)

        if self._config['rl']['mode'] == 'test' or self._config['rl']['mode'] == 'debug':
            print(f"the agent is on lane {self._agent_lane}, and the instructed lane is {self._instructed_lane}\n")

        return self._get_obs()

    def render(self, mode="rgb_array"):
        rgb, _ = self._env_cam.render()
        return rgb

    def step(self, action):
        # Action a from the last state s
        action[0] = self.normalise(action[0], -1, 1, *self._model.actuator_ctrlrange[self._agent_x_motor_mjidx, :])

        self._data.ctrl[self._agent_x_motor_mjidx] = action[0]

        # Advance the simulation --> transit to the next state s'
        mujoco.mj_step(self._model, self._data, self._frame_skip)
        self._steps += 1

        # Get the agent's current position
        agent_x_pos = self._data.qpos[self._agent_joint_x_mjidx]
        # On the left lane
        if agent_x_pos < self._left_lane_threshold:
            self._agent_lane = 0
        # On the right lane
        elif agent_x_pos > self._right_lane_threshold:
            self._agent_lane = 1
        # Transitioning between lanes
        else:
            self._agent_lane = -1

        # Update agent's lane dwell timesteps
        if self._agent_lane == self._instructed_lane:
            self._agent_on_lane_timesteps += 1
        else:
            self._agent_on_lane_timesteps = 0

        # Get the step-wise reward
        reward = 0 if self._agent_lane == self._instructed_lane else -0.1

        # Get termination condition
        terminate = False
        if self._steps >= self.ep_len or self._agent_on_lane_timesteps >= self._dwell_timesteps:
            terminate = True

            # Estimate the reward for choosing the correct lane
            if self._agent_on_lane_timesteps >= self._dwell_timesteps:
                reward += 10

        # Print logs in the testing and debugging mode
        if self._in_hrl == False:
            if self._config['rl']['mode'] == 'test' or self._config['rl']['mode'] == 'debug':
                print("Agent's lane: ", self._agent_lane)
                print("Instructed lane: ", self._instructed_lane)
                print("Agent on lane timesteps: ", self._agent_on_lane_timesteps)
                print("Steps: ", self._steps)
                print("Reward: ", reward)
                print("")

        return self._get_obs(), reward, terminate, {}

    @staticmethod
    def normalise(x, x_min, x_max, a, b):
        # Normalise x (which is assumed to be in range [x_min, x_max]) to range [a, b]
        return (b - a) * ((x - x_min) / (x_max - x_min)) + a

    def _get_obs(self):
        """ Get the observation of the environment """
        # Compute the proprioception observation
        # Update the stack of frames of the proprioception channel
        # Remove the locomotion value since it is not normalized
        needed_qpos = self._data.qpos[self._agent_joint_x_mjidx].copy()
        self._qpos_frames.append(needed_qpos)
        while len(self._qpos_frames) < self._num_stk_frm:
            self._qpos_frames.append(self._qpos_frames[-1])
        # Get the proprioception observation
        qpos = np.stack(self._qpos_frames, axis=0)
        qpos = qpos.reshape((1, -1))
        # Remove the locomotion control since it is not normalized
        wanted_ctrl = self._data.ctrl[self._agent_joint_x_mjidx].copy()
        ctrl = wanted_ctrl.reshape((1, -1))
        proprioception = np.concatenate([qpos.flatten(), ctrl.flatten()], axis=0)

        # Compute the stateful information observation - normalize to [-1, 1]
        remaining_ep_len_norm = (self.ep_len - self._steps) / self.ep_len * 2 - 1
        remaining_dwell_timesteps_norm = (self._dwell_timesteps - self._agent_on_lane_timesteps) / self._dwell_timesteps * 2 - 1
        agent_lane_norm = self._agent_lane
        instructed_lane_norm = self._instructed_lane

        stateful_info = np.array(
            [remaining_ep_len_norm,
             remaining_dwell_timesteps_norm,
             # agent_lane_norm,
             instructed_lane_norm]
        )

        # Observation space check
        if proprioception.shape != self.observation_space["proprioception"].shape:
            raise ValueError(f"The shape of proprioception observation is not correct! "
                             f"Should be {self.observation_space['proprioception'].shape}, but got {proprioception.shape}")
        if stateful_info.shape != self.observation_space["stateful information"].shape:
            raise ValueError(f"The shape of stateful information observation is not correct!"
                             f"Should be {self.observation_space['stateful information'].shape}, but got {stateful_info.shape}")

        return {"proprioception": proprioception, "stateful information": stateful_info}
