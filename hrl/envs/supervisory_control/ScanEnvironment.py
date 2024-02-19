import numpy as np
import mujoco
import os

from gym import Env
from gym.spaces import Box, Dict

import yaml
from collections import deque
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from stable_baselines3 import PPO

from hrl.utils.rendering import Camera, Context
from hrl.utils.write_video import write_video
from hrl.envs.supervisory_control.OcularMotorControl import OcularMotorControl
from hrl.envs.supervisory_control.LocomotionControl import LocomotionControl


class ScanEnvironment(Env):

    def __init__(self):
        """
        Model when the attention is allocated off the smart glasses, i.e., the reading is interrupted,
            the agent has to determine the correct sign to scan and obtain the environmental event information as soon as possible,
            then instruct the locomotion control (change the lane or keep walking on the same lane, stop and watch the scene).

        Action: whether to look at the sign or not (very simple action space),
            and learn to get back to the reading content on the smart glasses after gaining information from the sign.
            The key is: when need the environmental information, the agent should look at the sign,
            when the information is obtained, the agent should look back at the smart glasses.
            And when trying to scan the sign, it would be better for the agent to stop walking, and resume walking after the scanning.
        """
        # Get directory of this file
        directory = os.path.dirname(os.path.realpath(__file__))

        # Read the configurations from the YAML file.
        root_dir = os.path.dirname(os.path.dirname(directory))
        with open(os.path.join(root_dir, "config.yaml")) as f:
            self._config = yaml.load(f, Loader=yaml.FullLoader)

        # Initialize the agent's state
        self._SMART_GLASSES = 'smart_glasses'
        self._ENV_NOTHING = 'env_nothing'
        self._ENV_EVENT = 'env_event'
        self._WALK = 1
        self._STOP = 0
        self._attention_choices = {
            self._SMART_GLASSES: -1,
            self._ENV_NOTHING: 0,
            self._ENV_EVENT: 1,
        }
        self._chosen_attention_target = None
        self._obtain_env_event_information = None      # Which lane to walk on
        self._mobility = None                          # Whether to walk or stop
        # The noise of the walking to the visual perception, prompted by the walking perturbation's adverse effect on reading
        self._walking_noise_to_visual_perception = 0.2

        # Dwell time on the environmental sign
        self._dwell_time = 20    # Step-wise dwell time: 20 steps
        self._steps_on_sign = None

        # Initialize the RL training related parameters
        self._steps = None
        self.ep_len = 50

        # Define the observation space
        self._num_stateful_info = 4
        self.observation_space = Box(low=-1, high=1, shape=(self._num_stateful_info,))

        # Define the action space: where to look at, and whether to stop or not
        self.action_space = Box(low=-1, high=1, shape=(2,))
        self._ZERO = 0
        self._ONE = 1
        self._TWO = 2
        self._THREE = 3
        self._threshold_walk_stop = 0

        # # Initialize the pre-trained RL models
        # if self._config['rl']['mode'] == 'test':
        #     self._omc_env = OcularMotorControl()
        #     self._lmc_env = LocomotionControl()
        #
        #     # The ocular motor control model
        #     omc_checkpoints_dir_name = ""
        #     omc_loaded_model_name = ""
        #     omc_model_path = os.path.join(root_dir, "training", "saved_models", omc_checkpoints_dir_name, omc_loaded_model_name)
        #
        #     self._omc_model = PPO.load(omc_model_path)
        #     self._omc_tuples = None
        #     self._omc_params = None
        #     self.omc_images = None
        #
        #     # The locomotion control model
        #     lmc_checkpoints_dir_name = ""
        #     lmc_loaded_model_name = ""
        #     lmc_model_path = os.path.join(root_dir, "training", "saved_models", lmc_checkpoints_dir_name, lmc_loaded_model_name)
        #
        #     self._lmc_model = PPO.load(lmc_model_path)
        #     self._lmc_tuples = None
        #     self._lmc_params = None
        #     self.lmc_images = None

    def reset(self):

        self._steps = 0

        # if self._config['rl']['mode'] == 'test':
        #     pass
        # else:
        self._chosen_attention_target = self._attention_choices[self._SMART_GLASSES]
        self._obtain_env_event_information = False
        self._mobility = self._WALK
        self._steps_on_sign = 0

        return self._get_obs()

    def render(self, mode="rgb_array"):
        pass

    def step(self, action):

        # Action a
        action[0] = self.normalise(action[0], -1, 1, self._ZERO, self._THREE)
        if self._ZERO <= action[0] < self._ONE:
            self._chosen_attention_target = self._attention_choices[self._SMART_GLASSES]
        elif self._ONE <= action[0] < self._TWO:
            self._chosen_attention_target = self._attention_choices[self._ENV_NOTHING]
        elif self._TWO <= action[0] <= self._THREE:
            self._chosen_attention_target = self._attention_choices[self._ENV_EVENT]
        else:
            raise ValueError(f"Invalid action: {action[0]}")

        mobility_action = action[1]
        if mobility_action < self._threshold_walk_stop:
            self._mobility = self._STOP
        else:
            self._mobility = self._WALK

        # State s'
        self._steps += 1

        if self._chosen_attention_target == self._attention_choices[self._ENV_EVENT]:
            if self._mobility == self._STOP:
                self._steps_on_sign += 1
            else:
                get_info = np.random.choice([True, False], p=[1-self._walking_noise_to_visual_perception,
                                                              self._walking_noise_to_visual_perception])
                if get_info:
                    self._steps_on_sign += 1
                else:
                    self._steps_on_sign = 0

        # When did not get the background event information, the agent should look at the background event/sign
        if not self._obtain_env_event_information:
            if self._config['rl']['mode'] == 'test':
                pass
                # TODO get the pretrained omc and lmc model
            else:
                if self._steps_on_sign >= self._dwell_time:
                    # Assume the info is obtained using the ocular motor control
                    self._obtain_env_event_information = True
                    # Assume the locomotion has been checked

        # Reward r and Terminate
        reward = -0.1

        terminate = False
        if self._steps >= self.ep_len or (self._obtain_env_event_information and self._chosen_attention_target == self._attention_choices[self._SMART_GLASSES]):
            terminate = True

            if self._obtain_env_event_information:
                catch_event_reward = 5
                reward += catch_event_reward

        return self._get_obs(), reward, terminate, {}

    @staticmethod
    def normalise(x, x_min, x_max, a, b):
        # Normalise x (which is assumed to be in range [x_min, x_max]) to range [a, b]
        return (b - a) * ((x - x_min) / (x_max - x_min)) + a

    def _get_obs(self):
        # Get the stateful information observation - normalize to [-1, 1]
        remaining_ep_len_norm = (self.ep_len - self._steps) / self.ep_len * 2 - 1
        chosen_attention_target_norm = self._chosen_attention_target
        obtained_background_event_information_norm = 1 if self._obtain_env_event_information else -1
        mobility_norm = 1 if self._mobility == 1 else -1

        stateful_info = np.array([remaining_ep_len_norm, chosen_attention_target_norm,
                                  obtained_background_event_information_norm, mobility_norm])

        # Observation space check
        if stateful_info.shape[0] != self._num_stateful_info:
            raise ValueError(f"The shape of stateful information observation is not correct! "
                             f"The allocated shape is: {self._num_stateful_info}, "
                             f"the actual shape is: {stateful_info.shape[0]}")

        return stateful_info
