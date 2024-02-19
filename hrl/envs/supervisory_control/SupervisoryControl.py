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
from hrl.envs.supervisory_control.WordSelection import WordSelection
from hrl.envs.supervisory_control.ScanEnvironment import ScanEnvironment


class SupervisoryControl(Env):

    def __init__(self):
        """
        Model the supervisory control of attention allocation in the context of walking and reading from smart glasses.
        One step in this task can be defined as many seconds in the real world.
        """
        # Get directory of this file
        directory = os.path.dirname(os.path.realpath(__file__))

        # Read the configurations from the YAML file.
        root_dir = os.path.dirname(os.path.dirname(directory))
        with open(os.path.join(root_dir, "config.yaml")) as f:
            self._config = yaml.load(f, Loader=yaml.FullLoader)

        # Register the reading content layout - has to be consistent with the layout in the WordSelection environment
        self._reading_content_per_page = {
            'num_rows': 3,
            'num_cols': 4,
            'num_words': 12,
        }

        self._L0 = 'L0'
        self._L50 = 'L50'
        self._L100 = 'L100'
        self._reading_content_layouts = {
            self._L0: -1,
            self._L50: 0,
            self._L100: 1,
        }
        self._reading_content_layout_name = None

        # These are parameters normalized from the empirical results
        self._word_selection_time_costs = {
            self._L0: 1,
            self._L50: -0.33,
            self._L100: -1,
        }
        self._word_selection_time_cost = None
        # self._word_selection_time_cost_noise = 0.25     # Not needed now

        self._word_selection_error_costs = {
            self._L0: 1,
            self._L50: 0.11,
            self._L100: -1,
        }
        self._word_selection_error_cost = None
        # self._word_selection_error_cost_noise = 0.25      # Not needed now

        # Initialize the reading and walking tasks related parameters
        # self._reading_task_weight_range = [0.5, 1]
        # self._reading_task_weight = None
        # self._walking_task_weight = None
        self._total_reading_words = 100   # The number of cells/words that needed to be read
        self._word_wise_reading_progress = None
        self._prev_word_wise_reading_progress = None
        self._MARGINS = 'margins'
        self._MIDDLE = 'middle'
        self._reading_positions = {
            self._MARGINS: -1,  # The beginning and the end of the reading task
            self._MIDDLE: 1,
        }
        self._reading_position = None
        self._reading_position_cost_factors = {
            self._MARGINS: 0.5,
            self._MIDDLE: 1,
        }
        self._reading_position_cost_factor = None

        self._background_events = {
            'left_lane': -1,
            'right_lane': 1,
        }
        self._background_event = None
        self._prev_background_event_step = None

        self._walking_lane = None
        self._prev_walking_lane = None
        self._steps_on_current_lane = None
        self._total_steps_on_incorrect_lane = None

        self._RECENT = 'recent'
        self._DISTANT = 'distant'
        self._background_last_check_flags = {
            self._RECENT: -1,
            self._DISTANT: 1,
        }
        self._background_last_check_step_wise_threshold = 10
        self._background_last_check_flag = None
        self._prev_background_last_check_step = None
        self._background_last_check_duration = None

        self._background_last_check_with_different_event_flags = {
            self._RECENT: -1,
            self._DISTANT: 1,
        }
        self._background_last_check_with_different_event_step_wise_threshold = 10
        self._background_last_check_with_different_event_flag = None
        self._observed_background_event = None
        self._prev_observed_background_event = None
        self._prev_background_last_check_with_different_event_step = None
        self._background_last_check_with_different_event_duration = None

        # Initialize the walking task environment related parameters
        self._attention_switch_to_background = None
        # Step-wise background information update frequency levels
        self._SHORT = 'short'
        self._MIDDLE = 'middle'
        self._LONG = 'long'
        self._background_event_intervals = {
            self._SHORT: 10,
            self._MIDDLE: 20,
            self._LONG: 30,
        }
        self._background_event_uncertainty = 0.382
        self._background_event_update_steps_list = None
        self._background_event_interval_level = None
        self._background_event_interval = None
        self._bg_event_interval_noise = None

        # Initialize the belief of the current state
        self._beliefs = None

        # Initialize the RL training related parameters
        self._steps = None
        self.ep_len = int(2*self._total_reading_words)
        self._epsilon = 1e-100
        self._info = None

        # Define the observation space
        self._num_stateful_info = 8
        self.observation_space = Box(low=-1, high=1, shape=(self._num_stateful_info,))

        # Define the action space - the attention allocation: reading on smart glasses (0) or reading on the environment (1)
        self.action_space = Box(low=-1, high=1, shape=(1,))
        self._ZERO_THRESHOLD = 0
        self._action_uncertainty = 0.333

        # # Initialize the pre-trained middle level RL models when testing the supervisory control
        # if self._config['rl']['mode'] == 'test':
        #     # Initialize the RL middle level task environments
        #     self._read_sg_env = WordSelection()
        #     self._read_bg_env = ReadBackground()
        #
        #     # Load the pre-trained RL middle level task models - reading on smart glasses
        #     read_sg_checkpoints_dir_name = ""
        #     read_sg_loaded_model_name = ""
        #     read_sg_model_path = os.path.join(root_dir, 'training', 'saved_models',
        #                                       read_sg_checkpoints_dir_name, read_sg_loaded_model_name)
        #     self._read_sg_model = PPO.load(read_sg_model_path)
        #     self._read_sg_tuples = None
        #     self._read_sg_params = None
        #     self.read_sg_images = None
        #
        #     # Load the pre-trained RL middle level task models - reading on the background/environment
        #     read_bg_checkpoints_dir_name = ""
        #     read_bg_loaded_model_name = ""
        #     read_bg_model_path = os.path.join(root_dir, 'training', 'saved_models',
        #                                       read_bg_checkpoints_dir_name, read_bg_loaded_model_name)
        #     self._read_bg_model = PPO.load(read_bg_model_path)
        #     self._read_bg_tuples = None
        #     self._read_bg_params = None
        #     self.read_bg_imgaes = None

    def reset(self, grid_search_params=None):

        self._steps = 0

        self._background_event_update_steps_list = []

        # For the test mode, evaluate with the pre-trained RL middle level task models
        if self._config['rl']['mode'] == 'test':
            # Embed the middle level task models into the supervisory control model
            # Reset the reading task related parameters
            self._word_wise_reading_progress = 0
            self._prev_word_wise_reading_progress = 0
            self._reading_position = self._reading_positions[self._MARGINS]
            self._reading_position_cost_factor = self._reading_position_cost_factors[self._MARGINS]
            self._reading_content_layout_name = self._L100 if grid_search_params is None else grid_search_params['layout']
            self._word_selection_time_cost = self._word_selection_time_costs[self._reading_content_layout_name]
            self._word_selection_error_cost = self._word_selection_error_costs[self._reading_content_layout_name]

            # Reset the walking and background related parameters
            self._background_event = np.random.choice(list(self._background_events.values()))
            self._walking_lane = self._background_event
            self._prev_walking_lane = self._walking_lane
            self._steps_on_current_lane = 0
            self._total_steps_on_incorrect_lane = 0
            self._prev_background_event_step = 0
            self._observed_background_event = self._background_event
            self._background_last_check_flag = self._background_last_check_flags[self._RECENT]
            self._prev_background_last_check_step = 0
            self._background_last_check_duration = 0

            self._background_last_check_with_different_event_flag = self._background_last_check_with_different_event_flags[self._RECENT]
            self._prev_observed_background_event = self._background_last_check_with_different_event_flag
            self._prev_background_last_check_with_different_event_step = 0
            self._background_last_check_with_different_event_duration = 0

            # Randomly initialize the attention allocation
            self._attention_switch_to_background = False

            # Randomly initialize the background information update frequency level
            self._background_event_interval_level = self._LONG if grid_search_params is None else grid_search_params['event_update_level']
            self._background_event_interval = self._background_event_intervals[self._background_event_interval_level]

            # # Randomly initialize the reading task weight and walking task weight - describes the perceived importance of the two tasks
            # # self._reading_task_weight = np.random.uniform(self._reading_task_weight_range[0],
            # #                                               self._reading_task_weight_range[1])
            # self._reading_task_weight = 0.5  # Start from the simple case
            # self._walking_task_weight = 1 - self._reading_task_weight

            self._update_beliefs()

            self._info = {
                'num_attention_switches': 0,
                'attention_switch_timesteps': [],
                'total_timesteps': 0,
                'num_steps_on_incorrect_lane': 0,
                'num_attention_switches_on_margins': 0,
                'num_attention_switches_on_middle': 0,
                'reading_speed': 0,
                'layout': self._reading_content_layout_name,
                'event interval': self._background_event_interval_level,
            }
        # For the training mode, only focus on this model, deploy a lot of stochasticity
        else:
            # Reset the reading task related parameters
            self._word_wise_reading_progress = 0
            self._prev_word_wise_reading_progress = 0
            self._reading_position = self._reading_positions[self._MARGINS]
            self._reading_position_cost_factor = self._reading_position_cost_factors[self._MARGINS]
            self._reading_content_layout_name = np.random.choice(list(self._reading_content_layouts.keys()))
            self._word_selection_time_cost = self._word_selection_time_costs[self._reading_content_layout_name]
            self._word_selection_error_cost = self._word_selection_error_costs[self._reading_content_layout_name]

            # Reset the walking and background related parameters
            self._background_event = np.random.choice(list(self._background_events.values()))
            self._walking_lane = self._background_event
            self._prev_walking_lane = self._walking_lane
            self._total_steps_on_incorrect_lane = 0
            self._steps_on_current_lane = 0
            self._prev_background_event_step = 0
            self._observed_background_event = self._background_event
            self._background_last_check_flag = self._background_last_check_flags[self._RECENT]
            self._background_last_check_duration = 0
            self._prev_background_last_check_step = 0
            self._background_last_check_with_different_event_flag = self._background_last_check_with_different_event_flags[self._RECENT]
            self._background_last_check_with_different_event_duration = 0
            self._prev_observed_background_event = self._background_last_check_with_different_event_flag
            self._prev_background_last_check_with_different_event_step = 0

            # Randomly initialize the attention allocation
            self._attention_switch_to_background = False

            # Randomly initialize the background information update frequency level
            self._background_event_interval_level = np.random.choice(list(self._background_event_intervals.keys()))
            self._background_event_interval = self._background_event_intervals[self._background_event_interval_level]

            # # Randomly initialize the reading task weight and walking task weight - describes the perceived importance of the two tasks
            # # self._reading_task_weight = np.random.uniform(self._reading_task_weight_range[0],
            # #                                               self._reading_task_weight_range[1])
            # # Start from the simple case - TODO make it dynamic later when static model works out
            # self._reading_task_weight = 0.5
            # self._walking_task_weight = 1 - self._reading_task_weight

            self._update_beliefs()

            self._info = {
                'num_attention_switches': 0,
                'attention_switch_timesteps': [],
                'total_timesteps': 0,
                'num_steps_on_incorrect_lane': 0,
                'num_attention_switches_on_margins': 0,
                'num_attention_switches_on_middle': 0,
                'reading_speed': 0,
            }

        return self._get_obs()

    def render(self, mode="rgb_array"):
        pass

    def step(self, action):

        # Action a - decision-making on every time step
        if action[0] <= self._ZERO_THRESHOLD:
            self._attention_switch_to_background = False
            action_name = 'continue_reading'
        else:
            # Add uncertainty to agent's decision-making / action
            switch_attention = np.random.choice([True, False], p=[1-self._action_uncertainty, self._action_uncertainty])
            if switch_attention:
                self._attention_switch_to_background = True
                action_name = 'switch_to_background'
                # Log the data
                self._info['attention_switch_timesteps'].append(self._steps)
                self._info['num_attention_switches'] += 1
                if self._reading_position == self._reading_positions[self._MARGINS]:
                    self._info['num_attention_switches_on_margins'] += 1
                else:
                    self._info['num_attention_switches_on_middle'] += 1
            else:
                self._attention_switch_to_background = False
                action_name = 'continue_reading'

        # State s'
        self._steps += 1

        # Update the background events
        self._update_background_events()

        # Apply actions
        # Continue reading
        if not self._attention_switch_to_background:
            # Testing mode
            if self._config['rl']['mode'] == 'test':
                self._prev_word_wise_reading_progress = self._word_wise_reading_progress
                # Continue the reading task, read one more cells/words
                self._word_wise_reading_progress += 1
                # Update the reading position
                self._update_reading_position()
            else:
                self._prev_word_wise_reading_progress = self._word_wise_reading_progress
                # Continue the reading task, read one more cells/words
                self._word_wise_reading_progress += 1
                # Update the reading position
                self._update_reading_position()
        # Switch attention to the environment, interrupt the reading task
        else:
            # Testing mode
            if self._config['rl']['mode'] == 'test':
                # Update the previous reading progress
                self._prev_word_wise_reading_progress = self._word_wise_reading_progress
                # Update the background last check - no matter whether observe a changed event instruction of not
                self._prev_background_last_check_step = self._steps
                # Update the observation of the background event
                self._observed_background_event = self._background_event
                # If the instruction in the background has changed, the agent will update its state
                if self._observed_background_event != self._prev_observed_background_event:
                    self._prev_observed_background_event = self._observed_background_event
                    self._prev_background_last_check_with_different_event_step = self._steps

                    # The background information has been read, then move to the correct lane
                    self._walking_lane = self._background_event
                    if self._prev_walking_lane != self._walking_lane:
                        self._steps_on_current_lane = 0
                        self._prev_walking_lane = self._walking_lane

                    # Resume reading - word selection
                    self._update_word_selection_costs()
            # All other non-testing modes
            else:
                # Update the previous reading progress
                self._prev_word_wise_reading_progress = self._word_wise_reading_progress
                # Update the background last check - no matter whether observe a changed event instruction of not
                self._prev_background_last_check_step = self._steps
                # Update the observation of the background event
                self._observed_background_event = self._background_event
                # If the instruction in the background has changed, the agent will update its state
                if self._observed_background_event != self._prev_observed_background_event:
                    self._prev_observed_background_event = self._observed_background_event
                    self._prev_background_last_check_with_different_event_step = self._steps

                    # The background information has been read, then move to the correct lane
                    self._walking_lane = self._background_event
                    if self._prev_walking_lane != self._walking_lane:
                        self._steps_on_current_lane = 0
                        self._prev_walking_lane = self._walking_lane

                    # Resume reading - word selection
                    self._update_word_selection_costs()

        # Update the step relevant flags
        self._update_background_check_flags()

        # Update the steps on the incorrect lane
        self._info['num_steps_on_incorrect_lane'] += 1 if self._walking_lane != self._background_event else 0

        # Update the steps on the current lane
        self._steps_on_current_lane += 1

        # Update beliefs
        self._update_beliefs()

        # Reward r
        reward = self._get_reward()

        # Termination
        terminate = False
        if self._steps >= self.ep_len or self._word_wise_reading_progress >= self._total_reading_words:
            terminate = True
            self._info['reading_speed'] = np.round(self._word_wise_reading_progress / self._steps, 4)
            self._info['total_timesteps'] = self._steps

        if self._config['rl']['mode'] == 'debug' or self._config['rl']['mode'] == 'test':
            print(f"The reading content layout name is {self._reading_content_layout_name}, "
                  f"the background event interval is {self._background_event_interval}\n"
                  f"The action name is {action_name}, the action value is {action}, \n"
                  f"The step is {self._steps}, \n"
                  f"The reading progress is {self._word_wise_reading_progress}, "
                  f"the reading position is: {[k for k, v in self._reading_positions.items() if v == self._reading_position]}, \n"
                  f"Walking on the lane {self._walking_lane}, the assigned lane: {self._background_event}, "
                  f"the total steps of walking incorrectly: {self._total_steps_on_incorrect_lane}\n"
                  f"The update background event list is {self._background_event_update_steps_list}, \n"
                  f"The reward is {reward}, \n")

        return self._get_obs(), reward, terminate, self._info

    # TODO Prepare to collect data for drawing the trajectory of the attention switches

    @staticmethod
    def normalise(x, x_min, x_max, a, b):
        # Normalise x (which is assumed to be in range [x_min, x_max]) to range [a, b]
        return (b - a) * ((x - x_min) / (x_max - x_min)) + a

    def _get_obs(self):
        # Get the stateful information observation - normalize to [-1, 1]
        remaining_ep_len_norm = (self.ep_len - self._steps) / self.ep_len * 2 - 1
        # Explicitly tell the agent the selected background event update intervals
        event_update_interval_norm = self.normalise(self._background_event_interval, 0, self.ep_len, -1, 1)
        # # Get reading and walking task's weight
        # reading_task_weight_norm = self._reading_task_weight
        # walking_task_weight_norm = self._walking_task_weight
        # Get the belief observation - normalize to [-1, 1]
        belief = self._beliefs.copy()

        stateful_info = np.array([remaining_ep_len_norm, event_update_interval_norm,
                                  # reading_task_weight_norm, walking_task_weight_norm,
                                  *belief])

        # Observation space check
        if stateful_info.shape[0] != self._num_stateful_info:
            raise ValueError(f"The shape of stateful information observation is not correct! "
                             f"The allocated shape is: {self._num_stateful_info}, "
                             f"the actual shape is: {stateful_info.shape[0]}")

        return stateful_info

    def _update_background_events(self):
        # Update the background event updates, we assume when the background events are updated,
        #   if the agent switch its attention on it, the agent can observe it immediately
        if self._steps - self._prev_background_event_step >= self._background_event_interval:
            # Add uncertainty to the background event updates
            change_event = np.random.choice([True, False],
                                            p=[1 - self._background_event_uncertainty, self._background_event_uncertainty])
            if change_event:
                # Update the background event update steps list
                self._background_event_update_steps_list.append(self._steps)
                # Update the previous background event
                prev_background_event = self._background_event
                # From the current background event, sample a new background event
                while True:
                    self._background_event = np.random.choice(list(self._background_events.values()))
                    if self._background_event != prev_background_event:
                        break
                self._prev_background_event_step = self._steps

    def _update_reading_position(self):
        # If the remains is not 0, then determine its position from a sentence's perspective
        sentence_remain = self._word_wise_reading_progress % self._reading_content_per_page['num_cols']
        # If the remained word is the first or last word in a sentence, then the reading position is at the margin
        if sentence_remain == 1 or sentence_remain == 0:
            reading_position_key = self._MARGINS
        # If the remained word is in the middle of a sentence, then the reading position is at the middle
        else:
            reading_position_key = self._MIDDLE
        # Update the reading position and the reading position cost factor
        self._reading_position = self._reading_positions[reading_position_key]
        self._reading_position_cost_factor = self._reading_position_cost_factors[reading_position_key]

    def _update_word_selection_costs(self):
        # Start from training with simple cases: deterministic word selection costs.
        #   Then move to the more complex cases where costs varies across trials
        self._word_selection_time_cost = self._word_selection_time_cost
        self._word_selection_error_cost = self._word_selection_error_cost

    def _update_background_check_flags(self):
        # Check and update the last checks
        # TODO to generalize the beliefs indicating when should the agent pay attention to the environment,
        #  maybe use continuous (blurred) probability instead of discrete flags
        if self._steps - self._prev_background_last_check_step >= self._background_last_check_step_wise_threshold:
            self._background_last_check_flag = self._background_last_check_flags[self._RECENT]
        else:
            self._background_last_check_flag = self._background_last_check_flags[self._RECENT]

        # Check and update the perceived background events
        if self._steps - self._prev_background_last_check_with_different_event_step >= self._background_last_check_with_different_event_step_wise_threshold:
            self._background_last_check_with_different_event_flag = self._background_last_check_with_different_event_flags[self._RECENT]
        else:
            self._background_last_check_with_different_event_flag = self._background_last_check_with_different_event_flags[self._RECENT]

        self._background_last_check_duration = self._steps - self._prev_background_last_check_step
        self._background_last_check_with_different_event_duration = self._steps - self._prev_background_last_check_with_different_event_step

    def _update_beliefs(self):
        # Update the LHS's reading related beliefs, containing the reading progress, reading position, estimated costs of switching attention
        reading_progress_norm = self.normalise(self._word_wise_reading_progress, 0, self._total_reading_words, -1, 1)
        reading_position_norm = self._reading_position
        reading_content_layout_norm = self._reading_content_layouts[self._reading_content_layout_name]

        # TODO not sure if we should include this in here, maybe should just appear in the reward function
        # word_selection_time_cost_norm = self._word_selection_time_cost * self._reading_position_cost_factor
        # word_selection_error_cost_norm = self._word_selection_error_cost * self._reading_position_cost_factor
        reading_related_beliefs = [reading_progress_norm, reading_position_norm, reading_content_layout_norm,
                                   # word_selection_time_cost_norm, word_selection_error_cost_norm
                                   ]

        # Update the RHS's environmental awareness/Walking related beliefs
        background_last_check_duration_norm = self.normalise(self._background_last_check_duration, 0, self.ep_len, -1, 1)
        background_last_check_with_different_event_duration_norm = self.normalise(self._background_last_check_with_different_event_duration, 0, self.ep_len, -1, 1)
        walking_lane_norm = self._walking_lane
        # TODO add the steps on the current lane in later, maybe it can speed up the training,
        #  since with this, the agent just need to map between the steps on the current lane and the event update interval
        steps_on_current_lane_norm = self.normalise(self._steps_on_current_lane, 0, self.ep_len, -1, 1)
        walking_related_beliefs = [background_last_check_duration_norm,
                                   background_last_check_with_different_event_duration_norm, walking_lane_norm]

        # Update the beliefs
        self._beliefs = reading_related_beliefs + walking_related_beliefs

    def _get_reward(self, reward=0):

        # Time cost for being there
        reward_time_cost = -0.1

        # Customized reward function, coefficients are to be tuned/modified
        reading_coefficient = 1
        if self._word_wise_reading_progress > self._prev_word_wise_reading_progress:
            reward_reading_making_progress = reading_coefficient * 1
        else:
            reward_reading_making_progress = 0

        # Make sure that the attention switch cost and reading resumption cost are not too high; otherwise,
        #   they will overshadow the other rewards and deter the agent from ever switching attention.
        # Define the tunable attention switch factor
        attention_switch_coefficient = 0.1
        if self._attention_switch_to_background:
            reward_attention_switch_cost = -0.25   # Can be proportional to the time cost
            reward_word_selection_time_cost = -self._reading_position_cost_factor * self.normalise(self._word_selection_time_cost, -1, 1, 0.1, 0.25)
            reward_word_selection_error_cost = -self._reading_position_cost_factor * self.normalise(self._word_selection_error_cost, -1, 1, 0.1, 0.25)
        else:
            reward_attention_switch_cost = 0
            reward_word_selection_time_cost = 0
            reward_word_selection_error_cost = 0

        # Define the reward related to walking, firstly define the tunable walking task factor
        walk_coefficient = 5
        if self._walking_lane == self._background_event:
            reward_walk_on_correct_lane = 0
        else:
            # Capture the nuances of multitasking behavior.
            #   An agent who hasn't checked the environment for a very long time might receive a bigger penalty if they are in the wrong lane.
            time_elapsed = self._background_last_check_duration
            reward_walk_on_correct_lane = walk_coefficient * (-0.5 + 2.5 * (np.exp(-0.04 * time_elapsed) - 1))

        # reward_reading = self._reading_task_weight * reward_reading_making_progress
        # reward_walking = self._walking_task_weight * reward_walk_on_correct_lane

        reward_reading = reward_reading_making_progress
        reward_walking = reward_walk_on_correct_lane
        reward_attention_switch = attention_switch_coefficient * (
                reward_attention_switch_cost +
                reward_word_selection_time_cost +
                reward_word_selection_error_cost
        )
        reward += reward_time_cost + reward_reading + reward_walking + reward_attention_switch

        # TODO to further differentiate different layouts, scale up the weight and range of the word selection costs

        if self._config['rl']['mode'] == 'debug' or self._config['rl']['mode'] == 'test':
            print(f"The reward components are:\n"
                  f"reward_reading_making_progress: {reward_reading_making_progress}\n"
                  f"reward_word_selection_time_cost: {reward_word_selection_time_cost}\n"
                  f"reward_word_selection_error_cost: {reward_word_selection_error_cost}\n"
                  f"reward_walk_on_correct_lane: {reward_walk_on_correct_lane}\n"
                  f"reward_attention_switch_cost: {reward_attention_switch_cost}, reward_time_cost: {reward_time_cost}\n"
                  f"reward: {reward}\n")

        return reward


# class SupervisoryControlWalkControl(Env):
#
#     def __init__(self):
#         """
#         Model the supervisory control of attention allocation in the context of walking and reading on OHMDs.
#         Study: Study 4. The class name contains WalkControl is because this supervisor's locomotion control
#             is walking speed control, instead of unrealistic lane-change, but the walking speed control.
#         Objective:
#             1. a unified Study that evaluates all components of the model.
#             2. a unified Study that has human data for evaluation.
#             3. a realistic scenario - endorsed by the published literature.
#
#         Suspended on 30 Nov 2023
#         """
#         # Get directory of this file
#         directory = os.path.dirname(os.path.realpath(__file__))
#
#         # Read the configurations from the YAML file.
#         root_dir = os.path.dirname(os.path.dirname(directory))
#         with open(os.path.join(root_dir, "config.yaml")) as f:
#             self._config = yaml.load(f, Loader=yaml.FullLoader)
#
#         # Initialize the environment - to be noted: one time step is one second
#         self.action_sample_freq = 1
#
#         # Walking related states
#         self._preferred_walking_speed = 1.5  # We assumed that the average walking speed is set to 1.5 m/s, ref: ChatGPT4. We do not have data replication on this metrics, so empirically set an average value.
#         self._PPWS = None  # Start try with the discrete levels: very slow, slow, relative slow, normal; and each one will correspond to a perturbation level, then finally result in the decrease in the readability from the oculomotor control
#         self._PPWS_ratio_intervals = {
#             # The intervals of the percentage of the preferred walking speed, ratios of the two free-parameters
#             #   in the oculomotor control: perturbation_amp_noise_scale and perturbation_amp_tuning_factor
#             # 'very slow': [[0, 0.2], [1, 0.85]],
#             # 'slow': [[0.2, 0.5], [0.85, 0.6]],
#             # 'relative slow': [[0.5, 0.8], [0.6, 0.4]],
#             # 'normal': [[0.8, 1], [0.4, 0.25]],
#             # Empirical setting - trial structure: [the PPWS, the ratio of readability/reading speed]
#             'very slow': [0.1, 0.9],    # Previously: [0.1, 0.9]
#             'slow': [0.5, 0.85],
#             # 'relative slow': [0.7, 0.5],
#             'normal': [0.9, 0.3],
#         }
#         # TODO: design the function between the readability <--> head perturbation <--> walking speed,
#         #  reference: Effects of walking velocity on vertical head and body movements during locomotion,
#         #  https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=9dfdcd7be78e40413a884fef601f04271e779e08
#         #  (might need to retrain the oculomotor control model to get more dynamic conditions and corresponding data)
#         # TODO or we just run the study for long time, and get agent's performance, in terms of:
#         #   1. the distribution --> later we can sample performance from.
#         #   2. the mean value --> more computationally efficient.
#         # TODO: then read-in data from the asset, the format should be: across all free-parameters, what are the performance (normalize to a range).
#         # Right now: assign it as a linear model: the higher ratio, the more perturbation, the higher two free-parameters,
#         #   and the lower readability is now hard-encoded into the environment, later need to get from the oculomotor control results,
#         #   ultimately directly get from the performance of oculomotor control model's interaction with the environment.
#
#         # Attention related states
#         self._attention_target = None   # The position of where the attention is allocated voluntarily
#         # The position of the actual attention, it differentiates with the above one,
#         # since the attention can be allocated to the environment, but the agent might not be able to perceive it,
#         # e.g., when the agent is seeing the sign in the environment
#         self._attention_actual_position = None
#         self._NA = 'NA'
#         self._OHMD = 'OHMD'
#         self._SIGN = 'Sign'
#         self._attention_positions = {
#             self._NA: -1,
#             self._OHMD: 0,
#             self._SIGN: 1,
#         }
#
#         # Reading related states
#         self._text_length = 360     # The number of words in the text is set with an average value 360, ref: Not All Spacings are Created Equal
#         self._reading_progress = None   # The ratio of the words that have been read in the text, ranges from 0 to 1
#         self._prev_reading_progress = None
#         self._reading_speed = 2     # Empirical setting - the estimated reading speed: 2 words per second.
#         self._reading_speed_ratio = None    # The ratio of the reading speed, ranges from 0 to 1, get from the above empirical setting
#
#         # Experimental environment related states - Maybe use a two-dimensional array to represent the rectangular-path environment
#         # Firstly try to use a one-dimensional array to represent the path environment
#         self._walking_path_perimeter = 30   # The perimeter of the path is 30 meters in the experiment, ref: Not All Spacings are Created Equal
#         self._walking_position = None   # The unit is also meter
#         self._prev_walking_position = None
#         self._total_walking_rounds = 2  # The number of rounds of walking in the experiment is set to 2, ref: Not All Spacings are Created Equal
#         self._current_walking_rounds = None
#         self._total_walking_path_length = self._walking_path_perimeter * self._total_walking_rounds
#         self._sign_distance = 2.5       # Same as in the experiment - 2.5 meters away from the path
#         # Empirical setting - the extreme perceivable distance of seeing the sign in the environment
#         self._perceivable_distance = 3  # FIXME: This might be a free-parameter that we can tune later on
#         half_perceivable_range_on_path = self._sign_distance * np.tan(np.arccos(self._sign_distance / self._perceivable_distance))
#         self._sign_1 = 3.5
#         self._sign_2 = 11
#         self._sign_3 = 18.5
#         self._sign_4 = 26
#         self._sign_5 = 33.5
#         self._sign_6 = 41
#         self._sign_7 = 48.5
#         self._sign_8 = 56
#         # Get the normalized sign positions
#         self._normalised_sign_positions = [
#             self.normalise(self._sign_1, 0, self._total_walking_path_length, -1, 1),
#             self.normalise(self._sign_2, 0, self._total_walking_path_length, -1, 1),
#             self.normalise(self._sign_3, 0, self._total_walking_path_length, -1, 1),
#             self.normalise(self._sign_4, 0, self._total_walking_path_length, -1, 1),
#             self.normalise(self._sign_5, 0, self._total_walking_path_length, -1, 1),
#             self.normalise(self._sign_6, 0, self._total_walking_path_length, -1, 1),
#             self.normalise(self._sign_7, 0, self._total_walking_path_length, -1, 1),
#             self.normalise(self._sign_8, 0, self._total_walking_path_length, -1, 1),
#         ]
#         self._sign_perceivable_locations = {
#             'sign_1': [self._sign_1 - half_perceivable_range_on_path, self._sign_1],
#             'sign_2': [self._sign_2 - half_perceivable_range_on_path, self._sign_2],
#             'sign_3': [self._sign_3 - half_perceivable_range_on_path, self._sign_3],
#             'sign_4': [self._sign_4 - half_perceivable_range_on_path, self._sign_4],
#             'sign_5': [self._sign_5 - half_perceivable_range_on_path, self._sign_5],
#             'sign_6': [self._sign_6 - half_perceivable_range_on_path, self._sign_6],
#             'sign_7': [self._sign_7 - half_perceivable_range_on_path, self._sign_7],
#             'sign_8': [self._sign_8 - half_perceivable_range_on_path, self._sign_8],
#         }
#         self._prev_seen_signs = None
#         self._seen_signs = None     # Should be a list
#         self._perceive_signs_duration = 3   # Need to cumulatively perceive the sign for 3 seconds to be able to read it
#         self._steps_on_sign = None
#         self._sign_perceivable = None
#         self._is_failed = None      # An indicator of whether the agent has failed in the task: walk by a sign without perceiving it
#
#         # The weight that balances the reading and walking tasks, ranges from 0 to 1
#         self._weight = None
#
#         # Initialize the RL training related parameters
#         self._steps = None
#         self.ep_len = int(2 * self._text_length)    # 100 - failed to finish the task
#         self._epsilon = 1e-100
#         self._info = None
#
#         # Define the observation space
#         self._num_stateful_info = 20
#         self.observation_space = Box(low=-1, high=1, shape=(self._num_stateful_info,))
#
#         # Define the action space - 1st: attention allocation; 2nd: walking speed control
#         self.action_space = Box(low=-1, high=1, shape=(2,))
#         self._action_attention_thresholds = {
#             self._NA: [-1, -0.3],
#             self._OHMD: [-0.3, 0.3],
#             self._SIGN: [0.3, 1],
#         }
#         # self._action_walking_speed_thresholds = {
#         #     'very slow': [-1, -0.5],
#         #     'slow': [-0.5, 0],
#         #     'relative slow': [0, 0.5],
#         #     'normal': [0.5, 1],
#         # }
#         self._action_walking_speed_thresholds = {
#             'very slow': [-1, -0.3],
#             'slow': [-0.3, 0.3],
#             'normal': [0.3, 1],
#         }
#
#         # Determine the information loggers
#         self._step_indexes = None
#         self._step_wise_walking_positions = None
#         self._step_wise_attentions = None
#         self._step_wise_walking_speeds = None
#         self._step_wise_reading_ratios = None
#         self._step_wise_reading_progress = None
#
#         # # Initialize the pre-trained middle level RL models when testing the supervisory control
#         # if self._config['rl']['mode'] == 'test':
#         #     # Initialize the RL middle level task environments
#         #     self._read_sg_env = WordSelection()
#         #     self._read_bg_env = ReadBackground()
#         #
#         #     # Load the pre-trained RL middle level task models - reading on smart glasses
#         #     read_sg_checkpoints_dir_name = ""
#         #     read_sg_loaded_model_name = ""
#         #     read_sg_model_path = os.path.join(root_dir, 'training', 'saved_models',
#         #                                       read_sg_checkpoints_dir_name, read_sg_loaded_model_name)
#         #     self._read_sg_model = PPO.load(read_sg_model_path)
#         #     self._read_sg_tuples = None
#         #     self._read_sg_params = None
#         #     self.read_sg_images = None
#         #
#         #     # Load the pre-trained RL middle level task models - reading on the background/environment
#         #     read_bg_checkpoints_dir_name = ""
#         #     read_bg_loaded_model_name = ""
#         #     read_bg_model_path = os.path.join(root_dir, 'training', 'saved_models',
#         #                                       read_bg_checkpoints_dir_name, read_bg_loaded_model_name)
#         #     self._read_bg_model = PPO.load(read_bg_model_path)
#         #     self._read_bg_tuples = None
#         #     self._read_bg_params = None
#         #     self.read_bg_imgaes = None
#
#     def reset(self, params=None):
#
#         self._steps = 0
#
#         # Initialize the variables
#         self._attention_target = self._NA
#         self._attention_actual_position = self._NA
#         self._PPWS = self._PPWS_ratio_intervals['normal'][0]
#         self._reading_speed_ratio = self._PPWS_ratio_intervals['normal'][-1]
#         self._reading_progress = 0
#         self._prev_reading_progress = 0
#         self._walking_position = 0
#         self._prev_walking_position = 0
#         self._current_walking_rounds = 0
#         self._prev_seen_signs = []
#         self._seen_signs = []
#         self._steps_on_sign = 0
#         self._sign_perceivable = False
#         self._is_failed = False
#
#         if params is None:
#             self._weight = np.random.uniform(0, 1)
#         else:
#             if self._config['rl']['mode'] == 'test':
#                 self._weight = params['weight']
#
#         # Log related information
#         self._info = {}
#         self._step_indexes = []
#         self._step_wise_walking_positions = []
#         self._step_wise_attentions = []
#         self._step_wise_walking_speeds = []
#         self._step_wise_reading_ratios = []
#         self._step_wise_reading_progress = []
#
#         return self._get_obs()
#
#     def render(self, mode="rgb_array"):
#         pass
#
#     def step(self, action):
#
#         self._prev_walking_position = self._walking_position
#         self._prev_seen_signs = self._seen_signs.copy()
#         self._prev_reading_progress = self._reading_progress
#
#         self._steps += 1
#
#         action_attention = action[0]
#         action_walking_speed = action[1]
#
#         # Determine the walking speed
#         if action_walking_speed <= self._action_walking_speed_thresholds['very slow'][-1]:
#             self._PPWS = self._PPWS_ratio_intervals['very slow'][0]
#             self._reading_speed_ratio = self._PPWS_ratio_intervals['very slow'][-1]
#         elif self._action_walking_speed_thresholds['slow'][0] < action_walking_speed <= self._action_walking_speed_thresholds['slow'][-1]:
#             self._PPWS = self._PPWS_ratio_intervals['slow'][0]
#             self._reading_speed_ratio = self._PPWS_ratio_intervals['slow'][-1]
#         # elif self._action_walking_speed_thresholds['relative slow'][0] < action_walking_speed <= self._action_walking_speed_thresholds['relative slow'][-1]:
#         #     self._PPWS = self._PPWS_ratio_intervals['relative slow'][0]
#         #     self._reading_speed_ratio = self._PPWS_ratio_intervals['relative slow'][-1]
#         elif self._action_walking_speed_thresholds['normal'][0] < action_walking_speed <= self._action_walking_speed_thresholds['normal'][-1]:
#             self._PPWS = self._PPWS_ratio_intervals['normal'][0]
#             self._reading_speed_ratio = self._PPWS_ratio_intervals['normal'][-1]
#         else:
#             raise ValueError(f"The action value of walking speed is not in the range of [-1, 1]! "
#                              f"The action value is: {action_walking_speed}")
#         # Update the walking position
#         self._walking_position += self._PPWS * self._preferred_walking_speed
#
#         # Update the perceivable range of the sign in the environment out of the loop for in-time update
#         # Determine whether it can see the sign - whether in the perceivable range
#         self._sign_perceivable, sign_name = self._determine_sign_perceivable()
#
#         # Determine the attention allocation
#         if action_attention <= self._action_attention_thresholds[self._NA][-1]:
#             # Do nothing, the agent is looking no where
#             self._attention_target = self._NA
#             self._attention_actual_position = self._NA
#         elif self._action_attention_thresholds[self._OHMD][0] < action_attention <= self._action_attention_thresholds[self._OHMD][-1]:
#             # The agent is reading on the OHMD
#             if self._reading_progress < self._text_length:
#                 self._reading_progress += self._reading_speed_ratio * self._reading_speed
#                 self._reading_progress = np.clip(self._reading_progress, 0, self._text_length)
#             self._attention_target = self._OHMD
#             self._attention_actual_position = self._OHMD
#         elif self._action_attention_thresholds[self._SIGN][0] < action_attention <= self._action_attention_thresholds[self._SIGN][-1]:
#             # The agent is reading the sign in the environment
#             # # Determine whether it can see the sign - whether in the perceivable range - update here will make it not in-time disable the flag
#             # self._sign_perceivable, sign_name = self._determine_sign_perceivable()
#             self._attention_target = self._SIGN
#             self._attention_actual_position = self._NA
#             if self._sign_perceivable:
#                 self._attention_actual_position = self._SIGN
#                 # If the sign is in the perceivable range, then the agent can read it
#                 if sign_name not in self._seen_signs and sign_name is not None:
#                     self._steps_on_sign += 1
#                     if self._steps_on_sign >= self._perceive_signs_duration:
#                         # # If the agent has perceived the sign for 3 seconds, then the agent can read it, but only append the sign that has not been seen before
#                         # if sign_name not in self._seen_signs and sign_name is not None:
#                         # self._prev_seen_signs = self._seen_signs.copy()
#                         self._seen_signs.append(sign_name)
#                         self._steps_on_sign = 0
#
#         terminate = self._get_terminate()
#         reward = self._get_reward(terminate=terminate)
#
#         # Log the information
#         self._step_indexes.append(self._steps)
#         # self._step_wise_attentions.append(self._attention)
#         self._step_wise_walking_positions.append(self._walking_position)
#         self._step_wise_attentions.append(self._attention_actual_position)
#         self._step_wise_walking_speeds.append(self._PPWS)
#         self._step_wise_reading_ratios.append(self._reading_speed_ratio)
#         self._step_wise_reading_progress.append(self._reading_progress)
#
#         # if self._config['rl']['mode'] == 'debug' or self._config['rl']['mode'] == 'test':
#         #     # Step-wise print out debugging
#         #     print(f"Step: {self._steps}, "
#         #           f"Attention: {self._attention_actual_position}, "
#         #           f"Steps on sign: {self._steps_on_sign}, "
#         #           f"Sign perceivable: {self._sign_perceivable}, "
#         #           f"Walking positions: {self._walking_position}, ")
#
#         if terminate and (self._config['rl']['mode'] == 'debug' or self._config['rl']['mode'] == 'test'):
#             # Episode-wise print out debugging
#             self._info = {
#                 'steps': self._step_indexes,
#                 'weight': self._weight,
#                 'walking_path_finished': self._walking_position >= self._total_walking_path_length,
#                 'signs_read': self._seen_signs,
#                 'sign_positions': [self._sign_1, self._sign_2, self._sign_3, self._sign_4, self._sign_5, self._sign_6, self._sign_7, self._sign_8],
#                 'step_wise_attentions': self._step_wise_attentions,
#                 'step_wise_walking_positions': self._step_wise_walking_positions,
#                 'step_wise_walking_speeds': self._step_wise_walking_speeds,
#                 'step_wise_reading_ratios': self._step_wise_reading_ratios,
#                 'step_wise_reading_progress': self._step_wise_reading_progress,
#             }
#             print(self._info)
#             # print(self._info['step_wise_walking_positions'])
#
#         return self._get_obs(), reward, terminate, self._info
#
#     @staticmethod
#     def normalise(x, x_min, x_max, a, b):
#         # Normalise x (which is assumed to be in range [x_min, x_max]) to range [a, b]
#         return (b - a) * ((x - x_min) / (x_max - x_min)) + a
#
#     def _determine_sign_perceivable(self):
#         # Determine whether the agent is walking in the perceivable range of the sign
#         for sign in self._sign_perceivable_locations.keys():
#             perceivable_range = self._sign_perceivable_locations[sign]
#             if perceivable_range[0] <= self._walking_position <= perceivable_range[1]:
#                 # If the agent is in the perceivable range, then the agent perceives the sign
#                 return True, sign
#             else:
#                 continue
#         return False, None
#
#     def _get_obs(self):
#
#         # Get the stateful information observation - normalize to [-1, 1]
#         remaining_ep_len_norm = (self.ep_len - self._steps) / self.ep_len * 2 - 1
#
#         # Get the attention allocation
#         norm_attention_target = self._attention_positions[self._attention_target]
#         norm_attention_actual_position = self._attention_positions[self._attention_actual_position]
#
#         # Get the walking position - where the agent is and where the destination is
#         norm_walking_position = self.normalise(self._walking_position, 0, self._total_walking_path_length, -1, 1)
#
#         # Get the sign positions
#         norm_sign_positions = self._normalised_sign_positions
#
#         # Get the walking speed
#         norm_walking_speed = self.normalise(self._PPWS, 0, 1, -1, 1)
#
#         # Get the reading speed
#         norm_reading_speed = self.normalise(self._reading_speed_ratio, 0, 1, -1, 1)
#
#         # Get the reading progress
#         norm_reading_progress = self.normalise(self._reading_progress, 0, self._text_length, -1, 1)
#         norm_prev_reading_progress = self.normalise(self._prev_reading_progress, 0, self._text_length, -1, 1)
#
#         # Get the sign related observation
#         norm_sign_perceivable = 1 if self._sign_perceivable else -1
#         norm_num_seen_signs = self.normalise(len(self._seen_signs), 0, len(self._sign_perceivable_locations), -1, 1)
#
#         # Get the task related information
#         norm_fail_task = 1 if self._is_failed else -1
#
#         # Get the tasks' weight
#         w = self._weight
#
#         stateful_info = np.array([remaining_ep_len_norm, norm_attention_target, norm_attention_actual_position, norm_walking_position,
#                                   norm_walking_speed, norm_reading_speed,
#                                   norm_reading_progress, norm_prev_reading_progress, norm_sign_perceivable, norm_num_seen_signs,
#                                   *norm_sign_positions,
#                                   norm_fail_task, w,
#                                   ])
#
#         # Observation space check
#         if stateful_info.shape[0] != self._num_stateful_info:
#             raise ValueError(f"The shape of stateful information observation is not correct! "
#                              f"The allocated shape is: {self._num_stateful_info}, "
#                              f"the actual shape is: {stateful_info.shape[0]}")
#
#         return stateful_info
#
#     def _get_terminate(self):
#
#         terminate = False
#
#         # Normal termination: determine by the walking position
#         if self._walking_position >= self._total_walking_path_length:
#             terminate = True
#
#         # Abnormal termination: determine by exceeding the maximum steps
#         if self._steps >= self.ep_len:
#             terminate = True
#             if self._is_failed:
#                 # If the agent spends too much steps on finishing the task (now set 100 steps as the episode length)
#                 self._is_failed = True
#
#         # Abnormal termination: determine by failing the task
#         for sign in self._sign_perceivable_locations.keys():
#             perceivable_range = self._sign_perceivable_locations[sign]
#             if self._walking_position > perceivable_range[1]:
#                 # If the agent is in the perceivable range, then the agent perceives the sign
#                 if sign not in self._seen_signs and sign is not None:
#                     self._is_failed = True
#                     terminate = True
#                     break
#
#         return terminate
#
#     # def _get_reward(self, terminate=False):
#     #     # Bai Yunpeng's version of reward function - suspended for now
#     #
#     #     # TODO: the back-up plan: tune the weight for this
#     #     # Time cost
#     #     time_cost = -1 + 1 * (np.exp(-0.035 * self._steps) - 1)
#     #
#     #     # Reading related rewards
#     #     reading_making_progress = 1 * (self._reading_progress - self._prev_reading_progress)
#     #
#     #     # Walking progress related rewards
#     #     # walking_making_progress = 0.1 * (self._walking_position - self._prev_walking_position)
#     #     walking_making_progress = 0
#     #
#     #     # Walking task finished related rewards
#     #     bonus_finish_task = 0
#     #     if terminate is True:
#     #         if self._is_failed:
#     #             # Punish if failed the task - did not read all the signs along the way
#     #             bonus_finish_task = -100
#     #         else:
#     #             if self._walking_position >= self._total_walking_path_length:
#     #                 # Finished the walking task
#     #                 bonus_finish_task = 100
#     #             else:
#     #                 # Punish if not read all the signs
#     #                 bonus_finish_task = -100
#     #
#     #     reward = time_cost + reading_making_progress + walking_making_progress + bonus_finish_task
#     #
#     #     return reward
#
#     def _get_reward(self, terminate=False):
#
#         # Aleksi's version of reward function
#         w = self._weight
#         # r1 = self._reading_speed_ratio if self._attention_actual_position == self._OHMD else 0
#         r1 = self._PPWS
#         # r2 = self._reading_speed_ratio if self._attention_actual_position == self._OHMD else 0
#         if self._attention_actual_position == self._OHMD and self._reading_progress < self._text_length:
#             r2 = self._reading_speed_ratio
#         else:
#             r2 = 0
#
#         # reward = w * r1 - (1 - w) * r2
#         reward = w * r1 + (1 - w) * r2
#
#         # [GPT4] Normalization Benefits: Normalizing rewards can help in stabilizing training,
#         # especially in environments where the scale of rewards can vary significantly.
#         # It helps in ensuring that the gradient updates during training are neither too large (causing instability)
#         # nor too small (slowing down learning).
#
#         return reward


class SupervisoryControlWalkControl(Env):

    def __init__(self):
        """
        Model the supervisory control of attention allocation in the context of walking and reading on OHMDs.
        Study: Study 4. The class name contains WalkControl is because this supervisor's locomotion control
            is walking speed control, instead of unrealistic lane-change, but the walking speed control.
        Objective:
            1. a unified Study that evaluates all components of the model.
            2. a unified Study that has human data for evaluation.
            3. a realistic scenario - endorsed by the published literature.

        Initiated on 30 Nov 2023.
        """
        # Get directory of this file
        directory = os.path.dirname(os.path.realpath(__file__))

        # Read the configurations from the YAML file.
        root_dir = os.path.dirname(os.path.dirname(directory))
        with open(os.path.join(root_dir, "config.yaml")) as f:
            self._config = yaml.load(f, Loader=yaml.FullLoader)

        # Initialize the environment - to be noted: one time step is one second
        self.action_sample_freq = 1

        # Walking related states
        self._preferred_walking_speed = 1.5  # We assumed that the average walking speed is set to 1.5 m/s, ref: ChatGPT4. We do not have data replication on this metrics, so empirically set an average value.
        self._PPWS = None  # Start try with the discrete levels: very slow, slow, relative slow, normal; and each one will correspond to a perturbation level, then finally result in the decrease in the readability from the oculomotor control
        self._PPWS_ratio_intervals = {
            # The intervals of the percentage of the preferred walking speed, ratios of the two free-parameters
            #   in the oculomotor control: perturbation_amp_noise_scale and perturbation_amp_tuning_factor
            # 'very slow': [[0, 0.2], [1, 0.85]],
            # 'slow': [[0.2, 0.5], [0.85, 0.6]],
            # 'relative slow': [[0.5, 0.8], [0.6, 0.4]],
            # 'normal': [[0.8, 1], [0.4, 0.25]],
            # Empirical setting - trial structure: [the PPWS, the ratio of readability/reading speed]
            'very slow': [0.1, 0.9],    # Previously: [0.1, 0.9]
            'slow': [0.5, 0.85],
            # 'relative slow': [0.7, 0.5],
            'normal': [0.9, 0.3],
        }

        # Attention related states
        self._attention_target = None   # Where the agent wants to see
        self._attention_actual_position = None  # Where the agent actually sees
        self._ENV = 'ENV'
        self._OHMD = 'OHMD'
        self._SIGN = 'Sign'
        self._attention_positions = {
            self._ENV: -1,
            self._OHMD: 0,
            self._SIGN: 1,
        }

        # Reading related states
        self._text_length = 360     # The number of words in the text is set with an average value 360, ref: Not All Spacings are Created Equal
        self._reading_progress = None   # The ratio of the words that have been read in the text, ranges from 0 to 1
        self._prev_reading_progress = None
        self._reading_speed = 2     # Empirical setting - the estimated reading speed: 2 words per second.
        self._reading_speed_ratio = None    # The ratio of the reading speed, ranges from 0 to 1, get from the above empirical setting

        # Experimental environment related states - Maybe use a two-dimensional array to represent the rectangular-path environment
        # Firstly try to use a one-dimensional array to represent the path environment
        self._walking_path_perimeter = 30   # The perimeter of the path is 30 meters in the experiment, ref: Not All Spacings are Created Equal
        self._walking_position = None   # The unit is also meter
        self._prev_walking_position = None
        self._total_walking_rounds = 2  # The number of rounds of walking in the experiment is set to 2, ref: Not All Spacings are Created Equal
        self._current_walking_rounds = None
        self._total_walking_path_length = self._walking_path_perimeter * self._total_walking_rounds
        self._sign_distance = 2.5       # Same as in the experiment - 2.5 meters away from the path
        # Empirical setting - the extreme perceivable distance of seeing the sign in the environment
        self._perceivable_distance = 3  # FIXME: This might be a free-parameter that we can tune later on
        half_perceivable_range_on_path = self._sign_distance * np.tan(np.arccos(self._sign_distance / self._perceivable_distance))
        self._sign_1 = 3.5
        self._sign_2 = 11
        self._sign_3 = 18.5
        self._sign_4 = 26
        self._sign_5 = 33.5
        self._sign_6 = 41
        self._sign_7 = 48.5
        self._sign_8 = 56
        # Get the normalized sign positions
        self._normalised_sign_positions = [
            self.normalise(self._sign_1, 0, self._total_walking_path_length, -1, 1),
            self.normalise(self._sign_2, 0, self._total_walking_path_length, -1, 1),
            self.normalise(self._sign_3, 0, self._total_walking_path_length, -1, 1),
            self.normalise(self._sign_4, 0, self._total_walking_path_length, -1, 1),
            self.normalise(self._sign_5, 0, self._total_walking_path_length, -1, 1),
            self.normalise(self._sign_6, 0, self._total_walking_path_length, -1, 1),
            self.normalise(self._sign_7, 0, self._total_walking_path_length, -1, 1),
            self.normalise(self._sign_8, 0, self._total_walking_path_length, -1, 1),
        ]
        self._sign_perceivable_locations = {
            'sign_1': [self._sign_1 - half_perceivable_range_on_path, self._sign_1],
            'sign_2': [self._sign_2 - half_perceivable_range_on_path, self._sign_2],
            'sign_3': [self._sign_3 - half_perceivable_range_on_path, self._sign_3],
            'sign_4': [self._sign_4 - half_perceivable_range_on_path, self._sign_4],
            'sign_5': [self._sign_5 - half_perceivable_range_on_path, self._sign_5],
            'sign_6': [self._sign_6 - half_perceivable_range_on_path, self._sign_6],
            'sign_7': [self._sign_7 - half_perceivable_range_on_path, self._sign_7],
            'sign_8': [self._sign_8 - half_perceivable_range_on_path, self._sign_8],
        }
        self._prev_seen_signs = None
        self._seen_signs = None     # Should be a list
        self._step_wise_perceive_signs_duration = 3   # Need to cumulatively perceive the sign for 3 seconds to be able to read it
        self._steps_on_sign = None
        self._sign_perceivable = None
        self._is_failed = None      # An indicator of whether the agent has failed in the task: walk by a sign without perceiving it

        # The weight that balances the reading and walking tasks, ranges from 0 to 1
        self._weight = None

        # Initialize the RL training related parameters
        self._steps = None
        self.ep_len = int(2 * self._text_length)    # 100 - failed to finish the task
        self._epsilon = 1e-100
        self._info = None

        # Define the observation space
        self._num_stateful_info = 20
        self.observation_space = Box(low=-1, high=1, shape=(self._num_stateful_info,))

        # Define the action space - 1st: attention allocation; 2nd: walking speed control
        self.action_space = Box(low=-1, high=1, shape=(2,))
        self._action_attention_thresholds = {
            self._OHMD: [-1, 0],
            self._ENV: [0, 1],
        }
        self._action_walking_speed_thresholds = {
            'very slow': [-1, -0.3],
            'slow': [-0.3, 0.3],
            'normal': [0.3, 1],
        }

        # Determine the information loggers
        self._step_indexes = None
        self._step_wise_walking_positions = None
        self._step_wise_attentions = None
        self._step_wise_walking_speeds = None
        self._step_wise_reading_ratios = None
        self._step_wise_reading_progress = None

    def reset(self, params=None):

        self._steps = 0

        # Initialize the variables
        self._attention_target = self._ENV
        self._attention_actual_position = self._ENV
        self._PPWS = self._PPWS_ratio_intervals['normal'][0]
        self._reading_speed_ratio = self._PPWS_ratio_intervals['normal'][-1]
        self._reading_progress = 0
        self._prev_reading_progress = 0
        self._walking_position = 0
        self._prev_walking_position = 0
        self._current_walking_rounds = 0
        self._prev_seen_signs = []
        self._seen_signs = []
        self._steps_on_sign = 0
        self._sign_perceivable = False
        self._is_failed = False

        if params is None:
            self._weight = np.random.uniform(0, 1)
        else:
            if self._config['rl']['mode'] == 'test':
                self._weight = params['weight']

        # Log related information
        self._info = {}
        self._step_indexes = []
        self._step_wise_walking_positions = []
        self._step_wise_attentions = []
        self._step_wise_walking_speeds = []
        self._step_wise_reading_ratios = []
        self._step_wise_reading_progress = []

        return self._get_obs()

    def render(self, mode="rgb_array"):
        pass

    def step(self, action):

        self._prev_walking_position = self._walking_position
        self._prev_seen_signs = self._seen_signs.copy()
        self._prev_reading_progress = self._reading_progress

        self._steps += 1

        action_attention = action[0]
        action_walking_speed = action[1]

        # Determine the walking speed
        if action_walking_speed <= self._action_walking_speed_thresholds['very slow'][-1]:
            self._PPWS = self._PPWS_ratio_intervals['very slow'][0]
            self._reading_speed_ratio = self._PPWS_ratio_intervals['very slow'][-1]
        elif self._action_walking_speed_thresholds['slow'][0] < action_walking_speed <= self._action_walking_speed_thresholds['slow'][-1]:
            self._PPWS = self._PPWS_ratio_intervals['slow'][0]
            self._reading_speed_ratio = self._PPWS_ratio_intervals['slow'][-1]
        elif self._action_walking_speed_thresholds['normal'][0] < action_walking_speed <= self._action_walking_speed_thresholds['normal'][-1]:
            self._PPWS = self._PPWS_ratio_intervals['normal'][0]
            self._reading_speed_ratio = self._PPWS_ratio_intervals['normal'][-1]
        else:
            raise ValueError(f"The action value of walking speed is not in the range of [-1, 1]! "
                             f"The action value is: {action_walking_speed}")
        # Update the walking position
        self._walking_position += self._PPWS * self._preferred_walking_speed

        # Update the perceivable range of the sign in the environment out of the loop for in-time update
        # Determine whether it can see the sign - whether in the perceivable range
        self._sign_perceivable, sign_name = self._determine_sign_perceivable()

        # Determine the attention allocation
        if action_attention <= self._action_attention_thresholds[self._OHMD][-1]:
            self._attention_target = self._OHMD
            self._attention_actual_position = self._OHMD
            # The agent is reading on the OHMD
            if self._reading_progress < self._text_length:
                self._reading_progress += self._reading_speed_ratio * self._reading_speed
                self._reading_progress = np.clip(self._reading_progress, 0, self._text_length)
        elif self._action_attention_thresholds[self._ENV][0] < action_attention <= self._action_attention_thresholds[self._ENV][-1]:
            self._attention_target = self._ENV
            if self._sign_perceivable:
                self._attention_actual_position = self._SIGN
                # If the sign is in the perceivable range, then the agent can read it
                if sign_name not in self._seen_signs and sign_name is not None:
                    self._steps_on_sign += 1
                    if self._steps_on_sign >= self._step_wise_perceive_signs_duration:
                        self._seen_signs.append(sign_name)
                        self._steps_on_sign = 0
            else:
                self._attention_actual_position = self._ENV

        terminate = self._get_terminate()
        reward = self._get_reward(terminate=terminate)

        # Log the information
        self._step_indexes.append(self._steps)
        # self._step_wise_attentions.append(self._attention)
        self._step_wise_walking_positions.append(self._walking_position)
        self._step_wise_attentions.append(self._attention_actual_position)
        self._step_wise_walking_speeds.append(self._PPWS)
        self._step_wise_reading_ratios.append(self._reading_speed_ratio)
        self._step_wise_reading_progress.append(self._reading_progress)

        if terminate and (self._config['rl']['mode'] == 'debug' or self._config['rl']['mode'] == 'test'):
            # Episode-wise print out debugging
            self._info = {
                'steps': self._step_indexes,
                'weight': self._weight,
                'walking_path_finished': self._walking_position >= self._total_walking_path_length,
                'signs_read': self._seen_signs,
                'sign_positions': [self._sign_1, self._sign_2, self._sign_3, self._sign_4, self._sign_5, self._sign_6, self._sign_7, self._sign_8],
                'step_wise_attentions': self._step_wise_attentions,
                'step_wise_walking_positions': self._step_wise_walking_positions,
                'step_wise_walking_speeds': self._step_wise_walking_speeds,
                'step_wise_reading_ratios': self._step_wise_reading_ratios,
                'step_wise_reading_progress': self._step_wise_reading_progress,
            }
            print(self._info['weight'])
            # print(self._info['step_wise_walking_positions'])

        return self._get_obs(), reward, terminate, self._info

    @staticmethod
    def normalise(x, x_min, x_max, a, b):
        # Normalise x (which is assumed to be in range [x_min, x_max]) to range [a, b]
        return (b - a) * ((x - x_min) / (x_max - x_min)) + a

    def _determine_sign_perceivable(self):
        # Determine whether the agent is walking in the perceivable range of the sign
        for sign in self._sign_perceivable_locations.keys():
            perceivable_range = self._sign_perceivable_locations[sign]
            if perceivable_range[0] <= self._walking_position <= perceivable_range[1]:
                # If the agent is in the perceivable range, then the agent perceives the sign
                return True, sign
            else:
                continue
        return False, None

    def _get_obs(self):

        # Get the stateful information observation - normalize to [-1, 1]
        remaining_ep_len_norm = (self.ep_len - self._steps) / self.ep_len * 2 - 1

        # Get the attention allocation
        norm_attention_target = self._attention_positions[self._attention_target]
        norm_attention_actual_position = self._attention_positions[self._attention_actual_position]

        # Get the walking position - where the agent is and where the destination is
        norm_walking_position = self.normalise(self._walking_position, 0, self._total_walking_path_length, -1, 1)

        # Get the sign positions
        norm_sign_positions = self._normalised_sign_positions

        # Get the walking speed
        norm_walking_speed = self.normalise(self._PPWS, 0, 1, -1, 1)

        # Get the reading speed
        norm_reading_speed = self.normalise(self._reading_speed_ratio, 0, 1, -1, 1)

        # Get the reading progress
        norm_reading_progress = self.normalise(self._reading_progress, 0, self._text_length, -1, 1)
        norm_prev_reading_progress = self.normalise(self._prev_reading_progress, 0, self._text_length, -1, 1)

        # Get the sign related observation
        norm_sign_perceivable = 1 if self._sign_perceivable else -1
        norm_num_seen_signs = self.normalise(len(self._seen_signs), 0, len(self._sign_perceivable_locations), -1, 1)

        # Get the task related information
        norm_fail_task = 1 if self._is_failed else -1

        # Get the tasks' weight
        w = self._weight

        stateful_info = np.array([remaining_ep_len_norm, norm_attention_target, norm_attention_actual_position, norm_walking_position,
                                  norm_walking_speed, norm_reading_speed,
                                  norm_reading_progress, norm_prev_reading_progress, norm_sign_perceivable, norm_num_seen_signs,
                                  *norm_sign_positions,
                                  norm_fail_task, w,
                                  ])

        # Observation space check
        if stateful_info.shape[0] != self._num_stateful_info:
            raise ValueError(f"The shape of stateful information observation is not correct! "
                             f"The allocated shape is: {self._num_stateful_info}, "
                             f"the actual shape is: {stateful_info.shape[0]}")

        return stateful_info

    def _get_terminate(self):

        terminate = False

        # Normal termination: determine by the walking position
        if self._walking_position >= self._total_walking_path_length:
            terminate = True

        # Abnormal termination: determine by exceeding the maximum steps
        if self._steps >= self.ep_len:
            terminate = True

        # Abnormal termination: determine by failing the task
        for sign in self._sign_perceivable_locations.keys():
            perceivable_range = self._sign_perceivable_locations[sign]
            if self._walking_position > perceivable_range[1]:
                # If the agent is in the perceivable range, then the agent perceives the sign
                if sign not in self._seen_signs and sign is not None:
                    self._is_failed = True
                    terminate = True
                    break

        return terminate

    def _get_reward(self, terminate=False):

        # Aleksi's version of reward function
        w = self._weight
        r1 = -1     # self._PPWS
        if self._attention_actual_position == self._OHMD and self._reading_progress < self._text_length:
            r2 = self._reading_speed_ratio
        else:
            r2 = 0

        reward = w * r1 + (1 - w) * r2

        # My reflection: the difference between r1=walking speed and r1=-1 (time penalty) is that the lateral will
        #   definitely try to end the episode sooner. While this is not certain case for the former, rather, it is dependent
        #   on the episode length (or more general, the task).

        return reward


class SupervisoryControlWalkControlElapsedTime(Env):

    def __init__(self):
        """
        Model the supervisory control of attention allocation in the context of walking and reading on OHMDs.
        Study: Study 4. The class name contains WalkControl is because this supervisor's locomotion control
            is walking speed control, instead of unrealistic lane-change, but the walking speed control.
        Objective:
            1. a unified Study that evaluates all components of the model.
            2. a unified Study that has human data for evaluation.
            3. a realistic scenario - endorsed by the published literature.

        Initiated on 30 Nov 2023. For another training paradigm of training with termination
            only based on the episode length.
        """
        # Get directory of this file
        directory = os.path.dirname(os.path.realpath(__file__))

        # Read the configurations from the YAML file.
        root_dir = os.path.dirname(os.path.dirname(directory))
        with open(os.path.join(root_dir, "config.yaml")) as f:
            self._config = yaml.load(f, Loader=yaml.FullLoader)

        # Initialize the environment - to be noted: one time step is one second
        self.action_sample_freq = 1

        # Walking related states
        # self._preferred_walking_speed = 1.3
        # S.L. Whitney, ... A. Alghadir, in Handbook of Clinical Neurology, 2016; Collectively,
        #   the range for normal WS for adults is between 1.2 and 1.4 m/s [73].
        self._preferred_walking_speed = None
        self._preferred_walking_speed_range = [0.8, 1.5]    # [1.2, 1.4]
        self._preferred_walking_speed_random_factor = None
        self._walking_speed_stop_threshold = 0.6    # 0.6 m/s is the threshold of walking speed to stop walking, ref: Effects of walking velocity on vertical head and body movements during locomotion [hirasaki1999effects]
        self._PPWS = None  # Start try with the discrete levels: very slow, slow, relative slow, normal; and each one will correspond to a perturbation level, then finally result in the decrease in the readability from the oculomotor control

        # Walking head perturbation's pitch values
        # Ref: Frequency and velocity of rotational head perturbations during locomotion
        self._theoretical_pitch_peak_amp = 3  # Theoretical pitch amplitude of the head during walking is 3 degrees - keep consistency with the oculomotor control model
        self._pitch_amp_ratio_stop_threshold = self.f(self._walking_speed_stop_threshold) / self._theoretical_pitch_peak_amp

        # Attention related states
        self._attention_target = None   # Where the agent wants to see
        self._attention_actual_position = None  # Where the agent actually sees
        self._ENV = 'ENV'
        self._OHMD = 'OHMD'
        self._SIGN = 'Sign'
        self._attention_positions = {
            self._ENV: -1,
            self._OHMD: 0,
            self._SIGN: 1,
        }

        # Reading related states
        # self._text_length = 360     # The number of words in the text is set with an average value 360, ref: Not All Spacings are Created Equal
        self._reading_progress = None   # The ratio of the words that have been read in the text, ranges from 0 to 1
        # self._prev_reading_progress = None
        self._reading_speed = 2     # Empirical setting - the estimated reading speed: 2 words per second.
        self._reading_speed_ratio = None    # The ratio of the reading speed, ranges from 0 to 1, get from the above empirical setting

        # Experimental environment related states - Maybe use a two-dimensional array to represent the rectangular-path environment
        # Firstly try to use a one-dimensional array to represent the path environment
        self._long_side = 8   # Remaining is even number
        self._short_side = 7    # Remaining is odd number
        # self._walking_path_perimeter = 30   # The perimeter of the path is 30 meters in the experiment, ref: Not All Spacings are Created Equal
        self._walking_position = None   # The unit is also meter
        self._prev_walking_position = None
        # self._total_walking_rounds = 2  # The number of rounds of walking in the experiment is set to 2, ref: Not All Spacings are Created Equal
        # self._current_walking_rounds = None
        # self._total_walking_path_length = self._walking_path_perimeter * self._total_walking_rounds
        self._void_total_walking_path_length = 1000     # A void value for normalising the walking positions and the sign positions
        self._sign_distance = 2.5       # Same as in the experiment - 2.5 meters away from the path
        # Empirical setting - the extreme perceivable distance of seeing the sign in the environment
        self._perception_factor = None
        self._perceivable_distance_range = [2.5, 2.7]       # [2.5, 3]
        self._perceivable_distance = None
        self._half_perceivable_range = None

        self._next_sign_position = None
        self._next_sign_number = None   # Start from 1
        self._prev_seen_signs = None
        self._seen_signs = None     # Should be a list
        # Sign read duration related states
        # self._sign_read_steps_range = [2, 6]
        # self._sign_read_step_factor = None
        self._sign_read_steps = 2   #3  # Need to cumulatively perceive the sign for 3 seconds to be able to read it
        # self._remaining_sign_read_steps = None
        # self._sign_read_duration_normalise_denominator = 10     # Use as the denominator to normalise the sign read duration
        self._steps_on_sign = None
        self._sign_perceivable = None
        self._is_failed = None      # An indicator of whether the agent has failed in the task: walk by a sign without perceiving it

        # The weight that balances the reading and walking tasks, ranges from 0 to 1
        self._weight = None

        # Initialize the RL training related parameters
        self._steps = None
        self._ep_len_range = [100, 500]
        self.ep_len = None    # Randomized episode length
        self._epsilon = 1e-100
        self._info = None

        # Define the observation space
        self._num_stateful_info = 12
        self.observation_space = Box(low=-1, high=1, shape=(self._num_stateful_info,))

        # Define the action space - 1st: attention allocation; 2nd: walking speed control
        self.action_space = Box(low=-1, high=1, shape=(2,))
        self._action_attention_thresholds = {
            self._OHMD: [-1, 0],
            self._ENV: [0, 1],
        }

        # Determine the information loggers
        self._step_indexes = None
        self._step_wise_walking_positions = None
        self._step_wise_attentions = None
        self._step_wise_walking_speeds = None
        self._step_wise_reading_ratios = None
        self._step_wise_reading_progress = None

    def reset(self, params=None):

        self._steps = 0

        if params is None:
            # Default setting: randomize the weight and episode length
            self._weight = np.random.uniform(0, 1)
            self._preferred_walking_speed_random_factor = np.random.uniform(0, 1)
            self._perception_factor = np.random.uniform(0, 1)
            self.ep_len = np.random.randint(self._ep_len_range[0], self._ep_len_range[1])
        else:
            if self._config['rl']['mode'] == 'test':
                self._weight = params['weight']
                self._preferred_walking_speed_random_factor = params['walk_factor']
                self._perception_factor = params['perception_factor']
                self.ep_len = self._ep_len_range[1]    # Set the episode length to the maximum

        # Initialize the variables
        self._attention_target = self._ENV
        self._attention_actual_position = self._ENV
        # self._PPWS = self._PPWS_ratio_intervals['normal 1'][0]
        # self._reading_speed_ratio = self._PPWS_ratio_intervals['normal 1'][-1]
        # self._preferred_walking_speed_random_factor = np.random.uniform(0, 1)
        self._preferred_walking_speed = self._preferred_walking_speed_range[0] + self._preferred_walking_speed_random_factor * (self._preferred_walking_speed_range[1] - self._preferred_walking_speed_range[0])
        self._PPWS = 1
        self._reading_speed_ratio = self._get_reading_speed_ratio(walking_speed_ratio=self._PPWS)   # Get the reading speed ratio from the walking speed ratio - continuous values
        self._reading_progress = 0
        # self._sign_read_steps = 3
        self._perceivable_distance = self._perceivable_distance_range[0] + self._perception_factor * (self._perceivable_distance_range[1] - self._perceivable_distance_range[0])
        self._half_perceivable_range = self._sign_distance * np.tan(np.arccos(self._sign_distance / self._perceivable_distance))
        # self._remaining_sign_read_steps = self._sign_read_steps
        # self._prev_reading_progress = 0
        self._walking_position = 0
        self._prev_walking_position = 0
        # self._current_walking_rounds = 0
        self._prev_seen_signs = []
        self._seen_signs = []
        self._steps_on_sign = 0
        self._sign_perceivable = False
        self._is_failed = False

        self._next_sign_position, self._next_sign_number = self._get_next_sign_position()

        # Log related information
        self._info = {}
        self._step_indexes = []
        self._step_wise_walking_positions = []
        self._step_wise_attentions = []
        self._step_wise_walking_speeds = []
        self._step_wise_reading_ratios = []
        self._step_wise_reading_progress = []

        return self._get_obs()

    def render(self, mode="rgb_array"):
        pass

    def step(self, action):

        self._prev_walking_position = self._walking_position
        self._prev_seen_signs = self._seen_signs.copy()
        # self._prev_reading_progress = self._reading_progress

        self._steps += 1

        # Continuous actions
        action_attention = action[0]
        action_walking_speed = self.normalise(action[1], -1, 1, 0, 1)
        self._PPWS = action_walking_speed
        self._reading_speed_ratio = self._get_reading_speed_ratio(walking_speed_ratio=self._PPWS)

        # Update the walking position
        self._walking_position += self._PPWS * self._preferred_walking_speed

        # Update the next sign position
        self._next_sign_position, self._next_sign_number = self._get_next_sign_position()

        # Update the perceivable range of the sign in the environment out of the loop for in-time update
        # Determine whether it can see the sign - whether in the perceivable range
        self._sign_perceivable, sign_name = self._get_sign_perceptibility()

        # Determine the attention allocation
        if action_attention <= self._action_attention_thresholds[self._OHMD][-1]:
            self._attention_target = self._OHMD
            self._attention_actual_position = self._OHMD
            # Update the reading progress
            self._reading_progress += self._reading_speed_ratio * self._reading_speed
        elif self._action_attention_thresholds[self._ENV][0] < action_attention <= self._action_attention_thresholds[self._ENV][-1]:
            self._attention_target = self._ENV
            if self._sign_perceivable:
                self._attention_actual_position = self._SIGN
                # If the sign is in the perceivable range, then the agent can read it
                if sign_name not in self._seen_signs and sign_name is not None:
                    self._steps_on_sign += 1
                    if self._steps_on_sign >= self._sign_read_steps:
                        self._seen_signs.append(sign_name)
                        self._steps_on_sign = 0
            else:
                self._attention_actual_position = self._ENV

        terminate = self._get_terminate()
        reward = self._get_reward(terminate=terminate)

        # Log the information
        self._step_indexes.append(self._steps)
        # self._step_wise_attentions.append(self._attention)
        self._step_wise_walking_positions.append(self._walking_position)
        self._step_wise_attentions.append(self._attention_actual_position)
        self._step_wise_walking_speeds.append(self._PPWS)
        self._step_wise_reading_ratios.append(self._reading_speed_ratio)
        self._step_wise_reading_progress.append(self._reading_progress)

        if terminate and (self._config['rl']['mode'] == 'debug' or self._config['rl']['mode'] == 'test'):
            # Episode-wise print out debugging
            self._info = {
                'steps': self._step_indexes,
                'weights': self._weight,
                'walk_factors': self._preferred_walking_speed_random_factor,
                'perception_factors': self._perception_factor,
                'preferred_walking_speed': self._preferred_walking_speed,
                'rectangle_path_length': 2 * 2 * (self._long_side + self._short_side),
                'ep_len': self.ep_len,
                'sign_positions': [4, 11.5, 19, 26.5, 34, 41.5, 49, 56.5],  # When testing only test two rounds.
                'walking_path_finished': True if len(self._seen_signs) == self._next_sign_number - 1 else False,
                'signs_read': self._seen_signs,
                'step_wise_attentions': self._step_wise_attentions,
                'step_wise_walking_positions': self._step_wise_walking_positions,
                'step_wise_walking_speeds': self._step_wise_walking_speeds,
                'step_wise_reading_ratios': self._step_wise_reading_ratios,
                'step_wise_reading_progress': self._step_wise_reading_progress,
            }
            print(f'weight: {self._info["weights"]}, '
                  f'\nwalk_factor: {self._info["walk_factors"]}, preferred_walking_speed: {self._info["preferred_walking_speed"]}, '
                  f'\nperception_factors: {self._info["perception_factors"]}, perceivable_distance: {self._perceivable_distance}, '
                  f'half_perceivable_range: {self._half_perceivable_range}, ')

        return self._get_obs(), reward, terminate, self._info

    @staticmethod
    def normalise(x, x_min, x_max, a, b):
        # Normalise x (which is assumed to be in range [x_min, x_max]) to range [a, b]
        return (b - a) * ((x - x_min) / (x_max - x_min)) + a

    def _get_reading_speed_ratio(self, walking_speed_ratio):

        # Get the walking speed
        walking_speed = walking_speed_ratio * self._preferred_walking_speed

        # Get the pitch amplitude
        pitch_amp = self._get_pitch_amp_from_walking_speed(walking_speed=walking_speed)

        # Get the reading speed ratio
        reading_speed_ratio = self._get_reading_speed_ratio_from_pitch_amp(pitch_amp=pitch_amp)

        return reading_speed_ratio

    @staticmethod
    def f(x):
        # The quadratic polynomial function that approximates the relationship between the walking speed and the pitch amplitude
        return 0.2233395 * x**2 - 0.03672422 * x + 1.0099814

    @staticmethod
    def g(x):
        # The quadratic polynomial function that approximates the relationship between the pitch amplitude and the reading speed ratio

        # return -0.18624534 * x**2 - 0.2453667 * x + 0.81824108
        # Use the second function later

        return -0.2094 * x**2 - 0.4340 * x + 1.1123

    def _get_pitch_amp_from_walking_speed(self, walking_speed):
        # Rough data: Effects of walking velocity on vertical head and body movements during locomotion [hirasaki1999effects]
        # Approximated data from: WebPlotDigitizer https://apps.automeris.io/wpd/
        # Rationality of using a quadratic polynomial function to approximate: quadratic polynomial functions are often
        #   used as a default approximation tool because of their simplicity and flexibility
        #   [Introduction to optimum design][arora2004introduction]
        # The log in slides: https://docs.google.com/presentation/d/1qfmHrfqYp4SHzDiCw_vjatLaCZ40jS08PoyUtvQeZyU/edit#slide=id.g262ae9c5a48_0_44

        # if walking_speed <= self._walking_speed_stop_threshold:
        #     return 0
        # else:
        #     return self.f(walking_speed)
        # Remove the threshold

        return self.f(walking_speed)

    def _get_reading_speed_ratio_from_pitch_amp(self, pitch_amp):

        pitch_amp_ratio = pitch_amp / self._theoretical_pitch_peak_amp

        # if pitch_amp_ratio <= self._pitch_amp_ratio_stop_threshold:
        #     return 1
        # else:
        #     return self.g(pitch_amp_ratio)
        # Remove the threshold

        return self.g(pitch_amp_ratio)

    def _get_next_sign_position(self):
        # Get the next sign position based on the current walking position
        num_half_laps = int(self._walking_position // (self._long_side + self._short_side))
        walked_path_length = num_half_laps * (self._long_side + self._short_side)
        remaining_two_sides_length = self._walking_position - walked_path_length
        if remaining_two_sides_length <= self._long_side / 2:
            return walked_path_length + self._long_side / 2, 2 * num_half_laps + 1
        elif self._long_side / 2 < remaining_two_sides_length <= self._long_side + self._short_side / 2:
            return walked_path_length + self._long_side + self._short_side / 2, 2 * num_half_laps + 2
        elif remaining_two_sides_length > self._long_side + self._short_side / 2:
            return walked_path_length + self._long_side + self._short_side + self._long_side / 2, 2 * num_half_laps + 3
        else:
            raise ValueError(f"The next sign position is not correctly determined! "
                             f"The current walking position is: {self._walking_position}, "
                             f"the next sign position is: {self._next_sign_position}")

    def _get_sign_perceptibility(self):
        # Determine whether the agent is walking in the perceivable range of the sign
        perceivable_range = [self._next_sign_position - self._half_perceivable_range, self._next_sign_position + self._half_perceivable_range]
        if perceivable_range[0] <= self._walking_position <= perceivable_range[1]:
            return True, f'sign_{self._next_sign_number}'
        else:
            return False, None

    def _get_obs(self):

        # Get the stateful information observation - normalize to [-1, 1]
        remaining_ep_len_norm = (self.ep_len - self._steps) / self.ep_len * 2 - 1

        # Get the attention allocation
        norm_attention_target = self._attention_positions[self._attention_target]
        norm_attention_actual_position = self._attention_positions[self._attention_actual_position]

        # Get the walking position - where the agent is and where the destination is
        norm_walking_position = self.normalise(self._walking_position, 0, self._void_total_walking_path_length, -1, 1)

        # Get the sign positions
        norm_next_sign_position = self.normalise(self._next_sign_position, 0, self._void_total_walking_path_length, -1, 1)

        # Get the walking speed
        norm_walking_speed_factor = self.normalise(self._preferred_walking_speed_random_factor, 0, 1, -1, 1)

        # Get the walking speed ratio
        norm_walking_speed_ratio = self.normalise(self._PPWS, 0, 1, -1, 1)

        # Get the reading speed
        norm_reading_speed = self.normalise(self._reading_speed_ratio, 0, 1, -1, 1)

        # Get the reading progress
        # norm_reading_progress = self.normalise(self._reading_progress, 0, self._text_length, -1, 1)
        # norm_prev_reading_progress = self.normalise(self._prev_reading_progress, 0, self._text_length, -1, 1)

        # Get the sign related observation
        norm_perception_factor = self.normalise(self._perception_factor, 0, 1, -1, 1)
        norm_sign_perceivable = 1 if self._sign_perceivable else -1
        # norm_num_seen_signs = self.normalise(len(self._seen_signs), 0, len(self._sign_perceivable_locations), -1, 1)
        # Action taken: change this to whether he has seen the last sign, but this can be represented by the following norm_fail_task

        # Get the task related information
        norm_fail_task = 1 if self._is_failed else -1

        # Get the tasks' weight
        w = self._weight

        stateful_info = np.array([remaining_ep_len_norm, norm_attention_target, norm_attention_actual_position, norm_walking_position,
                                  norm_walking_speed_factor, norm_walking_speed_ratio, norm_reading_speed,
                                  # norm_reading_progress, norm_prev_reading_progress, norm_num_seen_signs,
                                  norm_sign_perceivable, norm_perception_factor,
                                  norm_next_sign_position,
                                  norm_fail_task, w,
                                  ])

        # Observation space check
        if stateful_info.shape[0] != self._num_stateful_info:
            raise ValueError(f"The shape of stateful information observation is not correct! "
                             f"The allocated shape is: {self._num_stateful_info}, "
                             f"the actual shape is: {stateful_info.shape[0]}")

        return stateful_info

    def _get_terminate(self):

        terminate = False

        # Normal termination: determine by exceeding the maximum steps
        if self._steps >= self.ep_len:
            terminate = True

        # Abnormal termination: determine by failing the task
        if len(self._seen_signs) < self._next_sign_number - 1:
            terminate = True
            self._is_failed = True

        return terminate

    def _get_reward(self, terminate=False):

        # Aleksi's version of reward function
        w = self._weight
        r1 = self._PPWS
        if self._attention_actual_position == self._OHMD:
            r2 = self._reading_speed_ratio
        else:
            r2 = 0

        reward = w * r1 + (1 - w) * r2

        return reward
