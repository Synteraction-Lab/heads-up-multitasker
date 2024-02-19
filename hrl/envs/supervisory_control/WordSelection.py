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


class WordSelection(Env):

    def __init__(self):
        """ Model select the last read word from the memory and read it again. Using Bayesian inference """
        # Get directory of this file
        directory = os.path.dirname(os.path.realpath(__file__))

        # Read the configurations from the YAML file.
        root_dir = os.path.dirname(os.path.dirname(directory))
        with open(os.path.join(root_dir, "config.yaml")) as f:
            self._config = yaml.load(f, Loader=yaml.FullLoader)

        # Load the MuJoCo model for this task
        self._model = mujoco.MjModel.from_xml_path(os.path.join(directory, "12cells-3layouts-small-font-v1.xml"))
        self._data = mujoco.MjData(self._model)
        mujoco.mj_forward(self._model, self._data)

        self.action_sample_freq = 20

        # Get the smart glasses plane mjidx in MuJoCo
        self._sgp_ils100_body_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY,
                                                        "smart-glass-pane-interline-spacing-100")
        self._sgp_ils50_body_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY,
                                                       "smart-glass-pane-interline-spacing-50")
        self._sgp_ils0_body_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY,
                                                      "smart-glass-pane-interline-spacing-0")
        # Get geom mjidxs (geoms that belong to "smart-glass-pane-interline-spacing-100")
        self._ils100_cells_mjidxs = np.where(self._model.geom_bodyid == self._sgp_ils100_body_mjidx)[0]
        self._ils50_cells_mjidxs = np.where(self._model.geom_bodyid == self._sgp_ils50_body_mjidx)[0]
        self._ils0_cells_mjidxs = np.where(self._model.geom_bodyid == self._sgp_ils0_body_mjidx)[0]

        # Initialize the MuJoCo layout related parameters
        self._L100 = "L100"
        self._L50 = "L50"
        self._L0 = "L0"
        self._FOUR = 4  # The number of cells in a row
        self._layouts = [self._L100, self._L50, self._L0]
        self._layout = None
        self._cells_mjidxs = None

        # Initialize the true last word mjidx
        self._true_last_word_mjidx = None
        # Initialize the gaze mjidx
        self._gaze_mjidx = None

        # Initialize the prior probability distribution
        self._prior_prob_dist = None
        # Initialize the likelihood probability distribution
        self._likelihood_prob_dist = None
        # Initialize the posterior probability distribution
        self._posterior_prob_dist = None
        # Initialize the belief probability distribution, actually it is the posterior probability distribution
        self._belief = None

        # Initialize the memory model that can be used to update the prior probability distribution (activation level from ACT-R)
        # Initial sigma position memory
        # Ref: Modeling Touch-based Menu Selection Performance of Blind Users via Reinforcement Learning
        self._init_sigma_position_memory_range = [0.5, 5]
        self._init_sigma_position_memory = None
        # The initial time interval of attention switch
        # 2s - time spent on the environmental task;
        # 4s - according to empirical results, most word selection time is finished within 2s; 2+2=4
        # Ref: Not all spacings are created equally
        self._init_delta_t_range = [2, 4]
        self._init_delta_t = None
        self._delta_t = None
        # The decaying factor Bi formular is a simplified version of the learning component from ACT-R
        # Ref: Modeling Touch-based Menu Selection Performance of Blind Users via Reinforcement Learning
        # Ref: Adapting User Interfaces with Model-based Reinforcement Learning
        self._rho = 0.5
        self._sigma_position_memory = None

        # Initialize the likelihood related uncertainty
        # Relate this to the fovea size, e.g., 2 times of the fovea vision (2 * +- 1~2 degrees --> 0.026 ~ 0.0523)
        # Ref: Adapting User Interfaces with Model-based Reinforcement Learning
        # We assume agent samples the words that are nearer to the true last word
        self._fovea_degrees = 2
        self._fovea_size = None
        self._spatial_dist_coeff_range = [1, 5]
        self._spatial_dist_coeff = None
        self._sigma_likelihood = None

        # Initialize the memory decay weight
        self._weight_memory_decay_range = [0.1, 1]
        self._weight_memory_decay = None

        # Initialize the dwell time
        self._dwell_time = 0.2  # The time to dwell on a target

        # Initialize the log related parameters
        self._true_last_word_belief_list = None
        self._true_last_word_memory_decay_list = None

        # Initialise RL related thresholds and counters
        self._steps = None
        self.ep_len = 100
        # Define the training related parameters
        self._epsilon = 1e-100

        # Define the observation space
        self._num_stateful_info = 14
        self.observation_space = Box(low=-1, high=1, shape=(self._num_stateful_info,))

        # Define the action space - decision on attention distribution;
        self._action_gaze_idx = 0
        self._action_eye_rotate_x_idx = 1
        self._action_eye_rotate_y_idx = 2
        self._action_up_range = [0, 1]
        self._action_down_range = [1, 2]
        self._action_left_range = [2, 3]
        self._action_right_range = [3, 4]
        self._action_select_range = [4, 5]
        self._action_name = None
        self.action_space = Box(low=-1, high=1, shape=(1,))

        # Initialize the pre-trained RL model
        if self._config['rl']['mode'] == 'test':
            # Load the MuJoCo model for the ocular motor control task (omc) - we create 1 env for each training env
            self._omc_env = OcularMotorControl()

            # Load the pre-trained low level task model - Ocular motor control model
            self._checkpoints_dir_name = "0730_hrl_ocular_motor_control_100m"
            self._loaded_model_name = "rl_model_50000000_steps"
            self._loaded_model_path = os.path.join(root_dir, 'training', 'saved_models',
                                                   self._checkpoints_dir_name, self._loaded_model_name)
            # omc stands for ocular motor control
            self._omc_model = PPO.load(self._loaded_model_path, self._omc_env)

            # Initialize the variables for the ocular motor control task
            self._omc_tuples = None
            self._omc_params = None
            self.omc_images = None

    def reset(self, grid_search_params=None, load_model_params=None):

        # Reset MuJoCo sim
        mujoco.mj_resetData(self._model, self._data)

        # Reset the log related parameters
        self._true_last_word_belief_list = []
        self._true_last_word_memory_decay_list = []
        self._action_name = ''

        # Reset the variables and counters
        self._steps = 0

        # TODO add hrl load model mode

        # Initialize the stochastic memory model related parameters
        self._init_delta_t = np.random.uniform(*self._init_delta_t_range)
        self._delta_t = self._init_delta_t
        self._init_sigma_position_memory = np.random.uniform(*self._init_sigma_position_memory_range)
        self._weight_memory_decay = np.random.uniform(*self._weight_memory_decay_range)

        # Initialize the stochastic word selection destination related parameters
        self._layout = np.random.choice(self._layouts)

        # Initialize the stochastic local search - likelihood function related parameters
        mujoco.mj_forward(self._model, self._data)
        self._fovea_size = np.tan(np.radians(self._fovea_degrees / 2)) * self._data.geom(self._ils0_cells_mjidxs[0]).xpos[1]
        self._spatial_dist_coeff = np.random.uniform(*self._spatial_dist_coeff_range)
        self._sigma_likelihood = self._fovea_size * self._spatial_dist_coeff

        # Configure the stochastic hyperparameters in test mode
        if self._config['rl']['mode'] == 'test':
            # Normal testings
            if grid_search_params is None:
                self._init_delta_t = 2
                self._init_sigma_position_memory = 0.5
                self._weight_memory_decay = 0.8
                self._layout = self._L0
                self._spatial_dist_coeff = 4
                self._sigma_likelihood = self._fovea_size * self._spatial_dist_coeff
            # Testing - grid search
            else:
                self._init_delta_t = grid_search_params['init_delta_t']
                self._init_sigma_position_memory = grid_search_params['init_sigma_position_memory']
                self._weight_memory_decay = grid_search_params['weight_memory_decay']
                self._layout = grid_search_params['layout']
                self._spatial_dist_coeff = grid_search_params['spatial_dist_coeff']
                self._sigma_likelihood = self._fovea_size * self._spatial_dist_coeff
                print(f"Grid Search Testing with params: {grid_search_params}\n")

        # Initialize the scene after deciding the layout
        if self._layout == self._L100:
            self._cells_mjidxs = self._ils100_cells_mjidxs
        elif self._layout == self._L50:
            self._cells_mjidxs = self._ils50_cells_mjidxs
        elif self._layout == self._L0:
            self._cells_mjidxs = self._ils0_cells_mjidxs
        else:
            raise ValueError("The layout name is not correct!")

        # Initialize the true last word read mjidx
        self._true_last_word_mjidx = np.random.choice(self._cells_mjidxs)

        # Initialize the gaze mjidx according to the vicinity of the true last word read mjidx probability distribution
        # If randomly sample, the likelihood function will be very destructive, preventing the agent from updating the correct belief
        # If one needs to let agent make mistakes, may use gaze initialization with more noise/stochasticity/increasing the sigma
        self._init_gaze_mjidx()

        # Initialize the probability distributions as uniform distributions
        self._prior_prob_dist = np.ones(len(self._cells_mjidxs)) / len(self._cells_mjidxs)
        self._posterior_prob_dist = np.ones(len(self._cells_mjidxs)) / len(self._cells_mjidxs)

        # Update the internal representation - belief
        self._update_belief()

        # Set up the whole scene by confirming the initializations
        mujoco.mj_forward(self._model, self._data)

        if self._config['rl']['mode'] == 'test':
            # Reset the ocular motor control agent
            self._omc_params = {
                'cells_mjidxs': self._cells_mjidxs.copy(),
                'perturbation_amp_tuning_factor': 0,
                'perturbation_amp_noise_scale': 0,
                'dwell_time': self._dwell_time,
                # 'eye_x_rotation': 0,    # Store the last step's qos values, for smooth eyeball fixation transitions across cellss
                # 'eye_y_rotation': 0,
                'target_mjidx': self._gaze_mjidx,
                'layout': self._layout,
            }

            # Reset the pre-trained omc agent
            self._omc_tuples = {
                'obs': self._omc_env.reset(load_model_params=self._omc_params),
                'done': False,
                'score': 0,
                'info': {},
            }

            if self._config['rl']['mode'] == 'test' or self._config['rl']['mode'] == 'debug':
                self.omc_images = []
                self.omc_images.append(self._omc_env.render()[0])

        return self._get_obs()

    def render(self, mode="rgb_array"):
        return

    def step(self, action):
        # Action a from state s - attention allocation - Reading is a sequential process (MDP)
        # Action for attention deployment - Make a decision - Take the current one to read or keep searching
        # Move one word left, right, up, down, select the word

        action_gaze = self.normalise(action[self._action_gaze_idx], -1, 1, 0, 5)

        finish_search = False

        # Select the word, finish searching
        if self._action_select_range[0] < action_gaze <= self._action_select_range[1]:
            finish_search = True
            self._action_name = 'select'
        # Move the gaze to the next word above
        elif self._action_up_range[0] <= action_gaze <= self._action_up_range[1]:
            if self._gaze_mjidx - self._FOUR >= self._cells_mjidxs[0]:
                self._gaze_mjidx -= self._FOUR
            self._action_name = 'up'
        # Move the gaze to the next word below
        elif self._action_down_range[0] < action_gaze <= self._action_down_range[1]:
            if self._gaze_mjidx + self._FOUR <= self._cells_mjidxs[-1]:
                self._gaze_mjidx += self._FOUR
            self._action_name = 'down'
        # Move the gaze to the next word left
        elif self._action_left_range[0] < action_gaze <= self._action_left_range[1]:
            if self._gaze_mjidx - 1 >= self._cells_mjidxs[0]:
                self._gaze_mjidx -= 1
            self._action_name = 'left'
        # Move the gaze to the next word right
        elif self._action_right_range[0] < action_gaze <= self._action_right_range[1]:
            if self._gaze_mjidx + 1 <= self._cells_mjidxs[-1]:
                self._gaze_mjidx += 1
            self._action_name = 'right'
        else:
            raise ValueError(f"The action is not correct! It is: {action_gaze}")

        # Start a new fixation (a macro action)
        # Update the gaze position by moving the eyeball in the scene
        # In the training mode we don't do anything
        if self._config['rl']['mode'] == 'train' or self._config['rl']['mode'] == 'continual_train':
            pass
        # In the testing mode we move the eyeball in the scene using the pre-trained model
        elif self._config['rl']['mode'] == 'test' or self._config['rl']['mode'] == 'debug':
            # Update the ocular motor control agent parameters
            self._omc_params['target_mjidx'] = self._gaze_mjidx
            obs = self._omc_env.reset(load_model_params=self._omc_params)
            done = False
            while not done:
                action, _states = self._omc_model.predict(obs, deterministic=True)
                obs, reward, done, info = self._omc_env.step(action)
                self.omc_images.append(self._omc_env.render()[0])
            print(f"Using the established OMC model\n")
        else:
            raise ValueError(f"The mode is not correct! It is: {self._config['rl']['mode']}")

        # State s'
        self._steps += 1
        reward = 0
        terminate = False
        info = {}

        # Update the mental state / internal representation / belief
        self._update_belief()

        # Apply the time penalty
        mt_step_wise = self._dwell_time * self.action_sample_freq
        time_penalty = -0.1 * mt_step_wise
        reward += time_penalty

        # If all materials are read, give a big bonus reward
        if self._steps >= self.ep_len or finish_search:
            # Termination of the episode
            terminate = True

            # Reward estimation - final milestone rewards
            # Reward estimate - reward shaping based on the word selection (gaze) accuracy
            euclidean_distance = self.euclidean_distance(self._gaze_mjidx, self._true_last_word_mjidx)
            selection_accuracy = 10 * (np.exp(-0.1 * euclidean_distance) - 1)
            reward += selection_accuracy

            # Info updating
            info['steps'] = self._steps
            info['error'] = euclidean_distance

            if self._config['rl']['mode'] == 'test' and \
                    self._config['rl']['test']['grid_search_selection']['enable'] == False \
                    or self._config['rl']['mode'] == 'debug':
                # # Plot the belief and memory decay
                # self._plot_belief_and_memory_decay()
                # # Record the video of ocular motor control
                # self._write_omc_video()
                pass

        if self._config['rl']['mode'] == 'test' and \
                self._config['rl']['test']['grid_search_selection']['enable'] == False \
                or self._config['rl']['mode'] == 'debug':
            # Print logs
            gaze_idx = np.where(self._cells_mjidxs == self._gaze_mjidx)[0][0]
            true_last_word_idx = np.where(self._cells_mjidxs == self._true_last_word_mjidx)[0][0]
            print(
                  f"The current layout is: {self._cells_mjidxs[0]}\n"
                  f"Last step's action a is: {action_gaze}, {self._action_name}  "
                  f"The current steps is: {self._steps}, "
                  f"Finish search is: {finish_search}, "
                  # f"The prior probability distribution is: {self._prior_prob_dist},\n"
                  # f"The likelihood is: {self._likelihood_prob_dist},\n"
                  # f"The s' belief is: {self._belief}\n"
                  f"the r(s'|a, s) reward is: {reward}, \n"
                  f"the gaze position is: {self._gaze_mjidx}, its belief is {self._belief[gaze_idx]}\n"
                  f"the true last word is: {self._true_last_word_mjidx}, its belief is {self._belief[true_last_word_idx]}\n"
            )

        return self._get_obs(), reward, terminate, info

    @staticmethod
    def normalise(x, x_min, x_max, a, b):
        # Normalise x (which is assumed to be in range [x_min, x_max]) to range [a, b]
        return (b - a) * ((x - x_min) / (x_max - x_min)) + a

    @staticmethod
    def euclidean_distance(x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    @staticmethod
    def detect_invalid_array(array, array_name):
        if not np.all(array):
            print(f"Array contains NaN or Inf, the array is: {array_name}\n"
                  f"{array}\n"
                  f"")

    def _get_obs(self):
        """ Get the observation of the environment state """
        # Get the stateful information observation - normalize to [-1, 1]
        remaining_ep_len_norm = (self.ep_len - self._steps) / self.ep_len * 2 - 1

        gaze_target_mjidx_norm = self.normalise(self._gaze_mjidx, self._cells_mjidxs[0], self._cells_mjidxs[-1], -1, 1)
        belief = self._belief.copy()

        stateful_info = np.array(
            [remaining_ep_len_norm, gaze_target_mjidx_norm,
             *belief]
        )

        # Observation space check
        if stateful_info.shape[0] != self._num_stateful_info:
            raise ValueError(f"The shape of stateful information observation is not correct! "
                             f"The allocated shape is: {self._num_stateful_info}, the actual shape is: {stateful_info.shape[0]}")

        return stateful_info

    def _init_gaze_mjidx(self):
        # Set everything
        mujoco.mj_forward(self._model, self._data)

        # Initialize the gaze position probability distribution
        gaze_prob_distribution = np.zeros(len(self._cells_mjidxs))

        # Get the true attention target's position
        mu_xpos = self._data.geom(self._true_last_word_mjidx).xpos

        # Calculate every cell's distance to the target and get the attention distribution (probability)
        for mjidx in self._cells_mjidxs:
            xpos = self._data.geom(mjidx).xpos
            dist = np.linalg.norm(xpos - mu_xpos)
            idx = np.where(self._cells_mjidxs == mjidx)[0][0]
            gaze_prob_distribution[idx] = np.exp(-0.5 * (dist / self._sigma_likelihood) ** 2)
        # Normalization
        gaze_prob_distribution += self._epsilon
        gaze_prob_distribution /= np.sum(gaze_prob_distribution)

        # Initialize the gaze position
        while True:
            self._gaze_mjidx = np.random.choice(self._cells_mjidxs, p=gaze_prob_distribution)
            if self._gaze_mjidx != self._true_last_word_mjidx:
                break

    def _update_prior(self):
        """
         Get the prior probability distribution of the attention - the probability of attending to each word
         Last step's posterior is this step's prior
        """
        self._prior_prob_dist = self._posterior_prob_dist.copy()

        """
        Update the prior probability distribution of the attention corrupted by the memory decays
        """
        # Update the elapsed time in second
        # self._delta_t = self._init_delta_t + self._steps / self.action_sample_freq
        self._delta_t += self._dwell_time
        # Update the sigma position memory
        self._sigma_position_memory = self._init_sigma_position_memory / (1 + self._delta_t ** (-self._rho))

        # Corrupt the memory by adding uncertainty into the distribution -->
        # lower activation level B(mi), bigger sigma, then broader distribution
        true_last_word_idx = np.where(self._cells_mjidxs == self._true_last_word_mjidx)[0][0]
        mu = true_last_word_idx
        init_memory_decay_prob_dist = np.zeros(self._cells_mjidxs.shape[0])
        memory_decay_prob_dist = np.exp(-(np.arange(init_memory_decay_prob_dist.shape[0]) - mu)**2 / (2 * self._sigma_position_memory**2))
        memory_decay_prob_dist += self._epsilon     # Prevent invalid values in array
        memory_decay_prob_dist /= np.sum(memory_decay_prob_dist)
        self.detect_invalid_array(memory_decay_prob_dist, "memory_decay_prob_dist")     # Detect in debug, in case crash during training

        # Update the prior probability distribution with the weighted memory decay
        weight_decay = self._weight_memory_decay
        weight_prior = 1 - weight_decay
        self._prior_prob_dist = weight_decay * memory_decay_prob_dist + weight_prior * self._prior_prob_dist
        self._prior_prob_dist += self._epsilon   # Prevent invalid values in array
        # Normalise the prior probability distribution
        self._prior_prob_dist /= np.sum(self._prior_prob_dist)
        self.detect_invalid_array(self._prior_prob_dist, "prior_prob_dist")

        # Log the memory decay in the test mode
        if self._config['rl']['mode'] == 'test' and \
                self._config['rl']['test']['grid_search_selection']['enable'] == False:
            idx = np.where(self._cells_mjidxs == self._true_last_word_mjidx)[0][0]
            self._true_last_word_memory_decay_list.append(memory_decay_prob_dist[idx])

            # print(f"The memory decay weight is: {weight_decay}, the sigma likelihood is: {self._sigma_likelihood}\n"
            #       f"The memory decay is: {memory_decay_prob_dist}, the target's is: {memory_decay_prob_dist[idx]}\n"
            #       f"The updated prior probability distribution is: {self._prior_prob_dist}, "
            #       f"the target's is: {self._prior_prob_dist[idx]}\n")

    def _update_likelihood(self):
        """
        Calculate the likelihood function: assumption
        agents' strategy is to sample the word close to the target last word, and its probability is associated with
        the distance between the target last word and the sampled word

        It feels like model the environment.
        """
        # Set up the simulation
        mujoco.mj_forward(self._model, self._data)

        # Get the observation's spatial position of the agent's gaze
        gaze_xpos = self._data.geom(self._gaze_mjidx).xpos

        # Reset the attention distribution
        self._likelihood_prob_dist = np.zeros(self._cells_mjidxs.shape[0])

        # Get the probability distribution of P(observe gaze | "last word" = w1~wn)
        for i in range(len(self._cells_mjidxs)):
            mjidx = self._cells_mjidxs[i]
            xpos = self._data.geom(mjidx).xpos
            # Calculate the distance
            dist = np.linalg.norm(xpos - gaze_xpos)
            # Get the likelihood
            self._likelihood_prob_dist[i] = np.exp(-0.5 * (dist / self._sigma_likelihood) ** 2)

        # Prevent invalid values in array
        self._likelihood_prob_dist += self._epsilon

        # Normalization
        self._likelihood_prob_dist /= np.sum(self._likelihood_prob_dist)
        self.detect_invalid_array(self._likelihood_prob_dist, "likelihood_prob_dist")

    def _update_posterior(self):
        """Get the posterior probability distribution of the attention according to the Bayes' rules"""
        # Get the posterior probability distribution = prior * likelihood
        self._posterior_prob_dist = self._prior_prob_dist.copy() * self._likelihood_prob_dist.copy()
        # Prevent invalid values in array
        self._posterior_prob_dist += self._epsilon
        # Normalization
        self._posterior_prob_dist /= np.sum(self._posterior_prob_dist)
        self.detect_invalid_array(self._posterior_prob_dist, "posterior_prob_dist")

    def _update_belief(self):
        """Get the belief of the agent's attention distribution"""
        self._update_prior()
        self._update_likelihood()
        self._update_posterior()
        self._belief = self._posterior_prob_dist.copy()

        # Log the belief in the test mode
        if self._config['rl']['mode'] == 'test' and \
                self._config['rl']['test']['grid_search_selection']['enable'] == False:
            idx = np.where(self._cells_mjidxs == self._true_last_word_mjidx)[0][0]
            self._true_last_word_belief_list.append(self._belief[idx])

    def _plot_belief_and_memory_decay(self):
        # Create a new figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

        # Plot the belief update on the first subplot
        x_values_belief = range(1, len(self._true_last_word_belief_list) + 1)
        ax1.plot(x_values_belief, self._true_last_word_belief_list, marker='o', linestyle='-', color='b')
        ax1.set_xlabel('Times of Actions')
        ax1.set_ylabel('Belief of the True Last Word')
        ax1.set_title('Belief Update')
        ax1.grid(True)

        # Plot the memory decay on the second subplot
        x_values_memory_decay = range(1, len(self._true_last_word_memory_decay_list) + 1)
        ax2.plot(x_values_memory_decay, self._true_last_word_memory_decay_list, marker='o', linestyle='-', color='b')
        ax2.set_xlabel('Times of Actions')
        ax2.set_ylabel('Memory decay of the True Last Word')
        ax2.set_title('Memory Decay')
        ax2.grid(True)

        # Adjust layout to prevent overlapping titles and labels
        plt.tight_layout()

        # Save the plot to the specified location (replace 'save_path' with the desired file path)
        directory = os.path.dirname(os.path.realpath(__file__))
        save_path = os.path.join(directory, "results", "belief_and_memory_decay.png")
        plt.savefig(save_path)
        print(f"The combined plot is saved to: {save_path}")

    def _write_omc_video(self):
        directory = os.path.dirname(os.path.realpath(__file__))
        video_path = os.path.join(directory, "results", 'ocular_motor_control_agent.avi')
        rgb_images = self.omc_images
        write_video(
            filepath=video_path,
            fps=int(self._omc_env.action_sample_freq),
            rgb_images=rgb_images,
        )
