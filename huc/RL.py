import os

import yaml
import cv2
import csv
from tqdm import tqdm
import numpy as np
from typing import Callable

import gym
from gym import spaces

import torch as th
from torch import nn

import itertools
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from huc.utils.write_video import write_video
from huc.envs.mobile_reading.LocomotionRead import Read, PerturbationRead, WalkRead
from huc.envs.mobile_reading.StudyDemos import StudiesDemo
from huc.envs.locomotion.Locomotion import StraightWalk, SignWalk
from huc.envs.mobile_reading.MDPRead import MDPRead, MDPEyeRead, MDPResumeRead
from huc.envs.mobile_reading.POMDPRead import POMDPSelect

_MODES = {
    'train': 'train',
    'continual_train': 'continual_train',
    'test': 'test',
    'debug': 'debug',
    'interact': 'interact'
}


class VisionExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        """
        The custom cnn feature extractor.
        Ref: https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html#custom-feature-extractor
        :param observation_space: (gym.Space)
        :param features_dim: (int) Number of features extracted.
            This corresponds to the number of unit for the last layer.
        """
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=n_input_channels, out_channels=8, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2)),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2)),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2)),
            nn.LeakyReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            # (batch_size, hidden_channels * changed_width * changed_height)
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


class ProprioceptionExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        """
        Ref: Aleksi - https://github.com/BaiYunpeng1949/uitb-headsup-computing/blob/bf58d715b99ffabae4c2652f20898bac14a532e2/huc/RL.py#L75
        """
        super().__init__(observation_space, features_dim)
        # We assume a 1D tensor

        self.net = nn.Sequential(
            nn.Linear(in_features=observation_space.shape[0], out_features=features_dim),
            nn.LeakyReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.net(observations)


class StatefulInformationExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume a 1D tensor

        self.net = nn.Sequential(
            nn.Linear(in_features=observation_space.shape[0], out_features=features_dim),
            nn.LeakyReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.net(observations)


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, vision_features_dim: int = 256, proprioception_features_dim: int = 256, stateful_information_features_dim: int = 256):
        """
        Ref: Aleksi's code - https://github.com/BaiYunpeng1949/uitb-headsup-computing/blob/bf58d715b99ffabae4c2652f20898bac14a532e2/huc/RL.py#L90
        """
        super().__init__(observation_space, features_dim=vision_features_dim+proprioception_features_dim+stateful_information_features_dim)

        self.extractors = nn.ModuleDict({
            "vision": VisionExtractor(observation_space["vision"], vision_features_dim),
            "proprioception": ProprioceptionExtractor(observation_space["proprioception"], proprioception_features_dim),
            "stateful information": StatefulInformationExtractor(observation_space["stateful information"], stateful_information_features_dim),
        })

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, features_dim=vision_features_dim+proprioception_features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)


class NoVisionCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, proprioception_features_dim: int = 256, stateful_information_features_dim: int = 256):

        super().__init__(observation_space, features_dim=proprioception_features_dim+stateful_information_features_dim)

        self.extractors = nn.ModuleDict({
            "proprioception": ProprioceptionExtractor(observation_space["proprioception"], proprioception_features_dim),
            "stateful information": StatefulInformationExtractor(observation_space["stateful information"], stateful_information_features_dim),
        })

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, features_dim=vision_features_dim+proprioception_features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)


def linear_schedule(initial_value: float, min_value: float, threshold: float = 1.0) -> Callable[[float], float]:
    """
    Linear learning rate schedule. Adapted from the example at
    https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#learning-rate-schedule

    :param initial_value: Initial learning rate.
    :param min_value: Minimum learning rate.
    :param threshold: Threshold (of progress) when decay begins.
    :return: schedule that computes
    current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        if progress_remaining > threshold:
            return initial_value
        else:
            return min_value + (progress_remaining/threshold) * (initial_value - min_value)

    return func


class RL:
    def __init__(self, config_file):
        """
        This is the reinforcement learning pipeline where MuJoCo environments are created, and models are trained and tested.
        This pipeline is derived from my trials: context_switch.

        Args:
            config_file: the YAML configuration file that records the configurations.
        """
        # Read the configurations from the YAML file.
        with open(config_file) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        try:
            self._config_rl = config['rl']
        except ValueError:
            print('Invalid configurations. Check your config.yaml file.')

        # Specify the pipeline mode.
        self._mode = self._config_rl['mode']

        # Print the configuration
        if 'foveate' in self._config_rl['train']['checkpoints_folder_name']:
            print('Configuration:\n    The foveated vision is applied.')
        else:
            print('Configuration:\n    The foveated vision is NOT applied.')
        print(
            f"    The mode is: {self._config_rl['mode']} \n"
            f"        WARNING: The grid search is: {self._config_rl['test']['grid_search_selection']['enable']}\n"
            f"    The layout name is: {self._config_rl['test']['layout_name']}"
        )
        if self._mode == _MODES['continual_train'] or self._mode == _MODES['test']:
            print(
                f"    The loaded model checkpoints folder name is: {self._config_rl['train']['checkpoints_folder_name']}\n"
                f"    The loaded model checkpoint is: {self._config_rl['test']['loaded_model_name']}\n"
            )

        # Get an env instance for further constructing parallel environments.
        self._env = POMDPSelect()   #StudiesDemo()   # WalkRead()    # MDPEyeRead()      # SignWalk(), Read()

        # Initialise parallel environments
        self._parallel_envs = make_vec_env(
            env_id=self._env.__class__,
            n_envs=self._config_rl['train']["num_workers"],
            seed=None,
            vec_env_cls=SubprocVecEnv,
        )

        # Identify the modes and specify corresponding initiates.
        # Train the model, and save the logs and modes at each checkpoints.
        if self._mode == _MODES['train']:
            # Pipeline related variables.
            self._training_logs_path = os.path.join('training', 'logs')
            self._checkpoints_folder_name = self._config_rl['train']['checkpoints_folder_name']
            self._models_save_path = os.path.join('training', 'saved_models', self._checkpoints_folder_name)
            self._models_save_file_final = os.path.join(self._models_save_path,
                                                        self._config_rl['train']['checkpoints_folder_name'])
            # RL training related variable: total time-steps.
            self._total_timesteps = self._config_rl['train']['total_timesteps']

            if isinstance(self._env, POMDPSelect):
                policy_kwargs = dict(
                    features_extractor_class=StatefulInformationExtractor,
                    features_extractor_kwargs=dict(features_dim=128),
                    activation_fn=th.nn.LeakyReLU,
                    net_arch=[256, 256],
                    log_std_init=-1.0,
                    normalize_images=False
                )
                policy = 'MlpPolicy'
            else:
                # Configure the model - Initialise model that is run with multiple threads
                policy_kwargs = dict(
                    features_extractor_class=CustomCombinedExtractor,
                    features_extractor_kwargs=dict(vision_features_dim=128,
                                                   proprioception_features_dim=32,
                                                   stateful_information_features_dim=64),
                    activation_fn=th.nn.LeakyReLU,
                    net_arch=[256, 256],
                    log_std_init=-1.0,
                    normalize_images=False
                )
                policy = 'MultiInputPolicy'

            self._model = PPO(
                policy=policy,     # CnnPolicy, MlpPolicy, MultiInputPolicy
                env=self._parallel_envs,
                verbose=1,
                policy_kwargs=policy_kwargs,
                tensorboard_log=self._training_logs_path,
                n_steps=self._config_rl['train']["num_steps"],
                batch_size=self._config_rl['train']["batch_size"],
                # clip_range=linear_schedule(self._config_rl['train']["clip_range"]),
                # ent_coef=self._config_rl['train']["ent_coef"],
                # n_epochs=self._config_rl['train']["n_epochs"],
                learning_rate=linear_schedule(
                    initial_value=float(self._config_rl['train']["learning_rate"]["initial_value"]),
                    min_value=float(self._config_rl['train']["learning_rate"]["min_value"]),
                    threshold=float(self._config_rl['train']["learning_rate"]["threshold"]),
                ),
                device=self._config_rl['train']["device"],
                seed=42,
            )
        # Load the pre-trained models and test.
        elif self._mode == _MODES['test'] or self._mode == _MODES['continual_train']:
            # Pipeline related variables.
            self._loaded_model_name = self._config_rl['test']['loaded_model_name']
            self._checkpoints_folder_name = self._config_rl['train']['checkpoints_folder_name']
            self._models_save_path = os.path.join('training', 'saved_models', self._checkpoints_folder_name)
            self._loaded_model_path = os.path.join(self._models_save_path, self._loaded_model_name)
            # RL testing related variable: number of episodes and number of steps in each episodes.
            self._num_episodes = self._config_rl['test']['num_episodes']
            self._num_steps = self._env.ep_len
            # Load the model
            if self._mode == _MODES['test']:
                self._model = PPO.load(self._loaded_model_path, self._env)
            elif self._mode == _MODES['continual_train']:
                # Logistics.
                # Pipeline related variables.
                self._training_logs_path = os.path.join('training', 'logs')
                self._checkpoints_folder_name = self._config_rl['train']['checkpoints_folder_name']
                self._models_save_path = os.path.join('training', 'saved_models', self._checkpoints_folder_name)
                self._models_save_file_final = os.path.join(self._models_save_path,
                                                            self._config_rl['train']['checkpoints_folder_name'])
                # RL training related variable: total time-steps.
                self._total_timesteps = self._config_rl['train']['total_timesteps']
                # Model loading and register.
                self._model = PPO.load(self._loaded_model_path)
                self._model.set_env(self._parallel_envs)
        # The MuJoCo environment debugs. Check whether the environment and tasks work as designed.
        elif self._mode == _MODES['debug']:
            self._num_episodes = self._config_rl['test']['num_episodes']
            self._loaded_model_name = 'debug'
            # self._num_steps = self._env.num_steps
        # The MuJoCo environment demo display with user interactions, such as mouse interactions.
        elif self._mode == _MODES['interact']:
            pass
        else:
            pass

    def _train(self):
        """Add comments """
        # Save a checkpoint every certain steps, which is specified by the configuration file.
        # Ref: https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html
        # To account for the multi-envs' steps, save_freq = max(save_freq // n_envs, 1).
        save_freq = self._config_rl['train']['save_freq']
        n_envs = self._config_rl['train']['num_workers']
        save_freq = max(save_freq // n_envs, 1)
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=self._models_save_path,
            name_prefix='rl_model',
        )

        # Train the RL model and save the logs. The Algorithm and policy were given,
        # but it can always be upgraded to a more flexible pipeline later.
        self._model.learn(
            total_timesteps=self._total_timesteps,
            callback=checkpoint_callback,
        )

    def _continual_train(self):
        """
        This method perform the continual trainings.
        Ref: https://github.com/hill-a/stable-baselines/issues/599#issuecomment-569393193
        """
        save_freq = self._config_rl['train']['save_freq']
        n_envs = self._config_rl['train']['num_workers']
        save_freq = max(save_freq // n_envs, 1)
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=self._models_save_path,
            name_prefix='rl_model_continual',
        )

        self._model.learn(
            total_timesteps=self._total_timesteps,
            callback=checkpoint_callback,
            log_interval=1,
            tb_log_name=self._config_rl['test']['continual_logs_name'],
            reset_num_timesteps=False,
        )

        # Save the model as the rear guard.
        self._model.save(self._models_save_file_final)

    def _test(self):
        """
        This method generates the RL env testing results with or without a pre-trained RL model in a manual way.
        """
        grid_search_perturbation = self._config_rl['test']['grid_search_perturbation']['enable']
        grid_search_selection = self._config_rl['test']['grid_search_selection']['enable']
        if self._mode == _MODES['debug']:
            print('\nThe MuJoCo env and tasks baseline: ')
        elif self._mode == _MODES['test']:
            print('\nThe pre-trained RL model testing: ')
            if grid_search_perturbation:
                self._grid_search_perturbation()
            if grid_search_selection:
                self._grid_search_selection()

        if (grid_search_perturbation or grid_search_selection) and self._mode == _MODES['test']:
            pass
        else:
            imgs = []
            imgs_eye = []
            for episode in range(1, self._num_episodes + 1):
                obs = self._env.reset()
                imgs.append(self._env.render()[0])
                imgs_eye.append(self._env.render()[1])
                done = False
                score = 0
                info = None

                while not done:
                    if self._mode == _MODES['debug']:
                        action = self._env.action_space.sample()
                    elif self._mode == _MODES['test']:
                        action, _states = self._model.predict(obs, deterministic=True)
                    else:
                        action = 0
                    obs, reward, done, info = self._env.step(action)
                    imgs.append(self._env.render()[0])
                    imgs_eye.append(self._env.render()[1])
                    score += reward

                print(
                    f'Episode:{episode}     Score:{score} \n'
                    f'***************************************************************************************************\n'
                )

            return imgs, imgs_eye

        # if self._mode == _MODES['test']:
        #     # Use the official evaluation tool.
        #     evl = evaluate_policy(self._model, self._parallel_envs, n_eval_episodes=self._num_episodes, render=False)
        #     print('The evaluation results are: Mean {}; STD {}'.format(evl[0], evl[1]))

    def _grid_search_perturbation(self):
        # Download the configurations
        dwell_steps_range = self._config_rl['test']['grid_search']['dwell_steps'][0]
        dwell_steps_stride = self._config_rl['test']['grid_search']['dwell_steps'][1]
        dwell_steps_range[1] += dwell_steps_stride
        amp_tuning_factor_range = self._config_rl['test']['grid_search']['amp_tuning_factor'][0]
        amp_tuning_factor_stride = self._config_rl['test']['grid_search']['amp_tuning_factor'][1]
        amp_tuning_factor_range[1] += amp_tuning_factor_stride
        perturbation_amp_noise_scale_range = self._config_rl['test']['grid_search']['perturbation_amp_noise_scale'][0]
        perturbation_amp_noise_scale_stride = self._config_rl['test']['grid_search']['perturbation_amp_noise_scale'][1]
        perturbation_amp_noise_scale_range[1] += perturbation_amp_noise_scale_stride

        # Initialize the lists for storing parameters
        dwell_steps_list = []
        perturbation_amp_noise_scale_list = []
        perturbation_amp_tuning_factor_list = []
        end_steps_list = []
        csv_directory = None
        num_cells = None
        action_sample_freq = None

        for dwell_steps in np.arange(*dwell_steps_range, dwell_steps_stride):
            for perturbation_amp_tuning_factor in np.arange(*amp_tuning_factor_range, amp_tuning_factor_stride):
                for perturbation_amp_noise_scale in np.arange(*perturbation_amp_noise_scale_range,
                                                              perturbation_amp_noise_scale_stride):
                    dwell_steps = round(dwell_steps, 2)
                    perturbation_amp_tuning_factor = round(perturbation_amp_tuning_factor, 2)
                    perturbation_amp_noise_scale = round(perturbation_amp_noise_scale, 3)

                    dwell_steps_list.append(dwell_steps)
                    perturbation_amp_tuning_factor_list.append(perturbation_amp_tuning_factor)
                    perturbation_amp_noise_scale_list.append(perturbation_amp_noise_scale)

                    params = {
                        'dwell_steps': dwell_steps,
                        'perturbation_amp_tuning_factor': perturbation_amp_tuning_factor,
                        'perturbation_amp_noise_scale': perturbation_amp_noise_scale,
                        'mode': 1,
                    }

                    obs = self._env.reset(params=params)
                    done = False
                    score = 0
                    info = None

                    while not done:
                        action, _states = self._model.predict(obs, deterministic=True)
                        obs, reward, done, info = self._env.step(action)
                        score += reward

                    end_steps_list.append(info['end_steps'])
                    csv_directory = info['save_folder']
                    num_cells = info['num_cells']
                    action_sample_freq = info['action_sample_freq']

        # Write parameters and results to a dataframe
        df = pd.DataFrame({
            'dwell_steps': dwell_steps_list,
            'perturbation_amp_tuning_factor': perturbation_amp_tuning_factor_list,
            'perturbation_amp_noise_scale': perturbation_amp_noise_scale_list,
            'end_steps': end_steps_list,
        })
        # Process the data in dataframe, add reading speed and mobile reading speed degradation
        df['reading_speed_wps'] = num_cells * action_sample_freq / df['end_steps']
        # Select the row with the desired criteria
        reference_row = df[
            (df['perturbation_amp_tuning_factor'] == 0) & (df['perturbation_amp_noise_scale'] == 0)]
        # Group the dataframe by 'dwell_steps' and divide each group by the reference row
        df['walk_over_stand_percent'] = df.groupby('dwell_steps')['end_steps'].transform(lambda x: x.iloc[0] / x)
        # Save the plot as an image file (e.g., PNG, JPEG, PDF)
        directory = csv_directory
        if not os.path.exists(directory):
            os.makedirs(directory)
        csv_save_path = os.path.join(directory, "results.csv")
        df.to_csv(csv_save_path, index=False)
        print(
            f"\n--------------------------------------------------------------------------------------------"
            f"\nThe grid search results are stored in {csv_save_path}")

    def _grid_search_selection(self):
        # Download the configurations
        init_delta_t_range = self._config_rl['test']['grid_search_selection']['init_delta_t'][0]
        init_delta_t_stride = self._config_rl['test']['grid_search_selection']['init_delta_t'][1]
        # Add stride to the end of the range, or the last value will be missed
        init_delta_t_range[1] += init_delta_t_stride

        init_sigma_position_memory_range = self._config_rl['test']['grid_search_selection']['init_sigma_position_memory'][0]
        init_sigma_position_memory_stride = self._config_rl['test']['grid_search_selection']['init_sigma_position_memory'][1]
        init_sigma_position_memory_range[1] += init_sigma_position_memory_stride

        weight_memory_decay_range = self._config_rl['test']['grid_search_selection']['weight_memory_decay'][0]
        weight_memory_decay_stride = self._config_rl['test']['grid_search_selection']['weight_memory_decay'][1]
        weight_memory_decay_range[1] += weight_memory_decay_stride

        spatial_dist_coeff_range = self._config_rl['test']['grid_search_selection']['spatial_dist_coeff'][0]
        spatial_dist_coeff_stride = self._config_rl['test']['grid_search_selection']['spatial_dist_coeff'][1]
        spatial_dist_coeff_range[1] += spatial_dist_coeff_stride

        layouts = self._config_rl['test']['grid_search_selection']['layouts']
        num_episodes = self._config_rl['test']['grid_search_selection']['num_episodes']

        aggregate_df_columns = [
            'init_delta_t',
            'init_sigma_position_memory',
            'weight_memory_decay',
            'spatial_dist_coeff',
            'layout',
            'steps',
            'error'
        ]

        individual_df_columns = [
            'init_delta_t',
            'init_sigma_position_memory',
            'weight_memory_decay',
            'spatial_dist_coeff',
            'layout',
            'steps',
            'error',
            'true_last_word_mjidx',
            'gaze_positions_list',
            'true_last_word_memory_list',
            'gaze_positions_list',
            'all_words_belief_list',
        ]

        # Create or open CSV file with the column headers - Data collection of the aggregated results across all episodes/individuals
        csv_directory = "envs/mobile_reading/results/"
        if not os.path.exists(csv_directory):
            os.makedirs(csv_directory)
        csv_save_path = os.path.join(csv_directory, "selection_results.csv")

        if not os.path.isfile(csv_save_path):
            with open(csv_save_path, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(aggregate_df_columns)

        # Create or open CSV file with the column headers - Data collection of the results for each episode/individual
        individual_csv_save_path = os.path.join(csv_directory, "selection_individual_data.csv")
        individual_df_columns = individual_df_columns.copy()
        individual_df_columns.insert(5, 'episode_num')

        if not os.path.isfile(individual_csv_save_path):
            with open(individual_csv_save_path, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(individual_df_columns)

        for init_delta_t in np.arange(*init_delta_t_range, init_delta_t_stride):
            for init_sigma_position_memory in np.arange(*init_sigma_position_memory_range, init_sigma_position_memory_stride):
                for weight_memory_decay in np.arange(*weight_memory_decay_range, weight_memory_decay_stride):
                    for spatial_dist_coeff in np.arange(*spatial_dist_coeff_range, spatial_dist_coeff_stride):
                        for i in range(len(layouts)):
                            layout = layouts[i]

                            params = {
                                'init_delta_t': init_delta_t,
                                'init_sigma_position_memory': init_sigma_position_memory,
                                'weight_memory_decay': weight_memory_decay,
                                'spatial_dist_coeff': spatial_dist_coeff,
                                'layout': layout,
                            }

                            steps = []
                            errors = []

                            for episode in range(1, num_episodes + 1):
                                obs = self._env.reset(params=params)
                                done = False
                                score = 0
                                info = None

                                while not done:
                                    action, _states = self._model.predict(obs, deterministic=True)
                                    obs, reward, done, info = self._env.step(action)
                                    score += reward

                                # # Save individual episode data to its respective CSV   Uncomment when necessary
                                # with open(individual_csv_save_path, 'a') as f:
                                #     writer = csv.writer(f)
                                #     writer.writerow([
                                #         init_delta_t,
                                #         init_sigma_position_memory,
                                #         weight_memory_decay,
                                #         spatial_dist_coeff,
                                #         layout,
                                #         episode,  # Added episode number
                                #         info['steps'],
                                #         info['error'],
                                #         info['true_last_word_mjidx'],
                                #         info['gaze_word_belief_list'],
                                #         info['true_last_word_memory_list'],
                                #         info['gaze_positions_list'],
                                #         info['all_words_belief_list'],
                                #     ])

                                steps.append(info['steps'])
                                errors.append(info['error'])

                            avg_steps = np.mean(steps)
                            avg_errors = np.mean(errors)

                            # Save to CSV after each cluster of episodes is finished
                            with open(csv_save_path, 'a') as f:
                                writer = csv.writer(f)
                                writer.writerow([
                                    init_delta_t,
                                    init_sigma_position_memory,
                                    weight_memory_decay,
                                    spatial_dist_coeff,
                                    layouts[i],
                                    avg_steps,
                                    avg_errors
                                ])
        print(
            f"\n--------------------------------------------------------------------------------------------"
            f"\nThe grid search results are stored in {csv_save_path}")

    def run(self):
        """
        This method helps run the RL pipeline.
        Call it.
        """
        # Check train or not.
        if self._mode == _MODES['train']:
            self._train()
        elif self._mode == _MODES['continual_train']:
            self._continual_train()
        elif self._mode == _MODES['test'] or self._mode == _MODES['debug']:
            if self._config_rl['test']['grid_search_selection']['enable'] and self._mode == _MODES['test']:
                self._test()
            else:
                # Generate the results from the pre-trained model.
                rgb_images, rgb_eye_images = self._test()
                # Write a video. First get the rgb images, then identify the path.
                # video_folder_path = f"C:/Users/91584/Desktop/{self._config_rl['train']['checkpoints_folder_name']}"
                video_folder_path = os.path.join('training', 'videos', self._config_rl['train']['checkpoints_folder_name'])
                if os.path.exists(video_folder_path) is False:
                    os.makedirs(video_folder_path)
                layout_name = self._config_rl['test']['layout_name']
                video_name_prefix = self._mode + '_' + self._config_rl['train']['checkpoints_folder_name'] + '_' + self._loaded_model_name + '_' + layout_name
                video_path = os.path.join(video_folder_path, video_name_prefix + '.avi')
                write_video(
                    filepath=video_path,
                    fps=int(self._env._action_sample_freq),
                    rgb_images=rgb_images,
                    # width=rgb_images[0].shape[1],
                    # height=rgb_images[0].shape[0],
                )
                # Write the agent's visual perception
                video_path_eye = os.path.join(video_folder_path, video_name_prefix + '_eye.avi')
                write_video(
                    filepath=video_path_eye,
                    fps=int(self._env._action_sample_freq),
                    rgb_images=rgb_eye_images,
                    # width=rgb_eye_images[0].shape[1],
                    # height=rgb_eye_images[0].shape[0],
                )
        else:
            pass

    def __del__(self):
        # Close the environment.
        self._env.close()

        # Visualize the destructor.
        print(
            '\n\n***************************** RL pipeline ends. The MuJoCo environment of the pipeline has been destructed *************************************'
        )
