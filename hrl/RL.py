import os
import yaml
import csv
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Callable

import gym
from gym import spaces
import torch as th
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from hrl.utils.write_video import write_video

from hrl.envs.supervisory_control.OcularMotorControl import OcularMotorControl
from hrl.envs.supervisory_control.LocomotionControl import LocomotionControl
from hrl.envs.supervisory_control.WordSelection import WordSelection
from hrl.envs.supervisory_control.SupervisoryControl import SupervisoryControl, SupervisoryControlWalkControl, SupervisoryControlWalkControlElapsedTime
from hrl.envs.supervisory_control.ScanEnvironment import ScanEnvironment

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
        )
        if self._mode == _MODES['continual_train'] or self._mode == _MODES['test']:
            print(
                f"    The loaded model checkpoints folder name is: {self._config_rl['train']['checkpoints_folder_name']}\n"
                f"    The loaded model checkpoint is: {self._config_rl['test']['loaded_model_name']}\n"
            )

        # Get an env instance for further constructing parallel environments.
        self._env = SupervisoryControlWalkControlElapsedTime() # SupervisoryControlWalkControl()  # SupervisoryControl()    # OcularMotorControl()    # LocomotionControl()

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

            # Configure the model - HRL - Ocular motor control
            if isinstance(self._env, OcularMotorControl):
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
                policy = "MultiInputPolicy"
            # Configure the model - HRL - Locomotion Control
            elif isinstance(self._env, LocomotionControl):
                policy_kwargs = dict(
                    features_extractor_class=NoVisionCombinedExtractor,
                    features_extractor_kwargs=dict(proprioception_features_dim=32,
                                                   stateful_information_features_dim=64),
                    activation_fn=th.nn.LeakyReLU,
                    net_arch=[64, 64],
                    log_std_init=-1.0,
                    normalize_images=False
                )
                policy = "MultiInputPolicy"
            # Configure the model - HRL - Word Selection, Read Background, Supervisory Control
            elif isinstance(self._env, WordSelection) or isinstance(self._env, ScanEnvironment) or isinstance(self._env, SupervisoryControl) or isinstance(self._env, SupervisoryControlWalkControl) or isinstance(self._env, SupervisoryControlWalkControlElapsedTime):
                if isinstance(self._env, WordSelection) or isinstance(self._env, SupervisoryControl) or isinstance(self._env, SupervisoryControlWalkControl) or isinstance(self._env, SupervisoryControlWalkControlElapsedTime):
                    features_dim = 128
                    net_arch = [256, 256]
                else:
                    features_dim = 16
                    net_arch = [64, 64]
                policy_kwargs = dict(
                    features_extractor_class=StatefulInformationExtractor,
                    features_extractor_kwargs=dict(features_dim=features_dim),
                    activation_fn=th.nn.LeakyReLU,
                    net_arch=net_arch,
                    log_std_init=-1.0,
                    normalize_images=False
                )
                policy = "MlpPolicy"
            else:
                raise ValueError(f'Invalid environment {self._env}.')

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
        grid_search_supervisory_control = self._config_rl['test']['grid_search_supervisory_control']['enable']
        grid_test_study4 = self._config_rl['test']['grid_test_study4']['enable']

        if self._mode == _MODES['debug']:
            print('\nThe MuJoCo env and tasks baseline ')
        elif self._mode == _MODES['test']:
            print('\nThe pre-trained RL model testing ')
            if grid_search_perturbation:
                self._grid_search_perturbation()
            if grid_search_selection:
                self._grid_search_selection()
            if grid_search_supervisory_control:
                self._grid_search_supervisory_control()
            if grid_test_study4:
                print(f'Now grid testing the Study 4')
                self._grid_test_study4()

        if ((grid_search_perturbation or grid_search_selection or grid_search_supervisory_control or grid_test_study4)
                and self._mode == _MODES['test']):
            pass

        if self._mode == _MODES['debug']:
            imgs = []
            imgs_eye = []
            for episode in range(1, self._num_episodes + 1):
                obs = self._env.reset()
                if not isinstance(self._env, WordSelection):
                    if isinstance(self._env, LocomotionControl):
                        imgs.append(self._env.render())
                    elif isinstance(self._env, OcularMotorControl):
                        imgs.append(self._env.render()[0])
                        # imgs_eye.append(self._env.render()[1])
                    else:
                        pass
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
                    if not isinstance(self._env, WordSelection):
                        if isinstance(self._env, LocomotionControl):
                            imgs.append(self._env.render())
                        elif isinstance(self._env, OcularMotorControl):
                            imgs.append(self._env.render()[0])
                            # imgs_eye.append(self._env.render()[1])
                        else:
                            pass
                    score += reward

                if isinstance(self._env, WordSelection):
                    imgs.append(self._env.omc_images)

                print(
                    f'Episode:{episode}     Score:{score} \n'
                    f'***************************************************************************************************\n'
                )
            return imgs, imgs_eye

        if (not (grid_search_perturbation or grid_search_selection or grid_search_supervisory_control or grid_test_study4)
                and self._mode == _MODES['test']):
            imgs = []
            imgs_eye = []
            L100_index_array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
            L50_index_array = np.array([13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])
            L0_index_array = np.array([25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36])

            omc_params = {
                'cells_mjidxs': L0_index_array,
                'perturbation_amp_tuning_factor': 0,
                'perturbation_amp_noise_scale': 0,
                'dwell_time': 0.5,
                # 'eye_x_rotation': 0,    # Store the last step's qos values, for smooth eyeball fixation transitions across cellss
                # 'eye_y_rotation': 0,
                'target_mjidx': 24,
                'layout': "L0",
            }

            ep_info = {
                'num_attention_switches': 0,
                'num_steps_on_incorrect_lane': 0,
                'num_attention_switches_on_margins': 0,
                'num_attention_switches_on_middle': 0,
                'reading_speed': 0,
                'layout': None,
                'event interval': None,
            }

            ep_study4_info = {
                'sign_positions': [],
                'steps': [],
                'weights': [],
                'step_wise_walking_positions': [],
                'step_wise_attentions': [],
                'step_wise_walking_speeds': [],
                'step_wise_reading_ratios': [],
                'step_wise_reading_progress': [],
            }

            for episode in range(1, self._num_episodes + 1):
                omc_params['target_mjidx'] += 1
                # use these when testing the ocular motor control
                if not isinstance(self._env, WordSelection):
                    if isinstance(self._env, LocomotionControl):
                        obs = self._env.reset()
                        imgs.append(self._env.render())
                    elif isinstance(self._env, SupervisoryControl):
                        obs = self._env.reset()
                    else:
                        if isinstance(self._env, OcularMotorControl):
                            obs = self._env.reset(load_model_params=omc_params)
                            imgs.append(self._env.render()[0])
                            # imgs_eye.append(self._env.render()[1])
                        elif isinstance(self._env, SupervisoryControlWalkControl) or isinstance(self._env, SupervisoryControlWalkControlElapsedTime):
                            obs = self._env.reset()
                        else:
                            obs = self._env.reset()
                else:
                    obs = self._env.reset()

                done = False
                score = 0
                info = {}

                while not done:
                    action, _states = self._model.predict(obs, deterministic=True)
                    obs, reward, done, info = self._env.step(action)
                    if not isinstance(self._env, WordSelection):
                        if isinstance(self._env, LocomotionControl):
                            imgs.append(self._env.render())
                        elif isinstance(self._env, SupervisoryControl) or isinstance(self._env, SupervisoryControlWalkControl) or isinstance(self._env, SupervisoryControlWalkControlElapsedTime):
                            pass
                        else:
                            imgs.append(self._env.render()[0])
                            # imgs_eye.append(self._env.render()[1])
                    score += reward

                if isinstance(self._env, WordSelection):
                    imgs.append(self._env.omc_images)
                elif isinstance(self._env, SupervisoryControl):
                    ep_info['num_attention_switches'] += info['num_attention_switches']
                    ep_info['num_attention_switches_on_margins'] += info['num_attention_switches_on_margins']
                    ep_info['num_attention_switches_on_middle'] += info['num_attention_switches_on_middle']
                    ep_info['num_steps_on_incorrect_lane'] += info['num_steps_on_incorrect_lane']
                    ep_info['reading_speed'] += info['reading_speed']
                    ep_info['layout'] = info['layout']
                    ep_info['event interval'] = info['event interval']
                elif isinstance(self._env, SupervisoryControlWalkControl) or isinstance(self._env, SupervisoryControlWalkControlElapsedTime):
                    ep_study4_info['sign_positions'].append(info['sign_positions'])
                    ep_study4_info['steps'].append(info['steps'])
                    ep_study4_info['weights'].append(info['weight'])
                    ep_study4_info['step_wise_walking_positions'].append(info['step_wise_walking_positions'])
                    ep_study4_info['step_wise_attentions'].append(info['step_wise_attentions'])
                    ep_study4_info['step_wise_walking_speeds'].append(info['step_wise_walking_speeds'])
                    ep_study4_info['step_wise_reading_ratios'].append(info['step_wise_reading_ratios'])
                    ep_study4_info['step_wise_reading_progress'].append(info['step_wise_reading_progress'])

                print(
                    f'Episode:{episode}     Score:{score} \n'
                    f'***************************************************************************************************\n'
                )

            if isinstance(self._env, SupervisoryControl):
                print(f"The number of episodes is {self._num_episodes}, the current condition is: {ep_info['layout']} and {ep_info['event interval']}\n"
                      f"The average number of attention switches is {ep_info['num_attention_switches'] / self._num_episodes}\n"
                      f"The average number of attention switches on margins is {ep_info['num_attention_switches_on_margins'] / self._num_episodes}\n"       
                      f"The average number of attention switches on middle is {ep_info['num_attention_switches_on_middle'] / self._num_episodes}\n"
                      f"The average number of steps on incorrect lane is {ep_info['num_steps_on_incorrect_lane'] / self._num_episodes}\n"
                      f"The average reading speed is {ep_info['reading_speed'] / self._num_episodes}\n")
            elif isinstance(self._env, SupervisoryControlWalkControl) or isinstance(self._env, SupervisoryControlWalkControlElapsedTime):
                # Write to a csv file
                df = pd.DataFrame(ep_study4_info)
                dir = os.path.dirname(os.path.realpath(__file__))
                root_dir = os.path.dirname(dir)
                # Get current local time
                current_time = datetime.now()
                # Format the time as "MM-DD-HH"
                formatted_time = current_time.strftime("%m-%d-%H-%M")
                # Create a folder for the study data
                study_data_folder_path = os.path.join(root_dir, 'study data', 'Study 4', formatted_time)
                if not os.path.exists(study_data_folder_path):
                    os.makedirs(study_data_folder_path)
                # Write the data file name with the formatted time
                study_data_file_path = os.path.join(study_data_folder_path, f'{formatted_time}-study4_data.csv')
                df.to_csv(study_data_file_path, index=False)

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

        # Initialize the lists for storing parameters

        df_columns = [
            'init_delta_t',
            'init_sigma_position_memory',
            'weight_memory_decay',
            'spatial_dist_coeff',
            'layout',
            'steps',
            'error'
        ]

        # Create or open CSV file with the column headers
        csv_directory = "envs/supervisory_control/results/"
        if not os.path.exists(csv_directory):
            os.makedirs(csv_directory)
        csv_save_path = os.path.join(csv_directory, "selection_results.csv")

        if not os.path.isfile(csv_save_path):
            with open(csv_save_path, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(df_columns)

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
                                obs = self._env.reset(grid_search_params=params)
                                done = False
                                score = 0
                                info = None

                                while not done:
                                    action, _states = self._model.predict(obs, deterministic=True)
                                    obs, reward, done, info = self._env.step(action)
                                    score += reward

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

    def _grid_search_supervisory_control(self):
        agent_name = self._config_rl['test']['grid_search_supervisory_control']['agent_name']
        layouts = self._config_rl['test']['grid_search_supervisory_control']['layouts']
        event_update_levels = self._config_rl['test']['grid_search_supervisory_control']['event_update_levels']
        num_episodes = self._config_rl['test']['grid_search_supervisory_control']['num_episodes']

        df_columns = [
            'agent_name',
            'layout',
            'event_update_level',
            'num_attention_switches',
            'num_attention_switches_margins',
            'num_attention_switches_middle',
            'inform_loss',
            'reading_speed',
            'total_steps',
            'attention_switches_time_steps_list',
        ]

        # Create or open CSV file with the column headers
        csv_directory = "envs/supervisory_control/results/"
        # Get the current time and format it as a string
        current_time_str = datetime.now().strftime('%m-%d_%H-%M')
        # Create or open a directory with the current time as its name
        csv_directory = os.path.join(csv_directory, current_time_str)
        if not os.path.exists(csv_directory):
            os.makedirs(csv_directory)
        csv_save_path = os.path.join(csv_directory, agent_name + "_supervisory_control_results.csv")

        if not os.path.isfile(csv_save_path):
            with open(csv_save_path, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(df_columns)

        for i in range(len(event_update_levels)):
            event_update_level = event_update_levels[i]
            for j in range(len(layouts)):
                layout = layouts[j]

                params = {
                    'event_update_level': event_update_level,
                    'layout': layout,
                }

                num_attention_switches_list = []
                num_attention_switches_margins_list = []
                num_attention_switches_middle_list = []
                inform_loss_list = []
                reading_speed_list = []
                total_steps_list = []
                attention_switches_time_steps_list_list = []

                for episode in range(1, num_episodes + 1):
                    obs = self._env.reset(grid_search_params=params)
                    done = False
                    score = 0
                    info = None

                    while not done:
                        action, _states = self._model.predict(obs, deterministic=True)
                        obs, reward, done, info = self._env.step(action)
                        score += reward

                    num_attention_switches_list.append(info['num_attention_switches'])
                    num_attention_switches_margins_list.append(info['num_attention_switches_on_margins'])
                    num_attention_switches_middle_list.append(info['num_attention_switches_on_middle'])
                    inform_loss_list.append(info['num_steps_on_incorrect_lane'])
                    reading_speed_list.append(info['reading_speed'])
                    total_steps_list.append(info['total_timesteps'])
                    attention_switches_time_steps_list_list.append(info['attention_switch_timesteps'])

                    # Save to CSV after each episode is finished
                    with open(csv_save_path, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            agent_name,
                            layout,
                            event_update_level,
                            info['num_attention_switches'],
                            info['num_attention_switches_on_margins'],
                            info['num_attention_switches_on_middle'],
                            info['num_steps_on_incorrect_lane'],
                            info['reading_speed'],
                            info['total_timesteps'],
                            info['attention_switch_timesteps'],
                        ])

                # # Save to CSV after each cluster of episodes is finished
                # with open(csv_save_path, 'a') as f:
                #     writer = csv.writer(f)
                #     writer.writerow([
                #         agent_name,
                #         layout,
                #         event_update_level,
                #         num_attention_switches_list,
                #         num_attention_switches_margins_list,
                #         num_attention_switches_middle_list,
                #         inform_loss_list,
                #         reading_speed_list,
                #         total_steps_list,
                #         attention_switches_time_steps_list_list,
                #     ])
        print(
            f"\n--------------------------------------------------------------------------------------------"
            f"\nThe supervisory control grid search results are stored in {csv_save_path}")

    def _grid_test_study4(self):
        # Download the configurations
        weight_range = self._config_rl['test']['grid_test_study4']['weight'][0]
        weight_stride = self._config_rl['test']['grid_test_study4']['weight'][1]
        # Add stride to the end of the range, or the last value will be missed
        weight_range[1] += weight_stride

        walk_factor_range = self._config_rl['test']['grid_test_study4']['walk_factor'][0]
        walk_factor_stride = self._config_rl['test']['grid_test_study4']['walk_factor'][1]
        walk_factor_range[1] += walk_factor_stride

        perception_factor_range = self._config_rl['test']['grid_test_study4']['perception_factor'][0]
        perception_factor_stride = self._config_rl['test']['grid_test_study4']['perception_factor'][1]
        perception_factor_range[1] += perception_factor_stride

        num_episodes = self._config_rl['test']['grid_test_study4']['num_episodes']

        # Initialize the lists for storing parameters

        df_columns = [
            'walking_path_finished',
            'rectangle_path_length',
            'ep_len',
            'sign_read',
            'sign_positions',
            'steps',
            'weights',
            'walk_factors',
            'perception_factors',
            'preferred_walking_speed',
            'step_wise_walking_positions',
            'step_wise_attentions',
            'step_wise_walking_speeds',
            'step_wise_reading_ratios',
            'step_wise_reading_progress',
            'score',
        ]

        # Get the csv file path
        dir = os.path.dirname(os.path.realpath(__file__))
        root_dir = os.path.dirname(dir)
        # Get current local time
        current_time = datetime.now()
        # Format the time as "MM-DD-HH"
        formatted_time = current_time.strftime("%m-%d-%H-%M")
        # Create a folder for the study data
        study_data_folder_path = os.path.join(root_dir, 'study data', 'Study 4', formatted_time)
        if not os.path.exists(study_data_folder_path):
            os.makedirs(study_data_folder_path)
        study_data_file_path = os.path.join(study_data_folder_path, f'{formatted_time}-study4_data.csv')

        # Open the csv file and write as the models run - in case the program crashes/unexpected shutdown
        if not os.path.isfile(study_data_file_path):
            with open(study_data_file_path, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(df_columns)

        for weight in np.arange(*weight_range, weight_stride):
            for walk_factor in np.arange(*walk_factor_range, walk_factor_stride):
                for perception_factor in np.arange(*perception_factor_range, perception_factor_stride):
                    params = {
                        'weight': weight,
                        'walk_factor': walk_factor,
                        'perception_factor': perception_factor,
                    }

                    for episode in range(1, num_episodes + 1):
                        obs = self._env.reset(params=params)
                        done = False
                        score = 0
                        info = None

                        while not done:
                            action, _states = self._model.predict(obs, deterministic=True)
                            obs, reward, done, info = self._env.step(action)
                            score += reward

                        # Save to CSV after each episode is finished
                        with open(study_data_file_path, 'a') as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                info['walking_path_finished'],
                                info['rectangle_path_length'],
                                info['ep_len'],
                                info['signs_read'],
                                info['sign_positions'],
                                info['steps'],
                                info['weights'],
                                info['walk_factors'],
                                info['perception_factors'],
                                info['preferred_walking_speed'],
                                info['step_wise_walking_positions'],
                                info['step_wise_attentions'],
                                info['step_wise_walking_speeds'],
                                info['step_wise_reading_ratios'],
                                info['step_wise_reading_progress'],
                                score,
                            ])

        print(
            f"\n--------------------------------------------------------------------------------------------"
            f"\nThe Study 4's grid test results are stored in {study_data_file_path}")

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
            if (self._config_rl['test']['grid_search_selection']['enable'] or self._config_rl['test']['grid_search_supervisory_control']['enable']
                    and self._mode == _MODES['test']):
                self._test()
            else:
                print(f"HRL testing. The video is being generated.")
                if isinstance(self._env, SupervisoryControl) or isinstance(self._env, ScanEnvironment) or isinstance(self._env, SupervisoryControlWalkControl) or isinstance(self._env, SupervisoryControlWalkControlElapsedTime):
                    self._test()
                else:
                    # Generate the results from the pre-trained model.
                    rgb_images, rgb_eye_images = self._test()
                    # Write a video. First get the rgb images, then identify the path.
                    # video_folder_path = f"C:/Users/91584/Desktop/{self._config_rl['train']['checkpoints_folder_name']}"
                    video_folder_path = os.path.join('training', 'videos', self._config_rl['train']['checkpoints_folder_name'])
                    if os.path.exists(video_folder_path) is False:
                        os.makedirs(video_folder_path)
                    video_name_prefix = self._mode + '_' + self._config_rl['train']['checkpoints_folder_name'] + '_' + self._loaded_model_name + '_'
                    video_path = os.path.join(video_folder_path, video_name_prefix + '.avi')

                    # write_video(
                    #     filepath=video_path,
                    #     fps=int(self._env.action_sample_freq),
                    #     rgb_images=rgb_images,
                    # )
        else:
            pass

    def __del__(self):
        # Close the environment.
        self._env.close()

        # Visualize the destructor.
        print(
            '\n\n***************************** RL pipeline ends. The MuJoCo environment of the pipeline has been destructed *************************************'
        )
