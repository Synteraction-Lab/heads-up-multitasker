import numpy as np
from collections import Counter, deque
import mujoco
import os

from gym import Env
from gym.spaces import Box, Dict

import yaml
from collections import deque
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter

from huc.utils.rendering import Camera, Context


class MDPRead(Env):

    def __init__(self):
        """ Model sequential reading - read words in the layout one by one """
        # Get directory of this file
        directory = os.path.dirname(os.path.realpath(__file__))

        # Read the configurations from the YAML file.
        root_dir = os.path.dirname(os.path.dirname(directory))
        with open(os.path.join(root_dir, "config.yaml")) as f:
            self._config = yaml.load(f, Loader=yaml.FullLoader)

        # Load the MuJoCo model
        self._model = mujoco.MjModel.from_xml_path(os.path.join(directory, "mobile-read-v2.xml"))
        self._data = mujoco.MjData(self._model)
        mujoco.mj_forward(self._model, self._data)

        # Define how often policy is queried
        self._action_sample_freq = 20
        self._frame_skip = int((1 / self._action_sample_freq) / self._model.opt.timestep)

        # Get the joints idx in MuJoCo
        self._eye_joint_x_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "eye-joint-x")
        self._eye_joint_y_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "eye-joint-y")

        # Get the motors idx in MuJoCo
        self._eye_x_motor_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, "eye-x-motor")
        self._eye_x_motor_translation_range = self._model.actuator_ctrlrange[self._eye_x_motor_mjidx]
        self._eye_y_motor_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, "eye-y-motor")
        self._eye_y_motor_translation_range = self._model.actuator_ctrlrange[self._eye_y_motor_mjidx]

        self._sgp_ils100_body_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY,
                                                        "smart-glass-pane-interline-spacing-100")
        # Get targets (geoms that belong to "smart-glass-pane-interline-spacing-100")
        self._ils100_cells_mjidxs = np.where(self._model.geom_bodyid == self._sgp_ils100_body_mjidx)[0]

        self._MEMORY_PAD_VALUE = -2
        # The pool of targets that can be attended to, -1 stands for attention switch away from reading
        self._attention_pool = np.array([self._MEMORY_PAD_VALUE, *self._ils100_cells_mjidxs.copy()])

        # Mental state related variables
        self._deployed_attention_target_mjidx = None
        self._attention_action_threshold = 0
        self._mental_state = {  # The mental state of the agent, or it can be called the internal state
            # The previous state of the words that have been read
            'prev_reading_memory': np.full((len(self._ils100_cells_mjidxs),), self._MEMORY_PAD_VALUE),
            # The memory of the words that have been read
            'reading_memory': np.full((len(self._ils100_cells_mjidxs),), self._MEMORY_PAD_VALUE),
            # The memory of the words that have been read
            'attention': self._deployed_attention_target_mjidx,  # The intended target mjidx
            'page_finish': False,  # Whether the page is finished
        }

        # Define the target idx probability distribution
        self._VISUALIZE_RGBA = [1, 1, 0, 1]
        self._DFLT_RGBA = [0, 0, 0, 1]

        self._dwell_steps = int(0.2 * self._action_sample_freq)  # The number of steps to dwell on a target

        # Initialise RL related thresholds and counters
        self._steps = None
        self._on_target_steps = None
        self.ep_len = 100

        # Define the observation space
        width, height = 80, 80
        self._num_stk_frm = 1
        self._vision_frames = None
        self._qpos_frames = None
        self._num_stateful_info = 16

        self.observation_space = Box(low=-1, high=1, shape=(self._num_stateful_info,))

        # Define the action space
        # 1st: decision on attention distribution;
        # 2nd and 3rd: eyeball rotations;
        self._action_attention_deployment_idx = 0
        self._action_eye_rotate_x_idx = 1
        self._action_eye_rotate_y_idx = 2
        self.action_space = Box(low=-1, high=1, shape=(3,))

        # Initialize the context and camera
        context = Context(self._model, max_resolution=[1280, 960])
        self._eye_cam = Camera(context, self._model, self._data, camera_id="eye", resolution=[width, height],
                               maxgeom=100,
                               dt=1 / self._action_sample_freq)
        self._env_cam = Camera(context, self._model, self._data, camera_id="env", maxgeom=100,
                               dt=1 / self._action_sample_freq)
        self._eye_cam_fovy = self._model.cam_fovy[mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, "eye")]

    @staticmethod
    def normalise(x, x_min, x_max, a, b):
        # Normalise x (which is assumed to be in range [x_min, x_max]) to range [a, b]
        return (b - a) * ((x - x_min) / (x_max - x_min)) + a

    @staticmethod
    def euclidean_distance(x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def render(self, mode="rgb_array"):
        rgb, _ = self._env_cam.render()
        rgb_eye, _ = self._eye_cam.render()
        return rgb, rgb_eye

    def _get_obs(self):
        """ Get the observation of the environment state """
        # Get the stateful information observation - normalize to [-1, 1]
        remaining_ep_len_norm = (self.ep_len - self._steps) / self.ep_len * 2 - 1
        remaining_dwell_steps_norm = (self._dwell_steps - self._on_target_steps) / self._dwell_steps * 2 - 1
        attention_deploy_target_mjidx_norm = self.normalise(self._deployed_attention_target_mjidx,
                                                            self._attention_pool[0], self._attention_pool[-1], -1, 1)

        reading_memory = self._mental_state['reading_memory'].copy()
        reading_memory_norm = self.normalise(reading_memory, self._MEMORY_PAD_VALUE, self._ils100_cells_mjidxs[-1], -1,
                                             1)

        page_finish_norm = -1 if self._mental_state['page_finish'] == False else 1

        stateful_info = np.array(
            [remaining_ep_len_norm, remaining_dwell_steps_norm, attention_deploy_target_mjidx_norm, page_finish_norm,
             *reading_memory_norm]
        )

        # Observation space check
        if stateful_info.shape[0] != self._num_stateful_info:
            raise ValueError("The shape of stateful information observation is not correct!")

        return stateful_info

    def reset(self, params=None):

        # Reset MuJoCo sim
        mujoco.mj_resetData(self._model, self._data)

        # Reset the variables and counters
        self._steps = 0
        self._on_target_steps = 0

        # Mental state related variables
        # Randomly choose the intended target mjidx
        init_deployed_attention = np.random.choice(self._ils100_cells_mjidxs)
        # Initialize a number from 0 to len(self._ils100_cells_mjidxs)
        init_memory_length = int(np.random.randint(0, len(self._ils100_cells_mjidxs)))
        init_reading_memory = np.pad(self._ils100_cells_mjidxs[:init_memory_length],
                                     (0, len(self._ils100_cells_mjidxs) - init_memory_length),
                                     constant_values=self._MEMORY_PAD_VALUE)
        self._mental_state = {
            # The short term memory of the processed information
            'prev_reading_memory': init_reading_memory,
            'reading_memory': init_reading_memory,
            'attention': init_deployed_attention,
            'page_finish': False,
        }
        self._deployed_attention_target_mjidx = init_deployed_attention
        # To model memory loss, I can manipulate the internal state - mental state to simulate that

        # Set up the whole scene by confirming the initializations
        mujoco.mj_forward(self._model, self._data)

        return self._get_obs()

    def _get_focus(self, site_name):
        site = self._data.site(site_name)
        pnt = site.xpos
        vec = site.xmat.reshape((3, 3))[:, 2]
        # Exclude the body that contains the site, like in the rangefinder sensor
        bodyexclude = self._model.site_bodyid[site.id]
        geomid_out = np.array([-1], np.int32)
        distance = mujoco.mj_ray(
            self._model, self._data, pnt, vec, geomgroup=None, flg_static=1,
            bodyexclude=bodyexclude, geomid=geomid_out)
        return distance, geomid_out[0]

    def _reset_stm(self):
        self._deployed_attention_target_mjidx = self._ils100_cells_mjidxs[0]
        self._mental_state['prev_reading_memory'] = np.full_like(self._mental_state['prev_reading_memory'],
                                                                 self._MEMORY_PAD_VALUE)
        self._mental_state['reading_memory'] = np.full_like(self._mental_state['reading_memory'],
                                                            self._MEMORY_PAD_VALUE)

    def step(self, action):
        # Action t+1 from state t - attention allocation - Reading is a sequential process (MDP)
        # Action for attention deployment
        # 1. should the agent read the next word;
        # 2. should the agent get back to the first word uncovered in the short-term memory

        finish_reading = False

        action_attention_deployment = action[self._action_attention_deployment_idx]
        # Change the target of attention anytime
        if action_attention_deployment >= self._attention_action_threshold:
            if self._mental_state['page_finish'] is False:
                # Only when the agent has not finished reading
                if self._on_target_steps >= self._dwell_steps:
                    # Read the next word if the previous one has been processed
                    self._deployed_attention_target_mjidx += 1
                    # If the next read word exceeds the reading material limit,
                    # clear the short term memory and restart from the beginning, just like turning the page
                    if self._deployed_attention_target_mjidx > self._ils100_cells_mjidxs[-1]:
                        self._reset_stm()
                        # Page finish
                        finish_reading = True
                    # Reset the on target steps
                    self._on_target_steps = 0
            else:
                # Page / Reading finish, reset the memory
                # Actively finish reading
                finish_reading = True
        else:
            # Resample the latest word not read
            indices = np.where(self._mental_state['reading_memory'] != self._MEMORY_PAD_VALUE)[0]
            if len(indices) <= 0:
                # The memory is empty, reset the memory
                self._reset_stm()
            else:
                # Resample the latest word not read
                self._deployed_attention_target_mjidx = self._mental_state['reading_memory'][indices[-1]]
            # Reset the on target steps
            self._on_target_steps = 0

        # Eyeball movement
        xpos = self._data.geom(self._deployed_attention_target_mjidx).xpos
        x, y, z = xpos[0], xpos[1], xpos[2]
        self._data.ctrl[self._eye_x_motor_mjidx] = np.arctan(z / y)
        self._data.ctrl[self._eye_y_motor_mjidx] = np.arctan(-x / y)

        # Advance the simulation - State t+1
        mujoco.mj_step(self._model, self._data, self._frame_skip)
        self._steps += 1

        # State t+1
        reward = 0
        terminate = False
        info = {}

        # Update Eyeball movement driven by the intention / attention allocation
        # Eye-sight detection
        dist, geomid = self._get_focus(site_name="rangefinder-site")
        # Reset the scene first
        for mj_idx in self._ils100_cells_mjidxs:
            self._model.geom(mj_idx).rgba = self._DFLT_RGBA
        # Count the number of steps that the agent fixates on the target
        if geomid == self._deployed_attention_target_mjidx:
            self._on_target_steps += 1
            self._model.geom(self._deployed_attention_target_mjidx).rgba = self._VISUALIZE_RGBA

        # Update the mental state
        # Update the page finish flag
        if np.array_equal(self._mental_state['reading_memory'], self._ils100_cells_mjidxs):
            self._mental_state['page_finish'] = True

        # Update the previous reading memory
        self._mental_state['prev_reading_memory'] = self._mental_state['reading_memory'].copy()

        # Update the mental state only after the agent has read the word - the current reading memory
        if self._on_target_steps >= self._dwell_steps:
            # Judge whether the agent has read the word, only update the new read words
            if self._deployed_attention_target_mjidx not in self._mental_state['reading_memory']:
                # Find the first word that is not -2 in the memory
                indices = np.where(self._mental_state['reading_memory'] == self._MEMORY_PAD_VALUE)[0]
                if len(indices) > 0:
                    new_memory_slot_idx = indices[0]
                    self._mental_state['reading_memory'][new_memory_slot_idx] = self._deployed_attention_target_mjidx
                else:
                    pass

        # Reward shaping based on the reading progress - similarity
        # Estimate the reward
        if len(np.where(self._mental_state['reading_memory'] != self._MEMORY_PAD_VALUE)[0]) > len(
                np.where(self._mental_state['prev_reading_memory'] != self._MEMORY_PAD_VALUE)[0]):
            new_knowledge_gain = 1
        else:
            new_knowledge_gain = 0
        time_cost = -0.1
        reward += time_cost + new_knowledge_gain

        # If all materials are read, give a big bonus reward
        if self._steps >= self.ep_len or finish_reading:
            terminate = True

            # Get the comprehension reward
            reading_progress_seq = self._mental_state['reading_memory']
            euclidean_distance = self.euclidean_distance(reading_progress_seq, self._ils100_cells_mjidxs)
            reward += 10 * (np.exp(-0.1 * euclidean_distance) - 1)

        # # TODO debug delete later when training
        # print(f"The step is: {self._steps}, the finish page flag is: {self._mental_state['page_finish']}\n"
        #       f"The action is: {action[0]}, the deployed target is: {self._deployed_attention_target_mjidx}, the geomid is: {geomid}, \n"
        #       f"the on target steps is: {self._on_target_steps}, \n"
        #       f"the reading memory is: {self._mental_state['reading_memory']}, \n"
        #       f"The reward is: {reward}, \n")

        return self._get_obs(), reward, terminate, info


class MDPEyeRead(Env):

    def __init__(self):
        """ Model sequential reading - read words in the layout one by one """
        # Get directory of this file
        directory = os.path.dirname(os.path.realpath(__file__))

        # Read the configurations from the YAML file.
        root_dir = os.path.dirname(os.path.dirname(directory))
        with open(os.path.join(root_dir, "config.yaml")) as f:
            self._config = yaml.load(f, Loader=yaml.FullLoader)

        # Load the MuJoCo model
        self._model = mujoco.MjModel.from_xml_path(os.path.join(directory, "mobile-read-v2.xml"))
        self._data = mujoco.MjData(self._model)
        mujoco.mj_forward(self._model, self._data)

        # Define how often policy is queried
        self._action_sample_freq = 20
        self._frame_skip = int((1 / self._action_sample_freq) / self._model.opt.timestep)

        # Get the joints idx in MuJoCo
        self._eye_joint_x_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "eye-joint-x")
        self._eye_joint_y_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "eye-joint-y")

        # Get the motors idx in MuJoCo
        self._eye_x_motor_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, "eye-x-motor")
        self._eye_x_motor_translation_range = self._model.actuator_ctrlrange[self._eye_x_motor_mjidx]
        self._eye_y_motor_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, "eye-y-motor")
        self._eye_y_motor_translation_range = self._model.actuator_ctrlrange[self._eye_y_motor_mjidx]

        self._sgp_ils100_body_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY,
                                                        "smart-glass-pane-interline-spacing-100")
        # Get targets (geoms that belong to "smart-glass-pane-interline-spacing-100")
        self._ils100_cells_mjidxs = np.where(self._model.geom_bodyid == self._sgp_ils100_body_mjidx)[0]

        self._MEMORY_PAD_VALUE = -2
        # The pool of targets that can be attended to, -1 stands for attention switch away from reading
        self._attention_pool = np.array([self._MEMORY_PAD_VALUE, *self._ils100_cells_mjidxs.copy()])

        # Mental state related variables
        self._deployed_attention_target_mjidx = None
        self._attention_action_threshold = 0
        self._mental_state = {  # The mental state of the agent, or it can be called the internal state
            # The previous state of the words that have been read
            'prev_reading_memory': np.full((len(self._ils100_cells_mjidxs),), self._MEMORY_PAD_VALUE),
            # The memory of the words that have been read
            'reading_memory': np.full((len(self._ils100_cells_mjidxs),), self._MEMORY_PAD_VALUE),
            # The memory of the words that have been read
            'attention': self._deployed_attention_target_mjidx,  # The intended target mjidx
            'page_finish': False,  # Whether the page is finished
        }

        # Define the target idx probability distribution
        self._VISUALIZE_RGBA = [1, 1, 0, 1]
        self._DFLT_RGBA = [0, 0, 0, 1]

        self._dwell_steps = int(0.2 * self._action_sample_freq)  # The number of steps to dwell on a target

        # Initialise RL related thresholds and counters
        self._steps = None
        self._on_target_steps = None
        self.ep_len = 100

        # Define the observation space
        width, height = 80, 80
        self._num_stk_frm = 1
        self._vision_frames = None
        self._qpos_frames = None
        self._num_stateful_info = 16

        self.observation_space = Dict({
            "vision": Box(low=-1, high=1, shape=(self._num_stk_frm, width, height)),
            "proprioception": Box(low=-1, high=1, shape=(self._num_stk_frm * self._model.nq + self._model.nu,)),
            "stateful information": Box(low=-1, high=1, shape=(self._num_stateful_info,)),
        })

        # Define the action space
        # 1st: decision on attention distribution;
        # 2nd and 3rd: eyeball rotations;
        self._action_attention_deployment_idx = 0
        self._action_eye_rotate_x_idx = 1
        self._action_eye_rotate_y_idx = 2
        self.action_space = Box(low=-1, high=1, shape=(3,))

        # Initialize the context and camera
        context = Context(self._model, max_resolution=[1280, 960])
        self._eye_cam = Camera(context, self._model, self._data, camera_id="eye", resolution=[width, height],
                               maxgeom=100,
                               dt=1 / self._action_sample_freq)
        self._env_cam = Camera(context, self._model, self._data, camera_id="env", maxgeom=100,
                               dt=1 / self._action_sample_freq)
        self._eye_cam_fovy = self._model.cam_fovy[mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, "eye")]

    @staticmethod
    def normalise(x, x_min, x_max, a, b):
        # Normalise x (which is assumed to be in range [x_min, x_max]) to range [a, b]
        return (b - a) * ((x - x_min) / (x_max - x_min)) + a

    @staticmethod
    def euclidean_distance(x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def render(self, mode="rgb_array"):
        rgb, _ = self._env_cam.render()
        rgb_eye, _ = self._eye_cam.render()
        return rgb, rgb_eye

    def _get_obs(self):
        """ Get the observation of the environment state """
        # Compute the vision observation
        # Render the image
        rgb, _ = self._eye_cam.render()
        # Preprocess - H*W*C -> C*W*H
        rgb = np.transpose(rgb, [2, 1, 0])
        rgb_normalize = self.normalise(rgb, 0, 255, -1, 1)
        # Convert the rgb to grayscale - boost the training speed
        gray_normalize = rgb_normalize[0:1, :, :] * 0.299 + rgb_normalize[1:2, :, :] * 0.587 + rgb_normalize[2:3, :,
                                                                                               :] * 0.114
        gray_normalize = np.squeeze(gray_normalize, axis=0)

        # Update the stack of frames
        self._vision_frames.append(gray_normalize)
        # Remove the locomotion value since it is not normalized
        wanted_qpos = self._data.qpos.copy()
        # wanted_qpos = np.delete(wanted_qpos, self._agent_joint_y_mjidx)
        self._qpos_frames.append(wanted_qpos)
        # Replicate the newest frame if the stack is not full
        while len(self._vision_frames) < self._num_stk_frm:
            self._vision_frames.append(self._vision_frames[-1])
        while len(self._qpos_frames) < self._num_stk_frm:
            self._qpos_frames.append(self._qpos_frames[-1])

        # Reshape the stack of frames - Get vision observations
        vision = np.stack(self._vision_frames, axis=0)
        vision = vision.reshape((-1, vision.shape[-2], vision.shape[-1]))

        # vision = gray_normalize.reshape((-1, gray_normalize.shape[-2], gray_normalize.shape[-1]))

        # Get the proprioception observation
        qpos = np.stack(self._qpos_frames, axis=0)
        qpos = qpos.reshape((1, -1))
        # Remove the locomotion control since it is not normalized
        wanted_ctrl = self._data.ctrl.copy()
        # wanted_ctrl = np.delete(wanted_ctrl, self._agent_y_motor_mjidx)
        ctrl = wanted_ctrl.reshape((1, -1))
        proprioception = np.concatenate([qpos.flatten(), ctrl.flatten()], axis=0)

        # Get the stateful information observation - normalize to [-1, 1]
        remaining_ep_len_norm = (self.ep_len - self._steps) / self.ep_len * 2 - 1
        remaining_dwell_steps_norm = (self._dwell_steps - self._on_target_steps) / self._dwell_steps * 2 - 1
        attention_deploy_target_mjidx_norm = self.normalise(self._deployed_attention_target_mjidx,
                                                            self._attention_pool[0], self._attention_pool[-1], -1, 1)

        reading_memory = self._mental_state['reading_memory'].copy()
        reading_memory_norm = self.normalise(reading_memory, self._MEMORY_PAD_VALUE, self._ils100_cells_mjidxs[-1], -1,
                                             1)

        page_finish_norm = -1 if self._mental_state['page_finish'] == False else 1

        stateful_info = np.array(
            [remaining_ep_len_norm, remaining_dwell_steps_norm, attention_deploy_target_mjidx_norm, page_finish_norm,
             *reading_memory_norm]
        )

        # Observation space check
        if vision.shape != self.observation_space["vision"].shape:
            raise ValueError("The shape of vision observation is not correct!")
        if proprioception.shape != self.observation_space["proprioception"].shape:
            raise ValueError("The shape of proprioception observation is not correct!")
        if stateful_info.shape != self.observation_space["stateful information"].shape:
            raise ValueError("The shape of stateful information observation is not correct!")

        return {"vision": vision, "proprioception": proprioception, "stateful information": stateful_info}

    def reset(self, params=None):

        # Reset MuJoCo sim
        mujoco.mj_resetData(self._model, self._data)

        # Reset the variables and counters
        self._vision_frames = deque(maxlen=self._num_stk_frm)
        self._qpos_frames = deque(maxlen=self._num_stk_frm)
        self._steps = 0
        self._on_target_steps = 0

        # Mental state related variables
        # Randomly choose the intended target mjidx
        init_deployed_attention = np.random.choice(self._ils100_cells_mjidxs)
        # Initialize a number from 0 to len(self._ils100_cells_mjidxs)
        init_memory_length = int(np.random.randint(0, len(self._ils100_cells_mjidxs)))
        init_reading_memory = np.pad(self._ils100_cells_mjidxs[:init_memory_length],
                                     (0, len(self._ils100_cells_mjidxs) - init_memory_length),
                                     constant_values=self._MEMORY_PAD_VALUE)
        self._mental_state = {
            # The short term memory of the processed information
            'prev_reading_memory': init_reading_memory,
            'reading_memory': init_reading_memory,
            'attention': init_deployed_attention,
            'page_finish': False,
        }
        self._deployed_attention_target_mjidx = init_deployed_attention
        # To model memory loss, I can manipulate the internal state - mental state to simulate that

        # Initialize eyeball rotation angles
        init_eye_x_rotation = np.random.uniform(*self._eye_x_motor_translation_range)
        init_eye_y_rotation = np.random.uniform(*self._eye_y_motor_translation_range)
        self._data.qpos[self._eye_joint_x_mjidx] = init_eye_x_rotation
        self._data.ctrl[self._eye_x_motor_mjidx] = init_eye_x_rotation
        self._data.qpos[self._eye_joint_y_mjidx] = init_eye_y_rotation
        self._data.ctrl[self._eye_y_motor_mjidx] = init_eye_y_rotation

        # Set up the whole scene by confirming the initializations
        mujoco.mj_forward(self._model, self._data)

        return self._get_obs()

    def _get_focus(self, site_name):
        site = self._data.site(site_name)
        pnt = site.xpos
        vec = site.xmat.reshape((3, 3))[:, 2]
        # Exclude the body that contains the site, like in the rangefinder sensor
        bodyexclude = self._model.site_bodyid[site.id]
        geomid_out = np.array([-1], np.int32)
        distance = mujoco.mj_ray(
            self._model, self._data, pnt, vec, geomgroup=None, flg_static=1,
            bodyexclude=bodyexclude, geomid=geomid_out)
        return distance, geomid_out[0]

    def _reset_stm(self):
        self._deployed_attention_target_mjidx = self._ils100_cells_mjidxs[0]
        self._mental_state['prev_reading_memory'] = np.full_like(self._mental_state['prev_reading_memory'],
                                                                 self._MEMORY_PAD_VALUE)
        self._mental_state['reading_memory'] = np.full_like(self._mental_state['reading_memory'],
                                                            self._MEMORY_PAD_VALUE)

    @staticmethod
    def angle_between(v1, v2):
        # https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python/13849249#13849249
        def unit_vector(vec):
            return vec / np.linalg.norm(vec)

        v1_u = unit_vector(v1)
        v2_u = unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    def _angle_from_target(self, site_name, target_idx):
        """
        Return the angle between the vector pointing from the site to the target and the vector pointing from the site to the front
        ranges from 0 to pi.
        """
        # Get vector pointing direction from site
        site = self._data.site(site_name)
        pnt = site.xpos
        vec = pnt + site.xmat.reshape((3, 3))[:, 2]

        # Get vector pointing direction to target
        target_vec = self._data.geom(target_idx).xpos - pnt

        # Estimate distance as angle
        angle = self.angle_between(vec, target_vec)

        return angle

    def _show_target(self):
        for mj_idx in self._ils100_cells_mjidxs:
            self._model.geom(mj_idx).rgba = self._DFLT_RGBA
        self._model.geom(self._deployed_attention_target_mjidx).rgba = self._VISUALIZE_RGBA

    def step(self, action):
        # TODO maybe reconstruct the step method to make it clearer
        # Action t+1 from state t - attention allocation - Reading is a sequential process (MDP)
        # Action for attention deployment
        # 1. should the agent read the next word;
        # 2. should the agent get back to the first word uncovered in the short-term memory

        finish_reading = False

        action_attention_deployment = action[self._action_attention_deployment_idx]
        # Change the target of attention anytime
        if action_attention_deployment >= self._attention_action_threshold:
            if self._mental_state['page_finish'] is False:
                # Only when the agent has not finished reading
                if self._on_target_steps >= self._dwell_steps:
                    # Read the next word if the previous one has been processed
                    self._deployed_attention_target_mjidx += 1
                    # If the next read word exceeds the reading material limit,
                    # clear the short term memory and restart from the beginning, just like turning the page
                    if self._deployed_attention_target_mjidx > self._ils100_cells_mjidxs[-1]:
                        self._reset_stm()
                        # Page finish
                        finish_reading = True
                    # Reset the on target steps
                    self._on_target_steps = 0
            else:
                # Page / Reading finish, reset the memory
                # Actively finish reading
                finish_reading = True
        else:
            # Resample the latest word not read
            indices = np.where(self._mental_state['reading_memory'] != self._MEMORY_PAD_VALUE)[0]
            if len(indices) <= 0:
                # The memory is empty, reset the memory
                self._reset_stm()
            else:
                # Resample the latest word not read
                self._deployed_attention_target_mjidx = self._mental_state['reading_memory'][indices[-1]]
            # Reset the on target steps
            self._on_target_steps = 0

        # Control the eyeball movement
        action[self._action_eye_rotate_x_idx] = self.normalise(action[self._action_eye_rotate_x_idx], -1, 1,
                                                               *self._model.actuator_ctrlrange[self._eye_x_motor_mjidx,
                                                                :])
        action[self._action_eye_rotate_y_idx] = self.normalise(action[self._action_eye_rotate_y_idx], -1, 1,
                                                               *self._model.actuator_ctrlrange[self._eye_y_motor_mjidx,
                                                                :])

        # The simple version of the eyeball rotation control
        self._data.ctrl[self._eye_x_motor_mjidx] = action[self._action_eye_rotate_x_idx]
        self._data.ctrl[self._eye_y_motor_mjidx] = action[self._action_eye_rotate_y_idx]

        # Advance the simulation - State t+1
        mujoco.mj_step(self._model, self._data, self._frame_skip)
        self._steps += 1

        # State t+1
        reward = 0
        reward_attention_deployment = 0
        reward_eyeball_rotation = 0
        terminate = False
        info = {}

        # Update the external environment - scene hint
        self._show_target()

        # Update Eyeball movement driven by the intention / attention allocation
        # Eye-sight detection
        dist, geomid = self._get_focus(site_name="rangefinder-site")
        # Count the number of steps that the agent fixates on the target
        if geomid == self._deployed_attention_target_mjidx:
            self._on_target_steps += 1

        # Update the mental state
        # Update the page finish flag
        if np.array_equal(self._mental_state['reading_memory'], self._ils100_cells_mjidxs):
            self._mental_state['page_finish'] = True

        # Update the previous reading memory
        self._mental_state['prev_reading_memory'] = self._mental_state['reading_memory'].copy()

        # Update the mental state only after the agent has read the word - the current reading memory
        if self._on_target_steps >= self._dwell_steps:
            # # Once the agent has read the word, grant an eyeball rotation reward / ocular motor control reward
            # reward_eyeball_rotation = 1

            # Judge whether the agent has read the word, only update the new read words
            if self._deployed_attention_target_mjidx not in self._mental_state['reading_memory']:
                # Find the first word that is not -2 in the memory
                indices = np.where(self._mental_state['reading_memory'] == self._MEMORY_PAD_VALUE)[0]
                if len(indices) > 0:
                    new_memory_slot_idx = indices[0]
                    self._mental_state['reading_memory'][new_memory_slot_idx] = self._deployed_attention_target_mjidx
                else:
                    pass

        # Reward shaping based on the eyeball rotation
        reward_eyeball_rotation = 0.1 * (np.exp(
            -10 * self._angle_from_target(site_name="rangefinder-site",
                                          target_idx=self._deployed_attention_target_mjidx)) - 1)

        # Reward shaping based on the reading progress
        # Estimate the reward - new word read
        if len(np.where(self._mental_state['reading_memory'] != self._MEMORY_PAD_VALUE)[0]) > len(
                np.where(self._mental_state['prev_reading_memory'] != self._MEMORY_PAD_VALUE)[0]):
            new_knowledge_gain = 10
        else:
            new_knowledge_gain = 0
        # Memory similarity
        reading_progress_seq = self._mental_state['reading_memory']
        euclidean_distance = self.euclidean_distance(reading_progress_seq, self._ils100_cells_mjidxs)
        reward_shaping_memory = 0.1 * (np.exp(-0.1 * euclidean_distance) - 1)
        time_cost = -0.1
        reward_attention_deployment += time_cost + new_knowledge_gain + reward_shaping_memory

        reward += reward_attention_deployment + reward_eyeball_rotation

        # If all materials are read, give a big bonus reward
        if self._steps >= self.ep_len or finish_reading:
            terminate = True

            # # Get the comprehension reward
            # reading_progress_seq = self._mental_state['reading_memory']
            # euclidean_distance = self.euclidean_distance(reading_progress_seq, self._ils100_cells_mjidxs)
            # reward += 10 * (np.exp(-0.1 * euclidean_distance) - 1)

        # # TODO debug comment later when training
        # print(f"The step is: {self._steps}, the finish page flag is: {self._mental_state['page_finish']}\n"
        #       f"The action is: {action[0]}, the deployed target is: {self._deployed_attention_target_mjidx}, the geomid is: {geomid}, \n"
        #       f"the on target steps is: {self._on_target_steps}, \n"
        #       f"the reading memory is: {self._mental_state['reading_memory']}, \n"
        #       f"The reward attention deployment is: {reward_attention_deployment}, the reward eyeball rotation is: {reward_eyeball_rotation}, "
        #       f"The reward is: {reward}, \n")

        return self._get_obs(), reward, terminate, info


class MDPResumeRead(Env):

    def __init__(self):
        """ Model sequential reading - read words in the layout one by one """
        # Get directory of this file
        directory = os.path.dirname(os.path.realpath(__file__))

        # Read the configurations from the YAML file.
        root_dir = os.path.dirname(os.path.dirname(directory))
        with open(os.path.join(root_dir, "config.yaml")) as f:
            self._config = yaml.load(f, Loader=yaml.FullLoader)

        # Load the MuJoCo model
        self._model = mujoco.MjModel.from_xml_path(os.path.join(directory, "pomdp-resume-read-v1.xml"))
        self._data = mujoco.MjData(self._model)
        mujoco.mj_forward(self._model, self._data)

        # Define how often policy is queried
        self._action_sample_freq = 20
        self._frame_skip = int((1 / self._action_sample_freq) / self._model.opt.timestep)

        # Get the joints idx in MuJoCo
        self._eye_joint_x_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "eye-joint-x")
        self._eye_joint_y_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "eye-joint-y")

        # Get the motors idx in MuJoCo
        self._eye_x_motor_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, "eye-x-motor")
        self._eye_x_motor_translation_range = self._model.actuator_ctrlrange[self._eye_x_motor_mjidx]
        self._eye_y_motor_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, "eye-y-motor")
        self._eye_y_motor_translation_range = self._model.actuator_ctrlrange[self._eye_y_motor_mjidx]

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

        # Initialize the layout related parameters
        # Relate this to the fovea size, e.g., 2 times of the fovea vision (2 * +- 1~2 degrees --> 0.026 ~ 0.0523)
        # Ref: Adapting User Interfaces with Model-based Reinforcement Learning
        # Local search
        self._fovea_degrees = 2
        self._fovea_size = None
        self._delta_local_search_range = [1, 5]
        self._delta_local_search = None
        self._sigma_local_search = None
        self._L100 = "L100"
        self._L50 = "L50"
        self._L0 = "L0"
        self._layouts = [self._L100, self._L50, self._L0]
        self._cells_mjidxs = None

        # Initialize the mental state related parameters
        self._MEMORY_PAD_VALUE = None

        # Mental state related variables
        self._deployed_attention_target_mjidx = None
        self._true_attention_target_mjidx = None
        self._mental_state = {  # The mental state of the agent, or it can be called the internal state
            # The original memory before corruption
            'original_memory': np.full((len(self._ils100_cells_mjidxs),), 0),
            # The previous state of the words that have been read
            'prev_reading_memory': np.full((len(self._ils100_cells_mjidxs),), 0),
            # The memory of the words that have been read
            'reading_memory': np.full((len(self._ils100_cells_mjidxs),), 0),
            # The memory of the words that have been read
            'attention': self._deployed_attention_target_mjidx,  # The intended target mjidx
            'page_finish': False,  # Whether the page is finished
        }

        # Define the belief state (memory) uncertainty - sigma position memory
        # Initial sigma position memory
        # Ref: Modeling Touch-based Menu Selection Performance of Blind Users via Reinforcement Learning
        self._init_sigma_position_memory_range = [0.5, 5]
        self._init_sigma_position_memory = None
        # The initial time interval of attention switch
        # 2s - time spent on the environmental task;
        # 4s - according to empirical results, most word selection time is finished within 2s; 2+2=4
        # Ref: Not all spacings are created equally
        self._delta_t_range = [2, 4]
        self._delta_t = None
        # The decaying factor Bi formular is a simplified version of the learning component from ACT-R
        # Ref: Modeling Touch-based Menu Selection Performance of Blind Users via Reinforcement Learning
        # Ref: Adapting User Interfaces with Model-based Reinforcement Learning
        self._rho = 0.5
        self._sigma_position_memory = None

        # Define the target idx probability distribution
        self._VISUALIZE_RGBA = [1, 1, 0, 1]
        self._DFLT_RGBA = [0, 0, 0, 1]

        self._dwell_time = 0.2  # The time to dwell on a target
        self._dwell_steps = int(self._dwell_time * self._action_sample_freq)  # The number of steps to dwell on a target

        # Initialise RL related thresholds and counters
        self._steps = None
        self._on_target_steps = None
        self.ep_len = 100

        # Define the observation space
        width, height = 80, 80
        self._num_stk_frm = 1
        self._vision_frames = None
        self._qpos_frames = None
        self._num_stateful_info = 16

        self.observation_space = Box(low=-1, high=1, shape=(self._num_stateful_info,))

        # Define the action space
        # 1st: decision on attention distribution;
        # 2nd and 3rd: eyeball rotations;
        self._action_attention_deployment_idx = 0
        self._action_eye_rotate_x_idx = 1
        self._action_eye_rotate_y_idx = 2
        self.action_space = Box(low=-1, high=1, shape=(1,))

        # Initialize the context and camera
        context = Context(self._model, max_resolution=[1280, 960])
        self._eye_cam = Camera(context, self._model, self._data, camera_id="eye", resolution=[width, height],
                               maxgeom=100,
                               dt=1 / self._action_sample_freq)
        self._env_cam = Camera(context, self._model, self._data, camera_id="env", maxgeom=100,
                               dt=1 / self._action_sample_freq)
        self._eye_cam_fovy = self._model.cam_fovy[mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, "eye")]

    def reset(self, params=None):

        # Reset MuJoCo sim
        mujoco.mj_resetData(self._model, self._data)

        # Reset the variables and counters
        self._steps = 0
        self._on_target_steps = 0

        # Initialize the memory model related parameters
        self._delta_t = np.random.uniform(*self._delta_t_range)
        self._init_sigma_position_memory = np.random.uniform(*self._init_sigma_position_memory_range)

        # Reset all cells to transparent
        for mjidx in self._ils100_cells_mjidxs:
            self._model.geom(mjidx).rgba[3] = 0
        for mjidx in self._ils50_cells_mjidxs:
            self._model.geom(mjidx).rgba[3] = 0
        for mjidx in self._ils0_cells_mjidxs:
            self._model.geom(mjidx).rgba[3] = 0

        # Initialize the word selection destination related parameters
        layout_name = np.random.choice(self._layouts)
        if layout_name == self._L100:
            self._cells_mjidxs = self._ils100_cells_mjidxs
        elif layout_name == self._L50:
            self._cells_mjidxs = self._ils50_cells_mjidxs
        elif layout_name == self._L0:
            self._cells_mjidxs = self._ils0_cells_mjidxs
        else:
            raise ValueError("The layout name is not correct!")
        # Set up the scene after deciding the layout
        mujoco.mj_forward(self._model, self._data)

        # Initialize the local search related parameters
        self._fovea_size = np.tan(np.radians(self._fovea_degrees / 2)) * self._data.geom(self._cells_mjidxs[0]).xpos[1]
        self._delta_local_search = np.random.uniform(*self._delta_local_search_range)
        self._sigma_local_search = self._fovea_size * self._delta_local_search
        self._MEMORY_PAD_VALUE = int(self._cells_mjidxs[0] - 2)

        # Mental state related variables
        # Randomly choose the intended target mjidx
        init_deployed_attention = np.random.choice(self._cells_mjidxs)
        # Initialize a number from 1 to len(_cells_mjidxs)
        init_memory_length = int(np.random.randint(1, len(self._cells_mjidxs)))
        init_reading_memory = np.pad(self._cells_mjidxs[:init_memory_length],
                                     (0, len(self._cells_mjidxs) - init_memory_length),
                                     constant_values=self._MEMORY_PAD_VALUE)
        self._mental_state = {
            # The short term memory of the processed information
            'original_memory': init_reading_memory,
            'prev_reading_memory': init_reading_memory,
            'reading_memory': init_reading_memory,
            'attention': init_deployed_attention,
            'page_finish': False,
        }

        indexes = np.where(init_reading_memory != self._MEMORY_PAD_VALUE)[0]
        if len(indexes) > 0:
            idx = indexes[-1]
            self._true_attention_target_mjidx = init_reading_memory[idx]
        else:
            self._true_attention_target_mjidx = self._cells_mjidxs[0]

        # Corrupt the memory
        self._corrupt_memory()

        # Initialize the deployed attention target
        self._init_attention_deployment_mjidx_distribution()

        # Sample a deployed attention target mjidx
        self._sample_attention_deployment_mjidx()

        # Set up the whole scene by confirming the initializations
        mujoco.mj_forward(self._model, self._data)

        # # TODO comment later debug here
        # print(f"the mental state initialization is: {self._mental_state}, "
        #       f"the true attention target is: {self._true_attention_target_mjidx}\n"
        #       f"The deployed attention target is: {self._deployed_attention_target_mjidx}\n")

        return self._get_obs()

    def render(self, mode="rgb_array"):
        rgb, _ = self._env_cam.render()
        rgb_eye, _ = self._eye_cam.render()
        return rgb, rgb_eye

    def step(self, action):
        # Action t+1 from state t - attention allocation - Reading is a sequential process (MDP)
        # Action for attention deployment - Make a decision - Take the current one to read or keep searching
        # 1. should the agent continue to read the next word, do not search;
        # 2. should the agent search for the target word, however, the more agent searches, the more the agent forgets

        finish_reading = False

        action_attention_deployment = action[self._action_attention_deployment_idx]
        if action_attention_deployment >= 0:
            # Confirm and keep reading
            if self._mental_state['page_finish'] == False:
                # Only when the agent has not finished reading
                if self._on_target_steps >= self._dwell_steps:
                    # Read the next word if the previous one has been processed
                    self._deployed_attention_target_mjidx += 1
                    # Reset the on target steps
                    self._on_target_steps = 0
                    if self._deployed_attention_target_mjidx > self._cells_mjidxs[-1]:
                        # Page / Reading finish, reset the memory
                        finish_reading = True
                        self._deployed_attention_target_mjidx -= 1
            else:
                # Page / Reading finish, reset the memory
                # Actively finish reading
                finish_reading = True
        else:
            # Local search
            if self._on_target_steps >= self._dwell_steps:
                # Sample a new target to deploy attention based on the attention distribution
                self._sample_attention_deployment_mjidx()
                # Reset the on target steps
                self._on_target_steps = 0

        xpos = self._data.geom(self._deployed_attention_target_mjidx).xpos
        x, y, z = xpos[0], xpos[1], xpos[2]
        self._data.ctrl[self._eye_x_motor_mjidx] = np.arctan(z / y)
        self._data.ctrl[self._eye_y_motor_mjidx] = np.arctan(-x / y)

        # Advance the simulation - State t+1
        mujoco.mj_step(self._model, self._data, self._frame_skip)
        self._steps += 1

        # State t+1
        reward = 0
        terminate = False
        info = {}

        # Update Eyeball movement driven by the intention / attention allocation
        # Eye-sight detection
        dist, geomid = self._get_focus(site_name="rangefinder-site")
        # Reset the scene first
        for mj_idx in self._cells_mjidxs:
            self._model.geom(mj_idx).rgba = self._DFLT_RGBA
        # Count the number of steps that the agent fixates on the target
        if geomid == self._deployed_attention_target_mjidx:
            self._on_target_steps += 1
            self._model.geom(self._deployed_attention_target_mjidx).rgba = self._VISUALIZE_RGBA

        # Update the mental state
        # Update the previous reading memory
        self._mental_state['prev_reading_memory'] = self._mental_state['reading_memory'].copy()

        # Update the mental state only after the agent has read the word - the current reading memory
        if action[self._action_attention_deployment_idx] >= 0:
            if self._on_target_steps >= self._dwell_steps:
                # Update the reading memory with new read word
                if self._deployed_attention_target_mjidx not in self._mental_state['reading_memory']:
                    # Find the first word that is not PAD_VALUE in the memory
                    indices = np.where(self._mental_state['reading_memory'] == self._MEMORY_PAD_VALUE)[0]
                    if len(indices) > 0:
                        new_memory_slot_idx = indices[0]
                        self._mental_state['reading_memory'][new_memory_slot_idx] = self._deployed_attention_target_mjidx

                # Update the original memory with new read word
                if self._deployed_attention_target_mjidx not in self._mental_state['original_memory']:
                    # Find the first word that is not PAD_VALUE in the memory
                    indices = np.where(self._mental_state['original_memory'] == self._MEMORY_PAD_VALUE)[0]
                    if len(indices) > 0:
                        new_memory_slot_idx = indices[0]
                        self._mental_state['original_memory'][new_memory_slot_idx] = self._deployed_attention_target_mjidx

                # Check whether the agent has read all the words
                # if np.array_equal(self._mental_state['reading_memory'], self._cells_mjidxs):
                if self._deployed_attention_target_mjidx >= self._cells_mjidxs[-1]:
                    self._mental_state['page_finish'] = True

        # Reward shaping based on the reading progress - similarity
        # Estimate the reward
        if len(np.where(self._mental_state['reading_memory'] != self._MEMORY_PAD_VALUE)[0]) > len(
                np.where(self._mental_state['prev_reading_memory'] != self._MEMORY_PAD_VALUE)[0]):
            new_knowledge_gain = 1
        else:
            new_knowledge_gain = 0
        time_cost = -0.1
        reward += time_cost + new_knowledge_gain

        # If all materials are read, give a big bonus reward
        if self._steps >= self.ep_len or finish_reading:
            terminate = True

            # Get the comprehension reward
            reading_progress_seq = self._mental_state['original_memory']
            euclidean_distance = self.euclidean_distance(reading_progress_seq, self._cells_mjidxs)
            reward += 10 * (np.exp(-0.1 * euclidean_distance) - 1)

        # TODO debug comment later when training
        print(f"The step is: {self._steps}, the finish page flag is: {self._mental_state['page_finish']}\n"
              f"The action is: {action[0]}, the deployed target is: {self._deployed_attention_target_mjidx}, the geomid is: {geomid}, \n"
              f"the on target steps is: {self._on_target_steps}, \n"
              f"The attention distribution is: {self._attention_mjidx_distribution}, \n"
              f"The original memory is: {self._mental_state['original_memory']}, \n"
              f"The reading memory is: {self._mental_state['reading_memory']}, \n"
              f"The reward is: {reward}, \n")

        return self._get_obs(), reward, terminate, info

    @staticmethod
    def normalise(x, x_min, x_max, a, b):
        # Normalise x (which is assumed to be in range [x_min, x_max]) to range [a, b]
        return (b - a) * ((x - x_min) / (x_max - x_min)) + a

    @staticmethod
    def euclidean_distance(x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _corrupt_memory(self):
        """ Init/Update/Corrupt the memory """
        # Update the sigma position memory
        self._sigma_position_memory = self._init_sigma_position_memory / (1 + self._delta_t ** (-self._rho))

        # Corrupt the memory by adding noise to the true attention target - the true attention target is the last word
        original_memory = self._mental_state['original_memory'].copy()
        original_last_word_idx = np.where(original_memory == self._true_attention_target_mjidx)[0][0]
        # Due to distractions, the working memory is corrupted. Information stored might be lost or partially forgotten.
        # So the memory vector will be shortened by a random number of words
        corrupted_last_word_idx = int(original_last_word_idx - np.abs(np.random.normal(0, self._sigma_position_memory)))
        corrupted_last_word_idx = np.clip(corrupted_last_word_idx, 0, len(self._cells_mjidxs) - 1)
        length = int(corrupted_last_word_idx + 1)
        # Update the reading memory
        corrupted_memory = np.pad(self._cells_mjidxs[:length],
                                  (0, len(self._cells_mjidxs) - length),
                                  constant_values=self._MEMORY_PAD_VALUE)
        self._mental_state['prev_reading_memory'] = corrupted_memory
        self._mental_state['reading_memory'] = corrupted_memory

        # # TODO debug delete later
        # print(
        #     f"\nThe sigma position memory is: {self._sigma_position_memory}, "
        #     f"the init sigma is: {self._init_sigma_position_memory} the delta t is: {self._delta_t},"
        #     f"\nOriginal_memory: {original_memory}, corrupted_memory: {corrupted_memory}")

    def _get_obs(self):
        """ Get the observation of the environment state """
        # Get the stateful information observation - normalize to [-1, 1]
        remaining_ep_len_norm = (self.ep_len - self._steps) / self.ep_len * 2 - 1
        remaining_dwell_steps_norm = (self._dwell_steps - self._on_target_steps) / self._dwell_steps * 2 - 1
        attention_deploy_target_mjidx_norm = self.normalise(self._deployed_attention_target_mjidx,
                                                            self._cells_mjidxs[0], self._cells_mjidxs[-1], -1, 1)

        reading_memory = self._mental_state['reading_memory'].copy()
        reading_memory_norm = self.normalise(reading_memory, self._MEMORY_PAD_VALUE, self._cells_mjidxs[-1], -1,
                                             1)

        page_finish_norm = -1 if not self._mental_state['page_finish'] else 1

        stateful_info = np.array(
            [remaining_ep_len_norm, remaining_dwell_steps_norm, attention_deploy_target_mjidx_norm, page_finish_norm,
             *reading_memory_norm]
        )

        # Observation space check
        if stateful_info.shape[0] != self._num_stateful_info:
            raise ValueError("The shape of stateful information observation is not correct!")

        return stateful_info

    def _get_focus(self, site_name):
        site = self._data.site(site_name)
        pnt = site.xpos
        vec = site.xmat.reshape((3, 3))[:, 2]
        # Exclude the body that contains the site, like in the rangefinder sensor
        bodyexclude = self._model.site_bodyid[site.id]
        geomid_out = np.array([-1], np.int32)
        distance = mujoco.mj_ray(
            self._model, self._data, pnt, vec, geomgroup=None, flg_static=1,
            bodyexclude=bodyexclude, geomid=geomid_out)
        return distance, geomid_out[0]

    def _init_attention_deployment_mjidx_distribution(self):
        # Set up the simulation
        mujoco.mj_forward(self._model, self._data)

        # Reset the attention distribution
        self._attention_mjidx_distribution = np.zeros(self._cells_mjidxs.shape[0])
        # Get the true attention target's position
        mu_xpos = self._data.geom(self._true_attention_target_mjidx).xpos

        # Calculate every cell's distance to the target and get the attention distribution (probability)
        for mjidx in self._cells_mjidxs:
            xpos = self._data.geom(mjidx).xpos
            dist = np.linalg.norm(xpos - mu_xpos)
            idx = np.where(self._cells_mjidxs == mjidx)[0][0]
            self._attention_mjidx_distribution[idx] = np.exp(-0.5 * (dist / self._sigma_local_search) ** 2)

        # Normalization
        self._attention_mjidx_distribution /= np.sum(self._attention_mjidx_distribution)

        # # TODO debug delete later
        # print(f"\n"
        #       f"The layout is: {self._cells_mjidxs}, the true attention target is: {self._true_attention_target_mjidx},"
        #       f"The original memory is: {self._mental_state['original_memory']}, the corrupted memory is: {self._mental_state['reading_memory']},"
        #       f"\nthe attention distribution is: {self._attention_mjidx_distribution}"
        #       f"\n")

    def _sample_attention_deployment_mjidx(self):
        # Sample a new target to deploy attention
        self._deployed_attention_target_mjidx = np.random.choice(self._cells_mjidxs,
                                                                 p=self._attention_mjidx_distribution)
        # Update the attention distribution, remove the deployed target by making its probability to 0
        if len(np.where(self._attention_mjidx_distribution != 0)[0]) > 1:
            idx = np.where(self._cells_mjidxs == self._deployed_attention_target_mjidx)[0][0]
            self._attention_mjidx_distribution[idx] = 0
        # Normalization
        self._attention_mjidx_distribution /= np.sum(self._attention_mjidx_distribution)

    def _reset_stm(self):
        self._deployed_attention_target_mjidx = self._cells_mjidxs[0]
        self._mental_state['prev_reading_memory'] = np.full_like(self._mental_state['prev_reading_memory'],
                                                                 self._MEMORY_PAD_VALUE)
        self._mental_state['reading_memory'] = np.full_like(self._mental_state['reading_memory'],
                                                            self._MEMORY_PAD_VALUE)
