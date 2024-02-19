import math

import gym
import numpy as np
from collections import Counter
import mujoco
import os

from gym import Env
from gym.spaces import Box, Dict

import yaml
from scipy.ndimage import gaussian_filter

from huc.utils.rendering import Camera, Context
from collections import deque

# Normalised the task modes -1 to 1
READ = 1
BG = 0
RELOC = -1

# Normalized layout index
ILS100 = 1
BC = 0
MR = -1


class Read(Env):

    def __init__(self):
        """ Model the reading with belief model/function """
        # Get directory of this file
        directory = os.path.dirname(os.path.realpath(__file__))

        # Read the configurations from the YAML file.
        root_dir = os.path.dirname(os.path.dirname(directory))
        with open(os.path.join(root_dir, "config.yaml")) as f:
            self._config = yaml.load(f, Loader=yaml.FullLoader)

        # Load the MuJoCo model
        self._model = mujoco.MjModel.from_xml_path(os.path.join(directory, "relocation-v1.xml"))
        self._data = mujoco.MjData(self._model)
        mujoco.mj_forward(self._model, self._data)

        # Define how often policy is queried
        self._action_sample_freq = 20
        self._frame_skip = int((1 / self._action_sample_freq) / self._model.opt.timestep)

        # Get the primitive idx in MuJoCo
        self._eye_joint_x_idx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "eye-joint-x")
        self._eye_joint_y_idx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "eye-joint-y")
        self._head_body_idx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "head")
        self._eye_body_idx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "eye")
        self._sgp_ils100_body_idx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY,
                                                      "smart-glass-pane-interline-spacing-100")

        # Get targets (geoms that belong to "smart-glass-pane-interline-spacing-100")
        self._ils100_cells_idxs = np.where(self._model.geom_bodyid == self._sgp_ils100_body_idx)[0]

        # Define the target idx probability distribution
        self._target_idx_prob = None  # The dynamic target idx probability distribution, should be updated with transition function
        self._sampled_target_idx = None  # The sampled target idx
        self._VISUALIZE_RGBA = [1, 1, 0, 1]
        self._DFLT_RGBA = [0, 0, 0, 1]

        self._dwell_steps = int(2 * self._action_sample_freq)  # 2 seconds

        # Define the observation space
        # Origin - https://github.com/BaiYunpeng1949/uitb-headsup-computing/blob/c9ef14a91febfcb258c4990ebef2246c972e8aaa/huc/envs/locomotion/RelocationStackFrame.py#L111
        width, height = self._config['mj_env']['width'], self._config['mj_env']['height']
        self._num_stk_frm = 1
        self._num_stateful_info = 4
        self.observation_space = Dict({
            "vision": Box(low=-1, high=1, shape=(self._num_stk_frm, width, height)),
            "proprioception": Box(low=-1, high=1, shape=(self._num_stk_frm * self._model.nq + self._model.nu,)),
            "stateful information": Box(low=-1, high=1, shape=(self._num_stateful_info,)),
        })

        # Define the action space
        self.action_space = Box(low=-1, high=1, shape=(self._model.nu,))

        # Initialize the context and camera
        context = Context(self._model, max_resolution=[1280, 960])
        self._eye_cam = Camera(context, self._model, self._data, camera_id="eye", resolution=[width, height],
                               maxgeom=100,
                               dt=1 / self._action_sample_freq)
        self._env_cam = Camera(context, self._model, self._data, camera_id="env", maxgeom=100,
                               dt=1 / self._action_sample_freq)
        self._eye_cam_fovy = self._model.cam_fovy[mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, "eye")]

        # Initialise thresholds and counters
        self._steps = None
        self._on_target_steps = None
        self._num_trials = None  # Cells are already been read
        self._max_trials = 5  # Maximum number of cells to read - more trials in one episode will boost the convergence

        self.ep_len = int(self._max_trials * self._dwell_steps * 2)

    @staticmethod
    def normalise(x, x_min, x_max, a, b):
        # Normalise x (which is assumed to be in range [x_min, x_max]) to range [a, b]
        return (b - a) * ((x - x_min) / (x_max - x_min)) + a

    def render(self, mode="rgb_array"):
        rgb, _ = self._env_cam.render()
        rgb_eye, _ = self._eye_cam.render()
        return rgb, rgb_eye

    def _get_obs(self):
        """ Get the observation of the environment """
        # Get the vision observation
        # Render the image
        rgb, _ = self._eye_cam.render()

        # Preprocess - H*W*C -> C*W*H
        rgb = np.transpose(rgb, [2, 1, 0])
        rgb_normalize = self.normalise(rgb, 0, 255, -1, 1)

        # Convert the rgb to grayscale - boost the training speed
        gray_normalize = rgb_normalize[0:1, :, :] * 0.299 + rgb_normalize[1:2, :, :] * 0.587 + rgb_normalize[2:3, :,
                                                                                               :] * 0.114
        gray_normalize = np.squeeze(gray_normalize, axis=0)
        vision = gray_normalize.reshape((-1, gray_normalize.shape[-2], gray_normalize.shape[-1]))

        # Get the proprioception observation
        proprioception = np.concatenate([self._data.qpos, self._data.ctrl])

        # Get the stateful information observation - normalize to [-1, 1]
        remaining_ep_len_norm = (self.ep_len - self._steps) / self.ep_len * 2 - 1
        remaining_dwell_steps_norm = (self._dwell_steps - self._on_target_steps) / self._dwell_steps * 2 - 1
        remaining_trials_norm = (self._max_trials - self._num_trials) / self._max_trials * 2 - 1
        sampled_target_idx_norm = self.normalise(self._sampled_target_idx, self._ils100_cells_idxs[0],
                                                 self._ils100_cells_idxs[-1], -1, 1)
        stateful_info = np.array(
            [remaining_ep_len_norm, remaining_dwell_steps_norm, remaining_trials_norm, sampled_target_idx_norm])

        if stateful_info.shape[0] != self._num_stateful_info:
            raise ValueError("The shape of stateful information is not correct!")

        return {"vision": vision, "proprioception": proprioception, "stateful information": stateful_info}

    def reset(self):

        # Reset MuJoCo sim
        mujoco.mj_resetData(self._model, self._data)

        # Reset the variables and counters
        self._steps = 0
        self._on_target_steps = 0
        self._num_trials = 0

        # Initialize eyeball rotation angles
        self._data.qpos[self._eye_joint_x_idx] = np.random.uniform(-0.5, 0.5)
        self._data.qpos[self._eye_joint_y_idx] = np.random.uniform(-0.5, 0.5)

        # Update the target idx probability distribution
        self._update_target_idx_prob(mj_target_idx=np.random.choice(self._ils100_cells_idxs.copy()))

        # Sample a target according to the target idx probability distribution
        self._sample_target()

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

    def _update_target_idx_prob(self, mj_target_idx):
        """TODO the memory and belief model will be built upon this later"""
        # Initialize the target idx probability distribution - make sure probability sum to 1
        self._target_idx_prob = np.zeros(len(self._ils100_cells_idxs.copy()))
        # Get the idx of mj_target_idx in self._ils100_cells_idxs
        idx = np.where(self._ils100_cells_idxs == mj_target_idx)[0][0]
        # Update the target idx probability distribution by assigning the probability of the target idx at the given mj_target_idx to be 1
        self._target_idx_prob[idx] = 1
        # Check whether the probability sum to 1
        if np.sum(self._target_idx_prob) != 1:
            raise ValueError("The target idx probability distribution does not sum to 1!")

    def _sample_target(self):
        # Sample a target from the cells according to the target idx probability distribution
        self._sampled_target_idx = np.random.choice(self._ils100_cells_idxs.copy(), p=self._target_idx_prob)

        # Reset the counter
        self._on_target_steps = 0

        # Update the number of remaining unread cells
        self._num_trials += 1

    def step(self, action):
        # Normalise action from [-1, 1] to actuator control range
        action[0] = self.normalise(action[0], -1, 1, *self._model.actuator_ctrlrange[0, :])
        action[1] = self.normalise(action[1], -1, 1, *self._model.actuator_ctrlrange[1, :])

        # Set motor control
        self._data.ctrl[:] = action

        # Advance the simulation
        mujoco.mj_step(self._model, self._data, self._frame_skip)
        self._steps += 1

        # Eye-sight detection
        dist, geomid = self._get_focus(site_name="rangefinder-site")

        # Reset the scene first
        for mj_idx in self._ils100_cells_idxs:
            self._model.geom(mj_idx).rgba = self._DFLT_RGBA
        # Apply the transition function - update the scene regarding the actions
        if geomid == self._sampled_target_idx:
            self._on_target_steps += 1
            self._model.geom(self._sampled_target_idx).rgba = self._VISUALIZE_RGBA

        # Update the transitions - get rewards and next state
        if self._on_target_steps >= self._dwell_steps:
            # Update the milestone bonus reward for finish reading a cell
            reward = 10
            # Update the target idx probability distribution - randomly choose a new target idx
            self._update_target_idx_prob(mj_target_idx=np.random.choice(self._ils100_cells_idxs.copy()))
            # Get the next target
            self._sample_target()
        else:
            reward = 0.1 * (np.exp(
                -10 * self._angle_from_target(site_name="rangefinder-site", target_idx=self._sampled_target_idx)) - 1)

        # Get termination condition
        terminate = False
        if self._steps >= self.ep_len or self._num_trials > self._max_trials:
            terminate = True

        # Update the scene to reflect the transition function
        mujoco.mj_forward(self._model, self._data)

        return self._get_obs(), reward, terminate, {}


class AttentionSwitch(Env):

    def __init__(self):
        """ Model the reading, attention switch, switch back, relocation, resume reading with a belief model/function """
        # Get directory of this file
        directory = os.path.dirname(os.path.realpath(__file__))

        # Read the configurations from the YAML file.
        root_dir = os.path.dirname(os.path.dirname(directory))
        with open(os.path.join(root_dir, "config.yaml")) as f:
            self._config = yaml.load(f, Loader=yaml.FullLoader)

        # Load the MuJoCo model
        self._model = mujoco.MjModel.from_xml_path(os.path.join(directory, "relocation-v2.xml"))
        self._data = mujoco.MjData(self._model)
        mujoco.mj_forward(self._model, self._data)

        # Define how often policy is queried
        self._action_sample_freq = 20
        self._frame_skip = int((1 / self._action_sample_freq) / self._model.opt.timestep)

        # Get the primitive idx in MuJoCo
        self._eye_joint_x_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "eye-joint-x")
        self._eye_joint_y_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "eye-joint-y")
        self._head_body_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "head")
        self._eye_body_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "eye")
        self._sgp_ils100_body_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY,
                                                        "smart-glass-pane-interline-spacing-100")
        self._bg_body_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "background-pane")

        # Get MuJoCo cell idxs (geoms that belong to "smart-glass-pane-interline-spacing-100")
        self._ils100_cells_mjidxs = np.where(self._model.geom_bodyid == self._sgp_ils100_body_mjidx)[0]
        # Get MuJoCo background idx (geoms that belong to "background-pane")
        self._bg_pane_mjidxs = np.where(self._model.geom_bodyid == self._bg_body_mjidx)[0]
        self._bg_mjidx = self._bg_pane_mjidxs[0]
        # Concatenate cell idxs and background idx
        self._fixations_mjidxs = np.concatenate((self._ils100_cells_mjidxs, np.array([self._bg_mjidx])))

        # Get the min and max x and y positions of the possible fixation cells
        self._fixation_cells_x_min = np.min([self._data.geom(mjidx).xpos[0] for mjidx in self._fixations_mjidxs])
        self._fixation_cells_x_max = np.max([self._data.geom(mjidx).xpos[0] for mjidx in self._fixations_mjidxs])
        self._fixation_cells_z_min = np.min([self._data.geom(mjidx).xpos[2] for mjidx in self._fixations_mjidxs])
        self._fixation_cells_z_max = np.max([self._data.geom(mjidx).xpos[2] for mjidx in self._fixations_mjidxs])

        # Define the target idx probability distribution - TODO 2 stages of belief - stage 1 sample a target idx, stage 2 sample a fixation idx
        self._true_target_idx = None  # The true target MuJoCo idx
        self._target_mjidx_memory = None  # The memory of where should the sampled target mjidx be - TODO specify the prob distribution later
        self._sampled_target_mjidx = None  # The target MuJoCo idx, should be sampled from memory
        self._target_mjidx_belief = None  # 'Belief': The dynamic target MuJoCo idx probability distribution
        self._sampled_fixation_mjidx = None  # The fixation MuJoCo idx, should be sampled from belief

        self._VIS_CELL_RGBA = [1, 1, 0, 1]
        self._DFLT_CELL_RGBA = [0, 0, 0, 1]
        self._SHOW_BG_RGBA = [1, 1, 1, 1]
        self._VIS_BG_RGBA = [0, 0, 1, 1]
        self._DFLT_BG_RGBA = [0, 0, 0, 0]
        self._VIS_RELOC_RGBA = [1, 0, 0, 1]

        self._dwell_cell_steps = int(2 * self._action_sample_freq)  # 2 seconds
        self._dwell_bg_steps = int(1 * self._action_sample_freq)  # 1 second
        self._dwell_reloc_steps = int(
            1 * self._action_sample_freq)  # 1 second     # TODO use Hick's law to determine this

        # Task mode
        self._task_mode = None

        # Attention switch flag on a certain trial
        self._attention_switch = None

        # Determine the radian of the visual spotlight for visual search, or 'neighbors'
        self._neighbour_radius = 0.0101  # Obtained empirically
        self._neighbors_mjidxs_list = None  # The MuJoCo idxs of the neighbors of the sampled target idx

        # Initialise thresholds and counters
        self._steps = None
        self._fixation_steps = None
        self._num_trials = None
        self._max_trials = 5
        self.ep_len = int(self._max_trials * self._dwell_cell_steps * 2)

        # Define the observation space
        width, height = self._config['mj_env']['width'], self._config['mj_env']['height']
        self._num_stk_frm = 1
        self._num_stateful_info = 7
        self.observation_space = Dict({
            "vision": Box(low=-1, high=1, shape=(self._num_stk_frm, width, height)),
            "proprioception": Box(low=-1, high=1, shape=(self._num_stk_frm * self._model.nq + self._model.nu,)),
            "stateful information": Box(low=-1, high=1, shape=(self._num_stateful_info,)),
        })

        # Define the action space
        self.action_space = Box(low=-1, high=1, shape=(self._model.nu,))

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

    def render(self, mode="rgb_array"):
        rgb, _ = self._env_cam.render()
        rgb_eye, _ = self._eye_cam.render()
        return rgb, rgb_eye

    def _get_obs(self):
        """ Get the observation of the environment """
        # Get the vision observation
        # Render the image
        rgb, _ = self._eye_cam.render()

        # Preprocess - H*W*C -> C*W*H
        rgb = np.transpose(rgb, [2, 1, 0])
        rgb_normalize = self.normalise(rgb, 0, 255, -1, 1)

        # Convert the rgb to grayscale - boost the training speed
        gray_normalize = rgb_normalize[0:1, :, :] * 0.299 + rgb_normalize[1:2, :, :] * 0.587 + rgb_normalize[2:3, :,
                                                                                               :] * 0.114
        gray_normalize = np.squeeze(gray_normalize, axis=0)
        vision = gray_normalize.reshape((-1, gray_normalize.shape[-2], gray_normalize.shape[-1]))

        # Get the proprioception observation
        proprioception = np.concatenate([self._data.qpos, self._data.ctrl])

        # Get the stateful information observation - normalize to [-1, 1]
        remaining_ep_len_norm = (self.ep_len - self._steps) / self.ep_len * 2 - 1
        remaining_dwell_steps_norm = (self._dwell_cell_steps - self._fixation_steps) / self._dwell_cell_steps * 2 - 1
        remaining_trials_norm = (self._max_trials - self._num_trials) / self._max_trials * 2 - 1
        task_mode_norm = self._task_mode
        attention_switch_norm = 1 if self._attention_switch == True else -1
        # sampled_fixation_mjidx_norm = self.normalise(self._sampled_fixation_mjidx, self._fixations_mjidxs[0], self._fixations_mjidxs[-1], -1, 1)
        sampled_fixation_x = self._data.geom(self._sampled_fixation_mjidx).xpos[0]
        sampled_fixation_x_norm = self.normalise(sampled_fixation_x, self._fixation_cells_x_min, self._fixation_cells_x_max, -1, 1)
        sampled_fixation_z = self._data.geom(self._sampled_fixation_mjidx).xpos[2]
        sampled_fixation_z_norm = self.normalise(sampled_fixation_z, self._fixation_cells_z_min, self._fixation_cells_z_max, -1, 1)
        # TODO if I am explicitly telling the agent where the target/fixation is, why bother to use RL to learn to fixate?
        # TODO what about the rough layout information? e.g. the target is in the left/right half of the screen
        # TODO thus later we can add a belief of target location, but blur it with a Gaussian - I was roughly at the 4th line..., this can be integrated into the memory model.
        # TODO the agent should be able to figure out where he should look at with just some relative information, not the exact coordinators that can be done by rule based manner
        # TODO but I found that if just use idx, hard to train to switch to the env.
        stateful_info = np.array(
            [remaining_ep_len_norm, remaining_dwell_steps_norm, remaining_trials_norm,
             sampled_fixation_x_norm, sampled_fixation_z_norm, task_mode_norm, attention_switch_norm]
        )
        if stateful_info.shape[0] != self._num_stateful_info:
            raise ValueError("The shape of stateful information is not correct!")

        return {"vision": vision, "proprioception": proprioception, "stateful information": stateful_info}

    def reset(self):

        # Reset MuJoCo sim
        mujoco.mj_resetData(self._model, self._data)

        # Reset the variables and counters
        self._steps = 0
        self._num_trials = 0
        # Reset and initialize the target belief
        self._target_mjidx_belief = np.zeros(self._fixations_mjidxs.shape[0])

        # Initialize eyeball rotation angles
        self._data.qpos[self._eye_joint_x_mjidx] = np.random.uniform(-0.5, 0.5)
        self._data.qpos[self._eye_joint_y_mjidx] = np.random.uniform(-0.5, 0.5)

        # Initialize the task mode
        self._task_mode = READ

        # Initialize whether this trial includes the attention switch
        self._attention_switch = np.random.choice([True, False])

        # Sample a target mjidx
        self._sample_target()

        # Sample a target according to the target idx probability distribution
        self._sample_fixation(target_mjidx=self._sampled_target_mjidx)

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

    def _sample_target(self):
        if self._task_mode == READ:
            self._sampled_target_mjidx = np.random.choice(self._ils100_cells_mjidxs.copy())
        elif self._task_mode == BG:
            self._sampled_target_mjidx = self._bg_mjidx
        elif self._task_mode == RELOC:
            # Update the memory function - belief on which cell should be the target mjidx
            # TODO to start with a simplified version, the target mjidx are determinist - the cut off cell/word
            self._sampled_target_mjidx = self._true_target_idx

        return self._sampled_target_mjidx

    def _sample_fixation(self, target_mjidx, previous_fixation_mjidx=None):
        # Update the target mjidx belief
        fixate_target = False
        # TODO use the fixation_mjidx to update the target mjidx belief
        # Deterministic belief on the READ and BG task
        if self._task_mode == READ or self._task_mode == BG:
            # Reset the target idx probability distribution to all 0 in the reading mode
            self._target_mjidx_belief = np.zeros(self._target_mjidx_belief.shape[0])
            # Allocate the probability of the sampled target mjidx to be 1
            idx = np.where(self._fixations_mjidxs == target_mjidx)[0][0]
            self._target_mjidx_belief[idx] = 1
        elif self._task_mode == RELOC:
            # Initializations - Determine the neighbours by identifying all cells that are within the preset neighbour radius
            if previous_fixation_mjidx == None:
                # Reset the target idx probability distribution to all 0 in the reading mode
                self._target_mjidx_belief = np.zeros(self._target_mjidx_belief.shape[0])
                self._neighbors_mjidxs_list = []
                center_xpos = self._data.geom(target_mjidx).xpos
                for mjidx in self._ils100_cells_mjidxs:
                    xpos = self._data.geom(mjidx).xpos
                    dist = np.linalg.norm(xpos - center_xpos)
                    if dist <= self._neighbour_radius:
                        self._neighbors_mjidxs_list.append(mjidx)
                # Leak probability to the neighbours
                # TODO make it a function of the elapsed time later
                center_prob = 0.5
                leak_prob_per_neighbour = float((1 - center_prob) / (len(self._neighbors_mjidxs_list) - 1))
                # Assign the probability to the neighbours and the center
                for mjidx in self._neighbors_mjidxs_list:
                    if mjidx == target_mjidx:
                        idx = np.where(self._fixations_mjidxs == mjidx)[0][0]
                        self._target_mjidx_belief[idx] = center_prob
                    else:
                        idx = np.where(self._fixations_mjidxs == mjidx)[0][0]
                        self._target_mjidx_belief[idx] = leak_prob_per_neighbour

            # Update the belief of according to actual fixations - the previous fixation mjidx is not None
            else:
                # Find the index of the previous fixation mjidx in the fixation mjidx list
                idx = np.where(self._fixations_mjidxs == previous_fixation_mjidx)[0][0]
                # When the current fixation mjidx is not the sampled target mjidx
                if previous_fixation_mjidx != target_mjidx:
                    prob = self._target_mjidx_belief[idx]
                    # Set 0 to the current fixation mjidx
                    self._target_mjidx_belief[idx] = 0
                    # Remove it from the neighbors_mjidx
                    self._neighbors_mjidxs_list.remove(previous_fixation_mjidx)
                    # Reallocate the probability to assure they sum to 1
                    leak_prob_per_neighbour = float(prob / (len(self._neighbors_mjidxs_list)))
                    for mjidx in self._neighbors_mjidxs_list:
                        index = np.where(self._fixations_mjidxs == mjidx)[0][0]
                        self._target_mjidx_belief[index] += leak_prob_per_neighbour
                # When the sampled fixation mjidx is the sampled target mjidx
                else:
                    self._target_mjidx_belief = np.zeros(self._target_mjidx_belief.shape[0])
                    self._target_mjidx_belief[idx] = 1
                    fixate_target = True
                    # Clear the list of neighbours
                    self._neighbors_mjidxs_list = []
        else:
            raise ValueError("The task mode is not correct!")

        # Make sure they sum to 1
        self._target_mjidx_belief /= np.sum(self._target_mjidx_belief)

        # Sample a fixation mjidx from the cells according to the target mjidx probability distribution (belief)
        self._sampled_fixation_mjidx = np.random.choice(self._fixations_mjidxs.copy(), p=self._target_mjidx_belief)
        # Reset the counter
        self._fixation_steps = 0

        return fixate_target

    def step(self, action):
        # Normalise action from [-1, 1] to actuator control range
        action[0] = self.normalise(action[0], -1, 1, *self._model.actuator_ctrlrange[0, :])
        action[1] = self.normalise(action[1], -1, 1, *self._model.actuator_ctrlrange[1, :])

        # Set motor control
        self._data.ctrl[:] = action

        # Advance the simulation
        mujoco.mj_step(self._model, self._data, self._frame_skip)
        self._steps += 1

        # Eye-sight detection
        dist, geomid = self._get_focus(site_name="rangefinder-site")

        # Reset the scene first - TODO optimize this part for the training speed
        for mj_idx in self._ils100_cells_mjidxs:
            self._model.geom(mj_idx).rgba = self._DFLT_CELL_RGBA
        if self._task_mode == BG:
            self._model.geom(self._bg_mjidx).rgba = self._SHOW_BG_RGBA
        else:
            self._model.geom(self._bg_mjidx).rgba = self._DFLT_BG_RGBA

        # Step with different task modes
        if self._task_mode == READ:
            if geomid == self._sampled_fixation_mjidx:
                self._fixation_steps += 1
                self._model.geom(self._sampled_fixation_mjidx).rgba = self._VIS_CELL_RGBA

            if self._fixation_steps >= self._dwell_cell_steps:
                # Update the milestone bonus reward for finish reading a cell
                reward = 10

                # Check whether to change the task mode
                # TODO to simplify, the attention switch always happen after a cell has been fixated for enough time
                if self._attention_switch == True:
                    self._task_mode = BG
                    self._true_target_idx = self._sampled_target_mjidx
                    # Get the next target
                    self._sample_target()
                    self._sample_fixation(target_mjidx=self._sampled_target_mjidx)
                    # Show the background pane
                    self._model.geom(self._bg_mjidx).rgba = self._SHOW_BG_RGBA
                else:
                    # Update the number of trials
                    self._num_trials += 1
                    # Sample the next target and fixation
                    self._task_mode = READ
                    self._attention_switch = np.random.choice([True, False])
                    # Sample a new target mjidx in the READ mode
                    self._sample_target()
                    # Get the next target
                    self._sample_fixation(target_mjidx=self._sampled_target_mjidx)
            else:
                reward = 0.1 * (np.exp(-10 * self._angle_from_target(site_name="rangefinder-site",
                                                                     target_idx=self._sampled_fixation_mjidx)) - 1)
        elif self._task_mode == BG:
            if geomid == self._sampled_fixation_mjidx:
                self._fixation_steps += 1
                self._model.geom(self._sampled_fixation_mjidx).rgba = self._VIS_BG_RGBA

            if self._fixation_steps >= self._dwell_bg_steps:
                reward = 10
                self._task_mode = RELOC
                # Sample a new target mjidx in the RELOC mode
                self._sample_target()
                # Update the target idx probability distribution - randomly choose a new target idx
                self._sample_fixation(target_mjidx=self._sampled_target_mjidx)
                # Hide the background pane
                self._model.geom(self._bg_mjidx).rgba = self._DFLT_BG_RGBA
            else:
                reward = 0.1 * (np.exp(-10 * self._angle_from_target(site_name="rangefinder-site",
                                                                     target_idx=self._sampled_fixation_mjidx)) - 1)

        elif self._task_mode == RELOC:
            if geomid == self._sampled_fixation_mjidx:
                self._fixation_steps += 1
                self._model.geom(self._sampled_fixation_mjidx).rgba = self._VIS_RELOC_RGBA

            if self._fixation_steps >= self._dwell_reloc_steps:
                # Relocation needs to sample fixations multiple times because it is doing the visual search
                fixate_target = self._sample_fixation(target_mjidx=self._sampled_target_mjidx,
                                                      previous_fixation_mjidx=self._sampled_fixation_mjidx)
                if fixate_target == True:
                    # Pick up the target
                    reward = 10
                    # Update the number of trials
                    self._num_trials += 1
                    # Sample another target for the reading task
                    self._task_mode = READ
                    self._attention_switch = np.random.choice([True, False])
                    self._sample_target()
                    self._sample_fixation(target_mjidx=self._sampled_target_mjidx)
                else:
                    reward = 0.1 * (np.exp(-10 * self._angle_from_target(site_name="rangefinder-site",
                                                                         target_idx=self._sampled_fixation_mjidx)) - 1)
            else:
                reward = 0.1 * (np.exp(-10 * self._angle_from_target(site_name="rangefinder-site",
                                                                     target_idx=self._sampled_fixation_mjidx)) - 1)
        else:
            raise ValueError("The task mode is not correct!")

        # Get termination condition
        terminate = False
        if self._steps >= self.ep_len or self._num_trials > self._max_trials:
            terminate = True

        # Update the scene to reflect the transition function
        mujoco.mj_forward(self._model, self._data)

        return self._get_obs(), reward, terminate, {}


class AttentionSwitch3Layouts(Env):

    def __init__(self):
        """ Model the reading, attention switch, switch back, relocation, resume reading with a belief model/function
        Train on three layouts mentioned in the paper, TODO reconstruct this later"""
        # Get directory of this file
        directory = os.path.dirname(os.path.realpath(__file__))

        # Read the configurations from the YAML file.
        root_dir = os.path.dirname(os.path.dirname(directory))
        with open(os.path.join(root_dir, "config.yaml")) as f:
            self._config = yaml.load(f, Loader=yaml.FullLoader)

        # Load the MuJoCo model
        self._model = mujoco.MjModel.from_xml_path(os.path.join(directory, "relocation-v3.xml"))
        self._data = mujoco.MjData(self._model)
        mujoco.mj_forward(self._model, self._data)

        # Define how often policy is queried
        self._action_sample_freq = 20
        self._frame_skip = int((1 / self._action_sample_freq) / self._model.opt.timestep)

        # Get the primitive idx in MuJoCo
        self._eye_joint_x_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "eye-joint-x")
        self._eye_joint_y_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "eye-joint-y")
        self._head_body_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "head")
        self._eye_body_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "eye")
        self._sgp_ils100_body_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY,
                                                        "smart-glass-pane-interline-spacing-100")
        self._sgp_bc_body_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "smart-glass-pane-bottom-center")
        self._sgp_mr_body_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "smart-glass-pane-middle-right")
        self._bg_body_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "background-pane")

        # Get MuJoCo cell idxs (geoms that belong to "smart-glass-pane-interline-spacing-100")
        self._ils100_cells_mjidxs = np.where(self._model.geom_bodyid == self._sgp_ils100_body_mjidx)[0]
        self._bc_cells_mjidxs = np.where(self._model.geom_bodyid == self._sgp_bc_body_mjidx)[0]
        self._mr_cells_mjidxs = np.where(self._model.geom_bodyid == self._sgp_mr_body_mjidx)[0]
        # Get MuJoCo background idx (geoms that belong to "background-pane")
        self._bg_pane_mjidxs = np.where(self._model.geom_bodyid == self._bg_body_mjidx)[0]
        self._bg_mjidx = self._bg_pane_mjidxs[0]
        # Concatenate cell idxs and background idx
        self._fixations_all_layouts_mjidxs = np.concatenate((self._ils100_cells_mjidxs, self._bc_cells_mjidxs,
                                                     self._mr_cells_mjidxs, np.array([self._bg_mjidx])))
        self._fixations_ils100_mjidxs = np.concatenate((self._ils100_cells_mjidxs, np.array([self._bg_mjidx])))
        self._fixations_bc_mjidxs = np.concatenate((self._bc_cells_mjidxs, np.array([self._bg_mjidx])))
        self._fixations_mr_mjidxs = np.concatenate((self._mr_cells_mjidxs, np.array([self._bg_mjidx])))

        # Get the min and max x and y positions of the possible fixation cells - ils100
        self._fixation_ils100_x_min = np.min([self._data.geom(mjidx).xpos[0] for mjidx in self._fixations_ils100_mjidxs])
        self._fixation_ils100_x_max = np.max([self._data.geom(mjidx).xpos[0] for mjidx in self._fixations_ils100_mjidxs])
        self._fixation_ils100_z_min = np.min([self._data.geom(mjidx).xpos[2] for mjidx in self._fixations_ils100_mjidxs])
        self._fixation_ils100_z_max = np.max([self._data.geom(mjidx).xpos[2] for mjidx in self._fixations_ils100_mjidxs])
        # bc
        self._fixation_bc_x_min = np.min([self._data.geom(mjidx).xpos[0] for mjidx in self._fixations_bc_mjidxs])
        self._fixation_bc_x_max = np.max([self._data.geom(mjidx).xpos[0] for mjidx in self._fixations_bc_mjidxs])
        self._fixation_bc_z_min = np.min([self._data.geom(mjidx).xpos[2] for mjidx in self._fixations_bc_mjidxs])
        self._fixation_bc_z_max = np.max([self._data.geom(mjidx).xpos[2] for mjidx in self._fixations_bc_mjidxs])
        # mr
        self._fixation_mr_x_min = np.min([self._data.geom(mjidx).xpos[0] for mjidx in self._fixations_mr_mjidxs])
        self._fixation_mr_x_max = np.max([self._data.geom(mjidx).xpos[0] for mjidx in self._fixations_mr_mjidxs])
        self._fixation_mr_z_min = np.min([self._data.geom(mjidx).xpos[2] for mjidx in self._fixations_mr_mjidxs])
        self._fixation_mr_z_max = np.max([self._data.geom(mjidx).xpos[2] for mjidx in self._fixations_mr_mjidxs])

        # Define the target idx probability distribution -
        # TODO the believed target idx in the relocation is the sampled fixation
        # TODO consolidate these terminologies later - memory - belief - fixation - target
        self._true_target_mjidx = None  # The true target MuJoCo idx
        self._target_mjidx_memory = None  # The perceived target MuJoCo idx, should be sampled from memory, which decays as time goes by, but the element should be the same as neighboring elements and the target belief
        self._memory_decay_rate = 0.2  # The memory decay rate - TODO a hyper-parameter, might need to fit to human data
        self._elapsed_visual_search_cells = None  # The number of visual search trials elapsed
        self._sampled_target_mjidx = None  # The target MuJoCo idx, should be sampled from memory
        self._target_mjidx_belief = None  # 'Belief': The dynamic target MuJoCo idx probability distribution
        self._sampled_fixation_mjidx = None  # The fixation MuJoCo idx, should be sampled from belief

        self._VIS_CELL_RGBA = [1, 1, 0, 1]
        self._DFLT_CELL_RGBA = [0, 0, 0, 1]
        self._SHOW_BG_RGBA = [1, 1, 1, 1]
        self._VIS_BG_RGBA = [0, 0, 1, 1]
        self._DFLT_BG_RGBA = [0, 0, 0, 0]
        self._VIS_RELOC_RGBA = [1, 0, 0, 1]

        self._dwell_cell_steps = int(2 * self._action_sample_freq)  # 2 seconds
        self._dwell_bg_steps = int(1 * self._action_sample_freq)  # 1 second
        self._dwell_reloc_steps = int(
            1 * self._action_sample_freq)  # 1 second     # TODO use Hick's law to determine this

        # Task mode
        self._task_mode = None

        # Layout
        self._layout_idx = None
        self._layout_fixations_mjidxs = None
        self._layout_cells_mjidxs = None

        # Attention switch flag on a certain trial
        self._attention_switch = None

        # Determine the radian of the visual spotlight for visual search, or 'neighbors'
        self._neighbour_radius = 0.0101  # Obtained empirically
        self._neighbors_mjidxs_list = None  # The MuJoCo idxs of the neighbors of the sampled target idx

        # Initialise thresholds and counters
        self._steps = None
        self._fixation_steps = None
        self._num_trials = None
        self._max_trials = 5
        self.ep_len = int(self._max_trials * self._dwell_cell_steps * 4)        # Make sure enough steps for tasks

        # Test-related variables
        self._test_switch_back_duration_list = None
        self._test_switch_back_step = None
        self._test_switch_back_error_list = None

        # Define the observation space
        width, height = self._config['mj_env']['width'], self._config['mj_env']['height']
        self._num_stk_frm = 1
        self._num_stateful_info = 8
        self.observation_space = Dict({
            "vision": Box(low=-1, high=1, shape=(self._num_stk_frm, width, height)),
            "proprioception": Box(low=-1, high=1, shape=(self._num_stk_frm * self._model.nq + self._model.nu,)),
            "stateful information": Box(low=-1, high=1, shape=(self._num_stateful_info,)),
        })

        # Define the action space
        self.action_space = Box(low=-1, high=1, shape=(self._model.nu,))

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

    def render(self, mode="rgb_array"):
        rgb, _ = self._env_cam.render()
        rgb_eye, _ = self._eye_cam.render()
        return rgb, rgb_eye

    def _get_obs(self):
        """ Get the observation of the environment """
        # Get the vision observation
        # Render the image
        rgb, _ = self._eye_cam.render()

        # Preprocess - H*W*C -> C*W*H
        rgb = np.transpose(rgb, [2, 1, 0])
        rgb_normalize = self.normalise(rgb, 0, 255, -1, 1)

        # Convert the rgb to grayscale - boost the training speed
        gray_normalize = rgb_normalize[0:1, :, :] * 0.299 + rgb_normalize[1:2, :, :] * 0.587 + rgb_normalize[2:3, :,
                                                                                               :] * 0.114
        gray_normalize = np.squeeze(gray_normalize, axis=0)
        vision = gray_normalize.reshape((-1, gray_normalize.shape[-2], gray_normalize.shape[-1]))

        # Get the proprioception observation
        proprioception = np.concatenate([self._data.qpos, self._data.ctrl])

        # Get the stateful information observation - normalize to [-1, 1]
        remaining_ep_len_norm = (self.ep_len - self._steps) / self.ep_len * 2 - 1
        if self._task_mode == READ:
            dwell_cell_steps = self._dwell_cell_steps
        elif self._task_mode == BG:
            dwell_cell_steps = self._dwell_bg_steps
        elif self._task_mode == RELOC:
            dwell_cell_steps = self._dwell_reloc_steps
        else:
            raise ValueError("Invalid task mode!")
        remaining_dwell_steps_norm = (dwell_cell_steps - self._fixation_steps) / dwell_cell_steps * 2 - 1
        remaining_trials_norm = (self._max_trials - self._num_trials) / self._max_trials * 2 - 1
        task_mode_norm = self._task_mode
        attention_switch_norm = 1 if self._attention_switch == True else -1
        layout_norm = self._layout_idx
        if self._layout_idx == ILS100:
            x_min = self._fixation_ils100_x_min
            x_max = self._fixation_ils100_x_max
            z_min = self._fixation_ils100_z_min
            z_max = self._fixation_ils100_z_max
        elif self._layout_idx == BC:
            x_min = self._fixation_bc_x_min
            x_max = self._fixation_bc_x_max
            z_min = self._fixation_bc_z_min
            z_max = self._fixation_bc_z_max
        elif self._layout_idx == MR:
            x_min = self._fixation_mr_x_min
            x_max = self._fixation_mr_x_max
            z_min = self._fixation_mr_z_min
            z_max = self._fixation_mr_z_max
        else:
            raise ValueError("Invalid layout index!")
        sampled_fixation_x = self._data.geom(self._sampled_fixation_mjidx).xpos[0]
        sampled_fixation_x_norm = self.normalise(sampled_fixation_x, x_min, x_max, -1, 1)
        sampled_fixation_z = self._data.geom(self._sampled_fixation_mjidx).xpos[2]
        sampled_fixation_z_norm = self.normalise(sampled_fixation_z, z_min, z_max, -1, 1)
        stateful_info = np.array(
            [remaining_ep_len_norm, remaining_dwell_steps_norm, remaining_trials_norm, layout_norm,
             sampled_fixation_x_norm, sampled_fixation_z_norm, task_mode_norm, attention_switch_norm]
        )
        if stateful_info.shape[0] != self._num_stateful_info:
            raise ValueError("The shape of stateful information is not correct!")

        return {"vision": vision, "proprioception": proprioception, "stateful information": stateful_info}

    def reset(self):

        # Reset MuJoCo sim
        mujoco.mj_resetData(self._model, self._data)

        # Reset the variables and counters
        self._steps = 0
        self._num_trials = 0

        # Initialize eyeball rotation angles
        self._data.qpos[self._eye_joint_x_mjidx] = np.random.uniform(-0.5, 0.5)
        self._data.qpos[self._eye_joint_y_mjidx] = np.random.uniform(-0.5, 0.5)

        # Reset some test-related variables
        self._test_switch_back_duration_list = []
        self._test_switch_back_step = self._steps
        self._test_switch_back_error_list = []

        # Initialize the layout - TODO to simplify the training, the layouts only refreshed once in one episode
        # self._layout_idx = np.random.choice([ILS100, BC, MR])
        self._layout_idx = BC   # TODO debug and test, choose from ILS100, BC, MR
        # Reset the scene - except the chosen layout, all the other layouts are hidden
        for mjidx in self._fixations_all_layouts_mjidxs:
            self._model.geom(mjidx).rgba[3] = 0
        if self._layout_idx == ILS100:
            self._layout_fixations_mjidxs = self._fixations_ils100_mjidxs
            self._layout_cells_mjidxs = self._ils100_cells_mjidxs
        elif self._layout_idx == BC:
            self._layout_fixations_mjidxs = self._fixations_bc_mjidxs
            self._layout_cells_mjidxs = self._bc_cells_mjidxs
        elif self._layout_idx == MR:
            self._layout_fixations_mjidxs = self._fixations_mr_mjidxs
            self._layout_cells_mjidxs = self._mr_cells_mjidxs
        else:
            raise ValueError("The layout index is not correct!")
        for mjidx in self._layout_cells_mjidxs:
            self._model.geom(mjidx).rgba[3] = 1

        # Reset and initialize the target belief
        self._target_mjidx_belief = np.zeros(self._layout_fixations_mjidxs.shape[0])
        # Reset and initialize the target memory
        self._target_mjidx_memory = np.zeros(self._layout_fixations_mjidxs.shape[0])

        # Reset the frequently updated variables
        self._elapsed_visual_search_cells = 0

        # Initialize the task mode
        self._task_mode = READ

        # Initialize whether this trial includes the attention switch
        self._attention_switch = np.random.choice([True, False])

        # Sample a target mjidx
        self._sample_target()

        # Sample a target according to the target idx probability distribution
        self._sample_fixation(target_mjidx=self._sampled_target_mjidx)

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

    def _sample_target(self):
        if self._task_mode == READ:
            self._sampled_target_mjidx = np.random.choice(self._layout_cells_mjidxs.copy())
        elif self._task_mode == BG:
            self._sampled_target_mjidx = self._bg_mjidx
        elif self._task_mode == RELOC:
            # Update the memory function - belief on which cell should be the target mjidx
            # Reset
            self._elapsed_visual_search_cells = 0
            # Update the target mjidx memory - before time cost on the visual search, there is no memory decay, the agent remembers where the target was with 1 prob
            self._target_mjidx_memory = np.zeros(self._target_mjidx_memory.shape[0])
            # TODO to start with a simplified version, the target mjidx are determinist - the cut off cell/word
            self._sampled_target_mjidx = self._true_target_mjidx
            idx = np.where(self._layout_fixations_mjidxs == self._sampled_target_mjidx)[0][0]
            self._target_mjidx_memory[idx] = 1


        return self._sampled_target_mjidx

    def _sample_fixation(self, target_mjidx, previous_fixation_mjidx=None):
        # Update the target mjidx belief
        fixate_on_target = False
        # Deterministic belief on the READ and BG task
        if self._task_mode == READ or self._task_mode == BG:
            # Reset the target idx probability distribution to all 0 in the reading mode
            self._target_mjidx_belief = np.zeros(self._target_mjidx_belief.shape[0])
            # Allocate the probability of the sampled target mjidx to be 1
            idx = np.where(self._layout_fixations_mjidxs == target_mjidx)[0][0]
            self._target_mjidx_belief[idx] = 1
        elif self._task_mode == RELOC:
            # Initializations - Determine the neighbours by identifying all cells that are within the preset neighbour radius
            if previous_fixation_mjidx == None:
                # Reset the target idx probability distribution to all 0 in the reading mode
                self._target_mjidx_belief = np.zeros(self._target_mjidx_belief.shape[0])
                self._neighbors_mjidxs_list = []
                center_xpos = self._data.geom(self._true_target_mjidx).xpos
                for mjidx in self._layout_cells_mjidxs:
                    xpos = self._data.geom(mjidx).xpos
                    dist = np.linalg.norm(xpos - center_xpos)
                    if dist <= self._neighbour_radius:
                        self._neighbors_mjidxs_list.append(mjidx)
                # Leak probability to the neighbours
                # TODO make it a function of the elapsed time later
                # center_prob = 0.5   # TODO a hyper-parameter, determine later
                # leak_prob_per_neighbour = float((1 - center_prob) / (len(self._neighbors_mjidxs_list) - 1))
                # # Assign the probability to the neighbours and the center
                # for mjidx in self._neighbors_mjidxs_list:
                #     if mjidx == target_mjidx:
                #         idx = np.where(self._layout_fixations_mjidxs == mjidx)[0][0]
                #         self._target_mjidx_belief[idx] = center_prob
                #     else:
                #         idx = np.where(self._layout_fixations_mjidxs == mjidx)[0][0]
                #         self._target_mjidx_belief[idx] = leak_prob_per_neighbour
                # Apply the even prob across all cells in the neighbour list -
                # TODO figure out a better way to do this - is it necessarily the even prob distribution?
                even_prob = float(1 / len(self._neighbors_mjidxs_list))
                for mjidx in self._neighbors_mjidxs_list:
                    idx = np.where(self._layout_fixations_mjidxs == mjidx)[0][0]
                    self._target_mjidx_belief[idx] = even_prob

            # Update the belief of according to actual fixations - the previous fixation mjidx is not None
            else:
                # Find the index of the previous fixation mjidx in the fixation mjidx list
                idx = np.where(self._layout_fixations_mjidxs == previous_fixation_mjidx)[0][0]
                # Seems right now does not matter whether the previous fixation mjidx is the target mjidx, since the target mjidix is updated along the visual search with fixations :>
                certainty_prob = self._target_mjidx_memory[idx]
                target_boolean = np.random.choice([True, False], p=[certainty_prob, 1 - certainty_prob])
                # If the previous fixation mjidx is regarded as the target mjidx
                if target_boolean == True:
                    return True, previous_fixation_mjidx
                else:
                    # Update the belief of the target mjidx - fixations
                    fixation_prob = self._target_mjidx_belief[idx]
                    # Set 0 to the current fixation mjidx
                    self._target_mjidx_belief[idx] = 0
                    self._target_mjidx_memory[idx] = 0
                    # Remove it from the neighbors_mjidx
                    self._neighbors_mjidxs_list.remove(previous_fixation_mjidx)
                    leak_prob_per_neighbour = float(fixation_prob / (len(self._neighbors_mjidxs_list)))
                    for mjidx in self._neighbors_mjidxs_list:
                        index = np.where(self._layout_fixations_mjidxs == mjidx)[0][0]
                        self._target_mjidx_belief[index] += leak_prob_per_neighbour
                    # Update the memory - the certainty prob states
                    true_target_idx = np.where(self._layout_fixations_mjidxs == self._true_target_mjidx)[0][0]
                    # If not the true target idx discarded
                    if self._true_target_mjidx in self._neighbors_mjidxs_list:
                        # self._target_mjidx_memory[idx] = 0
                        # Memory decay update to the true target idx
                        true_target_memory_after_decay = 1 * np.exp(-self._memory_decay_rate * self._elapsed_visual_search_cells)
                        self._target_mjidx_memory[true_target_idx] = true_target_memory_after_decay
                        # Update the rest certainty prob
                        for mjidx in self._neighbors_mjidxs_list:
                            if mjidx != self._true_target_mjidx:
                                index = np.where(self._layout_fixations_mjidxs == mjidx)[0][0]
                                self._target_mjidx_memory[index] = float((1 - true_target_memory_after_decay) / (len(self._neighbors_mjidxs_list) - 1))
                    # If the true target idx is discarded
                    else:
                        # True target memory update - only useful for the first time
                        self._target_mjidx_memory[true_target_idx] = 0
                        # self._target_mjidx_memory[idx] = 0
                        # Update the rest certainty prob
                        for mjidx in self._neighbors_mjidxs_list:
                            index = np.where(self._layout_fixations_mjidxs == mjidx)[0][0]
                            self._target_mjidx_memory[index] = float(1 / (len(self._neighbors_mjidxs_list)))
                    # Make sure the belief and memory sum to 1
                    self._target_mjidx_belief = self._target_mjidx_belief / np.sum(self._target_mjidx_belief)
                    self._target_mjidx_memory = self._target_mjidx_memory / np.sum(self._target_mjidx_memory)

                # # When the current fixation mjidx is not the sampled target mjidx
                # if previous_fixation_mjidx != target_mjidx:
                #     # Visual search starts by allocating various cells to be the fixations
                #     prob = self._target_mjidx_belief[idx]
                #     # Set 0 to the current fixation mjidx
                #     self._target_mjidx_belief[idx] = 0
                #     # Remove it from the neighbors_mjidx
                #     self._neighbors_mjidxs_list.remove(previous_fixation_mjidx)
                #     # Reallocate the probability to assure they sum to 1
                #     # TODO debug delete later
                #     if len(self._neighbors_mjidxs_list) == 0:
                #         print(f"Noooooooooo  The fixation mjidx is {previous_fixation_mjidx}, "
                #               f"the target memory is {self._target_mjidx_memory}, the target belief is {self._target_mjidx_belief}")
                #
                #     leak_prob_per_neighbour = float(prob / (len(self._neighbors_mjidxs_list)))
                #     for mjidx in self._neighbors_mjidxs_list:
                #         index = np.where(self._layout_fixations_mjidxs == mjidx)[0][0]
                #         self._target_mjidx_belief[index] += leak_prob_per_neighbour
                #     # The memory decay happens after the visual search -
                #     # TODO use the number of trials intead of the actual steps to simplify the problem and speed up training
                #     # memory_after_decay = 1 * np.exp(-self._memory_decay_rate * self._elapsed_visual_search_cells)   # TODO non-linearly
                #     memory_after_decay = 1 * (1 - self._memory_decay_rate * self._elapsed_visual_search_cells)   # TODO linearly
                #     if len(self._neighbors_mjidxs_list) - 1 > 0:
                #         # Update the memory after the visual search - leak the certainty of the true target to other visual search cell candidates
                #         certainty_leak = float(memory_after_decay / (len(self._neighbors_mjidxs_list) - 1))
                #         # Allocate the certainty to the neighbours
                #         for mjidx in self._neighbors_mjidxs_list:
                #             if mjidx == self._true_target_idx:
                #                 idx = np.where(self._layout_fixations_mjidxs == mjidx)[0][0]
                #                 self._target_mjidx_memory[idx] = memory_after_decay
                #             else:
                #                 idx = np.where(self._layout_fixations_mjidxs == mjidx)[0][0]
                #                 self._target_mjidx_memory[idx] = certainty_leak
                #     # Make sure the memory prob sum to 1
                #     self._target_mjidx_memory /= np.sum(self._target_mjidx_memory)
                #     # Sample a new target
                #     self._sampled_target_mjidx = np.random.choice(self._layout_fixations_mjidxs.copy(), p=self._target_mjidx_memory)
                # # When the sampled fixation mjidx is the sampled target mjidx
                # else:
                #     # Reset the belief buffer and the neighbours list
                #     self._target_mjidx_belief = np.zeros(self._target_mjidx_belief.shape[0])
                #     self._neighbors_mjidxs_list = []
                #     self._target_mjidx_memory = np.zeros(self._target_mjidx_memory.shape[0])
                #     fixate_on_target = True
                #     return fixate_on_target
        else:
            raise ValueError("The task mode is not correct!")

        # Make sure they sum to 1
        self._target_mjidx_belief /= np.sum(self._target_mjidx_belief)

        # Sample a fixation mjidx from the cells according to the target mjidx probability distribution (belief)
        self._sampled_fixation_mjidx = np.random.choice(self._layout_fixations_mjidxs.copy(), p=self._target_mjidx_belief)
        # Reset the counter
        self._fixation_steps = 0

        return False, previous_fixation_mjidx

    def step(self, action):
        # Normalise action from [-1, 1] to actuator control range
        action[0] = self.normalise(action[0], -1, 1, *self._model.actuator_ctrlrange[0, :])
        action[1] = self.normalise(action[1], -1, 1, *self._model.actuator_ctrlrange[1, :])

        # # TODO debug delete/comment this part later
        # mjidx = self._sampled_fixation_mjidx
        # xpos = self._data.geom(mjidx).xpos
        # x, y, z = xpos[0], xpos[1], xpos[2]
        # action = [z/y, -x/y]

        # Set motor control
        self._data.ctrl[:] = action

        # Advance the simulation
        mujoco.mj_step(self._model, self._data, self._frame_skip)
        self._steps += 1

        # Eye-sight detection
        dist, geomid = self._get_focus(site_name="rangefinder-site")

        # Reset the scene first - TODO optimize this part for the training speed
        for mj_idx in self._layout_fixations_mjidxs:
            self._model.geom(mj_idx).rgba = self._DFLT_CELL_RGBA
        if self._task_mode == BG:
            self._model.geom(self._bg_mjidx).rgba = self._SHOW_BG_RGBA
        else:
            self._model.geom(self._bg_mjidx).rgba = self._DFLT_BG_RGBA

        # Step with different task modes
        if self._task_mode == READ:
            if geomid == self._sampled_fixation_mjidx:
                self._fixation_steps += 1
                self._model.geom(self._sampled_fixation_mjidx).rgba = self._VIS_CELL_RGBA

            if self._fixation_steps >= self._dwell_cell_steps:
                # Update the milestone bonus reward for finish reading a cell
                reward = 10

                # Check whether to change the task mode
                # TODO to simplify, the attention switch always happen after a cell has been fixated for enough time
                if self._attention_switch == True:
                    self._task_mode = BG
                    self._true_target_mjidx = self._sampled_target_mjidx
                    # Get the next target
                    self._sample_target()
                    self._sample_fixation(target_mjidx=self._sampled_target_mjidx)
                    # Show the background pane
                    self._model.geom(self._bg_mjidx).rgba = self._SHOW_BG_RGBA
                else:
                    # Update the number of trials
                    self._num_trials += 1
                    # Sample the next target and fixation
                    self._task_mode = READ
                    self._attention_switch = np.random.choice([True, False])
                    # Sample a new target mjidx in the READ mode
                    self._sample_target()
                    # Get the next target
                    self._sample_fixation(target_mjidx=self._sampled_target_mjidx)
            else:
                reward = 0.1 * (np.exp(-10 * self._angle_from_target(site_name="rangefinder-site",
                                                                     target_idx=self._sampled_fixation_mjidx)) - 1)
        elif self._task_mode == BG:
            if geomid == self._sampled_fixation_mjidx:
                self._fixation_steps += 1
                self._model.geom(self._sampled_fixation_mjidx).rgba = self._VIS_BG_RGBA

            if self._fixation_steps >= self._dwell_bg_steps:
                reward = 10
                self._task_mode = RELOC
                self._test_switch_back_step = self._steps
                # Sample a new target mjidx in the RELOC mode
                self._sample_target()
                # Update the target idx probability distribution - randomly choose a new target idx
                self._sample_fixation(target_mjidx=self._sampled_target_mjidx)
                # Hide the background pane
                self._model.geom(self._bg_mjidx).rgba = self._DFLT_BG_RGBA

                # # TODO debug delete it later
                # print(
                #     f"The true target is {self._true_target_mjidx}, the sampled target is {self._sampled_target_mjidx}, The sampled fixation is {self._sampled_fixation_mjidx},"
                #     f"The target mjidx memory is {self._target_mjidx_memory}, the target idx belief is {self._target_mjidx_belief},"
                # )

            else:
                reward = 0.1 * (np.exp(-10 * self._angle_from_target(site_name="rangefinder-site",
                                                                     target_idx=self._sampled_fixation_mjidx)) - 1)

        elif self._task_mode == RELOC:
            if geomid == self._sampled_fixation_mjidx:
                self._fixation_steps += 1
                self._model.geom(self._sampled_fixation_mjidx).rgba = self._VIS_RELOC_RGBA

            if self._fixation_steps >= self._dwell_reloc_steps:
                # Update the fixation trials during the relocation visual search
                self._elapsed_visual_search_cells += 1
                # Relocation needs to sample fixations multiple times because it is doing the visual search
                fixate_on_target, pre_fixation_mjidx = self._sample_fixation(target_mjidx=self._sampled_target_mjidx,
                                                      previous_fixation_mjidx=self._sampled_fixation_mjidx)

                # # TODO debug delete it later
                # print(
                #     f"The true target is {self._true_target_mjidx}, the sampled target is {self._sampled_target_mjidx}, The sampled fixation is {self._sampled_fixation_mjidx},"
                #     f"The target mjidx memory is {self._target_mjidx_memory}, the target idx belief is {self._target_mjidx_belief},"
                # )

                if fixate_on_target == True:
                    # Pick up the target -
                    # TODO since we are modeling both correct and incorrect relocations,
                    #  we should also give rewards for incorrect relocations, try punish this later -
                    #  but I guess the agent will develop a policy of cutting off at some points, which is not my objective of modeling it, too detailed
                    reward = 10
                    # Reset the counter
                    self._elapsed_visual_search_cells = 0
                    # Update the number of trials
                    self._num_trials += 1
                    # Sample another target for the reading task
                    self._task_mode = READ
                    self._test_switch_back_duration_list.append(self._steps - self._test_switch_back_step)
                    self._test_switch_back_error_list.append(np.abs(pre_fixation_mjidx - self._true_target_mjidx))
                    self._attention_switch = np.random.choice([True, False])
                    self._sample_target()
                    self._sample_fixation(target_mjidx=self._sampled_target_mjidx)
                else:
                    reward = 0.1 * (np.exp(-10 * self._angle_from_target(site_name="rangefinder-site",
                                                                         target_idx=self._sampled_fixation_mjidx)) - 1)
            else:
                reward = 0.1 * (np.exp(-10 * self._angle_from_target(site_name="rangefinder-site",
                                                                     target_idx=self._sampled_fixation_mjidx)) - 1)
        else:
            raise ValueError("The task mode is not correct!")

        # Get termination condition
        terminate = False
        if self._steps >= self.ep_len or self._num_trials >= self._max_trials:
            terminate = True
            # TODO print the switch back error rate and the switch back durations - testify whether such modelings make sense
            if self._config["rl"]["mode"] == "test":
                print(f"The switch back avg duration is {np.mean(self._test_switch_back_duration_list)}, "
                      f"The switch back avg error is {np.mean(self._test_switch_back_error_list)}")

        # Update the scene to reflect the transition function
        mujoco.mj_forward(self._model, self._data)

        return self._get_obs(), reward, terminate, {}


# TODO extract the relocation + memory function from the eyeball fixations later if agent cannot learn well.
class AttentionSwitchMemory(Env):

    def __init__(self):
        """
        Model the reading, attention switch, switch back, relocation, resume reading with stochastics.
        The belief will sample an intended focus of attention; Then according to the confidence, which is calculated based
        on the decaying memory, agent decides whether to determine the focused item as the true target to pick up and resume reading from.
        """
        # Get directory of this file
        directory = os.path.dirname(os.path.realpath(__file__))

        # Read the configurations from the YAML file.
        root_dir = os.path.dirname(os.path.dirname(directory))
        with open(os.path.join(root_dir, "config.yaml")) as f:
            self._config = yaml.load(f, Loader=yaml.FullLoader)

        # Load the MuJoCo model
        self._model = mujoco.MjModel.from_xml_path(os.path.join(directory, "relocation-v3.xml"))
        self._data = mujoco.MjData(self._model)
        mujoco.mj_forward(self._model, self._data)

        # Define how often policy is queried
        self._action_sample_freq = 20
        self._frame_skip = int((1 / self._action_sample_freq) / self._model.opt.timestep)

        # Get the primitive idx in MuJoCo
        self._eye_joint_x_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "eye-joint-x")
        self._eye_joint_y_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "eye-joint-y")
        self._head_body_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "head")
        self._eye_body_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "eye")
        self._sgp_ils100_body_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY,
                                                        "smart-glass-pane-interline-spacing-100")
        self._sgp_bc_body_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "smart-glass-pane-bottom-center")
        self._sgp_mr_body_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "smart-glass-pane-middle-right")
        self._bg_body_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "background-pane")

        # Get MuJoCo cell idxs (geoms that belong to "smart-glass-pane-interline-spacing-100")
        self._ils100_cells_mjidxs = np.where(self._model.geom_bodyid == self._sgp_ils100_body_mjidx)[0]
        self._bc_cells_mjidxs = np.where(self._model.geom_bodyid == self._sgp_bc_body_mjidx)[0]
        self._mr_cells_mjidxs = np.where(self._model.geom_bodyid == self._sgp_mr_body_mjidx)[0]
        # Get MuJoCo background idx (geoms that belong to "background-pane")
        self._bg_pane_mjidxs = np.where(self._model.geom_bodyid == self._bg_body_mjidx)[0]
        self._bg_mjidx = self._bg_pane_mjidxs[0]
        # Concatenate cell idxs and background idx
        self._fixations_all_layouts_mjidxs = np.concatenate((self._ils100_cells_mjidxs, self._bc_cells_mjidxs,
                                                     self._mr_cells_mjidxs, np.array([self._bg_mjidx])))
        self._fixations_ils100_mjidxs = np.concatenate((self._ils100_cells_mjidxs, np.array([self._bg_mjidx])))
        self._fixations_bc_mjidxs = np.concatenate((self._bc_cells_mjidxs, np.array([self._bg_mjidx])))
        self._fixations_mr_mjidxs = np.concatenate((self._mr_cells_mjidxs, np.array([self._bg_mjidx])))

        # Get the min and max x and y positions of the possible fixation cells - ils100
        self._fixation_ils100_x_min = np.min([self._data.geom(mjidx).xpos[0] for mjidx in self._fixations_ils100_mjidxs])
        self._fixation_ils100_x_max = np.max([self._data.geom(mjidx).xpos[0] for mjidx in self._fixations_ils100_mjidxs])
        self._fixation_ils100_z_min = np.min([self._data.geom(mjidx).xpos[2] for mjidx in self._fixations_ils100_mjidxs])
        self._fixation_ils100_z_max = np.max([self._data.geom(mjidx).xpos[2] for mjidx in self._fixations_ils100_mjidxs])
        # bc
        self._fixation_bc_x_min = np.min([self._data.geom(mjidx).xpos[0] for mjidx in self._fixations_bc_mjidxs])
        self._fixation_bc_x_max = np.max([self._data.geom(mjidx).xpos[0] for mjidx in self._fixations_bc_mjidxs])
        self._fixation_bc_z_min = np.min([self._data.geom(mjidx).xpos[2] for mjidx in self._fixations_bc_mjidxs])
        self._fixation_bc_z_max = np.max([self._data.geom(mjidx).xpos[2] for mjidx in self._fixations_bc_mjidxs])
        # mr
        self._fixation_mr_x_min = np.min([self._data.geom(mjidx).xpos[0] for mjidx in self._fixations_mr_mjidxs])
        self._fixation_mr_x_max = np.max([self._data.geom(mjidx).xpos[0] for mjidx in self._fixations_mr_mjidxs])
        self._fixation_mr_z_min = np.min([self._data.geom(mjidx).xpos[2] for mjidx in self._fixations_mr_mjidxs])
        self._fixation_mr_z_max = np.max([self._data.geom(mjidx).xpos[2] for mjidx in self._fixations_mr_mjidxs])

        # Define the target idx probability distribution -
        # TODO Apply the Gaussian distribution to the belief and confidence distributions
        self._true_target_mjidx = None  # The true target MuJoCo idx
        self._buffer_true_target_mjidx = None  # The buffer for the true target MuJoCo idx
        self._target_confidence_distribution = None  # The perceived target MuJoCo idx, should be sampled from memory, which decays as time goes by, but the element should be the same as neighboring elements and the target belief
        self._memory_decay_rate = 0.4  # The memory decay rate - TODO a hyper-parameter, might need to fit to human data
        self._num_elapsed_visual_searched_cells = None  # The number of visual search trials elapsed
        self._target_position_belief_distribution = None  # 'Belief': The dynamic target MuJoCo idx probability distribution
        self._sampled_intended_focus_mjidx = None  # The fixation MuJoCo idx, should be sampled from belief

        self._VIS_CELL_RGBA = [1, 1, 0, 1]
        self._DFLT_CELL_RGBA = [0, 0, 0, 1]
        self._SHOW_BG_RGBA = [1, 1, 1, 1]
        self._VIS_BG_RGBA = [0, 0, 1, 1]
        self._DFLT_BG_RGBA = [0, 0, 0, 0]
        self._VIS_RELOC_RGBA = [1, 0, 0, 1]

        self._dwell_sg_steps = int(2 * self._action_sample_freq)  # 2 seconds
        self._dwell_bg_steps = int(1 * self._action_sample_freq)  # 1 second
        self._reloc_identification_steps = int(
            1 * self._action_sample_freq)  # 1 second
        # TODO In terms of mathematical representation, crowding could be modeled as a function of the distance between
        #  the intended focus word and its neighboring words, the size of the words, and the contrast between words and the background.
        #  However, the exact form of this function would depend on the specific details of the crowding phenomenon,
        #  which are still under investigation.
        # TODO: Characterize this way: the required identification time of a given word/cell is reversely proportional
        #  to the distance between the intended focus and its surrounding flankers [Visual Crowding].

        # Task mode
        self._task_mode = None

        # Layout
        self._sampled_layout_idx = None
        self._sampled_layout_sg_bg_mjidx_list = None
        self._sampled_layout_sg_mjidx_list = None

        # Attention switch flag on a certain trial
        self._attention_switch = None

        # Determine the radian of the visual spotlight for visual search, or 'neighbors'
        self._neighbour_radius = 0.0101  # Obtained empirically TODO make it more valid using text crowding and central vision models later
        self._neighbors_mjidxs_list = None  # The MuJoCo idxs of the neighbors of the sampled target idx
        self._neighbors_mjidxs_list_buffer = None

        # Initialise thresholds and counters
        self._steps = None
        self._focus_steps = None
        self._num_trials = None
        self._max_trials = 5
        self.ep_len = int(self._max_trials * self._dwell_sg_steps * 4)        # Make sure enough steps for tasks

        # Test-related variables - TODO delete later
        self._test_switch_back_duration_list = None
        self._test_switch_back_step = None
        self._test_switch_back_error_list = None

        # Define the observation space
        # TODO can we try 40*40 resolution to speed up trainings since we are not only relying on the vision channel?
        width, height = self._config['mj_env']['width'], self._config['mj_env']['height']
        self._num_stk_frm = 1
        self._num_stateful_info = 8
        self.observation_space = Dict({
            "vision": Box(low=-1, high=1, shape=(self._num_stk_frm, width, height)),
            "proprioception": Box(low=-1, high=1, shape=(self._num_stk_frm * self._model.nq + self._model.nu,)),
            "stateful information": Box(low=-1, high=1, shape=(self._num_stateful_info,)),
        })

        # Define the action space - 2 dof eyeball rotation control + decision to relocate or not
        self.action_space = Box(low=-1, high=1, shape=(self._model.nu + 1,))

        # Initialize the context and camera
        context = Context(self._model, max_resolution=[1280, 960])
        self._eye_cam = Camera(context, self._model, self._data, camera_id="eye", resolution=[width, height],
                               maxgeom=100,
                               dt=1 / self._action_sample_freq)
        self._env_cam = Camera(context, self._model, self._data, camera_id="env", maxgeom=100,
                               dt=1 / self._action_sample_freq)

    @staticmethod
    def normalise(x, x_min, x_max, a, b):
        # Normalise x (which is assumed to be in range [x_min, x_max]) to range [a, b]
        return (b - a) * ((x - x_min) / (x_max - x_min)) + a

    def render(self, mode="rgb_array"):
        rgb, _ = self._env_cam.render()
        rgb_eye, _ = self._eye_cam.render()
        return rgb, rgb_eye

    def _get_obs(self):
        """ Get the observation of the environment """
        # Get the vision observation
        # Render the image
        rgb, _ = self._eye_cam.render()

        # Preprocess - H*W*C -> C*W*H
        rgb = np.transpose(rgb, [2, 1, 0])
        rgb_normalize = self.normalise(rgb, 0, 255, -1, 1)

        # Convert the rgb to grayscale - boost the training speed
        gray_normalize = rgb_normalize[0:1, :, :] * 0.299 + rgb_normalize[1:2, :, :] * 0.587 + rgb_normalize[2:3, :,
                                                                                               :] * 0.114
        gray_normalize = np.squeeze(gray_normalize, axis=0)
        vision = gray_normalize.reshape((-1, gray_normalize.shape[-2], gray_normalize.shape[-1]))

        # Get the proprioception observation
        proprioception = np.concatenate([self._data.qpos, self._data.ctrl])

        # Get the stateful information observation - normalize to [-1, 1]
        remaining_ep_len_norm = (self.ep_len - self._steps) / self.ep_len * 2 - 1
        if self._task_mode == READ:
            dwell_cell_steps = self._dwell_sg_steps
        elif self._task_mode == BG:
            dwell_cell_steps = self._dwell_bg_steps
        elif self._task_mode == RELOC:
            dwell_cell_steps = self._reloc_identification_steps
        else:
            raise ValueError("Invalid task mode!")
        remaining_dwell_steps_norm = (dwell_cell_steps - self._focus_steps) / dwell_cell_steps * 2 - 1
        remaining_trials_norm = (self._max_trials - self._num_trials) / self._max_trials * 2 - 1
        current_task_mode_norm = self._task_mode
        # attention_switch_norm = 1 if self._attention_switch == True else -1
        current_focus_idx = np.where(self._sampled_layout_sg_bg_mjidx_list == self._sampled_intended_focus_mjidx)[0][0]
        current_focus_confidence_norm = self._target_confidence_distribution[current_focus_idx]
        layout_norm = self._sampled_layout_idx
        if self._sampled_layout_idx == ILS100:
            x_min = self._fixation_ils100_x_min
            x_max = self._fixation_ils100_x_max
            z_min = self._fixation_ils100_z_min
            z_max = self._fixation_ils100_z_max
        elif self._sampled_layout_idx == BC:
            x_min = self._fixation_bc_x_min
            x_max = self._fixation_bc_x_max
            z_min = self._fixation_bc_z_min
            z_max = self._fixation_bc_z_max
        elif self._sampled_layout_idx == MR:
            x_min = self._fixation_mr_x_min
            x_max = self._fixation_mr_x_max
            z_min = self._fixation_mr_z_min
            z_max = self._fixation_mr_z_max
        else:
            raise ValueError("Invalid layout index!")
        sampled_intended_focus_x = self._data.geom(self._sampled_intended_focus_mjidx).xpos[0]
        sampled_intended_focus_x_norm = self.normalise(sampled_intended_focus_x, x_min, x_max, -1, 1)
        sampled_intended_focus_z = self._data.geom(self._sampled_intended_focus_mjidx).xpos[2]
        sampled_intended_focus_z_norm = self.normalise(sampled_intended_focus_z, z_min, z_max, -1, 1)
        stateful_info = np.array(
            [remaining_ep_len_norm, remaining_dwell_steps_norm, remaining_trials_norm, layout_norm,
             sampled_intended_focus_x_norm, sampled_intended_focus_z_norm, current_task_mode_norm,
             current_focus_confidence_norm]
        )
        if stateful_info.shape[0] != self._num_stateful_info:
            raise ValueError("The shape of stateful information is not correct!")

        return {"vision": vision, "proprioception": proprioception, "stateful information": stateful_info}

    def reset(self):

        # Reset MuJoCo sim
        mujoco.mj_resetData(self._model, self._data)

        # Reset the variables and counters
        self._steps = 0
        self._num_trials = 0

        # Initialize eyeball rotation angles
        self._data.qpos[self._eye_joint_x_mjidx] = np.random.uniform(-0.5, 0.5)
        self._data.qpos[self._eye_joint_y_mjidx] = np.random.uniform(-0.5, 0.5)

        # Reset some test-related variables - TODO debug delete later
        self._test_switch_back_duration_list = []
        self._test_switch_back_step = self._steps
        self._test_switch_back_error_list = []

        # Initialize the layout - TODO to simplify the training, the layouts only refreshed once in one episode
        # self._sampled_layout_idx = np.random.choice([ILS100, BC])
        self._sampled_layout_idx = BC       # TODO debug and test delete later

        # Reset the scene - except the chosen layout, all the other layouts are hidden
        for mjidx in self._fixations_all_layouts_mjidxs:
            self._model.geom(mjidx).rgba[3] = 0
        if self._sampled_layout_idx == ILS100:
            self._sampled_layout_sg_bg_mjidx_list = self._fixations_ils100_mjidxs
            self._sampled_layout_sg_mjidx_list = self._ils100_cells_mjidxs
        elif self._sampled_layout_idx == BC:
            self._sampled_layout_sg_bg_mjidx_list = self._fixations_bc_mjidxs
            self._sampled_layout_sg_mjidx_list = self._bc_cells_mjidxs
        elif self._sampled_layout_idx == MR:
            self._sampled_layout_sg_bg_mjidx_list = self._fixations_mr_mjidxs
            self._sampled_layout_sg_mjidx_list = self._mr_cells_mjidxs
        else:
            raise ValueError("The layout index is not correct!")
        for mjidx in self._sampled_layout_sg_mjidx_list:
            self._model.geom(mjidx).rgba[3] = 1

        # Reset and initialize the target belief
        self._target_position_belief_distribution = np.zeros(self._sampled_layout_sg_bg_mjidx_list.shape[0])
        # Reset and initialize the target memory
        self._target_confidence_distribution = np.zeros(self._sampled_layout_sg_bg_mjidx_list.shape[0])

        # Reset the frequently updated variables
        self._num_elapsed_visual_searched_cells = 0

        # Initialize the task mode
        self._task_mode = READ

        # Initialize whether this trial includes the attention switch
        self._attention_switch = np.random.choice([True, False])

        # Get the true target mjidx
        self._get_true_target()

        # Sample an intended focus according to the belief probability distribution
        self._sample_intended_focus()

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

    @staticmethod
    def angle_between(v1, v2):
        # https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python/13849249#13849249
        def unit_vector(vec):
            return vec / np.linalg.norm(vec)

        v1_u = unit_vector(v1)
        v2_u = unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    def _angle_from_focus(self, site_name, target_idx):
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

    def _get_true_target(self):
        if self._task_mode == READ:
            self._true_target_mjidx = np.random.choice(self._sampled_layout_sg_mjidx_list.copy())
        elif self._task_mode == BG:
            self._buffer_true_target_mjidx = self._true_target_mjidx
            self._true_target_mjidx = self._bg_mjidx
        elif self._task_mode == RELOC:
            # Reset
            self._num_elapsed_visual_searched_cells = 0
            self._true_target_mjidx = self._buffer_true_target_mjidx

    def _sample_intended_focus(self, visual_search_in_progress=False):
        """
        It's like the transition function in the POMDP,
        both target confidence and target position belief are updated here.
        """
        # Update the target mjidx belief
        # Deterministic belief and confidence on the READ and BG task
        if self._task_mode == READ or self._task_mode == BG:
            # Reset the target idx probability distribution to all 0 in the reading/background mode
            self._target_position_belief_distribution = np.zeros(self._target_position_belief_distribution.shape[0])
            # Allocate the probability of the sampled target mjidx to be 1
            idx = np.where(self._sampled_layout_sg_bg_mjidx_list == self._true_target_mjidx)[0][0]
            self._target_position_belief_distribution[idx] = 1
            # Update the target mjidx memory - before time cost on the visual search, there is no memory decay, the agent remembers where the target was with 1 prob
            self._target_confidence_distribution = np.zeros(self._target_confidence_distribution.shape[0])
            idx = np.where(self._sampled_layout_sg_bg_mjidx_list == self._true_target_mjidx)[0][0]
            self._target_confidence_distribution[idx] = 1
        elif self._task_mode == RELOC:
            # Initializations - The visual search has not started yet
            # Determine the neighbours by identifying all cells that are within the preset neighbour radius
            # TODO use the crowd text model/central view/peripheral view
            if visual_search_in_progress == False:
                # Reset the target idx probability distribution to all 0 in the reading mode
                self._target_position_belief_distribution = np.zeros(self._target_position_belief_distribution.shape[0])
                self._neighbors_mjidxs_list = []
                self._neighbors_mjidxs_list_buffer = []
                center_xpos = self._data.geom(self._true_target_mjidx).xpos
                for mjidx in self._sampled_layout_sg_mjidx_list:
                    xpos = self._data.geom(mjidx).xpos
                    dist = np.linalg.norm(xpos - center_xpos)
                    if dist <= self._neighbour_radius:
                        self._neighbors_mjidxs_list.append(mjidx)
                # Store the initial neighbour list in the buffer - in case the agent never find a target and need to sample one
                self._neighbors_mjidxs_list_buffer = self._neighbors_mjidxs_list.copy()
                # Apply the even prob across all cells in the neighbour list -
                # TODO figure out a better way to do this - is it necessarily the even prob distribution? -
                #  Apply a Gaussian distribution based on the geometric distances of neighbours
                even_prob = float(1 / len(self._neighbors_mjidxs_list))
                for mjidx in self._neighbors_mjidxs_list:
                    idx = np.where(self._sampled_layout_sg_bg_mjidx_list == mjidx)[0][0]
                    self._target_position_belief_distribution[idx] = even_prob

                # Update the target mjidx memory - before time cost on the visual search, there is no memory decay, the agent remembers where the target was with 1 prob
                self._target_confidence_distribution = np.zeros(self._target_confidence_distribution.shape[0])
                idx = np.where(self._sampled_layout_sg_bg_mjidx_list == self._true_target_mjidx)[0][0]
                self._target_confidence_distribution[idx] = 1

            # Update the belief and confidence - the previous fixation mjidx is not None
            elif visual_search_in_progress == True:
                # Find the index of the sampled intended focus mjidx in the layout candidate list
                focused_idx = np.where(self._sampled_layout_sg_bg_mjidx_list == self._sampled_intended_focus_mjidx)[0][0]
                # Update the belief of the target mjidx - intended focus cell mjidx
                focused_cell_prob = self._target_position_belief_distribution[focused_idx]
                # Set 0 to the current fixation mjidx
                self._target_position_belief_distribution[focused_idx] = 0
                # Update the confidence based on the memory decay
                self._target_confidence_distribution[focused_idx] = 0
                # Remove it from the neighbors_mjidx
                self._neighbors_mjidxs_list.remove(self._sampled_intended_focus_mjidx)
                # Only update the belief and confidence if there are still neighbours left
                if len(self._neighbors_mjidxs_list) > 0:
                    leak_prob_per_neighbour = float(focused_cell_prob / (len(self._neighbors_mjidxs_list)))
                    for mjidx in self._neighbors_mjidxs_list:
                        index = np.where(self._sampled_layout_sg_bg_mjidx_list == mjidx)[0][0]
                        self._target_position_belief_distribution[index] += leak_prob_per_neighbour

                    # If not the true target idx discarded
                    if self._true_target_mjidx in self._neighbors_mjidxs_list:
                        # Memory decay update to the true target idx
                        true_target_idx = np.where(self._sampled_layout_sg_bg_mjidx_list == self._true_target_mjidx)[0][
                            0]
                        # TODO Advancement: use self._num_elapsed_visual_searched_cells * self._reloc_identification_steps to update the memory decay rate
                        true_target_confidence_after_memory_decay = 1 * np.exp(-self._memory_decay_rate * self._num_elapsed_visual_searched_cells)
                        self._target_confidence_distribution[true_target_idx] = true_target_confidence_after_memory_decay
                        # Update the rest certainty prob
                        for mjidx in self._neighbors_mjidxs_list:
                            if mjidx != self._true_target_mjidx:
                                index = np.where(self._sampled_layout_sg_bg_mjidx_list == mjidx)[0][0]
                                self._target_confidence_distribution[index] = float((1 - true_target_confidence_after_memory_decay) / (len(self._neighbors_mjidxs_list) - 1))
                    # If the true target idx was discarded
                    else:
                        # # True target memory update - only useful for the first time
                        # self._target_confidence_distribution[true_target_idx] = 0
                        # Update the rest cells' confidence
                        for mjidx in self._neighbors_mjidxs_list:
                            index = np.where(self._sampled_layout_sg_bg_mjidx_list == mjidx)[0][0]
                            self._target_confidence_distribution[index] = float(1 / (len(self._neighbors_mjidxs_list)))
        else:
            raise ValueError("The task mode is not correct!")

        # Sample a fixation mjidx from the cells according to the target mjidx probability distribution (belief)
        # If all probs are 0, then sample from the neighbors_mjidxs_list_buffer,
        #  and force to perceive this as the true target mjidx thus end the trial
        #  no need to focus on that cell anymore because it has been focused before
        finish_trial = False
        if np.sum(self._target_position_belief_distribution) == 0:
            self._sampled_intended_focus_mjidx = np.random.choice(self._neighbors_mjidxs_list_buffer)
            finish_trial = True
        else:
            # Make sure they sum to 1
            self._target_position_belief_distribution /= np.sum(self._target_position_belief_distribution)
            self._target_confidence_distribution = self._target_confidence_distribution / np.sum(
                self._target_confidence_distribution)
            self._sampled_intended_focus_mjidx = np.random.choice(self._sampled_layout_sg_bg_mjidx_list.copy(), p=self._target_position_belief_distribution)

        # Reset the counter
        self._focus_steps = 0

        return finish_trial

    def step(self, action):
        # Normalise 2 eyeball rotation actions from [-1, 1] to actuator control range
        action[0] = self.normalise(action[0], -1, 1, *self._model.actuator_ctrlrange[0, :])
        action[1] = self.normalise(action[1], -1, 1, *self._model.actuator_ctrlrange[1, :])

        # # TODO debug delete/comment this part later
        # mjidx = self._sampled_intended_focus_mjidx
        # xpos = self._data.geom(mjidx).xpos
        # x, y, z = xpos[0], xpos[1], xpos[2]
        # action = [z/y, -x/y, -0.5]

        # Set motor control
        self._data.ctrl[:] = action[0:2]

        # Get the agent's decision from the action - only useful for the relocation task
        # TODO try the stochastic policy later - the output action is a probability distribution of calling to take the target or not.
        #  This not necessarily makes harder or slower to train, but might beneficial on increasing the exploration
        focused_is_regarded_as_target_boolean = True if action[2] >= 0 else False

        # Advance the simulation
        mujoco.mj_step(self._model, self._data, self._frame_skip)
        self._steps += 1

        # Focus detection
        dist, geomid = self._get_focus(site_name="rangefinder-site")

        # Reset the scene appropriately
        for mjidx in self._sampled_layout_sg_bg_mjidx_list:
            self._model.geom(mjidx).rgba = self._DFLT_CELL_RGBA
        if self._task_mode == BG:
            self._model.geom(self._bg_mjidx).rgba = self._SHOW_BG_RGBA
        else:
            self._model.geom(self._bg_mjidx).rgba = self._DFLT_BG_RGBA

        # Estimate general reward
        reward = 0.1 * (np.exp(-10 * self._angle_from_focus(site_name="rangefinder-site",
                                                            target_idx=self._sampled_intended_focus_mjidx)) - 1)

        # Step with different task modes
        if self._task_mode == READ:
            if geomid == self._sampled_intended_focus_mjidx:
                self._focus_steps += 1
                self._model.geom(self._sampled_intended_focus_mjidx).rgba = self._VIS_CELL_RGBA

            if self._focus_steps >= self._dwell_sg_steps:
                # Update the milestone bonus reward for finish reading a cell
                reward = 10

                # Check whether to change the task mode
                # TODO to simplify, the attention switch always happen after a cell has been fixated for enough time
                if self._attention_switch == True:
                    # Show the background pane
                    self._model.geom(self._bg_mjidx).rgba = self._SHOW_BG_RGBA
                    self._task_mode = BG
                else:
                    # Update the number of trials
                    self._num_trials += 1
                    # Sample the next target and fixation
                    self._task_mode = READ
                    self._attention_switch = np.random.choice([True, False])
                # Get a new target and sample the new intended focus
                self._get_true_target()
                self._sample_intended_focus()

        elif self._task_mode == BG:
            if geomid == self._sampled_intended_focus_mjidx:
                self._focus_steps += 1
                self._model.geom(self._sampled_intended_focus_mjidx).rgba = self._VIS_BG_RGBA

            if self._focus_steps >= self._dwell_bg_steps:
                reward = 10
                # Hide the background pane
                self._model.geom(self._bg_mjidx).rgba = self._DFLT_BG_RGBA
                self._task_mode = RELOC
                self._test_switch_back_step = self._steps
                # Sample a new target mjidx in the RELOC mode
                self._get_true_target()
                self._sample_intended_focus()

        elif self._task_mode == RELOC:
            if geomid == self._sampled_intended_focus_mjidx:
                self._focus_steps += 1
                self._model.geom(self._sampled_intended_focus_mjidx).rgba = self._VIS_RELOC_RGBA

            if self._focus_steps >= self._reloc_identification_steps:
                # Update the fixation trials during the relocation visual search
                self._num_elapsed_visual_searched_cells += 1
                # Store the mjidx before being re-sampled
                previous_focused_mjidx_buffer = self._sampled_intended_focus_mjidx

                # # TODO debug delete it later
                # print(
                #     f"The true target is {self._true_target_mjidx}, "
                #     f"the sampled intented focus is {self._sampled_intended_focus_mjidx}, \n"
                #     f"The confidence distribution is {self._target_confidence_distribution}, \n"
                #     f"the target position belief is {self._target_position_belief_distribution},"
                # )

                # Re-sample the intended focus - TODO improve here, might exist better logic
                finish_trial = self._sample_intended_focus(visual_search_in_progress=True)

                # Perceive the focused cell is the target - or force to sample a target after searching all but had no luck
                if focused_is_regarded_as_target_boolean == True or finish_trial == True:

                    # Update some testing statistics TODO do this before the updates
                    self._test_switch_back_duration_list.append(self._steps - self._test_switch_back_step)
                    self._test_switch_back_error_list.append(
                        np.abs(previous_focused_mjidx_buffer - self._true_target_mjidx))

                    # # TODO debug delete later
                    # print(f"I was here, the focus boolean is {focused_is_regarded_as_target_boolean}, "
                    #       f"The determination action is {action[2]}, "
                    #       f"the finish trial is {finish_trial}")

                    # Only reward the agent if the previous focused item is the same as the true target
                    # TODO some bug with the previous_focused_mjidx_buffer here, if finish_trial is True, then it should be the new sampled intended focus
                    if previous_focused_mjidx_buffer == self._true_target_mjidx:
                        reward = 25
                    else:
                        # Punish the agent if he makes the incorrect relocations - provide incentives to make wiser decisions
                        reward = -25

                        # # TODO debug delete later
                        # print(f"I was also here, the sampled intended focus is {focused_mjidx_buffer}, "
                        #       f"the true target is {self._true_target_mjidx}")

                    # Reset the counter
                    self._num_elapsed_visual_searched_cells = 0
                    # Update the number of trials
                    self._num_trials += 1
                    # Sample another target for the reading task
                    self._task_mode = READ
                    self._attention_switch = np.random.choice([True, False])
                    self._get_true_target()
                    self._sample_intended_focus()
        else:
            raise ValueError("The task mode is not correct!")

        # Get termination condition
        terminate = False
        if self._steps >= self.ep_len or self._num_trials >= self._max_trials:
            terminate = True
            # TODO print the switch back error rate and the switch back durations - testify whether such modelings make sense
            if self._config["rl"]["mode"] == "test" or self._config["rl"]["mode"] == "debug":
                if len(self._test_switch_back_duration_list) == 0 or len(self._test_switch_back_error_list) == 0:
                    print("No switch back trials!")
                else:
                    print(f"The switch back avg duration is {np.mean(self._test_switch_back_duration_list)}, "
                          f"The switch back avg error is {np.mean(self._test_switch_back_error_list)}")

        # Update the scene to reflect the transition function
        mujoco.mj_forward(self._model, self._data)

        return self._get_obs(), reward, terminate, {}


class RelocationMemory(Env):

    def __init__(self):
        """
        This is a trial model for the relocation task, which is a simplified version without visual focus trainings.
        The only action here is whether the agent takes the current focus as the target or not.
        """
        # Get directory of this file
        directory = os.path.dirname(os.path.realpath(__file__))

        # Read the configurations from the YAML file.
        root_dir = os.path.dirname(os.path.dirname(directory))
        with open(os.path.join(root_dir, "config.yaml")) as f:
            self._config = yaml.load(f, Loader=yaml.FullLoader)

        # Load the MuJoCo model
        self._model = mujoco.MjModel.from_xml_path(os.path.join(directory, "relocation-v4.xml"))
        self._data = mujoco.MjData(self._model)
        mujoco.mj_forward(self._model, self._data)

        # Define how often policy is queried
        self._action_sample_freq = 20
        self._frame_skip = int((1 / self._action_sample_freq) / self._model.opt.timestep)

        # Get the primitive idx in MuJoCo
        self._eye_joint_x_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "eye-joint-x")
        self._eye_joint_y_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "eye-joint-y")
        self._head_body_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "head")
        self._eye_body_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "eye")
        self._sgp_ils100_body_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY,
                                                        "smart-glass-pane-interline-spacing-100")
        self._sgp_bc_body_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "smart-glass-pane-bottom-center")

        # Get MuJoCo cell idxs (geoms that belong to "smart-glass-pane-interline-spacing-100")
        self._ils100_cells_mjidxs = np.where(self._model.geom_bodyid == self._sgp_ils100_body_mjidx)[0]
        self._bc_cells_mjidxs = np.where(self._model.geom_bodyid == self._sgp_bc_body_mjidx)[0]
        self._fixations_all_layouts_mjidxs = np.concatenate((self._ils100_cells_mjidxs, self._bc_cells_mjidxs))

        # Define the target idx probability distribution -
        self._true_target_mjidx = None  # The true target MuJoCo idx
        self._buffer_true_target_mjidx = None  # The buffer for the true target MuJoCo idx
        self._target_confidence_distribution = None  # The perceived target MuJoCo idx, should be sampled from memory, which decays as time goes by, but the element should be the same as neighboring elements and the target belief
        self._memory_decay_rate = 0.2  # The memory decay rate - not useful now, I used a std-increasing Gaussian to model the memory decay
        self._num_visual_search = None  # The number of visual search trials elapsed
        self._max_visual_search = None  # The maximum number of visual search trials
        self._target_position_belief_distribution = None  # 'Belief': The dynamic target MuJoCo idx probability distribution
        self._sampled_intended_focus_mjidx = None  # The fixation MuJoCo idx, should be sampled from belief

        self._BLACK = [0, 0, 0, 1]
        self._YELLOW = [1, 1, 0, 1]
        self._RED = [1, 0, 0, 1]

        # To avoid the delayed decision-making problem, I set the relocation identification steps as 1
        self._reloc_identification_steps = 1 # int(0.25*self._action_sample_freq)  # The number of steps to identify the relocation

        # Layout
        self._sampled_layout_idx = None
        self._sampled_layout_sg_bg_mjidx_list = None
        self._sampled_layout_sg_mjidx_list = None

        self._last_action = None

        # Determine the radian of the visual spotlight for visual search, or 'neighbors'
        # TODO hyper-parameters, might need to fit to human data - maybe link to the central vision and peripheral vision?
        self._neighbour_radius = 0.01  # Obtained empirically
        self._position_belief_std_sparse = float(self._neighbour_radius*1)
        self._position_belief_std_condense = float(self._neighbour_radius*1)
        self._position_belief_std = None
        self._initial_confidence_std = float(self._neighbour_radius/2)  # The initial confidence std
        self._max_confidence_std = float(self._neighbour_radius * 3)  # The max confidence std
        self._visual_searched_mjidx_list = None  # The MuJoCo idxs of the cells that have been visual searched

        # Initialise thresholds and counters
        self._steps = None
        self._focus_steps = None
        self._num_trials = None
        self._max_trials = 5
        self.ep_len = int(self._max_trials * self._reloc_identification_steps * 100)

        # Test-related variables
        self._test_switch_back_duration_list = None
        self._test_switch_back_step = None
        self._test_switch_back_error_list = None

        # Define the observation space
        width, height = 10, 10  # Use whatever resolution as small as you want
        self._num_stk_frm = 1
        self._num_stateful_info = 5
        self.observation_space = Box(low=-1, high=1, shape=(self._num_stateful_info,))

        # Define the action space - decision to relocate or not
        self.action_space = Box(low=-1, high=1, shape=(1,))

        # Initialize the context and camera
        context = Context(self._model, max_resolution=[1280, 960])
        self._eye_cam = Camera(context, self._model, self._data, camera_id="eye", resolution=[width, height],
                               maxgeom=1000,
                               dt=1 / self._action_sample_freq)
        self._env_cam = Camera(context, self._model, self._data, camera_id="env", maxgeom=1000,
                               dt=1 / self._action_sample_freq)

    @staticmethod
    def normalise(x, x_min, x_max, a, b):
        # Normalise x (which is assumed to be in range [x_min, x_max]) to range [a, b]
        return (b - a) * ((x - x_min) / (x_max - x_min)) + a

    def render(self, mode="rgb_array"):
        rgb, _ = self._env_cam.render()
        rgb_eye, _ = self._eye_cam.render()
        return rgb, rgb_eye

    def _get_obs(self):
        """ Get the observation of the environment """
        # Get the stateful information observation - normalize to [-1, 1]
        remaining_ep_len_norm = self.normalise((self.ep_len - self._steps), 0, self.ep_len, -1, 1)
        remaining_trials_norm = self.normalise((self._max_trials - self._num_trials), 0, self._max_trials, -1, 1)
        layout_norm = self._sampled_layout_idx
        # Normalize the current focus
        focus_idx = np.where(self._sampled_layout_sg_mjidx_list == self._sampled_intended_focus_mjidx)[0][0]
        focus_confidence_norm = self.normalise(self._target_confidence_distribution[focus_idx], 0, 1, -1, 1)
        # Normalize the number of searched items/cells/words
        searched_cells_norm = self.normalise(len(self._visual_searched_mjidx_list), 0, len(self._sampled_layout_sg_mjidx_list), -1, 1)

        stateful_info = np.array([remaining_ep_len_norm, remaining_trials_norm, layout_norm, focus_confidence_norm,
                                  searched_cells_norm])

        if stateful_info.shape[0] != self._num_stateful_info:
            raise ValueError(f"The shape of stateful information is not correct! The true shape is: {stateful_info.shape[0]}")

        # if self._config["rl"]["mode"] == "debug":
        #     print(
        #         f"The current focus confidence is {self._target_confidence_distribution[np.where(self._sampled_layout_sg_mjidx_list == self._sampled_intended_focus_mjidx)[0][0]]}, "
        #         f"the true target is: {self._true_target_mjidx}, "
        #         f"the sampled intended focus is: {self._sampled_intended_focus_mjidx}"
        #         f"\nStateful_info: {stateful_info}")

        return stateful_info

    def reset(self):

        # Reset MuJoCo sim
        mujoco.mj_resetData(self._model, self._data)

        # Reset the variables and counters
        self._steps = 0
        self._num_trials = 0

        # Reset some test-related variables
        self._test_switch_back_duration_list = []
        self._test_switch_back_step = self._steps
        self._test_switch_back_error_list = []

        # Initialize the layout
        self._sampled_layout_idx = np.random.choice([ILS100, BC])

        if self._config["rl"]["mode"] == "debug" or self._config["rl"]["mode"] == "test":
            self._sampled_layout_idx = ILS100
            print(f"NOTE, the current layout is: {self._sampled_layout_idx}")

        # Reset the scene - except the chosen layout, all the other layouts are hidden
        for mjidx in self._fixations_all_layouts_mjidxs:
            self._model.geom(mjidx).rgba[3] = 0
        if self._sampled_layout_idx == ILS100:
            self._sampled_layout_sg_mjidx_list = self._ils100_cells_mjidxs
            self._position_belief_std = self._position_belief_std_sparse
        elif self._sampled_layout_idx == BC:
            self._sampled_layout_sg_mjidx_list = self._bc_cells_mjidxs
            self._position_belief_std = self._position_belief_std_condense
        else:
            raise ValueError("The layout index is not correct!")

        for mjidx in self._sampled_layout_sg_mjidx_list:
            self._model.geom(mjidx).rgba = self._BLACK

        self._max_visual_search = self._sampled_layout_sg_mjidx_list.shape[0]

        # Reset and initialize the target belief
        self._target_position_belief_distribution = np.zeros(self._sampled_layout_sg_mjidx_list.shape[0])
        # Reset and initialize the target memory
        self._target_confidence_distribution = np.zeros(self._sampled_layout_sg_mjidx_list.shape[0])

        # Reset the frequently updated variables
        self._num_visual_search = 0

        # Get the true target mjidx
        self._get_true_target()

        # Sample an intended focus according to the belief probability distribution
        self._sample_intended_focus()

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

    @staticmethod
    def angle_between(v1, v2):
        # https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python/13849249#13849249
        def unit_vector(vec):
            return vec / np.linalg.norm(vec)

        v1_u = unit_vector(v1)
        v2_u = unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    def _angle_from_focus(self, site_name="rangefinder-site", target_idx=None):
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

    def _get_true_target(self):
        # Reset
        self._num_visual_search = 0
        # Randomly sample a target mjidx
        self._true_target_mjidx = np.random.choice(self._sampled_layout_sg_mjidx_list.copy())

    def _init_distributions(self):
        # Reset the neighobours mjidxs list and the visual searched mjidx list
        self._visual_searched_mjidx_list = []

        # Reset the target idx probability distribution to all 0 in the reading mode
        mujoco.mj_forward(self._model, self._data)
        self._target_position_belief_distribution = np.zeros(self._target_position_belief_distribution.shape[0])
        center_xpos = self._data.geom(self._true_target_mjidx).xpos
        for mjidx in self._sampled_layout_sg_mjidx_list:
            xpos = self._data.geom(mjidx).xpos
            dist = np.linalg.norm(xpos - center_xpos)
            # Initialize the position belief distribution - Calculate the probability using the Gaussian distribution - the pre-defined radius is std
            idx = np.where(self._sampled_layout_sg_mjidx_list == mjidx)[0][0]
            self._target_position_belief_distribution[idx] = np.exp(-0.5 * (dist / self._position_belief_std) ** 2)
            # Initialize the confidence distribution - use the Gaussian distribution - use a changing std
            self._target_confidence_distribution[idx] = np.exp(-0.5 * (dist / self._initial_confidence_std) ** 2)

        # Normalize distributions
        self._target_position_belief_distribution /= np.sum(self._target_position_belief_distribution)
        self._target_confidence_distribution /= np.sum(self._target_confidence_distribution)

    def _update_distributions(self):
        # Find the index of the sampled intended focus mjidx in the layout candidate list
        focused_idx = np.where(self._sampled_layout_sg_mjidx_list == self._sampled_intended_focus_mjidx)[0][0]
        # Update the position belief distribution - Set 0 to the current fixation mjidx
        self._target_position_belief_distribution[focused_idx] = 0

        # Update the confidence distribution
        updated_std = self._initial_confidence_std + (self._max_confidence_std - self._initial_confidence_std) * (self._num_visual_search / self._max_visual_search)
        center_xpos = self._data.geom(self._true_target_mjidx).xpos
        for mjidx in self._sampled_layout_sg_mjidx_list:
            xpos = self._data.geom(mjidx).xpos
            dist = np.linalg.norm(xpos - center_xpos)
            # Initialize the position belief distribution - Calculate the probability using the Gaussian distribution - the pre-defined radius is std
            idx = np.where(self._sampled_layout_sg_mjidx_list == mjidx)[0][0]
            # Initialize the confidence distribution - use the Gaussian distribution - use a changing std
            self._target_confidence_distribution[idx] = np.exp(-0.5 * (dist / updated_std) ** 2)
        # Set 0 to those cells that are already visual searched
        for mjidx in self._visual_searched_mjidx_list:
            idx = np.where(self._sampled_layout_sg_mjidx_list == mjidx)[0][0]
            self._target_confidence_distribution[idx] = 0

        # Normalize distributions
        if np.sum(self._target_position_belief_distribution) != 0 and np.sum(self._target_confidence_distribution) != 0:
            self._target_position_belief_distribution /= np.sum(self._target_position_belief_distribution)
            self._target_confidence_distribution /= np.sum(self._target_confidence_distribution)

    def _sample_intended_focus(self, visual_search_in_progress=False):
        """
        It's like the transition function in the POMDP,
        both target confidence and target position belief are updated here.
        """

        if self._config["rl"]["mode"] == "debug": # or self._config["rl"]["mode"] == "test":
            self._print_logs()

        # Initializations - The visual search has not started yet
        if visual_search_in_progress == False:
            self._init_distributions()
        # Update the belief and confidence - the previous fixation mjidx is not None
        else:
            self._update_distributions()

        # Randomly sample a target if the agent has visual searched all the cells but still not found the target
        last_target = None
        if np.sum(self._target_position_belief_distribution) == 0:
            # self._init_distributions()
            # last_target = np.random.choice(self._sampled_layout_sg_mjidx_list, p=self._target_position_belief_distribution)
            last_target = self._visual_searched_mjidx_list[-1]
            if len(self._visual_searched_mjidx_list) < len(self._sampled_layout_sg_mjidx_list):
                raise ValueError("The agent has not visual searched all the cells but still not found the target")

            if self._config["rl"]["mode"] == "debug":
                print(f"The random search applied, the randomly sampled target is {last_target} "
                      f"with the true target mjidx {self._true_target_mjidx}")

        else:
            self._sampled_intended_focus_mjidx = np.random.choice(self._sampled_layout_sg_mjidx_list.copy(), p=self._target_position_belief_distribution)

        # Reset the counter
        self._focus_steps = 0

        if self._config["rl"]["mode"] == "debug": # or self._config["rl"]["mode"] == "test":
            self._print_logs()

        return last_target

    def _reset_trial(self):
        # Update some testing statistics
        self._test_switch_back_duration_list.append(self._steps - self._test_switch_back_step)
        self._test_switch_back_error_list.append(
            np.abs(self._sampled_intended_focus_mjidx - self._true_target_mjidx))

        # Update some variables
        self._num_visual_search = 0
        self._visual_searched_mjidx_list = []
        self._test_switch_back_step = self._steps
        # Update the number of trials
        self._num_trials += 1
        # Sample the new target and the new intended focus for the next trial
        self._get_true_target()
        self._sample_intended_focus()

    def step(self, action):
        # Manually force the agent to focus on the intended focus mjidx
        mjidx = self._sampled_intended_focus_mjidx
        xpos = self._data.geom(mjidx).xpos
        x, y, z = xpos[0], xpos[1], xpos[2]
        intended_position = np.array([np.arctan(z/y), np.arctan(-x/y)])

        self._data.qpos[:] = intended_position
        mujoco.mj_forward(self._model, self._data)
        self._steps += 1

        # Stochastic actions - the probability is learnt
        # confidence = self.normalise(action[0], -1, 1, 0, 1)
        # focus_is_regarded_as_target_boolean = np.random.choice([True, False], p=[confidence, 1-confidence])
        # self._last_action = np.array([action[0], confidence, focus_is_regarded_as_target_boolean])

        # Deterministic actions - 0 or 1 is learnt
        focus_is_regarded_as_target_boolean = True if action[0] > 0 else False
        self._last_action = np.array([action[0], action[0], focus_is_regarded_as_target_boolean])

        if self._config["rl"]["mode"] == "debug" or self._config["rl"]["mode"] == "test":
            self._print_logs()

        # Focus detection
        dist, geomid = self._get_focus(site_name="rangefinder-site")

        # Reset the scene appropriately
        for mjidx in self._sampled_layout_sg_mjidx_list:
            self._model.geom(mjidx).rgba = self._BLACK
        self._model.geom(self._true_target_mjidx).rgba = self._YELLOW

        correct_relocation_bonus = 100
        decision_making_time_cost = -10
        reward = decision_making_time_cost

        if geomid == self._sampled_intended_focus_mjidx:
            self._focus_steps += 1
            self._model.geom(self._sampled_intended_focus_mjidx).rgba = self._RED

        if self._focus_steps >= self._reloc_identification_steps:
            # Update the fixation trials during the relocation visual search
            self._num_visual_search += 1
            self._visual_searched_mjidx_list.append(self._sampled_intended_focus_mjidx)

            if focus_is_regarded_as_target_boolean:
                # Update the rewards
                if self._sampled_intended_focus_mjidx == self._true_target_mjidx:
                    reward = correct_relocation_bonus
                # Update some variables
                self._reset_trial()
            else:
                # Keep searching
                last_target = self._sample_intended_focus(visual_search_in_progress=True)
                if last_target is not None:
                    # The last target is given if all the cells have been searched
                    if last_target == self._true_target_mjidx:
                        reward = correct_relocation_bonus
                    # Update some variables
                    self._reset_trial()

        # Get termination condition
        terminate = False
        if self._steps >= self.ep_len or self._num_trials >= self._max_trials:
            terminate = True

            # Print some testing statistics if in test mode or debug mode
            if self._config["rl"]["mode"] == "test" or self._config["rl"]["mode"] == "debug":
                print(f"The total steps is {self._steps}")
                if len(self._test_switch_back_duration_list) == 0 or len(self._test_switch_back_error_list) == 0:
                    print("No switch back trials!")
                else:
                    print(f"The switch back avg duration is {np.mean(self._test_switch_back_duration_list)}, "
                          f"The switch back avg error is {np.mean(self._test_switch_back_error_list)}")

        # Update the scene to reflect the transition function
        mujoco.mj_forward(self._model, self._data)

        return self._get_obs(), reward, terminate, {}

    def _print_logs(self):
        # print(
        #     f"\nThe current trial is: {self._num_trials}"
        #     f"   The CURRENT sampled intended focus mjidx is {self._sampled_intended_focus_mjidx}, the true target mjidx is {self._true_target_mjidx}"
        #     f"\nThe belief distribution is {self._target_position_belief_distribution}, "
        #     f"\nthe confidence distribution is {self._target_confidence_distribution}"
        #     f"\nThe LAST action tuple is {self._last_action}"
        #     f"\n--------------------------------------------------------------------------------------"
        # )
        pass
