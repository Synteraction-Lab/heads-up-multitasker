import math

import numpy as np
import mujoco
import os

from gym import Env
from gym.spaces import Box

from huc.utils.rendering import Camera, Context

import yaml


class ContextSwitchReplication(Env):

    def __init__(self):

        # Get directory of this file
        directory = os.path.dirname(os.path.realpath(__file__))

        # Read the configurations from the YAML file.
        root_dir = os.path.dirname(os.path.dirname(directory))
        with open(os.path.join(root_dir, "config.yaml")) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        self._mode = config['rl']['mode']

        # Open the mujoco model
        self._model = mujoco.MjModel.from_xml_path(os.path.join(directory,
                                                                "context-switch-12-inter-line-spacing-50-v1.xml"))
        self._data = mujoco.MjData(self._model)

        # Define how often policy is queried
        self._action_sample_freq = 20
        self._frame_skip = int((1 / self._action_sample_freq) / self._model.opt.timestep)  # 0.05/0.002=25

        # Initialise thresholds and counters
        self._target_switch_interval = 2 * self._action_sample_freq
        self._steps = 0
        self._max_trials = 1
        self._trials = 0

        # Get targets (geoms that belong to "smart-glass-pane")  TODO read how to handle nodes under nodes.
        sgp_body = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "smart-glass-pane")
        self._target_idxs = np.where(self._model.geom_bodyid == sgp_body)[0]
        self._targets = [self._model.geom(idx) for idx in self._target_idxs]
        self._target_idx = None
        self._target = None

        # Get the background (geoms that belong to "background-pane")
        bp_body = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "background-pane")
        self._background_idxs = np.where(self._model.geom_bodyid == bp_body)[0]
        self._background_idx0 = self._background_idxs[0]

        # Define the default text grid size and rgba from a sample grid idx=0
        self._DEFAULT_TEXT_SIZE = self._model.geom(self._target_idxs[0]).size[0:4].copy()
        self._DEFAULT_TEXT_RGBA = self._model.geom(self._target_idxs[0]).rgba[0:4].copy()
        self._HINT_SIZE = [self._DEFAULT_TEXT_SIZE[0]*4/3, self._DEFAULT_TEXT_SIZE[1], self._DEFAULT_TEXT_SIZE[2]*4/3]
        self._HINT_RGBA = [0.8, 0.8, 0, 1]
        self._DEFAULT_BACKGROUND_SIZE = self._model.geom(self._background_idx0).size[0:4].copy()
        self._DEFAULT_BACKGROUND_RGBA = self._model.geom(self._background_idx0).rgba[0:4].copy()
        self._EVENT_RGBA = [0.8, 0, 0, self._model.geom(self._background_idx0).rgba[3].copy()]

        # Define the idx of grids which needs to be traversed sequentially on the smart glass pane
        self._sequence_target_idxs = []
        # The reading result buffer - should has the same length
        self._sequence_results_idxs = []
        self._default_idx = -1
        self._num_targets = 0
        self._b_change = 0.2

        if (self._mode == 'train') or (self._mode == 'continual_train'):
            self._ep_len = 80        # TODO train mode
        else:
            self._ep_len = 800          # TODO test mode

        # Define the grids on the background pane
        self._background_on = False
        self._steps_background_on = 0
        self._steps_background_off = 0
        self._background_on_interval = 1 * self._action_sample_freq
        self._background_off_interval = 4 * self._action_sample_freq

        # Initialize the previous distance buffer for reward shaping
        self._pre_dist_to_target = 0

        # Define observation space
        self._width = 40
        self._height = 40
        self.observation_space = Box(low=0, high=255, shape=(3, self._width, self._height))  # width, height correctly set?

        # Define action space
        self.action_space = Box(low=-1, high=1, shape=(2,))

        # Define a cutoff for rangefinder (in meters, could be something like 3 instead of 0.1)
        self._rangefinder_cutoff = 0.1

        # Initialise context, cameras
        self._context = Context(self._model, max_resolution=[1280, 960])
        self._eye_cam = Camera(self._context, self._model, self._data, camera_id="eye",
                               resolution=[self._width, self._height], maxgeom=100,
                               dt=1 / self._action_sample_freq)
        self._env_cam = Camera(self._context, self._model, self._data, camera_id="env", maxgeom=100,
                               dt=1 / self._action_sample_freq)

    def _get_obs(self):

        # Render the image
        rgb, _ = self._eye_cam.render()
        # Preprocess
        rgb = np.transpose(rgb, [2, 0, 1])
        return self.normalise(rgb, 0, 255, -1, 1)

    def reset(self):

        # Reset mujoco sim
        mujoco.mj_resetData(self._model, self._data)

        # Reset counters
        self._steps = 0
        self._trials = 0

        # Reset the scene
        self._reset_scene()

        return self._get_obs()

    def _reset_scene(self):

        # Initialize eye ball rotation angles      # TODO change according to the settings
        if (self._mode == 'train') or (self._mode == 'continual_train'):
            eye_x_motor_init_range = [-0.5, 0.5]
            eye_y_motor_init_range = [-0.25, 0.25]
            action = [np.random.uniform(eye_x_motor_init_range[0], eye_x_motor_init_range[1]),
                      np.random.uniform(eye_y_motor_init_range[0], eye_y_motor_init_range[1])]
            # TODO try to use data.xmat directly set orientations. The model.quat should not be changed.
            for i in range(10):
                # Set motor control
                self._data.ctrl[:] = action
                # Advance the simulation
                mujoco.mj_step(self._model, self._data, self._frame_skip)

        # Reset the scene.
        for idx in self._target_idxs:
            self._model.geom(idx).rgba[0:4] = self._DEFAULT_TEXT_RGBA.copy()
            self._model.geom(idx).size[0:3] = self._DEFAULT_TEXT_SIZE.copy()

        if (self._mode == 'train') or (self._mode == 'continual_train'):
            self._sequence_target_idxs = np.random.choice(self._target_idxs.tolist(), 3, False)
        else:
            self._sequence_target_idxs = self._target_idxs.tolist()
        self._num_targets = len(self._sequence_target_idxs)
        # ------------------------------------------------------------------------------------------

        self._sequence_results_idxs = [self._default_idx for _ in self._sequence_target_idxs]

        self._switch_target(idx=self._sequence_target_idxs[0])

        # Initialize the previous distance buffer with a random generated initial action
        dist_to_target = self._dist_from_target(
            dist_plane=(-self._data.body("smart-glass-pane").xpos[1]),
            ray_site_name="rangefinder-site",
            target_id=self._target_idx
        )
        self._pre_dist_to_target = dist_to_target

    def _switch_target(self, idx):

        for _idx in self._target_idxs:
            if _idx != idx:
                self._model.geom(_idx).rgba[0:4] = self._DEFAULT_TEXT_RGBA.copy()
                self._model.geom(_idx).size[0:3] = self._DEFAULT_TEXT_SIZE.copy()

        self._model.geom(idx).rgba[0:4] = self._HINT_RGBA.copy()
        self._model.geom(idx).size[0:3] = self._HINT_SIZE.copy()

        # Update the target id
        self._target_idx = idx

        # Do a forward so everything will be set
        mujoco.mj_forward(self._model, self._data)

    def _update_background(self):

        # Update the steps of background on
        if self._background_on:
            self._steps_background_on += 1
            # Close the background change
            if self._steps_background_on >= self._background_on_interval:
                self._model.geom(self._background_idx0).rgba[0:4] = self._DEFAULT_BACKGROUND_RGBA.copy()
                self._model.geom(self._background_idx0).size[0:3] = self._DEFAULT_BACKGROUND_SIZE.copy()
                self._steps_background_off = 0
                self._background_on = False

        # Start deterministic background change
        if self._background_on is False:
            self._steps_background_off += 1
            # Start the background change
            if self._steps_background_off >= self._background_off_interval:
                self._model.geom(self._background_idx0).rgba[0:4] = self._EVENT_RGBA.copy()
                self._steps_background_on = 0
                self._background_on = True

        # Do a forward so everything will be set
        mujoco.mj_forward(self._model, self._data)

    def _ray_from_site(self, site_name):
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

    def _dist_from_target(self, dist_plane, ray_site_name, target_id):
        ray_site = self._data.site(ray_site_name)
        pnt = ray_site.xpos
        vec = ray_site.xmat.reshape((3, 3))[:, 2]

        # Define the x-z plane equation
        a, b, c = 0, 1, 0  # Normalize the vector of the x-z plane
        # dist_plane = - self._data.body("smart-glass-pane").xpos[1]  # Distance from origin to plane
        # Calculate the intersection point
        t = - (a * pnt[0] + b * pnt[1] + c * pnt[2] + dist_plane) / (a * vec[0] + b * vec[1] + c * vec[2])
        itsct_pnt = pnt + t * vec
        # Get the target point
        target_pnt = self._data.geom(target_id).xpos
        # Calculate the distance
        dist = math.sqrt((itsct_pnt[0]-target_pnt[0])**2 + (itsct_pnt[2]-target_pnt[2])**2)

        return dist

    def render(self, mode="rgb_array"):
        rgb, _ = self._env_cam.render()
        rgb_eye, _ = self._eye_cam.render()
        return rgb, rgb_eye

    @staticmethod
    def normalise(x, x_min, x_max, a, b):
        # Normalise x (which is assumed to be in range [x_min, x_max]) to range [a, b]
        return (b - a) * ((x - x_min) / (x_max - x_min)) + a

    def step(self, action):
        # Normalise action from [-1, 1] to actuator control range
        action[0] = self.normalise(action[0], -1, 1, *self._model.actuator_ctrlrange[0, :])
        action[1] = self.normalise(action[1], -1, 1, *self._model.actuator_ctrlrange[1, :])

        # Set motor control
        self._data.ctrl[:] = action

        # # Update the background changes
        # self._update_background()

        # Advance the simulation
        mujoco.mj_step(self._model, self._data, self._frame_skip)
        self._steps += 1

        dist, geomid = self._ray_from_site(site_name="rangefinder-site")

        # Estimate reward according to the distance change
        dist_to_target = self._dist_from_target(
            dist_plane=(-self._data.body("smart-glass-pane").xpos[1]),
            ray_site_name="rangefinder-site",
            target_id=self._target_idx
        )
        # Reward shaping according to the distance change, # TODO reward shaping
        # as long as the focus moves towards the target, positive reward will be applied
        a = 50
        b = 100
        del_dist_pct = (dist_to_target - self._pre_dist_to_target) / self._pre_dist_to_target
        if del_dist_pct >= 0:
            reward = (math.exp(-a * del_dist_pct) - 1) / b
        else:
            reward = - (math.exp(-a * np.abs(del_dist_pct)) - 1) / b

        # Update the previous distance buffer
        self._pre_dist_to_target = dist_to_target

        # TODO sparse rewards
        reward = 0

        # Estimate reward according to the collisions
        if geomid == self._target_idx:
            reward = 1
            # Update the environment
            acc = self._b_change / self._target_switch_interval
            self._model.geom(geomid).rgba[0:3] = [x + y for x, y in
                                                  zip(self._model.geom(geomid).rgba[0:3], [0, 0, acc])]
            # Do a forward so everything will be set
            mujoco.mj_forward(self._model, self._data)

        if self._steps >= self._ep_len:
            terminate = True
        else:
            terminate = False

            # Check whether the grid has been fixated for enough time
            if (self._model.geom(self._target_idx).rgba[2] >= self._b_change) and (self._target_idx not in self._sequence_results_idxs):
                self._trials += 1
                # Update the results
                for i in range(self._num_targets):
                    if self._sequence_results_idxs[i] == self._default_idx:
                        self._sequence_results_idxs[i] = self._target_idx
                        break

                # Under the training mode, just try once
                if (self._mode == 'train') or (self._mode == 'continual_train'):
                    if self._trials >= self._max_trials:
                        terminate = True

                # Switch a new target grid, check if one loop has finished
                if self._target_idx >= self._sequence_target_idxs[-1]:
                    # self._reset_scene()
                    terminate = True
                else:
                    self._switch_target(idx=self._target_idx+1)

        return self._get_obs(), reward, terminate, {}