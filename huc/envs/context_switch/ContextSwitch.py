import numpy as np
import mujoco
import os

from gym import Env
from gym.spaces import Box

from huc.utils.rendering import Camera, Context


class ContextSwitch(Env):

    def __init__(self):

        # Get directory of this file
        directory = os.path.dirname(os.path.realpath(__file__))

        # Open the mujoco model
        self._model = mujoco.MjModel.from_xml_path(os.path.join(directory, "context-switch.xml"))
        self._data = mujoco.MjData(self._model)

        # Define how often policy is queried
        self._action_sample_freq = 20
        self._frame_skip = int((1 / self._action_sample_freq) / self._model.opt.timestep)  # 0.05/0.002=25

        # Initialise thresholds and counters
        self._target_switch_interval = 2 * self._action_sample_freq
        self._steps = 0
        self._max_trials = 5
        self._trial_idx = 0

        # Get targets (geoms that belong to "smart-glass-pane")
        sgp_body = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "smart-glass-pane")
        self._target_idxs = np.where(self._model.geom_bodyid == sgp_body)[0]
        self._targets = [self._model.geom(idx) for idx in self._target_idxs]
        self._target_idx = None
        self._target = None

        # Get the background (geoms that belong to "background-pane")
        bp_body = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "background-pane")
        self._background_idxs = np.where(self._model.geom_bodyid == bp_body)[0]
        self._background_idx0 = self._background_idxs[0]

        # The smartglass pane: Determine the idx of grids which needs to be traversed in some sequence
        self._sequence_target_idxs = []
        # The reading result buffer - should has the same length
        self._sequence_results_idxs = []
        self._default_idx = -1
        self._num_targets = 0
        self._b_change = 0.2
        self._ep_len = 400

        # The background grids:
        self._background_on = False
        self._steps_background_on = 0
        self._steps_background_off = 0
        self._background_on_interval = 1 * self._action_sample_freq
        self._background_off_interval = 2 * self._action_sample_freq

        # Define observation space
        self._width = 80
        self._height = 80
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

        # Reset the scene
        self._reset_scene()

        return self._get_obs()

    # TODO there are some changable variables in this protected method, change it later
    def _reset_scene(self):

        # First reset the scene.
        for idx in self._target_idxs:
            self._model.geom(idx).rgba[0:4] = [0.15, 0.15, 0.15, 0.85]
            self._model.geom(idx).size[0:3] = [0.0025, 0.0001, 0.0025]

        self._sequence_target_idxs = self._target_idxs.tolist()
        self._num_targets = len(self._sequence_target_idxs)
        # ------------------------------------------------------------------------------------------

        self._sequence_results_idxs = [self._default_idx for _ in self._sequence_target_idxs]

        self._switch_target(idx=self._sequence_target_idxs[0])

    def _switch_target(self, idx):

        for _idx in self._target_idxs:
            if _idx != idx:
                self._model.geom(_idx).rgba[0:4] = [0.15, 0.15, 0.15, 0.85]
                self._model.geom(_idx).size[0:3] = [0.0025, 0.0001, 0.0025]

        self._model.geom(idx).rgba[0:4] = [0.8, 0.8, 0, 1]
        self._model.geom(idx).size[0:3] = [0.0045, 0.0001, 0.0045]

        self._target_idx = idx

        # Do a forward so everything will be set
        mujoco.mj_forward(self._model, self._data)

    def _update_background(self):

        # Update the steps of background on
        if self._background_on:
            self._steps_background_on += 1
            # Close the background change
            if self._steps_background_on >= self._background_on_interval:
                self._model.geom(self._background_idx0).rgba[0:4] = [0.5, 0.5, 0.5, 0.85]
                self._model.geom(self._background_idx0).size[0:3] = [0.0035, 0.0001, 0.0035]
                self._steps_background_off = 0
                self._background_on = False

        # Start deterministic background change
        if self._background_on is False:
            self._steps_background_off += 1
            # Start the background change
            if self._steps_background_off >= self._background_off_interval:
                self._model.geom(self._background_idx0).rgba[0:4] = [0.8, 0, 0, 1]
                self._model.geom(self._background_idx0).size[0:3] = [0.0065, 0.0001, 0.0065]
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

        # Update the background changes
        self._update_background()

        # Advance the simulation
        mujoco.mj_step(self._model, self._data, self._frame_skip)
        self._steps += 1

        dist, geomid = self._ray_from_site(site_name="rangefinder-site")

        # Check for collisions, estimate reward
        reward = 0

        if self._background_on:
            if geomid == self._background_idx0:
                reward = 1
                # Update the environment
                acc = self._b_change / self._background_on_interval
                self._model.geom(geomid).rgba[0:3] = [x + y for x, y in
                                                     zip(self._model.geom(geomid).rgba[0:3], [0, 0, acc])]
                # Do a forward so everything will be set
                mujoco.mj_forward(self._model, self._data)
        else:
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

                # Update the results
                for i in range(self._num_targets):
                    if self._sequence_results_idxs[i] == self._default_idx:
                        self._sequence_results_idxs[i] = self._target_idx
                        break

                # Switch a new target grid, check if one loop has finished
                if self._target_idx >= self._sequence_target_idxs[-1]:
                    self._reset_scene()
                else:
                    self._switch_target(idx=self._target_idx+1)

        return self._get_obs(), reward, terminate, {}