import numpy as np
import mujoco
import os

from gym import Env
from gym.spaces import Box

from huc.utils.rendering import Camera, Context


class MovingEye(Env):

    def __init__(self):

        # Get directory of this file
        directory = os.path.dirname(os.path.realpath(__file__))

        # Open the mujoco model
        self._model = mujoco.MjModel.from_xml_path(os.path.join(directory, "moving-eye.xml"))
        self._data = mujoco.MjData(self._model)

        # Define how often policy is queried
        self._action_sample_freq = 20
        self._frame_skip = int((1 / self._action_sample_freq) / self._model.opt.timestep)

        # Initialise thresholds and counters
        self._target_switch_interval = 2 * self._action_sample_freq
        self._steps = 0
        self._max_trials = 5
        self._trial_idx = 0
        self._ep_len = self._max_trials * self._target_switch_interval

        # Get targets (geoms that belong to "smart-glass-pane")
        sgp_body = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "smart-glass-pane")
        self._target_idxs = np.where(self._model.geom_bodyid == sgp_body)[0]
        self._targets = [self._model.geom(idx) for idx in self._target_idxs]
        self._target_idx = None
        self._target = None

        # Define observation space
        self._width = 80
        self._height = 80
        self.observation_space = Box(low=0, high=255,
                                     shape=(3, self._width, self._height))  # width, height correctly set?

        # Define action space
        self.action_space = Box(low=-1, high=1, shape=(2,))

        # Define a cutoff for rangefinder (in meters, could be something like 3 instead of 0.1)
        self._rangefinder_cutoff = 0.1

        # Initialise context, cameras
        self._context = Context(self._model, max_resolution=[1280, 960])
        self._eye_cam = Camera(self._context, self._model, self._data, camera_id="eye",
                               resolution=[self._width, self._height], maxgeom=100, dt=1 / self._action_sample_freq)
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
        self._trial_idx = 0

        # Choose one target at random
        self._switch_target()

        return self._get_obs()

    def _switch_target(self):

        # Sample a random target
        idx = np.random.choice(len(self._target_idxs))
        # self._target_idx = self._target_idxs[idx]
        self._target_idx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "target")
        self._target = self._targets[idx]

        # Set position of target (the yellow border)
        self._model.body("target").pos = self._model.body("smart-glass-pane").pos + self._target.pos

        # Do a forward so everything will be set
        mujoco.mj_forward(self._model, self._data)

    def render(self, mode="rgb_array"):
        rgb, _ = self._env_cam.render()
        rgb_eye, _ = self._eye_cam.render()
        return rgb, rgb_eye

    def normalise(self, x, x_min, x_max, a, b):
        # Normalise x (which is assumed to be in range [x_min, x_max]) to range [a, b]
        return (b - a) * ((x - x_min) / (x_max - x_min)) + a

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

    def step(self, action):

        # Normalise action from [-1, 1] to actuator control range
        action[0] = self.normalise(action[0], -1, 1, *self._model.actuator_ctrlrange[0, :])
        action[1] = self.normalise(action[1], -1, 1, *self._model.actuator_ctrlrange[1, :])

        # Set motor control
        self._data.ctrl[:] = action

        # Advance the simulation
        mujoco.mj_step(self._model, self._data, self._frame_skip)
        self._steps += 1

        dist, geomid = self._ray_from_site(site_name="rangefinder-site")

        # Check for collisions, estimate reward
        reward = 0
        if geomid == self._target_idx:
            reward = 1

        # Check whether we should terminate episode (if we have gone through enough trials)
        if self._trial_idx >= self._max_trials:
            terminate = True

        else:
            terminate = False
            # Check whether we should switch target
            if self._steps >= self._target_switch_interval:
                self._switch_target()
                self._trial_idx += 1
                self._steps = 0

        return self._get_obs(), reward, terminate, {}
