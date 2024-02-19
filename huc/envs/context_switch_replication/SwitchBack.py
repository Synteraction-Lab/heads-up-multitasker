import math

import numpy as np
import mujoco
import os

from gym import Env
from gym.spaces import Box

import yaml
from scipy.ndimage import gaussian_filter

from huc.utils.rendering import Camera, Context


class SwitchBack(Env):

    def __init__(self):

        # Get directory of this file
        directory = os.path.dirname(os.path.realpath(__file__))

        # Read the configurations from the YAML file.
        root_dir = os.path.dirname(os.path.dirname(directory))
        with open(os.path.join(root_dir, "config.yaml")) as f:
            self._config = yaml.load(f, Loader=yaml.FullLoader)
        self._mode = self._config['rl']['mode']

        # Open the mujoco model
        self._xml_path = os.path.join(directory, self._config['mj_env']['xml'])
        self._model = mujoco.MjModel.from_xml_path(self._xml_path)
        self._data = mujoco.MjData(self._model)

        # Define how often policy is queried
        self._action_sample_freq = 20
        self._frame_skip = int((1 / self._action_sample_freq) / self._model.opt.timestep)  # 0.05/0.002=25

        # Initialise thresholds and counters
        self._steps = None
        # The rgba cumulative change for finishing a fixation
        self._rgba_delta = 0.2

        # Get targets (geoms that belong to "smart-glass-pane")
        sgp_body = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "smart-glass-pane")
        self._reading_target_idxs = np.where(self._model.geom_bodyid == sgp_body)[0]
        self._reading_targets = [self._model.geom(idx) for idx in self._reading_target_idxs]
        self._reading_target_idx = None
        self._reading_target = None
        # Define the default text grid size and rgba from a sample grid idx=0, define the hint text size and rgba
        self._DEFAULT_TEXT_SIZE = self._model.geom(self._reading_target_idxs[0]).size[0:4].copy()
        self._DEFAULT_TEXT_RGBA = self._model.geom(self._reading_target_idxs[0]).rgba[0:4].copy()
        self._RUNTIME_TEXT_RGBA = None
        self._HINT_SIZE = [self._DEFAULT_TEXT_SIZE[0] * 4 / 3, self._DEFAULT_TEXT_SIZE[1],
                           self._DEFAULT_TEXT_SIZE[2] * 4 / 3]
        self._HINT_RGBA = [0.8, 0.8, 0, 1]

        # Get the background (geoms that belong to "background-pane")
        bp_body = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "background-pane")
        self._background_idxs = np.where(self._model.geom_bodyid == bp_body)[0]
        self._background_idx0 = self._background_idxs[0]
        # Define the default background grid size and rgba from a sample grid idx=0, define the event text size and rgba
        self._DEFAULT_BACKGROUND_SIZE = self._model.geom(self._background_idx0).size[0:4].copy()
        self._DEFAULT_BACKGROUND_RGBA = self._model.geom(self._background_idx0).rgba[0:4].copy()
        self._EVENT_RGBA = [0.8, 0, 0, self._model.geom(self._background_idx0).rgba[3].copy()]

        # Define the idx of grids which needs to be traversed sequentially on the smart glass pane
        self._reading_target_dwell_timesteps = 2 * self._action_sample_freq
        self._reading_targets_idxs = None
        self._change_rbga_reading_target = self._rgba_delta / self._reading_target_dwell_timesteps

        # Define the grids on the background pane
        self._background_on = None
        self._background_dwell_timesteps = self._reading_target_dwell_timesteps
        self._background_trials = None
        self._change_rgba_background = self._rgba_delta / self._background_dwell_timesteps

        # Define observation space
        self._width = self._config['mj_env']['width']
        self._height = self._config['mj_env']['height']
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
        self._cam_eye_fovy = self._model.cam_fovy[mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, "eye")]

    def _get_obs(self):

        # Render the image
        rgb, _ = self._eye_cam.render()

        # Preprocess
        # Foveated vision applied
        if 'foveate' in self._config['rl']['train']['checkpoints_folder_name']:
            rgb_foveated = self._foveate(img=rgb)
            rgb_foveated = np.transpose(rgb_foveated, [2, 0, 1])
            rgb_normalize = self.normalise(rgb_foveated, 0, 255, -1, 1)
            return rgb_normalize
        # Foveated vision not applied
        else:
            rgb = np.transpose(rgb, [2, 0, 1])
            rgb_normalize = self.normalise(rgb, 0, 255, -1, 1)
            return rgb_normalize

    def reset(self):

        # Reset mujoco sim
        mujoco.mj_resetData(self._model, self._data)

        # Reset counters
        self._steps = 0
        # self._trials = 0

        # Reset the scene
        self._reset_scene()

        return self._get_obs()

    def _reset_scene(self):

        # Reset the smart glass pane scene and variables
        for idx in self._reading_target_idxs:
            self._model.geom(idx).rgba[0:4] = self._DEFAULT_TEXT_RGBA.copy()
            self._model.geom(idx).size[0:3] = self._DEFAULT_TEXT_SIZE.copy()

        # Reset the background scene and variables
        self._background_trials = 0
        self._background_on = False
        self._model.geom(self._background_idx0).rgba[0:4] = self._DEFAULT_BACKGROUND_RGBA.copy()

    def _switch_target(self, idx):

        for _idx in self._reading_target_idxs:
            if _idx != idx:
                self._model.geom(_idx).rgba[0:4] = self._DEFAULT_TEXT_RGBA.copy()
                self._model.geom(_idx).size[0:3] = self._DEFAULT_TEXT_SIZE.copy()

        self._model.geom(idx).rgba[0:4] = self._HINT_RGBA.copy()
        self._model.geom(idx).size[0:3] = self._HINT_SIZE.copy()

        # Update the target id
        self._reading_target_idx = idx

        # Do a forward so everything will be set
        mujoco.mj_forward(self._model, self._data)

    def _update_background(self):
        return

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

    @staticmethod
    def normalise(x, x_min, x_max, a, b):
        # Normalise x (which is assumed to be in range [x_min, x_max]) to range [a, b]
        return (b - a) * ((x - x_min) / (x_max - x_min)) + a

    def _foveate(self, img):
        """
        Foveate the image, note that the shape of image has to be (height, width, 3)
        """

        # Define the blurring level
        sigma = 1

        # Define the foveal region
        fov = self._cam_eye_fovy
        foveal_size = 30
        foveal_pixels = int(foveal_size / 2 * img.shape[0] / fov)
        foveal_center = (img.shape[0] // 2, img.shape[1] // 2)

        # Define the blur kernel
        kernel_size = foveal_pixels * 2 + 1
        kernel = np.zeros((kernel_size, kernel_size))

        # Create a linear ramp for the blur kernel
        ramp = np.linspace(0, 1, kernel_size)
        kernel[:, foveal_pixels] = ramp
        kernel[foveal_pixels, :] = ramp

        # Create a circular mask for the foveal region
        y, x = np.ogrid[-foveal_center[0]:img.shape[0] - foveal_center[0], -foveal_center[1]:img.shape[1] - foveal_center[1]]
        mask = x ** 2 + y ** 2 <= (foveal_pixels ** 2)

        # Apply a Gaussian blur to each color channel separately
        blurred = np.zeros_like(img)
        for c in range(3):
            blurred_channel = gaussian_filter(img[:, :, c], sigma=sigma)
            blurred[:, :, c][~mask] = blurred_channel[~mask]

        # Combine the original image and the blurred image
        foveated = img.copy()
        foveated[~mask] = blurred[~mask]

        return foveated

    def render(self, mode="rgb_array"):
        rgb, _ = self._env_cam.render()
        rgb_eye, _ = self._eye_cam.render()
        return rgb, rgb_eye

    def step(self, action):
        return


class SwitchBackTrain(SwitchBack):

    def __init__(self):
        super().__init__()

        # Initialize the episode length
        self._ep_len = 100

        # Initialize the counter, and the max number of trials for the reading task
        self._reading_trials = None
        self._reading_max_trials = 1

        # Initialize the max number of trials for the background task
        self._background_max_trials = 1

        # Initialize the flag for the background task - show or not
        self._background_show_flag = None

    def reset(self):
        super().reset()

        self._reading_trials = 0

        return self._get_obs()

    def _reset_scene(self):
        super()._reset_scene()

        # Initialize eye ball rotation angles
        if 'context-switch-12-mid-right' in self._xml_path:
            eye_x_motor_init_range = [-0.5, 0.5]
            eye_y_motor_init_range = [-0.5, 0.25]
        elif 'context-switch-12-bottom-center' in self._xml_path:
            eye_x_motor_init_range = [-0.5, 0.5]
            eye_y_motor_init_range = [-0.25, 0.25]
        elif 'context-switch-12-inter-line-spacing-50' in self._xml_path:
            eye_x_motor_init_range = [-0.5, 0.5]
            eye_y_motor_init_range = [-0.25, 0.25]
        elif 'context-switch-12-inter-line-spacing-100' in self._xml_path:
            eye_x_motor_init_range = [-0.5, 0.5]
            eye_y_motor_init_range = [-0.25, 0.25]
        else:
            eye_x_motor_init_range = [-0.5, 0.5]
            eye_y_motor_init_range = [-0.5, 0.5]
        action = [np.random.uniform(eye_x_motor_init_range[0], eye_x_motor_init_range[1]),
                  np.random.uniform(eye_y_motor_init_range[0], eye_y_motor_init_range[1])]

        for i in range(10):
            # Set motor control
            self._data.ctrl[:] = action
            # Advance the simulation
            mujoco.mj_step(self._model, self._data, self._frame_skip)

        # Define the target reading grids
        self._sequence_target_idxs = np.random.choice(self._reading_target_idxs.tolist(), 3, False)

        # Define the background events
        self._background_show_flag = np.random.choice([True, False])

        # Decide whether or not show the reading grids
        if self._background_show_flag == False:
            self._switch_target(idx=self._sequence_target_idxs[0])

    def _update_background(self):
        super()._update_background()

        if self._background_on == False:
            if self._background_trials < self._background_max_trials:
                # Show the background event by changing to a brighter color
                if self._steps >= 0:
                    self._model.geom(self._background_idx0).rgba[0:4] = self._EVENT_RGBA.copy()
                    self._background_on = True
        else:
            # Identify whether should stop the background event
            if self._model.geom(self._background_idx0).rgba[2] >= self._rgba_delta:
                self._background_trials += 1
                self._model.geom(self._background_idx0).rgba[0:4] = self._DEFAULT_BACKGROUND_RGBA.copy()
                self._background_on = False

        mujoco.mj_forward(self._model, self._data)

    def step(self, action):
        super().step(action)

        # Normalise action from [-1, 1] to actuator control range
        action[0] = self.normalise(action[0], -1, 1, *self._model.actuator_ctrlrange[0, :])
        action[1] = self.normalise(action[1], -1, 1, *self._model.actuator_ctrlrange[1, :])

        # Set motor control
        self._data.ctrl[:] = action

        if self._background_show_flag == True:
            self._update_background()

        # Advance the simulation
        mujoco.mj_step(self._model, self._data, self._frame_skip)
        self._steps += 1

        # Eye-sight detection
        dist, geomid = self._ray_from_site(site_name="rangefinder-site")

        # Estimate reward for each step
        reward = 0

        # Specify the targets on different conditions
        if self._background_show_flag == True:
            target_idx = self._background_idx0
            acc = self._change_rgba_background
        else:
            target_idx = self._reading_target_idx
            acc = self._change_rbga_reading_target

        # Focus on targets detection
        if geomid == target_idx:
            reward = 1
            # Update the environment
            self._model.geom(geomid).rgba[0:3] = [x + y for x, y in
                                                  zip(self._model.geom(geomid).rgba[0:3], [0, 0, acc])]
            # Do a forward so everything will be set
            mujoco.mj_forward(self._model, self._data)

        # Check termination conditions
        if self._steps >= self._ep_len:
            terminate = True
        else:
            terminate = False

            if self._background_show_flag == True:
                if self._background_trials >= self._background_max_trials:
                    terminate = True
            else:
                if self._model.geom(self._reading_target_idx).rgba[2] >= self._rgba_delta:
                    self._reading_trials += 1
                    if self._reading_trials >= self._reading_max_trials:
                        terminate = True

        return self._get_obs(), reward, terminate, {}


class SwitchBackTest(SwitchBack):

    def __init__(self):
        super().__init__()

        self._ep_len = 2000

        self._init_reading_idx = -1

        self._background_max_trials = 4

        self._background_show_interval = 3 * self._reading_target_dwell_timesteps
        self._background_last_on_steps = None

        # Define the relocation distraction relevant variables
        self._relocating_dwell_interval = 0.5 * self._reading_target_dwell_timesteps
        self._relocating_steps = None
        self._relocating_neighbors = None
        self._relocating_incorrect_steps = None
        # Neighbor distance threshold
        self._relocating_dist_neighbor = 0.010 + 0.0001  # Actual inter-word distance + offset
        # Relocating - pick up point
        self._relocating_pickup_dwell_steps = int(0.25 * self._reading_target_dwell_timesteps)
        self._relocating_pickup_records = None
        self._relocating_incorrect_num = None
        self._off_background_step = None
        self._switch_back_durations = None

    def _reset_scene(self):
        super()._reset_scene()

        self._background_last_on_steps = 0

        # Relocating / pick-up issues
        self._relocating_steps = 0
        self._relocating_neighbors = []
        self._relocating_pickup_records = []
        self._relocating_incorrect_num = 0
        self._off_background_step = 0
        self._switch_back_durations = []

        # Reading grids
        self._sequence_target_idxs = self._reading_target_idxs.tolist()
        self._switch_target(idx=self._sequence_target_idxs[0])

        # Relocating
        self._relocating_incorrect_steps = 0

    def _update_background(self):
        super()._update_background()

        if self._background_on == False:
            if (
                    self._steps - self._background_last_on_steps) >= self._background_show_interval and self._background_trials < self._background_max_trials:
                # Background
                self._background_on = True
                self._model.geom(self._background_idx0).rgba[0:4] = self._EVENT_RGBA.copy()
                # Reading grids
                self._RUNTIME_TEXT_RGBA = self._model.geom(self._reading_target_idx).rgba[0:4].copy()
                for idx in self._sequence_target_idxs:
                    self._model.geom(idx).rgba[0:4] = self._DEFAULT_TEXT_RGBA.copy()
                    self._model.geom(idx).size[0:3] = self._DEFAULT_TEXT_SIZE.copy()
        else:
            if self._model.geom(self._background_idx0).rgba[2] >= self._rgba_delta:
                # Background
                self._background_trials += 1
                self._model.geom(self._background_idx0).rgba[0:4] = self._DEFAULT_BACKGROUND_RGBA.copy()
                self._background_on = False
                # Update the step buffer
                self._background_last_on_steps = self._steps
                # Reading grids distractions
                self._find_neighbors()  # TODO baseline vs issues on relocating

                # Reading without distractions
                # self._model.geom(self._reading_target_idx).rgba[0:4] = self._RUNTIME_TEXT_RGBA.copy()
                # self._model.geom(self._reading_target_idx).size[0:3] = self._HINT_SIZE.copy()
                # self._relocating_neighbors = [self._reading_target_idx]

                # Switch back off background timestamp
                if self._off_background_step == 0:
                    # Only update when last pick up happened and this time stamp was reset to 0
                    self._off_background_step = self._steps

        mujoco.mj_forward(self._model, self._data)

    def _find_neighbors(self):

        target_grid_idx = self._reading_target_idx
        target_xpos = self._data.geom(target_grid_idx).xpos

        neighbors = []

        for grid_idx in self._sequence_target_idxs:
            grid_xpos = self._data.geom(grid_idx).xpos
            dist = np.linalg.norm(grid_xpos - target_xpos)
            if dist <= self._relocating_dist_neighbor:
                neighbors.append(grid_idx)
                self._model.geom(grid_idx).rgba[0:4] = self._RUNTIME_TEXT_RGBA.copy()   # Continue to avoid infinite loop
                self._model.geom(grid_idx).size[0:3] = self._HINT_SIZE.copy()

        self._relocating_neighbors = neighbors.copy()

    def step(self, action):
        super().step(action)

        # Normalise action from [-1, 1] to actuator control range
        action[0] = self.normalise(action[0], -1, 1, *self._model.actuator_ctrlrange[0, :])
        action[1] = self.normalise(action[1], -1, 1, *self._model.actuator_ctrlrange[1, :])

        # Set motor control
        self._data.ctrl[:] = action

        # Update the background changes, non-trainings
        self._update_background()

        # Advance the simulation
        mujoco.mj_step(self._model, self._data, self._frame_skip)
        self._steps += 1

        # Eye-sight detection
        dist, geomid = self._ray_from_site(site_name="rangefinder-site")

        # Estimate reward for each step
        reward = 0

        # Specify the targets on different conditions
        if self._background_on == True:
            target_idx = self._background_idx0
            change = self._change_rgba_background
        else:
            target_idx = self._reading_target_idx
            change = self._change_rbga_reading_target

        # Focus on targets detection
        if geomid == target_idx:
            reward = 1
            # Update the environment
            self._model.geom(geomid).rgba[0:3] = [x + y for x, y in
                                                  zip(self._model.geom(geomid).rgba[0:3], [0, 0, change])]
            # Do a forward so everything will be set
            mujoco.mj_forward(self._model, self._data)

        # Check termination conditions
        if self._steps >= self._ep_len:
            terminate = True
        else:
            terminate = False

            # Check the relocating / pick-up status when the background event is off - version 0328
            if self._background_on == False:
                # Check the relocating / pick-up issues - version 0328
                if geomid in self._relocating_neighbors:
                    self._relocating_pickup_records.append(geomid)
                    for idx in self._relocating_neighbors:
                        # The first grid with cumulative fixations has not been found - TODO do it
                        if self._relocating_pickup_records.count(idx) > self._relocating_pickup_dwell_steps:
                            # Update the switch back duration
                            current_step = self._steps
                            switch_back_duration = current_step - self._off_background_step
                            self._switch_back_durations.append(switch_back_duration)

                            # Update the counters
                            if idx == self._reading_target_idx:
                                pass
                            else:
                                self._relocating_incorrect_num += 1

                            # Draw all distractions back to normal - version continue with the current grid
                            for _idx in self._relocating_neighbors:
                                if _idx != idx:
                                    self._model.geom(_idx).rgba[0:4] = self._DEFAULT_TEXT_RGBA.copy()
                                    self._model.geom(_idx).size[0:3] = self._DEFAULT_TEXT_SIZE.copy()
                            # Refresh the current target grid and the sequence results idxs if idx is not the previous target
                            if idx != self._reading_target_idx:
                                self._reading_target_idx = idx

                            # Clear the buffer
                            self._relocating_pickup_records = []
                            self._relocating_neighbors = []
                            self._off_background_step = 0
                            # Set to 0 to avoid one empty round of relocating: during one interval of the background task,
                            # it never picked up anything. This is related to the hyper-parameter of self._relocating_pickup_dwell_steps
                            break
            # Check at the start of the next background event, whether the pervious relocating trial was successful
            else:
                # The relocating did not pick up anything at the previous trial
                if self._off_background_step != 0:
                    self._relocating_incorrect_num += 1
                    self._switch_back_durations.append(self._background_show_interval)
                    print(
                        'One relocating trial was unable to pick up any grid at the target grid {}. '
                        'The relocating pickup records is: {}. '
                        'The switch back duration is: {}. '
                        'The background trial number is: {}.'
                            .format(self._reading_target_idx, self._background_show_interval,
                                    self._relocating_pickup_records, self._background_trials)
                    )
                    # Clear the buffer
                    self._relocating_pickup_records = []
                    self._relocating_neighbors = []
                    self._off_background_step = 0

            # Check whether the grid (background or reading) has been fixated for enough time
            if self._model.geom(self._reading_target_idx).rgba[2] >= self._rgba_delta:

                # Update the intervened relocating where relocating dwell was smaller than the remaining reading time
                if not self._relocating_neighbors:
                    pass
                else:  # When the relocating_neighbor is not empty - the relocation was not over that the relocating dwell was bigger than the remaining reading time
                    # Add new switch back durations
                    current_step = self._steps
                    switch_back_duration = current_step - self._off_background_step
                    self._switch_back_durations.append(switch_back_duration)

                    # Clear the buffer
                    self._relocating_pickup_records = []
                    self._relocating_neighbors = []
                    self._off_background_step = 0

                    print(
                        'One relocating trial was finished earlier at target grid {}. The switch back duration is: {}. '
                        'The background trial is: {}'.
                            format(self._reading_target_idx, switch_back_duration, self._background_trials)
                    )

                # Terminate the loop if all grids are traversed, or switch to the next grid
                if self._reading_target_idx >= self._sequence_target_idxs[-1]:
                    terminate = True
                else:
                    self._switch_target(idx=self._reading_target_idx + 1)

            if terminate:
                print('The total timesteps is: {}. \n'
                      'The switch back duration is: {}. The durations are: {} \n'
                      'The reading goodput is: {} (grids per timestep). \n'
                      'The switch back error rate is: {}%.'.
                      format(self._steps, np.sum(self._switch_back_durations), self._switch_back_durations,
                             round(len(self._sequence_target_idxs) / self._steps, 5),
                             round(100 * self._relocating_incorrect_num / self._background_max_trials, 2)))

        return self._get_obs(), reward, terminate, {}


