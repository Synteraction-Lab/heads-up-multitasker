import math

import numpy as np
from collections import Counter
import mujoco
import os

from gym import Env
from gym.spaces import Box, Dict

import yaml
from scipy.ndimage import gaussian_filter

from huc.utils.rendering import Camera, Context

READ = 'reading'
BG = 'background'
RELOC = 'relocating'


class LocoRelocBase(Env):

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
        # Forward pass to initialise the model, enable all variables
        mujoco.mj_forward(self._model, self._data)

        # Define how often policy is queried
        self._action_sample_freq = 20
        self._frame_skip = int((1 / self._action_sample_freq) / self._model.opt.timestep)  # 0.05/0.002=25

        # Initialise thresholds and counters
        self._steps = None
        # The rgba cumulative change for finishing a fixation
        self._rgba_delta = 0.2

        # Get the primitives idxs in MuJoCo
        self._eye_joint_x_idx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "eye-joint-x")
        self._eye_joint_y_idx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "eye-joint-y")
        self._head_joint_y_idx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "head-joint-y")
        self._head_joint_x_idx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "head-joint-x")
        self._head_body_idx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "head")
        self._eye_body_idx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "eye")
        self._sgp_ils100_body_idx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY,
                                                      "smart-glass-pane-interline-spacing-100")
        self._sgp_bc_body_idx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY,
                                                  "smart-glass-pane-bottom-center")
        self._sgp_mr_body_idx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "smart-glass-pane-mid-right")
        self._bgp_body_idx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "background-pane")

        # Get targets (geoms that belong to "smart-glass-pane")
        # Inter-line-spacing-100
        self._ils100_reading_target_idxs = np.where(self._model.geom_bodyid == self._sgp_ils100_body_idx)[0]
        # Bottom-center
        self._bc_reading_target_idxs = np.where(self._model.geom_bodyid == self._sgp_bc_body_idx)[0]
        # Middle-right
        self._mr_reading_target_idxs = np.where(self._model.geom_bodyid == self._sgp_mr_body_idx)[0]
        # All layouts
        self._all_layouts_reading_traget_idxs = np.concatenate((self._ils100_reading_target_idxs.copy(),
                                                                self._bc_reading_target_idxs.copy(),
                                                                self._mr_reading_target_idxs.copy()))
        # General reading target index
        self._reading_target_idxs = None  # The reading target idxs list
        self._reading_target_idx = None  # The exact reading target idx

        # Define the default text grid size and rgba from a sample grid idx=0, define the hint text size and rgba
        sample_grid_idx = self._ils100_reading_target_idxs[0].copy()
        self._DEFAULT_TEXT_SIZE = self._model.geom(sample_grid_idx).size[0:3].copy()
        self._DEFAULT_TEXT_RGBA = [0, 0, 0, 1]
        self._RUNTIME_TEXT_RGBA = None
        self._HINT_SIZE = [self._DEFAULT_TEXT_SIZE[0] * 4 / 3, self._DEFAULT_TEXT_SIZE[1],
                           self._DEFAULT_TEXT_SIZE[2] * 4 / 3]
        # self._HINT_SIZE = self._DEFAULT_TEXT_SIZE.copy()
        self._HINT_RGBA = [1, 1, 0, 1]

        # Get the background (geoms that belong to "background-pane")
        self._background_idxs = np.where(self._model.geom_bodyid == self._bgp_body_idx)[0]
        self._background_idx0 = self._background_idxs[0].copy()
        # Define the default background grid size and rgba from a sample grid idx=0, define the event text size and rgba
        self._DEFAULT_BACKGROUND_SIZE = self._model.geom(self._background_idx0).size[0:3].copy()
        self._DEFAULT_BACKGROUND_RGBA = self._model.geom(self._background_idx0).rgba[0:4].copy()
        self._EVENT_RGBA = [1, 0, 0, 1]

        # Define the idx of grids which needs to be traversed sequentially on the smart glass pane
        self._reading_target_dwell_timesteps = int(2 * self._action_sample_freq)
        self._reading_rgb_change_per_step = self._rgba_delta / self._reading_target_dwell_timesteps

        # Define the events on the background pane
        self._background_trials = None
        self._background_dwell_timesteps = self._reading_target_dwell_timesteps
        self._background_rgba_change_per_step = self._rgba_delta / self._background_dwell_timesteps

        # Define the tasks variables for the relocation task
        self._neighbor_dist_thres = 0.0101

        # Define the tasks variables for the relocation task
        self._relocation_target_idx = None
        self._relocating_dwell_steps_thres = self._reading_target_dwell_timesteps
        self._RELOCATION_HINT_RGBA = [1, 0, 1, 0.5]
        self._relocation_rgb_change_per_step = 1 / self._relocating_dwell_steps_thres
        self._relocation_alpha_change_per_step = (1 - self._RELOCATION_HINT_RGBA[
            3]) / self._relocating_dwell_steps_thres

        # Define the pseudo_locomotion variables
        self._disp_lower_bound = self._model.jnt_range[self._head_joint_y_idx][0].copy()
        self._disp_upper_bound = self._model.jnt_range[self._head_joint_y_idx][1].copy()
        self._nearest_head_xpos_y = self._data.body(self._head_body_idx).xpos[1].copy() + self._disp_lower_bound
        self._furthest_head_xpos_y = self._data.body(self._head_body_idx).xpos[
                                         1].copy() + self._disp_upper_bound
        self._head_disp_per_step = (self._disp_upper_bound - self._disp_lower_bound) / 400

        # Define observation space
        self._width = self._config['mj_env']['width']
        self._height = self._config['mj_env']['height']
        # self.observation_space = Box(low=0, high=255,
        #                              shape=(3, self._width, self._height))  # width, height correctly set?
        self.observation_space = Dict({
            "vision": Box(low=-1, high=1, shape=(3, self._width, self._height)),
            "proprioception": Box(low=-1, high=1, shape=(self._model.nq + self._model.nu,))})
        # TODO set "proprioception" low and high according to joint/control limits, or make sure to output normalized
        #  joint/control values as observations

        # Define action space
        self.action_space = Box(low=-1, high=1, shape=(2,))

        # Initialise context, cameras
        self._context = Context(self._model, max_resolution=[1280, 960])
        self._eye_cam = Camera(self._context, self._model, self._data, camera_id="eye",
                               resolution=[self._width, self._height], maxgeom=100,
                               dt=1 / self._action_sample_freq)
        self._env_cam = Camera(self._context, self._model, self._data, camera_id="env", maxgeom=100,
                               dt=1 / self._action_sample_freq)
        self._eye_cam_fovy = self._model.cam_fovy[mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, "eye")]

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

            # Get joint values (qpos) and motor set points (ctrl) -- call them proprioception for now
            # 0-translation slide joint head-joint-y, 1-hinge joint eye-joint-x, 2-hinge joint eye-joint-y
            # Ref - Aleksi's code - https://github.com/BaiYunpeng1949/uitb-headsup-computing/blob/bf58d715b99ffabae4c2652f20898bac14a532e2/huc/envs/context_switch_replication/SwitchBackLSTM.py#L96
            proprioception = np.concatenate([self._data.qpos, self._data.ctrl])
            return {"vision": rgb_normalize, "proprioception": proprioception}

    def reset(self):

        # Reset mujoco sim
        mujoco.mj_resetData(self._model, self._data)

        # Reset counters
        self._steps = 0

        # Reset the scene
        self._reset_scene()

        return self._get_obs()

    def _reset_scene(self):

        # Reset the all reading grids - hide
        for idx in self._all_layouts_reading_traget_idxs:
            self._model.geom(idx).rgba[3] = 0

        # Reset the background scene
        self._model.geom(self._background_idx0).rgba[0:4] = self._DEFAULT_BACKGROUND_RGBA.copy()

    def _switch_target(self, idx):

        # Check the validity of the target idx and switch to the first one if the target idx is out of range
        if idx > self._reading_target_idxs[-1]:
            idx = self._reading_target_idxs[0]

        # De-highlight all the distracting cells
        for _idx in self._reading_target_idxs:
            if _idx != idx:
                self._model.geom(_idx).rgba[0:4] = self._DEFAULT_TEXT_RGBA.copy()
                self._model.geom(_idx).size[0:3] = self._DEFAULT_TEXT_SIZE.copy()

        # Highlight the target cell
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

    @staticmethod
    def angle_between(v1, v2):
        # Ref: https://github.com/BaiYunpeng1949/uitb-headsup-computing/blob/bf58d715b99ffabae4c2652f20898bac14a532e2/huc/envs/context_switch_replication/SwitchBackLSTM.py#L162
        # https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python/13849249#13849249
        def unit_vector(vec):
            return vec / np.linalg.norm(vec)

        v1_u = unit_vector(v1)
        v2_u = unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    def _angle_from_target(self, site_name):
        # Ref: https://github.com/BaiYunpeng1949/uitb-headsup-computing/blob/bf58d715b99ffabae4c2652f20898bac14a532e2/huc/envs/context_switch_replication/SwitchBackLSTM.py#L162

        # Get vector pointing direction from site
        site = self._data.site(site_name)
        pnt = site.xpos
        vec = pnt + site.xmat.reshape((3, 3))[:, 2]

        # Get vector pointing direction to target
        target_vec = self._data.geom(self._reading_target_idx).xpos - pnt

        # Estimate distance as angle
        angle = self.angle_between(vec, target_vec)

        return angle

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
        dist = math.sqrt((itsct_pnt[0] - target_pnt[0]) ** 2 + (itsct_pnt[2] - target_pnt[2]) ** 2)
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
        fov = self._eye_cam_fovy
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
        y, x = np.ogrid[-foveal_center[0]:img.shape[0] - foveal_center[0],
               -foveal_center[1]:img.shape[1] - foveal_center[1]]
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


class LocoRelocTrain(LocoRelocBase):        # TODO initialize more transitions to train
    def __init__(self):
        super().__init__()

        # Initialize the episode length and training trial thresholds
        self._ep_len = 100
        self._max_trials = 1
        self._trials = 0

        # Initialize the steps on target: either reading or background
        self._steps_on_target = None

        # Initialize the relocation relevant variables
        self._relocating_center_grid_idx = None
        self._neighbors = None
        self._neighbors_steps = None

        # Define the transition of 3 modes: 1-reading, 2-background, 3-relocation
        self._task_mode = None
        self._layout = None

    def _reset_scene(self):
        super()._reset_scene()

        # Initializations
        self._trials = 0
        self._steps_on_target = 0

        # Eyeball rotation initialization
        # Initialize eye ball rotation angles
        self._data.qpos[self._eye_joint_x_idx] = np.random.uniform(-0.5, 0.5)
        self._data.qpos[self._eye_joint_y_idx] = np.random.uniform(-0.5, 0.5)

        # Define the target reading layouts, randomly choose one list to copy from self._ils100_reading_target_idxs, self._bc_reading_target_idxs, self._mr_reading_target_idxs
        # self._layout = np.random.choice(['interline-spacing-100', 'bottom-center', 'middle-right'], 1)
        self._layout = self._config['rl']['test']['layout_name']
        if self._layout == 'interline-spacing-100':
            self._reading_target_idxs = self._ils100_reading_target_idxs.copy()
        elif self._layout == 'bottom-center':
            self._reading_target_idxs = self._bc_reading_target_idxs.copy()
        elif self._layout == 'middle-right':
            self._reading_target_idxs = self._mr_reading_target_idxs.copy()
        else:
            raise NotImplementedError('Invalid layout name: {}'.format(self._layout))

        # Reset the smart glass pane scene and variables
        for idx in self._reading_target_idxs:
            self._model.geom(idx).rgba[0:4] = self._DEFAULT_TEXT_RGBA.copy()
            self._model.geom(idx).size[0:3] = self._DEFAULT_TEXT_SIZE.copy()
            mujoco.mj_forward(self._model, self._data)

        # Define the task mode: 1-reading, 2-background, 3-relocation (60%, 30%, 10%) respectively
        self._task_mode = np.random.choice([READ, BG, RELOC], 1, p=[0.6, 0.3, 0.1]).copy()
        # Reading task
        if self._task_mode == READ:
            # Highlight the target reading grids
            self._switch_target(idx=np.random.choice(self._reading_target_idxs.tolist()))
            # Initialize the locomotion position - randomize the head joint y position within the valid range
            self._data.qpos[self._head_joint_y_idx] = np.random.uniform(self._disp_lower_bound,
                                                                        self._disp_upper_bound)
        # Background task
        elif self._task_mode == BG:
            # Initialize the locomotion position - always at the upper bound
            self._data.qpos[self._head_joint_y_idx] = self._disp_upper_bound
        # Relocation task
        elif self._task_mode == RELOC:
            # Randomize the center grid index for evoking neighbors
            # Choose one grid from the reading target grids
            self._center_grid_idx = np.random.choice(self._reading_target_idxs.tolist().copy())
            self._neighbors, self._neighbors_steps = [], []
            # Find the neighbors and set up the scene
            self._find_neighbors()
            # Initialize the locomotion position - Always start from the lower bound
            self._data.qpos[self._head_joint_y_idx] = self._disp_lower_bound
        else:
            raise NotImplementedError('The task mode is not defined! Should be 1-reading, 2-background, 3-relocation!')

        mujoco.mj_forward(self._model, self._data)

    def _find_neighbors(self):
        # Find the neighbors of the center grid
        center_xpos = self._data.geom(self._center_grid_idx).xpos

        neighbors = []

        for grid_idx in self._reading_target_idxs:
            grid_xpos = self._data.geom(grid_idx).xpos
            dist = np.linalg.norm(grid_xpos - center_xpos)
            if dist <= self._neighbor_dist_thres:
                neighbors.append(grid_idx)
                self._model.geom(grid_idx).rgba[0:4] = self._RELOCATION_HINT_RGBA.copy()

        # Randomly choose one grid in the neighbors list to be the target
        self._relocation_target_idx = np.random.choice(neighbors, 1)[0].copy()

        # Update the neighbors list
        self._neighbors = neighbors.copy()

        # Initialize the steps on each neighbor
        self._neighbors_steps = [0] * len(neighbors)

    def _update_background(self):
        super()._update_background()

        # The background pane is showing events - the red color events show
        if self._task_mode == BG:
            # Show the background event by changing to a brighter color
            self._model.geom(self._background_idx0).rgba[0:4] = self._EVENT_RGBA.copy()
        # The background pane is not showing events - the head is moving
        else:
            # Move the head if it has not getting close to the background enough
            if self._data.body(self._head_body_idx).xpos[1] < self._furthest_head_xpos_y:
                # Move the head towards the background pane
                if self._data.qpos[self._head_joint_y_idx] + self._head_disp_per_step.copy() >= self._furthest_head_xpos_y:
                    self._data.qpos[self._head_joint_y_idx] = self._disp_upper_bound.copy()
                else:
                    self._data.qpos[self._head_joint_y_idx] += self._head_disp_per_step.copy()
            else:
                # Update the trial counter and terminate the episode when the head is close enough to the background
                self._trials += 1

        mujoco.mj_forward(self._model, self._data)

    def step(self, action):
        super().step(action)

        # Normalise action from [-1, 1] to actuator control range
        action[0] = self.normalise(action[0], -1, 1, *self._model.actuator_ctrlrange[0, :])
        action[1] = self.normalise(action[1], -1, 1, *self._model.actuator_ctrlrange[1, :])

        # Set motor control
        self._data.ctrl[:] = action

        # Update the background events
        self._update_background()

        # Advance the simulation
        mujoco.mj_step(self._model, self._data, self._frame_skip)
        self._steps += 1

        # Eye-sight detection
        dist, geomid = self._ray_from_site(site_name="rangefinder-site")

        # Estimate reward for each step
        reward = 0

        # Specify the targets on different conditions
        if self._task_mode == READ:
            target_idx = self._reading_target_idx
            change_rgba = 0  # self._reading_rgb_change_per_step
            thres = self._reading_target_dwell_timesteps
        elif self._task_mode == BG:
            target_idx = self._background_idx0
            change_rgba = self._background_rgba_change_per_step
            thres = self._background_dwell_timesteps
        elif self._task_mode == RELOC:
            target_idx = self._relocation_target_idx
            change_rgba = self._relocation_rgb_change_per_step
            thres = self._relocating_dwell_steps_thres
        else:
            raise NotImplementedError('The task mode is not defined! Should be reading, background, relocation!')

        # Single target scenarios - 1 reading and 2 background
        if self._task_mode == READ or self._task_mode == BG:
            # Focus on targets detection
            if geomid == target_idx:
                # Sparse reward
                reward = 1
                # Update the steps on target
                self._steps_on_target += 1
                # Update the environment
                self._model.geom(geomid).rgba[2] += change_rgba
                # Check if the target has been fixated enough
                if self._steps_on_target >= thres:
                    self._trials += 1

        # Multiple targets scenarios - 3 relocation
        else:
            if geomid in self._neighbors:
                # Update the steps on target
                self._neighbors_steps[self._neighbors.index(geomid)] += 1

                if geomid == target_idx:
                    # Sparse reward
                    reward = 1
                    # Update the steps on target
                    self._steps_on_target += 1
                    # Update the reading target - becomes more opaque
                    self._model.geom(geomid).rgba[3] += self._relocation_alpha_change_per_step
                else:
                    # Update the distractions - becomes dimmer
                    self._model.geom(geomid).rgba[1] += self._relocation_rgb_change_per_step

                # Update the environment
                # De-highlight the geom if it has been fixated enough
                if self._neighbors_steps[self._neighbors.index(geomid)] >= thres:
                    # Update the neighbors list
                    self._neighbors[self._neighbors.index(geomid)] = -2

                    # Check for the reading target
                    if geomid == target_idx:
                        # Update the reading target
                        self._trials += 1

        # Do a forward so everything will be set
        mujoco.mj_forward(self._model, self._data)

        # Check termination conditions
        if self._steps >= self._ep_len or self._trials >= self._max_trials:
            terminate = True
        else:
            terminate = False

        return self._get_obs(), reward, terminate, {}


class LocoRelocTest(LocoRelocBase):

    def __init__(self):
        super().__init__()

        # Initialize the length of the episode
        self._ep_len = 8000

        # Initialize the number of trials
        self._background_max_trials = 8
        self._background_trials = None

        self._steps_on_reading_target = None
        self._steps_on_relocation_target = None
        self._steps_on_background_target = None

        # Define the buffer for storing the number of goodput grids
        self._num_read_cells = None

        # Initialize the relocation relevant variables
        self._neighbors = None
        self._neighbors_steps = None

        # Initialize the context switch relevant variables
        self._off_background_step = None
        self._switch_back_durations = None
        self._switch_back_error_steps = None

        # Initialize the task mode and layout name
        self._task_mode = None
        self._layout_name = None

    def _reset_scene(self):
        super()._reset_scene()

        # Reset the permanent and temporary counters
        self._background_trials = 0
        self._num_read_cells = 0
        self._switch_back_durations = []
        self._switch_back_error_steps = []

        self._steps_on_reading_target = 0
        self._steps_on_relocation_target = 0
        self._steps_on_background_target = 0
        self._off_background_step = 0
        self._neighbors, self._neighbors_steps = [], []
        self._relocation_target_idx = -2

        # Define the target reading layouts, randomly choose one list to copy from self._ils100_reading_target_idxs, self._bc_reading_target_idxs, self._mr_reading_target_idxs
        self._layout_name = self._config['rl']['test']['layout_name']
        if self._layout_name == 'interline-spacing-100':
            self._reading_target_idxs = self._ils100_reading_target_idxs.copy()
        elif self._layout_name == 'bottom-center':
            self._reading_target_idxs = self._bc_reading_target_idxs.copy()
        elif self._layout_name == 'middle-right':
            self._reading_target_idxs = self._mr_reading_target_idxs.copy()
        else:
            raise ValueError('Invalid layout name.')

        # Reset the smart glass pane scene and variables
        for idx in self._reading_target_idxs:
            self._model.geom(idx).rgba[0:4] = self._DEFAULT_TEXT_RGBA.copy()
            self._model.geom(idx).size[0:3] = self._DEFAULT_TEXT_SIZE.copy()
            # Set everything so the we can find neighbors if needed
            mujoco.mj_forward(self._model, self._data)

        # Reading grids
        self._reading_target_idxs = self._reading_target_idxs.tolist()
        self._switch_target(idx=self._reading_target_idxs[0])
        self._task_mode = READ

        # Locomotion
        self._data.qpos[self._head_joint_y_idx] = self._disp_lower_bound.copy()

    def _update_background(self):
        super()._update_background()

        # Locomotion
        # If the head is not on the furthest position, it will move towards the background pane
        if self._data.body(self._head_body_idx).xpos[1] < self._furthest_head_xpos_y:
            # Move the head towards the background pane
            if self._data.qpos[self._head_joint_y_idx] + self._head_disp_per_step.copy() >= self._furthest_head_xpos_y:
                self._data.qpos[self._head_joint_y_idx] = self._disp_upper_bound.copy()
            else:
                self._data.qpos[self._head_joint_y_idx] += self._head_disp_per_step.copy()

        # If the head is on the furthest position, i.e., near the background, starts the red color event
        else:
            # Start the red color event
            if self._task_mode != BG:
                self._task_mode = BG
                # Set the red color
                self._model.geom(self._background_idx0).rgba[0:4] = self._EVENT_RGBA.copy()
                # De-highlight the previous job - reading
                self._RUNTIME_TEXT_RGBA = self._model.geom(self._reading_target_idx).rgba[0:4].copy()
                for idx in self._reading_target_idxs:
                    self._model.geom(idx).rgba[0:4] = self._DEFAULT_TEXT_RGBA.copy()
                    self._model.geom(idx).size[0:3] = self._DEFAULT_TEXT_SIZE.copy()

        mujoco.mj_forward(self._model, self._data)

    def _find_neighbors(self):
        # Find the neighbors of the center grid
        center_xpos = self._data.geom(self._reading_target_idx).xpos

        neighbors = []

        for grid_idx in self._reading_target_idxs:
            grid_xpos = self._data.geom(grid_idx).xpos
            dist = np.linalg.norm(grid_xpos - center_xpos)
            if dist <= self._neighbor_dist_thres:
                neighbors.append(grid_idx)
                self._model.geom(grid_idx).rgba[0:4] = self._RELOCATION_HINT_RGBA.copy()

        # Randomly choose one grid in the neighbors list to be the target
        self._relocation_target_idx = self._reading_target_idx

        # Update the neighbors list
        self._neighbors = neighbors.copy()

        # Initialize the steps on each neighbor
        self._neighbors_steps = [0] * len(neighbors)

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

        # Check the tasks and update accordingly
        if self._task_mode == READ:
            if geomid == self._reading_target_idx:
                # Grant an instant reward
                reward = 1

                # Update the reading target counter
                self._steps_on_reading_target += 1
                # Update the reading target color
                self._model.geom(geomid).rgba[2] += self._reading_rgb_change_per_step
                # Check the termination condition
                if self._steps_on_reading_target >= self._relocating_dwell_steps_thres:
                    # Update the number of reading target
                    self._num_read_cells += 1

                    # Reset the grid color
                    self._switch_target(idx=self._reading_target_idx + 1)
                    self._task_mode = READ
                    # Reset the reading target counter
                    self._steps_on_reading_target = 0

        elif self._task_mode == BG:
            if geomid == self._background_idx0:
                # Sparse reward
                reward = 1
                # Update the background target counter
                self._steps_on_background_target += 1
                # Update the background target color
                self._model.geom(geomid).rgba[2] += self._background_rgba_change_per_step
                # Check the termination condition
                if self._steps_on_background_target >= self._background_dwell_timesteps:
                    # Update the number of background trials
                    self._background_trials += 1
                    # Update the off background step
                    self._off_background_step = self._steps

                    # Reset the background target counter
                    self._steps_on_background_target = 0
                    # Reset the background variables: the color, status flag, the counter, and the head position
                    self._model.geom(self._background_idx0).rgba[0:4] = self._DEFAULT_BACKGROUND_RGBA.copy()
                    # Reset the pseudo_locomotion position
                    self._data.qpos[self._head_joint_y_idx] = self._disp_lower_bound.copy()

                    # Jump into the relocation task
                    self._find_neighbors()
                    self._task_mode = RELOC

        elif self._task_mode == RELOC:
            if geomid in self._neighbors:
                # Update the steps on target
                self._neighbors_steps[self._neighbors.index(geomid)] += 1

                if geomid == self._relocation_target_idx:
                    # Sparse reward
                    reward = 1
                    # Update the relocation target counter
                    self._steps_on_relocation_target += 1
                    # Update the relocation target color
                    self._model.geom(geomid).rgba[3] += self._relocation_alpha_change_per_step
                else:
                    # Update the distractions - becomes dimmer
                    self._model.geom(geomid).rgba[1] += self._relocation_rgb_change_per_step

                # Check the termination condition
                steps_on_relocation_target = self._neighbors_steps[self._neighbors.index(self._relocation_target_idx)]
                if steps_on_relocation_target >= self._relocating_dwell_steps_thres:
                    # Update the switch back duration
                    current_step = self._steps
                    switch_back_duration = current_step - self._off_background_step
                    self._switch_back_durations.append(switch_back_duration)

                    # Update the switch back error rate
                    # Get the total number of steps in self._neighbors_steps that does not corresponding to the relocation target
                    switch_back_error_steps = np.sum(self._neighbors_steps) - steps_on_relocation_target
                    # Update the switch back error rate list
                    self._switch_back_error_steps.append(switch_back_error_steps)

                    # Reset the relocation target counter
                    self._steps_on_relocation_target = 0

                    # Get back to the reading mode
                    self._task_mode = READ
                    # Update the reading target
                    self._switch_target(idx=self._reading_target_idx + 1)
                    # Reset the reading target counter
                    self._steps_on_reading_target = 0
        else:
            raise ValueError(f'Unknown task mode: {self._task_mode}')

        mujoco.mj_forward(self._model, self._data)

        # Check termination conditions
        if self._steps >= self._ep_len or self._background_trials > self._background_max_trials:
            terminate = True
        else:
            terminate = False

        if terminate:
            print(
                f'The total timesteps is: {self._steps}. The total number of cells traversed is: {self._num_read_cells} \n'
                f'The switch back duration is: {np.sum(self._switch_back_durations)}. The durations are: {self._switch_back_durations} \n'
                f'The reading goodput is: {round(self._num_read_cells / self._steps, 5)} (grids per timestep). \n'
                f'The switch back error rate is: {round(100 * np.sum(self._switch_back_error_steps) / (self._num_read_cells * self._reading_target_dwell_timesteps), 4)} %;'
                f'The switch back error steps are: {self._switch_back_error_steps} \n')

        return self._get_obs(), reward, terminate, {}
