import math

import numpy as np
from collections import Counter
import mujoco
import os

from gym import Env
from gym.spaces import Box

import yaml
from scipy.ndimage import gaussian_filter

from huc.utils.rendering import Camera, Context


class RelocationBase(Env):

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

        # Get the primitives idxs in MuJoCo
        self._eye_joint_x_idx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "eye-joint-x")
        self._eye_joint_y_idx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "eye-joint-y")
        self._head_joint_y_idx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "head-joint-y")
        self._head_joint_x_idx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "head-joint-x")
        self._head_body_idx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "head")
        self._eye_body_idx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "eye")
        self._sgp_ils100_body_idx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "smart-glass-pane-interline-spacing-100")
        self._sgp_bc_body_idx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "smart-glass-pane-bottom-center")
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
        self._reading_target_idxs = None    # The reading target idxs list
        self._reading_target_idx = None     # The exact reading target idx

        # Define the default text grid size and rgba from a sample grid idx=0, define the hint text size and rgba
        sample_grid_idx = self._ils100_reading_target_idxs[0].copy()
        self._DEFAULT_TEXT_SIZE = self._model.geom(sample_grid_idx).size[0:4].copy()
        self._DEFAULT_TEXT_RGBA = [0, 0, 0, 1]
        self._RUNTIME_TEXT_RGBA = None
        # self._HINT_SIZE = [self._DEFAULT_TEXT_SIZE[0] * 6 / 5, self._DEFAULT_TEXT_SIZE[1],
        #                    self._DEFAULT_TEXT_SIZE[2] * 6 / 5]
        self._HINT_SIZE = self._DEFAULT_TEXT_SIZE.copy()
        self._HINT_RGBA = [1, 1, 0, 0.5]      # Yellow: used to be [0.8, 0.8, 0, 1]

        # Get the background (geoms that belong to "background-pane")
        self._background_idxs = np.where(self._model.geom_bodyid == self._bgp_body_idx)[0]
        self._background_idx0 = self._background_idxs[0].copy()
        # Define the default background grid size and rgba from a sample grid idx=0, define the event text size and rgba
        self._DEFAULT_BACKGROUND_SIZE = self._model.geom(self._background_idx0).size[0:4].copy()
        self._DEFAULT_BACKGROUND_RGBA = self._model.geom(self._background_idx0).rgba[0:4].copy()
        self._EVENT_RGBA = [1, 0, 0, 1]      # Red: used to be [0.8, 0, 0, 1]

        # Color changes
        # The rgba cumulative change for finishing a fixation
        self._rgba_delta = 1
        # The rgba cumulative change for finishing a fixation of the real target
        self._alpha_delta = 1 - self._HINT_RGBA[3]

        # Define the idx of grids which needs to be traversed sequentially on the smart glass pane
        self._reading_target_dwell_timesteps = int(2 * self._action_sample_freq)        # TODO this has to be the relocation related variable in pseudo_locomotion class
        self._change_rbga_reading_target = self._rgba_delta / self._reading_target_dwell_timesteps
        self._change_alpha_reading_target = self._alpha_delta / self._reading_target_dwell_timesteps

        # Define the events on the background pane
        self._background_on = None
        self._background_trials = None
        self._background_dwell_timesteps = self._reading_target_dwell_timesteps
        self._change_rgba_background = self._rgba_delta / self._background_dwell_timesteps

        # Define the pseudo_locomotion variables
        self._displacement_lower_bound = self._model.jnt_range[self._head_joint_y_idx][0].copy()
        self._displacement_upper_bound = self._model.jnt_range[self._head_joint_y_idx][1].copy()
        self._nearest_head_xpos_y = self._data.body(self._head_body_idx).xpos[1].copy() + self._displacement_lower_bound
        self._furthest_head_xpos_y = self._data.body(self._head_body_idx).xpos[1].copy() + self._displacement_upper_bound
        self._head_disp_per_timestep = (self._displacement_upper_bound - self._displacement_lower_bound) / 400

        # Define observation space
        self._width = self._config['mj_env']['width']
        self._height = self._config['mj_env']['height']
        self.observation_space = Box(low=0, high=255, shape=(3, self._width, self._height))  # width, height correctly set?

        # Define action space
        self.action_space = Box(low=-1, high=1, shape=(2,))

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

        # Reset the scene
        self._reset_scene()

        return self._get_obs()

    def _reset_scene(self):

        # Reset the all reading grids - hide
        for idx in self._all_layouts_reading_traget_idxs:
            self._model.geom(idx).rgba[3] = 0

        # Reset the background scene
        self._model.geom(self._background_idx0).rgba[0:4] = self._DEFAULT_BACKGROUND_RGBA.copy()

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


class RelocationTrain(RelocationBase):

    def __init__(self):
        super().__init__()

        # Initialize the episode length
        self._ep_len = 500

        # Initialize the steps on target: either reading or background
        self._steps_on_target = None

        # Initialize the counter, and the max number of trials for the reading task
        self._reading_trials = None
        self._reading_max_trials = 1

        # Initialize the max number of trials for the background task
        self._background_max_trials = 1
        # Define the flag for the background task - show or not
        self._background_show_flag = None

        # Neighbor distance threshold
        self._neighbor_dist_thres = None    # 0.010 + 0.0001, or 0.0121
        # Neighbors list
        self._relocating_neighbors = None
        self._center_grid_idx = None
        self._neighbors = None
        self._neighbors_permanent_buffer = None
        self._neighbors_steps = None    # Each grids in neighbors list has a step counter

        # Define the initial displacement of the agent's head
        self._head_init_displacement_y = None

    def _reset_scene(self):
        super()._reset_scene()

        # Initialize eye ball rotation angles
        eye_x_motor_init_range = [-0.5, 0.5]
        eye_y_motor_init_range = [-0.5, 0.4]

        init_angle_x = np.random.uniform(eye_x_motor_init_range[0], eye_x_motor_init_range[1])
        init_angle_y = np.random.uniform(eye_y_motor_init_range[0], eye_y_motor_init_range[1])

        self._data.qpos[self._eye_joint_x_idx] = init_angle_x
        self._data.qpos[self._eye_joint_y_idx] = init_angle_y

        # Define the target reading layouts, randomly choose one list to copy from self._ils100_reading_target_idxs, self._bc_reading_target_idxs, self._mr_reading_target_idxs
        random_choice = np.random.choice([0, 1, 2], 1)
        if random_choice == 0:
            self._reading_target_idxs = self._ils100_reading_target_idxs.copy()
            self._neighbor_dist_thres = 0.0101
        elif random_choice == 1:
            self._reading_target_idxs = self._bc_reading_target_idxs.copy()
            self._neighbor_dist_thres = 0.0121
        else:
            self._reading_target_idxs = self._mr_reading_target_idxs.copy()
            self._neighbor_dist_thres = 0.0101

        # Reset the smart glass pane scene and variables
        for idx in self._reading_target_idxs:
            self._model.geom(idx).rgba[0:4] = self._DEFAULT_TEXT_RGBA.copy()
            self._model.geom(idx).size[0:3] = self._DEFAULT_TEXT_SIZE.copy()
            mujoco.mj_forward(self._model, self._data)

        # Define the target reading grids from the selected reading layouts
        self._reading_trials = 0

        # Define the central target idx, and start the distractions with a reading target embedded in it
        random_idxs = np.random.choice(self._reading_target_idxs.tolist().copy(), 3, False)
        self._center_grid_idx = random_idxs[0]
        self._neighbors = []
        self._neighbors_steps = []
        # Find the neighbors and set up the scene
        self._find_neighbors()

        # Initialize the steps on target
        self._steps_on_target = 0

        # Initialize the pseudo_locomotion slide displacement
        self._data.qpos[self._head_joint_y_idx] = 0.0

        mujoco.mj_forward(self._model, self._data)

    def _find_neighbors(self):

        center_grid_idx = self._center_grid_idx
        center_xpos = self._data.geom(center_grid_idx).xpos

        neighbors = []

        for grid_idx in self._reading_target_idxs:
            grid_xpos = self._data.geom(grid_idx).xpos
            dist = np.linalg.norm(grid_xpos - center_xpos)
            if dist <= self._neighbor_dist_thres:
                neighbors.append(grid_idx)
                self._model.geom(grid_idx).rgba[0:4] = self._HINT_RGBA.copy()   # Continue to avoid infinite loop
                self._model.geom(grid_idx).size[0:3] = self._HINT_SIZE.copy()

        # Update the neighbors list
        self._neighbors = neighbors.copy()
        self._neighbors_permanent_buffer = neighbors.copy()

        # Initialize the steps on each neighbor
        self._neighbors_steps = [0] * len(neighbors)

        # Randomly choose one grid in the neighbors list to be the target
        self._reading_target_idx = np.random.choice(neighbors, 1)[0].copy()

        if self._mode == "test":        # TODO debug delete later
            self._reading_target_idx = self._center_grid_idx

    def step(self, action):
        super().step(action)

        # Normalise action from [-1, 1] to actuator control range
        action[0] = self.normalise(action[0], -1, 1, *self._model.actuator_ctrlrange[0, :])
        action[1] = self.normalise(action[1], -1, 1, *self._model.actuator_ctrlrange[1, :])

        # Set motor control
        self._data.ctrl[:] = action

        # Advance the simulation
        mujoco.mj_step(self._model, self._data, self._frame_skip)
        self._steps += 1

        # Eye-sight detection
        dist, geomid = self._ray_from_site(site_name="rangefinder-site")

        # Estimate reward for each step
        reward = 0

        # Determine the targets on different conditions
        target_idx = self._reading_target_idx
        change_rgba = self._change_rbga_reading_target

        # Target detection
        if geomid in self._neighbors:

            # Update the steps on target
            self._neighbors_steps[self._neighbors.index(geomid)] += 1

            if geomid == target_idx:
                # Sparse reward
                reward = 1
                # Update the steps on target
                self._steps_on_target += 1
                # Update the reading target - becomes more opaque
                self._model.geom(geomid).rgba[2] += self._change_alpha_reading_target   # TODO a mistake, but it works, the index should be 3
            else:
                # Update the distractions - becomes dimmer
                self._model.geom(geomid).rgba[2] += change_rgba

            # Update the environment
            # De-highlight the geom if it has been fixated enough
            if self._neighbors_steps[self._neighbors.index(geomid)] >= self._reading_target_dwell_timesteps:    # TODO change to relocation threshold dwell timesteps
                # self._model.geom(geomid).rgba[0:4] = self._DEFAULT_TEXT_RGBA.copy()
                # self._model.geom(geomid).size[0:3] = self._DEFAULT_TEXT_SIZE.copy()
                # Update the neighbors list
                self._neighbors[self._neighbors.index(geomid)] = -2     # TODO cannot use -1, -1 is the default value for rangefinder

                # Check for the reading target
                if geomid == target_idx:
                    # Update the reading target
                    self._reading_trials += 1

            # Do a forward so everything will be set
            mujoco.mj_forward(self._model, self._data)

        # Check termination conditions
        if self._steps >= self._ep_len:
            terminate = True
        else:
            terminate = False

            # Terminate if all trials have been done - TODO if multiple trials, buffers need to be reset
            if self._reading_trials >= self._reading_max_trials:
                terminate = True

        # Print logs if on the testing mode - TODO debug delete
        if self._mode != 'train' and self._mode != 'continual_train':
            if terminate == True:
                print(f'The ending steps are: {self._steps}; \n'
                      f'The neighbors were defined as: {self._neighbors_permanent_buffer}; \n'
                      f'The number of elements in the neighborhood: {len(self._neighbors)}; \n'
                      f'The number of steps on the target: {self._steps_on_target}; \n'
                      f'The target idx is: {target_idx}; \n'
                      f'The neighbor fixations steps are: {self._neighbors_steps}; \n')

        return self._get_obs(), reward, terminate, {}
