import numpy as np
from collections import Counter, deque
import mujoco
import os

from gym import Env
from gym.spaces import Box, Dict

import yaml
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter

from huc.utils.rendering import Camera, Context


class OcularMotorControl(Env):

    def __init__(self):
        """
        Model the vision perception and ocular motor control of human

        The agent learns: with given target index and image input, control the eyeball to fixate on the target
        Ocular motor control has uncertainty, denoted by the ocular motor noise,
        which is from the paper: An Adaptive Model of Gaze-based Selection

        This should be the only model that directly interacts with the MuJoCo model
        """
        # Get directory of this file
        directory = os.path.dirname(os.path.realpath(__file__))

        # Read the configurations from the YAML file.
        root_dir = os.path.dirname(os.path.dirname(directory))
        with open(os.path.join(root_dir, "config.yaml")) as f:
            self._config = yaml.load(f, Loader=yaml.FullLoader)

        # Load the MuJoCo model
        self._model = mujoco.MjModel.from_xml_path(os.path.join(directory, "12cells-3layouts-small-font-v1.xml"))
        self._data = mujoco.MjData(self._model)
        mujoco.mj_forward(self._model, self._data)

        # Define how often policy is queried
        self.action_sample_freq = 20
        self._frame_skip = int((1 / self.action_sample_freq) / self._model.opt.timestep)

        # Get the joints idx in MuJoCo
        self._eye_joint_x_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "eye-joint-x")
        self._eye_joint_y_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "eye-joint-y")
        # Head joints for perturbation:
        # the head motion amplitudes were found to be approximately 11 degrees in the horizontal direction (yaw)
        # and 8 degrees in the vertical direction (pitch)
        # Ref: https://archivesphysiotherapy.biomedcentral.com/articles/10.1186/s40945-020-00077-9
        self._head_joint_x_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "head-joint-x")
        self._head_joint_z_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "head-joint-z")
        # Agent's locomotion joints
        self._agent_joint_y_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "agent-joint-y")

        # Get the motors idx in MuJoCo
        self._eye_x_motor_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, "eye-x-motor")
        self._eye_x_motor_translation_range = self._model.actuator_ctrlrange[self._eye_x_motor_mjidx]
        self._eye_y_motor_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, "eye-y-motor")
        self._eye_y_motor_rotation_range = self._model.actuator_ctrlrange[self._eye_y_motor_mjidx]
        self._head_x_motor_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, "head-x-motor")
        self._head_z_motor_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, "head-z-motor")
        self._agent_y_motor_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, "agent-y-motor")
        self._agent_y_rotation_range = self._model.actuator_ctrlrange[self._agent_y_motor_mjidx]

        self._L100_body_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "smart-glass-pane-interline-spacing-100")
        self._L50_body_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "smart-glass-pane-interline-spacing-50")
        self._L0_body_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "smart-glass-pane-interline-spacing-0")

        # Get geom mjidxs (geoms that belong to "smart-glass-pane-interline-spacing-100")
        self._L100_cells_mjidxs = np.where(self._model.geom_bodyid == self._L100_body_mjidx)[0]
        self._L50_cells_mjidxs = np.where(self._model.geom_bodyid == self._L50_body_mjidx)[0]
        self._L0_cells_mjidxs = np.where(self._model.geom_bodyid == self._L0_body_mjidx)[0]

        # Initialize the MuJoCo layout related parameters
        self._L100 = "L100"
        self._L50 = "L50"
        self._L0 = "L0"
        self._layouts = [self._L100, self._L50, self._L0]
        self._cells_mjidxs = None

        # Initialize the sampled target mjidx
        self._sampled_target_mjidx = None

        # Define the target idx probability distribution
        self._VISUALIZE_RGBA = [1, 1, 0, 1]
        self._DFLT_RGBA = [0, 0, 0, 1]

        # Initialize the dwell time parameters
        self._dwell_time_range = [0.2, 1]  # 200-1000 ms
        self._dwell_steps = None

        # Initialize the fixation related parameters
        self._last_step_saccade_qpos = None

        # Initialize the locomotion/translation parameters
        translation_speed = 2     # 2m/s for normal walking - 15 m/s will give you a fast view for demonstration
        self._step_wise_translation_speed = translation_speed / self.action_sample_freq

        # Initialize the perturbation parameters
        # Ref: Frequency and velocity of rotational head perturbations during locomotion
        self._theoretical_pitch_amp_peak = 0.0523599  # +-3 degrees in radians
        self._theoretical_yaw_amp_peak = 0.10472  # +-6 degrees in radians
        self._perturbation_amp_coeff_factor = None
        self._perturbation_amp_coeff_range = [0.1, 1]
        self._pitch_freq = 2  # 2 Hz
        self._yaw_freq = 1  # 1 Hz
        self._pitch_2nd_predominant_freq = 3  # 3.75 Hz
        self._yaw_2nd_predominant_freq = 2.75  # 3 Hz
        self._pitch_2nd_predominant_relative_amp = 0.1  # 10% of the 1st predominant frequency
        self._yaw_2nd_predominant_relative_amp = 0.1  # 10% of the 1st predominant frequency
        self._pitch_period_stepwise = int(self.action_sample_freq / self._pitch_freq)
        self._yaw_period_stepwise = int(self.action_sample_freq / self._yaw_freq)
        self._perturbation_amp_noise_scale = 0.015

        # Oculomotor control related parameters
        # Start from action noises - the integration of oculomotor noise and the drifts after fixations.
        # The oculomotor noise is formalized as SDN, the zero-mean Gaussian noise with a standard deviation of
        # the signal proportional to the magnitude of the signal itself.
        # Now the saccade is a ballistic movement and can be finished in 1 action taken step (50 ms)
        self._rho_ocular_motor = 0.08   # The proportionality constant from paper: An Adaptive Model of Gaze-based Selection
        self._fixate_on_target = None
        self._previous_fixate_on_target = None

        # Initialise RL related thresholds and counters
        self._steps = None
        self._on_target_steps = None
        self._num_trials = None  # Cells are already been read
        self._max_trials = 10  # Maximum number of cells to read - more trials in one episode will boost the convergence
        if self._config["rl"]["mode"] == "debug":
            self._max_trials = 1
        self.ep_len = int(self._max_trials * self._dwell_time_range[1] * self.action_sample_freq * 5)
        self._in_hrl = False

        # Define the observation space
        width, height = 80, 80
        self._num_stk_frm = 1
        self._vision_frames = None
        self._qpos_frames = None
        self._num_stateful_info = 6
        unwanted_qpos_ctrl = ['locomotion']
        self.observation_space = Dict({
            "vision": Box(low=-1, high=1, shape=(self._num_stk_frm, width, height)),
            "proprioception": Box(low=-1, high=1, shape=(self._num_stk_frm * (self._model.nq - len(unwanted_qpos_ctrl))
                                                         + (self._model.nu - len(unwanted_qpos_ctrl)),)),
            "stateful information": Box(low=-1, high=1, shape=(self._num_stateful_info,)),
        })

        # Define the action space
        self.action_space = Box(low=-1, high=1, shape=(2,))   # 2 dof eyeball rotation

        # Initialize the context and camera
        context = Context(self._model, max_resolution=[1280, 960])
        self._eye_cam = Camera(context, self._model, self._data, camera_id="eye", resolution=[width, height],
                               maxgeom=100,
                               dt=1 / self.action_sample_freq)
        self._env_cam = Camera(context, self._model, self._data, camera_id="env", maxgeom=100,
                               dt=1 / self.action_sample_freq)
        self._eye_cam_fovy = self._model.cam_fovy[mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, "eye")]

        # Define the training related parameters
        self._epsilon = 1e-100

    def reset(self, grid_search_params=None, load_model_params=None):

        # Reset MuJoCo sim
        mujoco.mj_resetData(self._model, self._data)

        # Initiate the stacked frames
        self._vision_frames = deque(maxlen=self._num_stk_frm)
        self._qpos_frames = deque(maxlen=self._num_stk_frm)

        # Reset the variables, flags, and counters
        self._steps = 0
        self._on_target_steps = 0
        self._num_trials = 0
        self._fixate_on_target = False
        self._in_hrl = False

        # Reset all cells to transparent
        for mjidx in self._L100_cells_mjidxs:
            self._model.geom(mjidx).rgba[3] = 0
        for mjidx in self._L50_cells_mjidxs:
            self._model.geom(mjidx).rgba[3] = 0
        for mjidx in self._L0_cells_mjidxs:
            self._model.geom(mjidx).rgba[3] = 0

        # The low level ocular motor control related training, testing (including the grid search), and debugging
        if load_model_params is None:
            # Initialize perturbation parameters for training
            self._perturbation_amp_coeff_factor = np.random.uniform(*self._perturbation_amp_coeff_range)
            self._dwell_steps = int(np.random.uniform(*self._dwell_time_range) * self.action_sample_freq)

            # Initialize eyeball rotation angles
            init_eye_x_rotation = np.random.uniform(*self._eye_x_motor_translation_range)
            init_eye_y_rotation = np.random.uniform(*self._eye_y_motor_rotation_range)
            self._data.qpos[self._eye_joint_x_mjidx] = init_eye_x_rotation
            self._data.ctrl[self._eye_x_motor_mjidx] = init_eye_x_rotation
            self._data.qpos[self._eye_joint_y_mjidx] = init_eye_y_rotation
            self._data.ctrl[self._eye_y_motor_mjidx] = init_eye_y_rotation

            # Initialize the layout
            layout = np.random.choice(self._layouts)
            self._get_layout_mjidxs(layout)

            # Configure the stochastic hyperparameters in the test mode
            if self._config["rl"]["mode"] == "test":
                self._data.qpos[self._eye_joint_x_mjidx] = 0
                self._data.qpos[self._eye_joint_y_mjidx] = 0

                if grid_search_params is None:
                    # The test-demo mode
                    self._perturbation_amp_coeff_factor = 0
                    self._perturbation_amp_noise_scale = 0
                    self._dwell_steps = int(0.5 * self.action_sample_freq)
                    print(f"The pert amp tuning factor was: {self._perturbation_amp_coeff_factor}, "
                          f"the pert amp noise factor is; {self._perturbation_amp_noise_scale}, "
                          f"the dwell steps is: {self._dwell_steps}")
                else:
                    # The test-grid-search mode
                    self._perturbation_amp_coeff_factor = grid_search_params["perturbation_amp_tuning_factor"]
                    self._perturbation_amp_noise_scale = grid_search_params["perturbation_amp_noise_scale"]
                    self._dwell_steps = int(grid_search_params["dwell_steps"] * self.action_sample_freq)

            # Initialize the saccade from qpos
            self._last_step_saccade_qpos = self._data.qpos[self._eye_joint_x_mjidx:self._eye_joint_x_mjidx + 2].copy()

            # Sample a target according to the target idx probability distribution
            self._sample_target()
        # Interacting with the environment using the loaded model
        else:
            self._cells_mjidxs = load_model_params["cells_mjidxs"]
            self._perturbation_amp_coeff_factor = load_model_params["perturbation_amp_tuning_factor"]
            self._perturbation_amp_noise_scale = load_model_params["perturbation_amp_noise_scale"]
            self._dwell_steps = int(load_model_params["dwell_time"] * self.action_sample_freq)

            # Initialize eyeball rotation angles
            init_eye_x_rotation = load_model_params["eye_x_rotation"]
            init_eye_y_rotation = load_model_params["eye_y_rotation"]
            self._data.qpos[self._eye_joint_x_mjidx] = init_eye_x_rotation
            self._data.ctrl[self._eye_x_motor_mjidx] = init_eye_x_rotation
            self._data.qpos[self._eye_joint_y_mjidx] = init_eye_y_rotation
            self._data.ctrl[self._eye_y_motor_mjidx] = init_eye_y_rotation

            # Initialize the saccade from qpos
            self._last_step_saccade_qpos = self._data.qpos[self._eye_joint_x_mjidx:self._eye_joint_x_mjidx + 2].copy()

            # Sample a target according to the target idx probability distribution
            self._sampled_target_mjidx = load_model_params["target_mjidx"]

            # Reset the scene
            for mjidx in self._cells_mjidxs:
                self._model.geom(mjidx).rgba = self._DFLT_RGBA
            self._model.geom(self._sampled_target_mjidx).rgba = self._VISUALIZE_RGBA

            self._in_hrl = True

        # Set up the whole scene by confirming the initializations
        mujoco.mj_forward(self._model, self._data)

        return self._get_obs()

    def render(self, mode="rgb_array"):
        rgb, _ = self._env_cam.render()
        rgb_eye, _ = self._eye_cam.render()
        return rgb, rgb_eye

    def step(self, action):
        # Apply the locomotion to the agent - non-stationary environment
        self._data.ctrl[self._agent_y_motor_mjidx] = np.clip(
            self._data.ctrl[self._agent_y_motor_mjidx]+self._step_wise_translation_speed, *self._agent_y_rotation_range
        )

        # Apply perturbations to agent's eyeball - non-stationary environment
        # Update the tunable hyperparameters, for training, I choose a big value to cover the whole range,
        # for testing, I choose a changing smaller value to fit human data
        amp_tuning_factor = self._perturbation_amp_coeff_factor
        perturbation_amp_noise_scale = self._perturbation_amp_noise_scale

        pitch_amp = self._theoretical_pitch_amp_peak * amp_tuning_factor
        yaw_amp = self._theoretical_yaw_amp_peak * amp_tuning_factor
        pitch = pitch_amp * np.sin(2 * np.pi * self._steps / self._pitch_period_stepwise) + \
            pitch_amp * self._pitch_2nd_predominant_relative_amp * np.sin(
            2 * np.pi * self._steps * self._pitch_2nd_predominant_freq / self.action_sample_freq)
        yaw = yaw_amp * np.sin(2 * np.pi * self._steps / self._yaw_period_stepwise) + \
            yaw_amp * self._yaw_2nd_predominant_relative_amp * np.sin(
            2 * np.pi * self._steps * self._yaw_2nd_predominant_freq / self.action_sample_freq)

        # Add some random noise
        pitch += np.random.normal(loc=0, scale=perturbation_amp_noise_scale, size=pitch.shape)
        yaw += np.random.normal(loc=0, scale=perturbation_amp_noise_scale, size=yaw.shape)

        self._data.ctrl[self._head_x_motor_mjidx] = np.clip(pitch, *self._model.actuator_ctrlrange[self._head_x_motor_mjidx])
        self._data.ctrl[self._head_z_motor_mjidx] = np.clip(yaw, *self._model.actuator_ctrlrange[self._head_z_motor_mjidx])

        # Action at t
        # Normalise action from [-1, 1] to actuator control range
        # The control range was set to [-0.7854, 0.7854] as corresponding to [-45, 45] degrees
        # Ref: "Head-fixed saccades can have amplitudes of up to 90°", https://en.wikipedia.org/wiki/Saccade
        action[0] = self.normalise(action[0], -1, 1, *self._model.actuator_ctrlrange[self._eye_x_motor_mjidx, :])
        action[1] = self.normalise(action[1], -1, 1, *self._model.actuator_ctrlrange[self._eye_y_motor_mjidx, :])

        dist, geomid = self._get_focus(site_name="rangefinder-site")
        if geomid == self._sampled_target_mjidx:
            self._fixate_on_target = True
        else:
            self._fixate_on_target = False
            # Get the ocular motor noise
            next_saccade_position = action[0:2].copy()
            last_saccade_position = self._last_step_saccade_qpos[0:2].copy()
            saccade_amplitude = np.abs(last_saccade_position - next_saccade_position)

            # The noises for 2 motors are sampled differently
            ocular_motor_noises = np.random.normal(0, np.abs(self._rho_ocular_motor * saccade_amplitude))
            action[0:2] += ocular_motor_noises

        self._data.ctrl[self._eye_x_motor_mjidx] = action[0]
        self._data.ctrl[self._eye_y_motor_mjidx] = action[1]

        # Log and save the action for saccades and fixations calculation
        self._previous_fixate_on_target = self._fixate_on_target

        # Advance the simulation
        mujoco.mj_step(self._model, self._data, self._frame_skip)
        self._steps += 1

        # Update the last step saccade qpos
        self._last_step_saccade_qpos = self._data.qpos[self._eye_joint_x_mjidx:self._eye_joint_x_mjidx+2].copy()

        # State at t+1 - transition function?
        # Eye-sight detection
        dist, geomid = self._get_focus(site_name="rangefinder-site")
        # Update the fixations
        if geomid == self._sampled_target_mjidx:
            self._on_target_steps += 1
            self._fixate_on_target = True
        else:
            self._fixate_on_target = False

        # In the separate training/testing scenario (not in the hrl):
        if self._in_hrl:
            # Update the transitions - get rewards and next state
            if self._on_target_steps >= self._dwell_steps:
                # Update the milestone bonus reward for finish reading a cell
                reward = 10
                self._num_trials += 1
            else:
                reward = 0.1 * (np.exp(
                    -10 * self._angle_from_target(site_name="rangefinder-site",
                                                  target_idx=self._sampled_target_mjidx)) - 1)
        else:
            # Update the transitions - get rewards and next state
            if self._on_target_steps >= self._dwell_steps:
                # Update the milestone bonus reward for finish reading a cell
                reward = 10
                # Get the next target
                self._sample_target()
            else:
                reward = 0.1 * (np.exp(
                    -10 * self._angle_from_target(site_name="rangefinder-site", target_idx=self._sampled_target_mjidx)) - 1)

        # Get termination condition
        terminate = False
        info = {
            'eye_x_rotation': self._data.qpos[self._eye_joint_x_mjidx],
            'eye_y_rotation': self._data.qpos[self._eye_joint_y_mjidx],
        }
        # # TODO debug delete later
        # print(f"****************************************************************************"
        #       f"   Ocular Motor Control:\n"
        #       f"the current layouts is: {self._cells_mjidxs},"
        #       f"the sampled target is: {self._sampled_target_mjidx}, the geomid is: {geomid}")

        if self._steps >= self.ep_len or self._num_trials > self._max_trials:
            terminate = True
            info = {
                'end_steps': self._steps,
                'save_folder': None,
                'num_cells': len(self._cells_mjidxs),
                'action_sample_freq': self.action_sample_freq,
                'eye_x_rotation': self._data.qpos[self._eye_joint_x_mjidx],
                'eye_y_rotation': self._data.qpos[self._eye_joint_y_mjidx],
            }

            if self._config["rl"]["mode"] == "debug" or self._config["rl"]["mode"] == "test":
                print(f"\nThe dwell steps is: {self._dwell_steps}, "
                      f"\nThe amp tuning factor is: {amp_tuning_factor}, "
                      f"\nThe perturbation amp noise scale is: {perturbation_amp_noise_scale}"
                      f"\nThe total time steps is: {self._steps}")

        return self._get_obs(), reward, terminate, info

    @staticmethod
    def normalise(x, x_min, x_max, a, b):
        # Normalise x (which is assumed to be in range [x_min, x_max]) to range [a, b]
        return (b - a) * ((x - x_min) / (x_max - x_min)) + a

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

    def _get_obs(self):
        """ Get the observation of the environment """
        # Compute the vision observation
        # Render the image
        rgb, _ = self._eye_cam.render()
        # Preprocess - H*W*C -> C*W*H
        rgb = np.transpose(rgb, [2, 1, 0])
        rgb_normalize = self.normalise(rgb, 0, 255, -1, 1)
        # Convert the rgb to grayscale - boost the training speed
        gray_normalize = rgb_normalize[0:1, :, :]*0.299 + rgb_normalize[1:2, :, :]*0.587 + rgb_normalize[2:3, :, :]*0.114
        gray_normalize = np.squeeze(gray_normalize, axis=0)

        # Update the stack of frames of the vision channel
        self._vision_frames.append(gray_normalize)
        # Replicate the newest frame if the stack is not full
        while len(self._vision_frames) < self._num_stk_frm:
            self._vision_frames.append(self._vision_frames[-1])
        # Reshape the stack of frames - Get vision observations
        vision = np.stack(self._vision_frames, axis=0)
        vision = vision.reshape((-1, vision.shape[-2], vision.shape[-1]))
        # vision = gray_normalize.reshape((-1, gray_normalize.shape[-2], gray_normalize.shape[-1]))

        # Compute the proprioception observation
        # Update the stack of frames of the proprioception channel
        # Remove the locomotion value since it is not normalized
        needed_qpos = self._data.qpos.copy()
        needed_qpos = np.delete(needed_qpos, self._agent_joint_y_mjidx)
        self._qpos_frames.append(needed_qpos)
        while len(self._qpos_frames) < self._num_stk_frm:
            self._qpos_frames.append(self._qpos_frames[-1])
        # Get the proprioception observation
        qpos = np.stack(self._qpos_frames, axis=0)
        qpos = qpos.reshape((1, -1))
        # Remove the locomotion control since it is not normalized
        wanted_ctrl = self._data.ctrl.copy()
        wanted_ctrl = np.delete(wanted_ctrl, self._agent_y_motor_mjidx)
        ctrl = wanted_ctrl.reshape((1, -1))
        proprioception = np.concatenate([qpos.flatten(), ctrl.flatten()], axis=0)

        # Compute the stateful information observation - normalize to [-1, 1]
        # TODO how to manage these in the supervisory control? We should not be able to know the remaining steps, e.g., when does the task end
        remaining_ep_len_norm = (self.ep_len - self._steps) / self.ep_len * 2 - 1
        remaining_dwell_steps_norm = (self._dwell_steps - self._on_target_steps) / self._dwell_steps * 2 - 1
        remaining_trials_norm = (self._max_trials - self._num_trials) / self._max_trials * 2 - 1
        sampled_target_mjidx_norm = self.normalise(self._sampled_target_mjidx, self._cells_mjidxs[0],
                                                   self._cells_mjidxs[-1], -1, 1)

        fixation_norm = 1 if self._fixate_on_target else -1
        previous_fixation_norm = 1 if self._previous_fixate_on_target else -1

        stateful_info = np.array(
            [remaining_ep_len_norm, remaining_dwell_steps_norm, remaining_trials_norm,
             sampled_target_mjidx_norm,
             fixation_norm, previous_fixation_norm]
        )

        # Observation space check
        if vision.shape != self.observation_space["vision"].shape:
            raise ValueError(f"The shape of vision observation is not correct! "
                             f"Should be {self.observation_space['vision'].shape}, but got {vision.shape}")
        if proprioception.shape != self.observation_space["proprioception"].shape:
            raise ValueError(f"The shape of proprioception observation is not correct! "
                             f"Should be {self.observation_space['proprioception'].shape}, but got {proprioception.shape}")
        if stateful_info.shape != self.observation_space["stateful information"].shape:
            raise ValueError(f"The shape of stateful information observation is not correct!"
                             f"Should be {self.observation_space['stateful information'].shape}, but got {stateful_info.shape}")

        return {"vision": vision, "proprioception": proprioception, "stateful information": stateful_info}

    def _get_layout_mjidxs(self, layout):
        if layout == self._L100:
            self._cells_mjidxs = self._L100_cells_mjidxs
        elif layout == self._L50:
            self._cells_mjidxs = self._L50_cells_mjidxs
        elif layout == self._L0:
            self._cells_mjidxs = self._L0_cells_mjidxs
        else:
            raise ValueError("The layout name is not correct!")
        # Set up the scene after determining the layout
        mujoco.mj_forward(self._model, self._data)

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

    def _sample_target(self):
        # Sample a target from the cells according to the target idx probability distribution
        new_sampled_target_mjidx = np.random.choice(self._cells_mjidxs.copy())

        # Make sure the sampled target is different from the previous one
        while True:
            if new_sampled_target_mjidx != self._sampled_target_mjidx:
                break
            else:
                new_sampled_target_mjidx = np.random.choice(self._cells_mjidxs.copy())
        self._sampled_target_mjidx = new_sampled_target_mjidx

        if self._config["rl"]["mode"] == "test":
            if self._num_trials <= self._max_trials - 1:
                self._sampled_target_mjidx = self._cells_mjidxs[self._num_trials]

        # Reset the counter
        self._on_target_steps = 0

        # Reset the scene
        for mjidx in self._cells_mjidxs:
            self._model.geom(mjidx).rgba = self._DFLT_RGBA
        self._model.geom(self._sampled_target_mjidx).rgba = self._VISUALIZE_RGBA

        # Update the number of remaining unread cells
        self._num_trials += 1
