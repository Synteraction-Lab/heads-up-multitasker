import numpy as np
import mujoco
import xml.etree.ElementTree as ET

from .reward_functions import NegativeExpDistanceWithHitBonus
from ..base import BaseTask


class ChoiceReaction(BaseTask):

  def __init__(self, model, data, end_effector, shoulder, **kwargs):
    super().__init__(model, data, **kwargs)

    # This task requires an end-effector and shoulder to be defined
    assert end_effector[0] == "geom", "end-effector must be a geom because contacts are between geoms"
    self._end_effector = end_effector[1]
    self._shoulder = shoulder

    # Get buttons
    self._buttons = [f"button-{idx}" for idx in range(4)]
    self._current_button = self._buttons[0]

    # Use early termination if target is not hit in time
    self._steps_since_last_hit = 0
    self._max_steps_without_hit = self._action_sample_freq*4

    # Define a maximum number of button presses
    self._trial_idx = 0
    self._max_trials = kwargs.get('max_trials', 10)
    self._targets_hit = 0

    # Used for logging states
    self._info = {"target_hit": False, "new_button_generated": False, "finished": False, "termination": False}

    # Define a default reward function
    self._reward_function = NegativeExpDistanceWithHitBonus()

    # Do a forward step so stuff like geom and body positions are calculated
    mujoco.mj_forward(model, data)

    # Get shoulder position
    shoulder_pos = getattr(data, self._shoulder[0])(self._shoulder[1]).xpos.copy()

    # Update button positions
    model.body("button-0").pos = shoulder_pos + [0.41, -0.07, -0.15]
    model.body("button-1").pos = shoulder_pos + [0.41, 0.07, -0.15]
    model.body("button-2").pos = shoulder_pos + [0.5, -0.07, -0.05]
    model.body("button-3").pos = shoulder_pos + [0.5, 0.07, -0.05]

    # Set camera angle
    model.cam_pos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'for_testing')] = np.array([1.1, -0.9, 0.95])
    model.cam_quat[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'for_testing')] = np.array(
      [0.6582, 0.6577, 0.2590, 0.2588])
    #model.cam_pos[model.camera_name2id('for_testing')] = np.array([-0.8, -0.6, 1.5])
    #model.cam_quat[model.camera_name2id('for_testing')] = np.array(
    #  [0.718027, 0.4371043, -0.31987, -0.4371043])

  @classmethod
  def initialise(cls, task_kwargs):

    assert "end_effector" in task_kwargs, "End-effector must be defined for this environment"
    end_effector = task_kwargs["end_effector"][1]

    # Parse xml file
    tree = ET.parse(cls.get_xml_file())
    root = tree.getroot()

    # Add contact between end-effector and buttons
    if root.find('contact') is None:
      root.append(ET.Element('contact'))
    for idx in range(4):
      root.find('contact').append(ET.Element('pair', name=f"ee-button-{idx}", geom1=end_effector,
                                             geom2=f"button-{idx}"))

    return tree

  def _update(self, model, data):

    # Set defaults
    finished = False
    self._info["new_button_generated"] = False

    # Check if the correct button has been pressed with suitable force
    force = data.sensor(self._current_button).data

    if 50 > force > 25:
      self._info["target_hit"] = True
      self._trial_idx += 1
      self._targets_hit += 1
      self._steps_since_last_hit = 0
      self._choose_button(model, data)
      self._info["new_button_generated"] = True

    else:

      self._info["target_hit"] = False

      # Check if time limit has been reached
      self._steps_since_last_hit += 1
      if self._steps_since_last_hit >= self._max_steps_without_hit:
        # Choose a new button
        self._steps_since_last_hit = 0
        self._trial_idx += 1
        self._choose_button(model, data)
        self._info["new_button_generated"] = True

    # Check if max number trials reached
    if self._trial_idx >= self._max_trials:
      finished = True
      self._info["termination"] = "max_trials_reached"

    # Get end-effector position and target position
    ee_position = data.geom(self._end_effector).xpos
    target_position = data.geom(self._current_button).xpos

    # Distance to target
    dist = np.linalg.norm(target_position - ee_position)

    # Calculate reward
    reward = self._reward_function.get(self, dist, self._info.copy())

    return reward, finished, self._info.copy()

  def _choose_button(self, model, data):

    # Choose a new button randomly, but don't choose the same button as previous one
    while True:
      new_button = self._rng.choice(self._buttons)
      if new_button != self._current_button:
        self._current_button = new_button
        break

    # Set color of screen
    model.geom("screen").rgba = model.geom(self._current_button).rgba

    mujoco.mj_forward(model, data)

  def _get_state(self, model, data):
    state = dict()
    state["current_button"] = self._current_button
    state["trial_idx"] = self._trial_idx
    state["targets_hit"] = self._targets_hit
    state.update(self._info)
    return state

  def _reset(self, model, data):

    # Reset counters
    self._steps_since_last_hit = 0
    self._trial_idx = 0
    self._targets_hit = 0

    self._info = {"target_hit": False, "new_button_generated": False, "finished": False, "termination": False}

    # Choose a new button
    self._choose_button(model, data)

  def get_stateful_information(self, model, data):
    # Time features
    targets_hit = -1.0 + 2*(self._trial_idx/self._max_trials)
    return np.array([targets_hit])
