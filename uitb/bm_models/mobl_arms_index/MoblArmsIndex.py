from ..base import BaseBMModel

import numpy as np
import mujoco


class MoblArmsIndex(BaseBMModel):
  """This model is based on the MoBL ARMS model, see https://simtk.org/frs/?group_id=657 for the original model in OpenSim,
  and https://github.com/aikkala/O2MConverter for the MuJoCo converted model. This model is the same as the one in uitb/bm_models/mobl_arms, except
  the index finger is flexed and it contains a force sensor. """

  def __init__(self, model, data, **kwargs):
    super().__init__(model, data, **kwargs)

    # Set shoulder variant; use "none" as default, use "patch-v1" for a qualitatively more reasonable looking movements (not thoroughly tested)
    self.shoulder_variant = kwargs.get("shoulder_variant", "none")

  def _update(self, model, data):

    # Update shoulder equality constraints
    if self.shoulder_variant.startswith("patch"):
      model.equality("shoulder1_r2_con").data[1] = \
        -((np.pi - 2 * data.joint('shoulder_elv').qpos) / np.pi)

      if self.shoulder_variant == "patch-v2":
        data.joint('shoulder_rot').range[:] = \
          np.array([-np.pi / 2, np.pi / 9]) - \
          2 * np.min((data.joint('shoulder_elv').qpos,
                      np.pi - data.joint('shoulder_elv').qpos)) / np.pi \
          * data.joint('elv_angle').qpos

      # Do a forward calculation
      mujoco.mj_forward(model, data)

  @classmethod
  def _get_floor(cls):
    return None
