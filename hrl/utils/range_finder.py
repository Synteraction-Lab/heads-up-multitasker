# # The distance detection
        # # Define the ray vector
        # p = self._data.site("rangefinder-site").xpos
        # direction_ = self._data.geom("fixate-point").xpos  # TODO this was not updated correctly
        # fixate_ray_len = abs(self._model.geom("fixate-point").pos[2])
        # projection_xy = abs(fixate_ray_len * math.cos(action[0]))
        # x = - projection_xy * math.sin(action[1])
        # y = projection_xy * math.cos(action[1])
        # z = fixate_ray_len * math.sin(action[0])
        # direction = np.array([x, y, z])
        # # Define the x-z plane equation
        # a, b, c = 0, 1, 0  # Normalize the vector of the x-z plane
        # dist_plane = - self._data.body("smart-glass-pane").xpos[1]  # Distance from origin to plane
        # # Calculate the intersection point
        # t = - (a * p[0] + b * p[1] + c * p[2] + dist_plane) / (a * direction[0] + b * direction[1] + c * direction[2])
        # intersection_xpos = p + t * direction
        #
        # geom2 = self._default_idx
        #
        # for idx in self._target_idxs:
        #     target_grid_xpos = self._data.geom(idx).xpos
        #     target_grid_size = self._model.geom(idx).size
        #     x_min = target_grid_xpos[0] - target_grid_size[0] / 2
        #     x_max = target_grid_xpos[0] + target_grid_size[0] / 2
        #     z_min = target_grid_xpos[2] - target_grid_size[2] / 2
        #     z_max = target_grid_xpos[2] + target_grid_size[2] / 2
        #
        #     if x_min <= intersection_xpos[0] <= x_max and z_min <= intersection_xpos[2] <= z_max:
        #         geom2 = idx
        #
        # if geom2 != self._target_idx:
        #     if geom2 in self._sequence_target_idxs and geom2 not in self._sequence_results_idxs:
        #         reward = 0.1
        #         # Update the environment
        #         acc = 0.8 / self._target_switch_interval
        #         self._model.geom(self._target_idx).rgba[0:3] = [x + y for x, y in zip(self._model.geom(self._target_idx).rgba[0:3], [0, 0, acc])]
        #         mujoco.mj_forward(self._model, self._data)