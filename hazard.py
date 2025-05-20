import numpy as np
# Module for environmental hazards such as tornadoes and meteors

class Hazard:
    """Parent class for environmental hazards"""
    def __init__(self, radius, base_damage, movement, x_pos=None, y_pos=None):
        self._radius = radius
        self._base_damage = base_damage
        self._movement = movement
        self._x_pos = x_pos
        self._y_pos = y_pos

    def get_x_pos(self):
        return self._x_pos

    def get_y_pos(self):
        return self._y_pos

    def get_radius(self):
        return self._radius

    def get_base_damage(self):
        return self._base_damage

    def determine_random_location(self, env_width, env_height, terrain=None) -> tuple:
        """
        Determines a random location in x, y coordinates within environment bounds
        :param env: the environment object
        :return: a tuple (x, y)
        """
        if terrain is not None:
            valid_mask = terrain >= 0
            valid_indices = np.argwhere(valid_mask)
            if valid_indices.size == 0:
                # fallback: random anywhere
                self._x_pos = np.random.randint(0, env_width)
                self._y_pos = np.random.randint(0, env_height)
            else:
                idx = np.random.choice(valid_indices.shape[0])
                self._y_pos, self._x_pos = valid_indices[idx]  # note: row = y, col = x
        else:
            self._x_pos = np.random.randint(0, env_width)
            self._y_pos = np.random.randint(0, env_height)

        return self._x_pos, self._y_pos


class Meteor(Hazard):
    """Meteor class with a set radius and base damage."""
    def __init__(self, radius=20, base_damage=100, movement=False, x_pos=None, y_pos=None):
        super().__init__(radius=radius, base_damage=base_damage, movement=movement, x_pos=x_pos, y_pos=y_pos)




