import numpy as np
import pygame
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
        self._start_x = None
        self._start_y = None
        self._progress = 0.0  # from 0.0 to 1.0
        self._landed = False
        self._trail = []  # stores recent positions for trail
        self._trail_max_len = 10

    def get_landed(self):
        return self._landed

    def launch_from(self, start_x, start_y):
        self._start_x = start_x
        self._start_y = start_y
        self._progress = 0.0
        self._landed = False

    def draw(self, surface, scale_x, scale_y, sidebar_width):
        if self._x_pos is None or self._y_pos is None:
            return

        # Animate flight
        if not self._landed:
            curr_x = (1 - self._progress) * self._start_x + self._progress * self._x_pos
            curr_y = (1 - self._progress) * self._start_y + self._progress * self._y_pos
            self._progress += 0.02
            if self._progress >= 1.0:
                self._progress = 1.0
                self._landed = True
        else:
            curr_x = self._x_pos
            curr_y = self._y_pos

        # Update trail
        self._trail.append((curr_x, curr_y))
        if len(self._trail) > self._trail_max_len:
            self._trail.pop(0)

        x = int(curr_x * scale_x) + sidebar_width
        y = int(curr_y * scale_y)
        radius = int(self._radius * ((scale_x + scale_y) / 2))

        for i, (tx, ty) in enumerate(self._trail):
            alpha = int(255 * (i + 1) / len(self._trail))  # fade from light to dark
            tx_px = int(tx * scale_x) + sidebar_width
            ty_px = int(ty * scale_y)
            trail_radius = max(2, int(self._radius * 0.2))  # small glow dot

            trail_surface = pygame.Surface((trail_radius * 2, trail_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(trail_surface, (255, 100, 0, alpha), (trail_radius, trail_radius), trail_radius)
            surface.blit(trail_surface, (tx_px - trail_radius, ty_px - trail_radius))

        # Draw jagged rocky appearance using polygon
        jagged_points = []
        segments = 50
        angle_step = 2 * np.pi / segments
        for i in range(segments):
            angle = i * angle_step
            noise = np.random.uniform(0.8, 1.2)
            r = radius * noise
            px = int(x + r * np.cos(angle))
            py = int(y + r * np.sin(angle))
            jagged_points.append((px, py))

        heat_color = (255, np.random.randint(90, 130), 0)
        if not self._landed:
            # Animated entry for meteor
            pygame.draw.polygon(surface, (255, 140, 0), jagged_points)
            pygame.draw.circle(surface, (255, 140, 0), (x, y), int(radius * 0.6))
        else:
            # Non-animated landed meteor
            crater_outer_radius = int(radius * 1.4)
            crater_inner_radius = int(radius * 1.2)

            # Draw crater ring (larger faded ring around the impact)
            pygame.draw.circle(surface, (50, 50, 50), (x, y), crater_outer_radius)
            pygame.draw.circle(surface, (30, 30, 30), (x, y), crater_inner_radius)

            # Draw the dark meteor body
            pygame.draw.circle(surface, (70, 70, 70), (x, y), int(radius * 0.6))





