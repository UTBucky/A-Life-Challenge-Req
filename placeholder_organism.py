# Placeholder organism
import numpy as np

class DummyOrganism:
    """
    Placeholder ogranism,  datamembers accessed directly
    if you use get_methods for encapsulation
    environment code will need to change
    """
    def __init__(self, position, speed):
        self.position = np.array(position, dtype=int)
        self.speed = speed

    def move(self, env):
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        np.random.shuffle(deltas)
        for dr, dc in deltas:
            r_new = self.position[0] + dr
            c_new = self.position[1] + dc
            if env.in_bounds(r_new, c_new):
                self.position = np.array([r_new, c_new])
                break