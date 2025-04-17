# A-Life Challenege
# Zhou, Prudent, Hagan
# Environment implementation via grid

import numpy as np


class GridEnvironment:
    """
    GridEnvironment(1 arg)
    size: size of grid desired
    2 different np arrays of size (size x size) terrain and occupancy
    - filled with 0s - dtype uint8
    organisms are stored in a python list datastructure
    """

    def __init__(self, size):
        """
        np arrays for terrain and occupancy
        python list for organisms
        """
        self._size = size
        self.terrain = np.zeros((size, size), dtype=np.uint8)
        self.occupancy = np.zeros((size, size), dtype=bool)
        self._organisms = []  # List of organism instances
        self._generation = 0

    def add_organism(self, organism):
        """
        Environment method that takes 1 arg
        organism: inp custom class that contains genes and a method for moving
        r, c are rows/columns
        """
        r, c = organism.position
        # Checks created organism is in bounds of board
        # and not within the occupancy array
        if self.in_bounds(r, c) and not self.occupancy[r, c]:
            self._organisms.append(organism)
            self.occupancy[r, c] = True
            return True
        return False

    def in_bounds(self, r, c):
        """
        Checks organism is within the board size
        """
        return 0 <= r < self._size and 0 <= c < self._size

    def step(self):
        """
        Used to progress the environment simulation by 1 generation
        """
        # Build speed-sorted queue (descending order: faster goes first)
        # key=lambda o: o.speed >>> tells inbult python sorting method sorted()
        # to use the speed method to sort (this documentation created before
        # full build organism was created)
        move_queue = sorted(
            self._organisms, key=lambda o: o.speed, reverse=True
            )

        # Track tiles visited this step to avoid collisions
        # zeroes_like creates an np array initialized with 0s of same shape
        # as inputted array, in this case it's the occupancy array
        visited = np.zeros_like(self.occupancy, dtype=bool)

        # ---If possible avoid python for loop in future
        for organism in move_queue:
            r_old, c_old = organism.position

            # Temporarily free current tile
            self.occupancy[r_old, c_old] = False

            # Let organism decide its own move
            organism.move(self)  # <-- internal logic modifies .position

            r_new, c_new = organism.position

            if not self.in_bounds(r_new, c_new):
                organism.position = np.array([r_old, c_old])  # Revert
                self.occupancy[r_old, c_old] = True
                continue

            # Must land on unoccupied & unvisited
            if (
                not self.occupancy[r_new, c_new] and
                not visited[r_new, c_new]
            ):
                self.occupancy[r_new, c_new] = True
                visited[r_new, c_new] = True
            else:
                # Revert move
                organism.position = np.array([r_old, c_old])
                self.occupancy[r_old, c_old] = True
        self._generation += 1

    def get_generation(self):
        """
        Returns generation for display in pygames rendering
        """
        return self._generation

    def get_size(self):
        """
        Returns size for rendering in pygames draw method
        """
        return self._size

    def get_organisms(self):
        """
        Returns organisms for rendering in pygames
        """
        return self._organisms
