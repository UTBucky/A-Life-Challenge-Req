# A-Life Challenege
# Zhou, Prudent, Hagan
# Organism implementation

import numpy as np


class Organism:
    """
    Represents an organism entity. Stores information such as name, genome,
    age, stats, and needs. Organism contains methods to move and reproduce.
    """

    def __init__(self, name, genome, curr_age, position):
        """
        Initialize an organism object.

        :param name: A string
        :param genome: An object
        :param curr_age: An integer
        """

        self._name = name
        self._genome = genome
        self._position = np.array(position, dtype=int)
        self._curr_age = curr_age

        # Extract gene values from genome
        self._size = genome.get_genes()["size"]
        self._speed = genome.get_genes()["speed"]
        self._max_age = genome.get_genes()["max_age"]
        self._energy_capacity = genome.get_genes()["energy_capacity"]
        self._move_eff = genome.get_genes()["move_eff"]
        self._reproduction_eff = genome.get_genes()["reproduction_eff"]
        self._min_temp_tol = genome.get_genes()["min_temp_tol"]
        self._max_temp_tol = genome.get_genes()["max_temp_tol"]
        self._energy_gathering = genome.get_genes()["energy_gathering"]
        self._energy_prod = genome.get_genes()["energy_prod"]
        self._movement_affordance = genome.get_genes()["move_affordance"]

    # Get methods for each private data member
    def get_name(self):
        return self._name

    def get_genome(self):
        return self._genome

    def get_curr_age(self):
        return self._curr_age

    def get_max_age(self):
        return self._max_age

    def get_size(self):
        return self._size

    def get_speed(self):
        return self._speed

    def get_min_temp_tol(self):
        return self._min_temp_tol

    def get_max_temp_tol(self):
        return self._max_temp_tol

    def get_hunger(self):
        return self._hunger

    def get_position(self):
        return self._position

    # Set methods
    def set_hunger(self, new_level):
        self._hunger = new_level

    def set_position(self, array):
        self._position = array

    def set_curr_age(self, new_age):
        self._curr_age = new_age

    # Other methods
    def reproduce(self, child_position) -> object:
        """
        Creates a child organism based on the genome of the parent.

        :param child_position: A tuple
        """

        child_genome = self._genome.replicate()
        return Organism(self._name, child_genome, 0, child_position)

    def move(self, env):
        """
        Organism takes its turn, returning its desired action and new location

        :param env: GridEnvironment Object

        :return: A string that declares action to take
        """

        if self._hunger <= 0 or self._curr_age >= self._max_age:
            return "die"

        elif self._hunger >= self._size:
            return "reproduce"

        else:
            deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            np.random.shuffle(deltas)

            i = 0
            dr, dc = deltas[i]
            r_new = self._position[0] + dr
            c_new = self._position[1] + dc

            while not env.in_bounds(r_new, c_new) and i < len(deltas):
                i += 1
                dr, dc = deltas[i]
                r_new = self._position[0] + dr
                c_new = self._position[1] + dc

            self._position = np.array([r_new, c_new])

            return "move"
