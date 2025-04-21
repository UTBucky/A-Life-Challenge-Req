# A-Life Challenege
# Zhou, Prudent, Hagan
# Gene implementation

import random


class Gene:
    """
    Represents a gene of an organism.
    """

    def __init__(self, name, val, min_val, max_val):
        """
        Initialize a gene object.

        :param name: A string
        :param val: An integer
        :param min_val: An integer
        :param max_val: An integer
        """

        self._name = name
        self._val = val
        self._min_val = min_val
        self._max_val = max_val

    def mutate(self):
        """
        Mutates the gene within the minimum and maximum value of the gene.
        """

        self._val = random.randint(self._min_val, self._max_val)
