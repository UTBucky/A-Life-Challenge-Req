# A-Life Challenege
# Zhou, Prudent, Hagan
# Gene implementation

import random


class Genome:

    def __init__(self, mutation_rate, genes):
        """
        Initialize a genome object.

        :param mutation_rate: A float
        :param genes: A dictionary of gene objects
        """

        self._mutation_rate = mutation_rate
        self._genes = genes

    def get_genes(self):
        return self._genes

    def replicate_genes(self) -> object:
        """
        Replicates genes during for organism reproduction,
        potentially causing mutations.
        """

        replicated_genome = {}

        for gene in self._genes:
            if random.random() > self._mutation_rate:
                replicated_genome[gene] = self._genes[gene]

            else:
                replicated_genome[gene] = self._genes[gene].mutate()

        child_genome = Genome(self._mutation_rate, replicated_genome)

        return child_genome


class Gene:
    """
    Represents a gene of an organism.
    """

    def __init__(self, name, val, min_val, max_val):
        """
        Initialize a gene object.

        :param name: A string
        :param val: An integer or float
        :param min_val: An integer or float
        :param max_val: An integer or float
        """

        self._name = name
        self._val = val
        self._min_val = min_val
        self._max_val = max_val

    def mutate(self) -> object:
        """
        Mutates the gene within the minimum and maximum value of the gene.
        """

        if self._val is int:
            new_val = random.randint(self._min_val, self._max_val)

        else:
            new_val = random.uniform(self._min_val, self._max_val)

        return Gene(self._name, new_val, self._min_val, self._max_val)

    # Get methods
    def get_name(self):
        return self._name

    def get_val(self):
        return self._val

    def get_min_val(self):
        return self._min_val

    def get_max_val(self):
        return self._max_val

    # Set methods
    def set_val(self, new_val):
        self._val = new_val


class EnergyGene(Gene):
    """
    Defines the energy gathering method of the organism.
    """

    def __init__(self, name, val, min_val, max_val, options):
        super().__init__(name, val, min_val, max_val)

        self._options = options

        if self._name not in self._options:
            raise ValueError(f"{self._name} is not a valid energy gene type.")

    def mutate(self) -> object:
        new_option = random.choice(self._options)
        new_val = random.uniform(self._min_val, self._max_val)
        return EnergyGene(new_option, new_val, self._min_val, self._max_val,
                          self._options)

    def get_options(self):
        return self._options


class MoveGene(Gene):
    """
    Defines movement affordances of organism.
    """

    def __init__(self, name, options, val=0, min_val=0, max_val=0):
        super().__init__(name, val, min_val, max_val)

        self._options = options

        if self._name not in self._options:
            raise ValueError(f"{self._name} is not a valid move gene type.")

    def get_options(self):
        return self._options

    def mutate(self) -> object:
        new_option = random.choice(self._options)
        return MoveGene(new_option, self._options)
