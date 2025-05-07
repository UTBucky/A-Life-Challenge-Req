# A-Life Challenege
# Zhou, Prudent, Hagan
# Gene implementation

import numpy as np


class Gene:
    """Base class: stores a mutation_rate and enforces a mutate() API."""

    def __init__(self, mutation_rate: float):
        self.mutation_rate = mutation_rate

    def mutate(self):
        raise NotImplementedError("Subclasses must implement mutate()")


class MorphologicalGenes(Gene):
    """
    Five continuous floats in [0,1]: size, camouflage, defense, attack, vision.
    Stored as a length-5 numpy array.
    """

    def __init__(self, mutation_rate: float, values: np.ndarray = None):
        super().__init__(mutation_rate)
        # values order: [size, camouflage, defense, attack, vision]
        self.values = values if values is not None else np.random.rand(5)

    @property
    def size(self) -> float:
        return float(self.values[0])

    @property
    def camouflage(self) -> float:
        return float(self.values[1])

    @property
    def defense(self) -> float:
        return float(self.values[2])

    @property
    def attack(self) -> float:
        return float(self.values[3])

    @property
    def vision(self) -> float:
        return float(self.values[4])

    def mutate(self) -> "MorphologicalGenes":
        mask = np.random.rand(5) < self.mutation_rate
        new_vals = self.values.copy()
        # draw fresh randoms for each mutated slot
        new_vals[mask] = np.random.rand(mask.sum())
        return MorphologicalGenes(self.mutation_rate, new_vals)


class MetabolicGenes(Gene):
    """
    Numeric array of [metabolism_rate, nutrient_efficiency] ∈ [0,1],
    plus a categorical diet_type.
    """
    DIET_TYPES = np.array(
        ["Herb", "Omni", "Carn", "Photo", "Parasite"], dtype=object)

    def __init__(
        self,
        mutation_rate: float,
        numeric: np.ndarray = None,
        diet_type: str = None
    ):
        super().__init__(mutation_rate)
        # numeric[0]=metabolism_rate, numeric[1]=nutrient_efficiency
        self.numeric = numeric if numeric is not None else np.random.rand(2)
        self.diet_type = (
            diet_type
            if diet_type is not None
            else np.random.choice(self.DIET_TYPES)
        )

    @property
    def metabolism_rate(self) -> float:
        return float(self.numeric[0])

    @property
    def nutrient_efficiency(self) -> float:
        return float(self.numeric[1])

    def mutate(self) -> "MetabolicGenes":
        mask = np.random.rand(2) < self.mutation_rate
        new_num = self.numeric.copy()
        new_num[mask] = np.random.rand(mask.sum())

        if np.random.rand() < self.mutation_rate:
            new_diet = np.random.choice(self.DIET_TYPES)
        else:
            new_diet = self.diet_type

        return MetabolicGenes(self.mutation_rate, new_num, new_diet)


class ReproductionGenes(Gene):
    """
    Continuous fertility_rate ∈ [0,1], integer offspring_count ∈ [1,10],
    and categorical reproduction_type.
    """
    REPRO_TYPES = np.array(["Sexual", "Asexual"], dtype=object)

    def __init__(
        self,
        mutation_rate: float,
        fertility_rate: float = None,
        offspring_count: int = None,
        reproduction_type: str = None
    ):
        super().__init__(mutation_rate)
        self.fertility_rate = (
            fertility_rate
            if fertility_rate is not None
            else float(np.random.rand())
        )
        self.offspring_count = (
            offspring_count
            if offspring_count is not None
            else int(np.random.randint(1, 11))
        )
        self.reproduction_type = (
            reproduction_type
            if reproduction_type is not None
            else np.random.choice(self.REPRO_TYPES)
        )

    def mutate(self) -> "ReproductionGenes":
        # fertility float
        fr = (
            float(np.random.rand())
            if np.random.rand() < self.mutation_rate
            else self.fertility_rate
        )
        # offspring int
        oc = (
            int(np.random.randint(1, 11))
            if np.random.rand() < self.mutation_rate
            else self.offspring_count
        )
        # reproduction type
        rt = (
            np.random.choice(self.REPRO_TYPES)
            if np.random.rand() < self.mutation_rate
            else self.reproduction_type
        )
        return ReproductionGenes(self.mutation_rate, fr, oc, rt)


class BehavioralGenes(Gene):
    """
    Two booleans: pack_behavior, symbiotic.
    Stored as a length-2 numpy boolean array.
    """

    def __init__(self, mutation_rate: float, bools: np.ndarray = None):
        super().__init__(mutation_rate)
        # [pack_behavior, symbiotic]
        if bools is None:
            self.bools = np.random.choice(
                [False, True], size=2
            )
        else:
            self.bools = bools

    @property
    def pack_behavior(self) -> bool:
        return bool(self.bools[0])

    @property
    def symbiotic(self) -> bool:
        return bool(self.bools[1])

    def mutate(self) -> "BehavioralGenes":
        mask = np.random.rand(2) < self.mutation_rate
        new_b = self.bools.copy()
        new_b[mask] = ~new_b[mask]
        return BehavioralGenes(self.mutation_rate, new_b)


class LocomotionGenes(Gene):
    """
    Three booleans [swim, walk, fly] and one float speed ∈ [0,1].
    """

    def __init__(
        self,
        mutation_rate: float,
        bools: np.ndarray = None,
        speed: float = None
    ):
        super().__init__(mutation_rate)
        # bools = [swim, walk, fly]
        self.bools = (
            bools
            if bools is not None
            else np.random.choice([False, True], size=3)
        )
        self.speed = (
            speed
            if speed is not None
            else float(np.random.rand())
        )

    @property
    def swim(self) -> bool:
        return bool(self.bools[0])

    @property
    def walk(self) -> bool:
        return bool(self.bools[1])

    @property
    def fly(self) -> bool:
        return bool(self.bools[2])

    def mutate(self) -> "LocomotionGenes":
        mask = np.random.rand(3) < self.mutation_rate
        new_b = self.bools.copy()
        new_b[mask] = ~new_b[mask]

        new_speed = (
            float(np.random.rand())
            if np.random.rand() < self.mutation_rate
            else self.speed
        )

        return LocomotionGenes(self.mutation_rate, new_b, new_speed)


class Genome:
    """
    Holds one instance of each gene-category.
    replicate_genes() returns a fresh Genome with each sub-gene mutated.
    """

    def __init__(self, mutation_rate: float):
        self.mutation_rate = mutation_rate
        self.morph = MorphologicalGenes(mutation_rate)
        self.metabolic = MetabolicGenes(mutation_rate)
        self.reproduction = ReproductionGenes(mutation_rate)
        self.behavioral = BehavioralGenes(mutation_rate)
        self.locomotion = LocomotionGenes(mutation_rate)

    def get_morphological(self) -> MorphologicalGenes:
        """Return the MorphologicalGenes instance."""
        return self.morph

    def get_metabolic(self) -> MetabolicGenes:
        """Return the MetabolicGenes instance."""
        return self.metabolic

    def get_reproduction(self) -> ReproductionGenes:
        """Return the ReproductionGenes instance."""
        return self.reproduction

    def get_behavioral(self) -> BehavioralGenes:
        """Return the BehavioralGenes instance."""
        return self.behavioral

    def get_locomotion(self) -> LocomotionGenes:
        """Return the LocomotionGenes instance."""
        return self.locomotion

    def replicate_genes(self) -> "Genome":
        child = Genome(self.mutation_rate)
        child.morph = self.morph.mutate()
        child.metabolic = self.metabolic.mutate()
        child.reproduction = self.reproduction.mutate()
        child.behavioral = self.behavioral.mutate()
        child.locomotion = self.locomotion.mutate()
        return child
