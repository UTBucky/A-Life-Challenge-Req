import numpy as np
from typing import Tuple, Dict


def copy_valid_count( 
    spawned: np.ndarray, 
    valid_count: np.ndarray,
    species_arr,
    size_arr,
    camouflage_arr,
    defense_arr,
    attack_arr,
    vision_arr,
    metabolism_rate_arr,
    nutrient_efficiency_arr,
    diet_type_arr,
    fertility_rate_arr,
    offspring_count_arr,
    reproduction_type_arr,
    pack_behavior_arr,
    symbiotic_arr,
    swim_arr,
    walk_arr,
    fly_arr,
    speed_arr,
    energy_arr,
    ) -> np.ndarray:
    spawned['species']            = species_arr[:valid_count]
    spawned['size']               = size_arr[:valid_count]
    spawned['camouflage']         = camouflage_arr[:valid_count]
    spawned['defense']            = defense_arr[:valid_count]
    spawned['attack']             = attack_arr[:valid_count]
    spawned['vision']             = vision_arr[:valid_count]
    spawned['metabolism_rate']    = metabolism_rate_arr[:valid_count]
    spawned['nutrient_efficiency']= nutrient_efficiency_arr[:valid_count]
    spawned['diet_type']          = diet_type_arr[:valid_count]
    spawned['fertility_rate']     = fertility_rate_arr[:valid_count]
    spawned['offspring_count']    = offspring_count_arr[:valid_count]
    spawned['reproduction_type']  = reproduction_type_arr[:valid_count]
    spawned['pack_behavior']      = pack_behavior_arr[:valid_count]
    spawned['symbiotic']          = symbiotic_arr[:valid_count]
    spawned['swim']               = swim_arr[:valid_count]
    spawned['walk']               = walk_arr[:valid_count]
    spawned['fly']                = fly_arr[:valid_count]
    spawned['speed']              = speed_arr[:valid_count]
    if energy_arr.any():
        spawned['energy']             = energy_arr[:valid_count]
    return spawned


def copy_parent_fields( 
    parents: np.ndarray, 
    offspring: np.ndarray,
    ) -> np.ndarray:
    offspring['species']           = parents['species']
    offspring['size']              = parents['size']
    offspring['camouflage']        = parents['camouflage']
    offspring['defense']           = parents['defense']
    offspring['attack']            = parents['attack']
    offspring['vision']            = parents['vision']
    offspring['metabolism_rate']   = parents['metabolism_rate']
    offspring['nutrient_efficiency']= parents['nutrient_efficiency']
    offspring['diet_type']         = parents['diet_type']
    offspring['fertility_rate']    = parents['fertility_rate']
    offspring['offspring_count']   = parents['offspring_count']
    offspring['reproduction_type'] = parents['reproduction_type']
    offspring['pack_behavior']     = parents['pack_behavior']
    offspring['symbiotic']         = parents['symbiotic']
    offspring['swim']              = parents['swim']
    offspring['walk']              = parents['walk']
    offspring['fly']               = parents['fly']
    offspring['speed']             = parents['speed']
    return 


def mutate_offspring(
    offspring, 
    flip_mask,
    gene_pool,
    m) -> np.ndarray:
    offspring['size'][flip_mask]               = np.random.uniform(
                                                    low=gene_pool['size'][0],
                                                    high=gene_pool['size'][1],
                                                    size=m
                                                ).astype(np.float32)

    offspring['camouflage'][flip_mask]         = np.random.uniform(
                                                    low=gene_pool['camouflage'][0],
                                                    high=gene_pool['camouflage'][1],
                                                    size=m
                                                ).astype(np.float32)

    offspring['defense'][flip_mask]            = np.random.uniform(
                                                    low=gene_pool['defense'][0],
                                                    high=gene_pool['defense'][1],
                                                    size=m
                                                ).astype(np.float32)

    offspring['attack'][flip_mask]             = np.random.uniform(
                                                    low=gene_pool['attack'][0],
                                                    high=gene_pool['attack'][1],
                                                    size=m
                                                ).astype(np.float32)

    offspring['vision'][flip_mask]             = np.random.uniform(
                                                    low=gene_pool['vision'][0],
                                                    high=gene_pool['vision'][1],
                                                    size=m
                                                ).astype(np.float32)

    offspring['metabolism_rate'][flip_mask]    = np.random.uniform(
                                                    low=gene_pool['metabolism_rate'][0],
                                                    high=gene_pool['metabolism_rate'][1],
                                                    size=m
                                                ).astype(np.float32)

    offspring['nutrient_efficiency'][flip_mask]= np.random.uniform(
                                                    low=gene_pool['nutrient_efficiency'][0],
                                                    high=gene_pool['nutrient_efficiency'][1],
                                                    size=m
                                                ).astype(np.float32)

    offspring['diet_type'][flip_mask]          = np.random.choice(
                                                    gene_pool['diet_type'],
                                                    size=m
                                                ).astype(np.str_)

    offspring['fertility_rate'][flip_mask]     = np.random.uniform(
                                                    low=gene_pool['fertility_rate'][0],
                                                    high=gene_pool['fertility_rate'][1],
                                                    size=m
                                                ).astype(np.float32)

    offspring['offspring_count'][flip_mask]    = np.random.randint(
                                                    gene_pool['offspring_count'][0],
                                                    gene_pool['offspring_count'][1] + 1,
                                                    size=m
                                                ).astype(np.int32)

    offspring['reproduction_type'][flip_mask]  = np.random.choice(
                                                    gene_pool['reproduction_type'],
                                                    size=m
                                                ).astype(np.str_)

    offspring['pack_behavior'][flip_mask]      = np.random.choice(
                                                    gene_pool['pack_behavior'],
                                                    size=m
                                                ).astype(np.bool_)

    offspring['symbiotic'][flip_mask]          = np.random.choice(
                                                    gene_pool['symbiotic'],
                                                    size=m
                                                ).astype(np.bool_)

    offspring['swim'][flip_mask]               = np.random.choice(
                                                    gene_pool['swim'],
                                                    size=m
                                                ).astype(np.bool_)

    offspring['walk'][flip_mask]               = np.random.choice(
                                                    gene_pool['walk'],
                                                    size=m
                                                ).astype(np.bool_)

    offspring['fly'][flip_mask]                = np.random.choice(
                                                    gene_pool['fly'],
                                                    size=m
                                                ).astype(np.bool_)

    offspring['speed'][flip_mask]              = np.random.uniform(
                                                    low=gene_pool['speed'][0],
                                                    high=gene_pool['speed'][1],
                                                    size=m
                                                ).astype(np.float32)


def initialize_default_traits(
    n: int,
    gene_pool: Dict[str, list]
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Create default (non-randomized) trait arrays for n organisms,
    using gene_pool for any categorical defaults.
    Returns in order:
        species_arr, size_arr, camouflage_arr, defense_arr, attack_arr,
        vision_arr, metabolism_rate_arr, nutrient_efficiency_arr,
        diet_type_arr, fertility_rate_arr, offspring_count_arr,
        reproduction_type_arr, pack_behavior_arr, symbiotic_arr,
        swim_arr, walk_arr, fly_arr, speed_arr, energy_arr
    """
    species_arr            = np.full((n,), "ORG", dtype=np.str_)
    size_arr               = np.full((n,), 1.0, dtype=np.float32)
    camouflage_arr         = np.zeros((n,),   dtype=np.float32)
    defense_arr            = np.zeros((n,),   dtype=np.float32)
    attack_arr             = np.zeros((n,),   dtype=np.float32)
    # or based on environment scale
    vision_arr             = np.full((n,), 15,  dtype=np.float32)
    metabolism_rate_arr    = np.full((n,), 1.0, dtype=np.float32)
    nutrient_efficiency_arr= np.full((n,), 1.0, dtype=np.float32)
    diet_type_arr          = np.full(
                                (n,),
                                gene_pool['diet_type'][0],
                                dtype=np.str_
                            )
    fertility_rate_arr     = np.full((n,), 0.1, dtype=np.float32)
    offspring_count_arr    = np.full((n,), 1,   dtype=np.int32)
    reproduction_type_arr  = np.full(
                                (n,),
                                gene_pool['reproduction_type'][0],
                                dtype=np.str_
                            )
    pack_behavior_arr      = np.full((n,), False, dtype=np.bool_)
    symbiotic_arr          = np.full((n,), False, dtype=np.bool_)
    swim_arr               = np.full((n,), False, dtype=np.bool_)
    walk_arr               = np.full((n,), True,  dtype=np.bool_)
    fly_arr                = np.full((n,), False, dtype=np.bool_)
    speed_arr              = np.full((n,), 1.0, dtype=np.float32)
    energy_arr             = np.full((n,), 20,  dtype=np.float32)
    #
    # Return in a tuple is ok because it's a wrapping of pointers
    #
    # Under the hood in python multicasting is construction and 
    # deconstruction of tuples anyways
    #
    return (
        species_arr, size_arr, camouflage_arr, defense_arr, attack_arr,
        vision_arr, metabolism_rate_arr, nutrient_efficiency_arr,
        diet_type_arr, fertility_rate_arr, offspring_count_arr,
        reproduction_type_arr, pack_behavior_arr, symbiotic_arr,
        swim_arr, walk_arr, fly_arr, speed_arr, energy_arr
    )


def initialize_random_traits(
    n: int,
    gene_pool: Dict[str, list]
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Create randomized trait arrays for n organisms based on gene_pool.
    Returns in order:
      species_arr, size_arr, camouflage_arr, defense_arr, attack_arr,
      vision_arr, metabolism_rate_arr, nutrient_efficiency_arr,
      diet_type_arr, fertility_rate_arr, offspring_count_arr,
      reproduction_type_arr, pack_behavior_arr, symbiotic_arr,
      swim_arr, walk_arr, fly_arr, speed_arr, energy_arr
    """
    # species label
    species_arr = np.full((n,), "ORG", dtype=np.str_)

    # helper for uniform floats
    def uni(key):
        lo, hi = gene_pool[key]
        return np.random.uniform(lo, hi, size=(n,)).astype(np.float32)

    # — MorphologicalGenes —
    size_arr        = uni('size')
    camouflage_arr  = uni('camouflage')
    defense_arr     = uni('defense')
    attack_arr      = uni('attack')
    vision_arr      = uni('vision')

    # — MetabolicGenes —
    metabolism_rate_arr    = uni('metabolism_rate')
    nutrient_efficiency_arr= uni('nutrient_efficiency')

    # diet choice with fixed probabilities
    p = [0.50, 0.20, 0.20, 0.05, 0.05]
    diet_type_arr = np.random.choice(
        gene_pool['diet_type'],
        size=n,
        p=p
    ).astype(np.str_)

    # — ReproductionGenes —
    fertility_rate_arr = uni('fertility_rate')
    offspring_count_arr = np.random.randint(
        gene_pool['offspring_count'][0],
        gene_pool['offspring_count'][1] + 1,
        size=(n,)
    ).astype(np.int32)
    reproduction_type_arr = np.random.choice(
        gene_pool['reproduction_type'],
        size=n
    ).astype(np.str_)

    # — BehavioralGenes —
    pack_behavior_arr = np.random.choice(
        gene_pool['pack_behavior'], size=n
    ).astype(bool)
    symbiotic_arr     = np.random.choice(
        gene_pool['symbiotic'],     size=n
    ).astype(bool)

    # — LocomotionGenes —
    swim_arr = np.random.choice(
        gene_pool['swim'], size=n
    ).astype(bool)
    walk_arr = np.random.choice(
        gene_pool['walk'], size=n
    ).astype(bool)
    fly_arr  = np.random.choice(
        gene_pool['fly'],  size=n
    ).astype(bool)

    # speed & energy
    speed_arr  = uni('speed')
    energy_arr = np.random.uniform(10, 30, size=(n,)).astype(np.float32)

    return (
        species_arr, size_arr, camouflage_arr, defense_arr, attack_arr,
        vision_arr, metabolism_rate_arr, nutrient_efficiency_arr,
        diet_type_arr, fertility_rate_arr, offspring_count_arr,
        reproduction_type_arr, pack_behavior_arr, symbiotic_arr,
        swim_arr, walk_arr, fly_arr, speed_arr, energy_arr
    )