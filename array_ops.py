import numpy as np
from scipy.spatial import cKDTree
from typing import Tuple, Dict

# Terrain avoidance constants
WATER_PUSH = 1.0
LAND_PUSH = 1.0

# Pack behavior constants
SEPARATION_WEIGHT = 2
SEPARATION_RADIUS = 1

#In-Place mutator
def copy_valid_count( 
    spawned:                    np.ndarray,
    valid_count:                np.ndarray,
    species_arr:                np.ndarray,
    size_arr:                   np.ndarray,
    camouflage_arr:             np.ndarray,
    defense_arr:                np.ndarray,
    attack_arr:                 np.ndarray,
    vision_arr:                 np.ndarray,
    metabolism_rate_arr:        np.ndarray,
    nutrient_efficiency_arr:    np.ndarray,
    diet_type_arr:              np.ndarray,
    fertility_rate_arr:         np.ndarray,
    offspring_count_arr:        np.ndarray,
    reproduction_type_arr:      np.ndarray,
    pack_behavior_arr:          np.ndarray,
    symbiotic_arr:              np.ndarray,
    swim_arr:                   np.ndarray,
    walk_arr:                   np.ndarray,
    fly_arr:                    np.ndarray,
    speed_arr:                  np.ndarray,
    energy_arr:                 np.ndarray,
    ) -> np.ndarray:
    """
    Sets the spawned founding organism attributes with the values generated in the inputted
    arrays. It only sets the attributes for the organisms that are within the valid_count mask.
    Parameters:
    -----------
        Spawned: np.ndarray
        species_arr: np.ndarray
        size_arr: np.ndarray
        camouflage_arr: np.ndarray
        defense_arr: np.ndarray
        attack_arr: np.ndarray
        vision_arr: np.ndarray
        metabolism_rate_arr: np.ndarray
        nutrient_efficiency_arr: np.ndarray
        diet_type_arr: np.ndarray
        fertility_rate_arr: np.ndarray
        offspring_count_arr: np.ndarray
        reproduction_type_arr: np.ndarray
        pack_behavior_arr: np.ndarray
        symbiotic_arr: np.ndarray
        swim_arr: np.ndarray
        walk_arr: np.ndarray
        fly_arr: np.ndarray
        speed_arr: np.ndarray
        energy_arr: np.ndarray
        
        
    """
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
    spawned['generation']         = np.full(valid_count, 0).astype(np.int32)
    if energy_arr.any():
        spawned['energy']             = energy_arr[:valid_count]
    return spawned

# In-Place mutator
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
    offspring['generation']        = parents['generation'] + 1
    return 

# In-Place mutator 
def mutate_offspring(
    offspring, 
    flip_mask,
    gene_pool,
    m) -> np.ndarray:

    # ===================== Continuous Genes =====================
    offspring['size'][flip_mask]               += np.random.uniform(
                                                    low=-0.1,
                                                    high=0.1,
                                                    size=m
                                                ).astype(np.float32)

    offspring['camouflage'][flip_mask]         += np.random.uniform(
                                                    low=1,
                                                    high=2,
                                                    size=m
                                                ).astype(np.float32)

    offspring['defense'][flip_mask]            += np.random.uniform(
                                                    low=-0.1,
                                                    high=0.3,
                                                    size=m
                                                ).astype(np.float32)

    offspring['attack'][flip_mask]             += np.random.uniform(
                                                    low=-0.1,
                                                    high=0.5,
                                                    size=m
                                                ).astype(np.float32)

    offspring['vision'][flip_mask]             += np.random.uniform(
                                                    low=-2,
                                                    high=2,
                                                    size=m
                                                ).astype(np.float32)

    offspring['metabolism_rate'][flip_mask]    += np.random.uniform(
                                                    low=-0.1,
                                                    high=0.2,
                                                    size=m
                                                ).astype(np.float32)

    offspring['nutrient_efficiency'][flip_mask]+= np.random.uniform(
                                                    low=-0.1,
                                                    high=0.2,
                                                    size=m
                                                ).astype(np.float32)

    offspring['fertility_rate'][flip_mask]     += np.random.uniform(
                                                    low=-0.1,
                                                    high=0.2,
                                                    size=m
                                                ).astype(np.float32)

    offspring['speed'][flip_mask]              += np.random.uniform(
                                                    low=-0.1,
                                                    high=0.2,
                                                    size=m
                                                ).astype(np.float32)


    # ---------------- Behavioral Genes ----------------
    pack_behavior_prob_arr = np.array([0.99, 0.01])       # False, True
    symbiotic_prob_arr = np.array([0.5, 0.5])           # False, True

    # ---------------- Locomotion Genes ----------------
    swim_prob_arr = np.array([0.01, 0.99])              # Mostly don't swim, small chance to swim
    fly_prob_arr = np.array([0.999, 0.001])               # Mostly don't fly, small chance to fly
    walk_prob_arr = np.array([0.99, 0.01])                # Mostly walk, small chance not to

    # ---------------- Diet Type ----------------
    diet_type_prob_arr = np.array([0.50, 0.20, 0.20, 0.05, 0.05])              # Herb, Omni, Carn, Photo, Parasite

    # ---------------- Reproduction Type ----------------
    reproduction_type_prob_arr = np.array([0.5, 0.5])  # Sexual vs Asexual

    # ===================== Discrete Genes =====================
    offspring['diet_type'][flip_mask]          = np.random.choice(
                                                    gene_pool['diet_type'],
                                                    size=m,
                                                    p=diet_type_prob_arr
                                                ).astype('U15')

    offspring['offspring_count'][flip_mask]    = np.random.randint(
                                                    gene_pool['offspring_count'][0],
                                                    gene_pool['offspring_count'][1] + 1,
                                                    size=m
                                                ).astype(np.int32)

    offspring['reproduction_type'][flip_mask]  = np.random.choice(
                                                    gene_pool['reproduction_type'],
                                                    size=m,
                                                    p=reproduction_type_prob_arr
                                                ).astype('U15')

    offspring['pack_behavior'][flip_mask]      = np.random.choice(
                                                    gene_pool['pack_behavior'],
                                                    size=m,
                                                    p=pack_behavior_prob_arr
                                                ).astype(np.bool_)

    offspring['symbiotic'][flip_mask]          = np.random.choice(
                                                    gene_pool['symbiotic'],
                                                    size=m,
                                                    p=symbiotic_prob_arr
                                                ).astype(np.bool_)

    offspring['swim'][flip_mask]               = np.random.choice(
                                                    gene_pool['swim'],
                                                    size=m,
                                                    p=swim_prob_arr
                                                ).astype(np.bool_)

    offspring['walk'][flip_mask]               = np.random.choice(
                                                    gene_pool['walk'],
                                                    size=m,
                                                    p=walk_prob_arr
                                                ).astype(np.bool_)

    offspring['fly'][flip_mask]                = np.random.choice(
                                                    gene_pool['fly'],
                                                    size=m,
                                                    p=fly_prob_arr
                                                ).astype(np.bool_)

# Returns new arrays
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
    if n <= 0:
        raise ValueError("Number of traits 'n' must be greater than 0.")

    species_arr            = np.full((n,), "ORG", dtype='U15')
    size_arr               = np.full((n,), 1.0, dtype=np.float32)
    camouflage_arr         = np.zeros((n,),   dtype=np.float32)
    defense_arr            = np.zeros((n,),   dtype=np.float32)
    attack_arr             = np.zeros((n,),   dtype=np.float32)
    # or based on environment scale
    vision_arr             = np.full((n,), 30,  dtype=np.float32)
    metabolism_rate_arr    = np.full((n,), 1.0, dtype=np.float32)
    nutrient_efficiency_arr= np.full((n,), 1.0, dtype=np.float32)
    diet_type_arr          = np.full(
                                (n,),
                                gene_pool['diet_type'][0],
                                dtype='U15'
                            )
    fertility_rate_arr     = np.full((n,), 0.1, dtype=np.float32)
    offspring_count_arr    = np.full((n,), 1,   dtype=np.int32)
    reproduction_type_arr  = np.full(
                                (n,),
                                gene_pool['reproduction_type'][0],
                                dtype='U15'
                            )
    pack_behavior_arr      = np.full((n,), False, dtype=np.bool_)
    symbiotic_arr          = np.full((n,), False, dtype=np.bool_)
    swim_arr               = np.full((n,), True, dtype=np.bool_)
    walk_arr               = np.full((n,), False,  dtype=np.bool_)
    fly_arr                = np.full((n,), False, dtype=np.bool_)
    speed_arr              = np.full((n,), 2, dtype=np.float32)
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

# Returns new arrays
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
    Returns:
    --------
    In order within a tuple:
    - species_arr              : Species label (default "ORG").
    - size_arr                 : Organism size.
    - camouflage_arr           : Camouflage effectiveness.
    - defense_arr              : Defensive capabilities.
    - attack_arr               : Attack strength.
    - vision_arr               : Vision radius.
    - metabolism_rate_arr      : Rate of metabolic consumption.
    - nutrient_efficiency_arr  : Efficiency in processing food.
    - diet_type_arr            : Type of diet.
    - fertility_rate_arr       : Rate of fertility.
    - offspring_count_arr      : Number of offspring per reproduction.
    - reproduction_type_arr    : Method of reproduction.
    - pack_behavior_arr        : Whether it displays pack behavior.
    - symbiotic_arr            : Whether it exhibits symbiotic behavior.
    - swim_arr                 : Capability to swim.
    - walk_arr                 : Capability to walk.
    - fly_arr                  : Capability to fly.
    - speed_arr                : Movement speed.
    - energy_arr               : Initial energy value.
    """
    if n <= 0:
        raise ValueError("Number of traits 'n' must be greater than 0.")

    # species label
    species_arr = np.full((n,), "ORG", dtype='U15')

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
    # Herb, Omni, Carn, Photo, Parasite
    diet_probs = [0.50, 0.20, 0.20, 0.05, 0.05]
    diet_type_arr               = np.random.choice(
                                        gene_pool['diet_type'],
                                        size=n,
                                        p=diet_probs
                                ).astype('U15')


 # ---------------- Reproduction Genes ----------------
    fertility_rate_arr          = uni('fertility_rate')
    # TODO Implement multi_offspring creation
    offspring_count_arr         = np.random.randint(
                                      low=gene_pool['offspring_count'][0],
                                      high=gene_pool['offspring_count'][1] + 1,
                                      size=n
                                ).astype(np.int32)
    # TODO Implement sexual reproduction
    reproduction_type_arr       = np.random.choice(
                                      gene_pool['reproduction_type'],
                                      size=n
                                ).astype('U15')

    # ---------------- Behavioral Genes ----------------
    pack_probs = [0.99, 0.01]
    pack_behavior_arr           = np.random.choice(
                                      gene_pool['pack_behavior'],
                                      size=n,
                                      p=pack_probs
                                ).astype(bool)
    # TODO Implement symbiotic energy distribution
    symbiotic_arr               = np.random.choice(
                                      gene_pool['symbiotic'],
                                      size=n
                                ).astype(bool)

    # ---------------- Locomotion Genes ----------------
    swim_probs = [0.01, 0.99]
    swim_arr                    = np.random.choice(
                                      gene_pool['swim'],
                                      size=n,
                                      p=swim_probs
                                ).astype(bool)

    walk_probs = [0.99, 0.01]    
    walk_arr                    = np.random.choice(
                                      gene_pool['walk'],
                                      size=n,
                                      p=walk_probs
                                ).astype(bool)

    fly_probs = [0.999,0.001]
    fly_arr                     = np.random.choice(
                                      gene_pool['fly'],
                                      size=n,
                                      p=fly_probs
                                ).astype(bool)

    # ---------------- Speed & Initial Energy ----------------
    speed_arr  = uni('speed')
    energy_arr = np.random.uniform(10, 30, size=(n,)).astype(np.float32)

    return (
        species_arr, size_arr, camouflage_arr, defense_arr, attack_arr,
        vision_arr, metabolism_rate_arr, nutrient_efficiency_arr,
        diet_type_arr, fertility_rate_arr, offspring_count_arr,
        reproduction_type_arr, pack_behavior_arr, symbiotic_arr,
        swim_arr, walk_arr, fly_arr, speed_arr, energy_arr
    )

# Returns new arrays
def calculate_valid_founder_terrain(
    env_terrain: np.ndarray,
    swim_arr: np.ndarray,
    walk_arr: np.ndarray,
    fly_arr: np.ndarray,
    env_width: int,
    n: int
) -> Tuple[np.ndarray, int]:
    """
    Generate n random candidate positions in a square world of width `env_width`,
    then filter them by locomotion capabilities against the terrain.

    Parameters:
    -----------
    env_terrain : np.ndarray
        2D array of shape (env_height, env_width) giving terrain heights.
        Negative values = water, non-negative = land.
    swim_arr : np.ndarray of bool
        Boolean mask of length n; True if the organism can swim.
    walk_arr : np.ndarray of bool
        Boolean mask of length n; True if the organism can walk.
    fly_arr : np.ndarray of bool
        Boolean mask of length n; True if the organism can fly.
    env_width : int
        Width (and height) of the square environment.
    n : int
        Number of candidate positions to sample.

    Returns:
    --------
    positions : np.ndarray of shape (valid_count, 2)
        Concatenated array of valid [x, y] positions for each locomotion type.
    valid_count : int
        Total number of valid positions.
    """
    if n <= 0:
        raise ValueError("Number of traits 'n' must be greater than 0.")
    if env_width < 200:
        raise ValueError("Environment width must be at least 200.")
    # 1) Sample random candidate positions
    positions = np.random.randint(0, env_width, size=(n, 2)).astype(np.int32)

    # 2) Lookup terrain at each candidate
    ix = positions[:, 0]
    iy = positions[:, 1]
    terrain_values = env_terrain[iy, ix]

    # 3) Build locomotion‐only masks
    swim_only = swim_arr & ~walk_arr & ~fly_arr
    walk_only = walk_arr & ~swim_arr & ~fly_arr
    # fly_arr itself indicates flyers

    # 4) Filter out invalid positions by locomotion & terrain
    valid_fly  = positions[fly_arr]
    valid_swim = positions[swim_only & (terrain_values <  0)]
    valid_walk = positions[walk_only & (terrain_values >= 0)]

    # 5) Concatenate all valid positions and count them
    positions = np.concatenate((valid_fly, valid_swim, valid_walk), axis=0)
    valid_count = positions.shape[0]

    return positions, valid_count

# Returns new arrays
def grab_move_arrays(
    organisms:np.ndarray
    ) -> Tuple[
        np.ndarray , np.ndarray , np.ndarray ,
        np.ndarray , np.ndarray , np.ndarray ,
        np.ndarray , np.ndarray , np.ndarray ,
        np.ndarray
    ]:
    """
    Grabs the necessary array items from the global organisms
    array. Compartmentalized into array_ops.py for readability.
    
    :Parameters:
    - organisms: array of organism_dtype objects, (np.ndarray)

    :Returns:
    diet_type - np.ndarray
    vision    - np.ndarray
    attack    - np.ndarray 
    defense   - np.ndarray 
    pack_flag - np.ndarray 
    species   - np.ndarray 
    fly_flag  - np.ndarray
    walk_flag - np.ndarray
    speed     - np.ndarray
    """
    diet_type = organisms['diet_type']
    vision = organisms['vision']
    attack = organisms['attack']
    defense = organisms['defense']
    pack_flag = organisms['pack_behavior']
    species = organisms['species']
    fly_flag = organisms['fly']
    swim_flag = organisms['swim']
    walk_flag = organisms['walk']
    speed = organisms['speed']

    return (diet_type, vision, attack, defense, pack_flag,
            species, fly_flag, swim_flag, walk_flag, speed)

# Returns new arrays
def get_coords_and_neighbors(                              
    orgs: np.ndarray,
    kdTree
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parameters:
    :param orgs: Pass in the global organisms array
    Returns:
    :coords: (n,2) dimensional numpy array of organism coordinates
    :neighbors: List of Lists of neighbors of organisms, based on cK-DTree query
    """
    coords = np.stack((orgs['x_pos'], orgs['y_pos']), axis=1)
    vision_radii = orgs['vision']
    neigh_lists = kdTree.query_ball_point(coords, vision_radii, workers=-1)

    return coords, neigh_lists


def movement_compute(
    organisms: np.ndarray,
    coords: np.ndarray, 
    neighs: np.ndarray, 
    width:  int, 
    length: int,
    avoid_land: np.ndarray, 
    avoid_water: np.ndarray
    ):
    (
    diet_type, vision, attack, defense, pack_flag, 
    species, fly_flag, swim_flag, walk_flag, speed
    ) = grab_move_arrays(organisms)

    avoidance_vec = np.zeros((organisms.shape[0], 2), dtype=np.float32)

    cannot_fly_swim_mask = (~fly_flag & ~swim_flag)
    cannot_fly_walk_mask = (~fly_flag & ~walk_flag)
    
    # === Apply Avoidance Logic Based on Masks ===
    # Only apply the terrain avoidance where the mask is True
    avoidance_vec[cannot_fly_swim_mask] += WATER_PUSH * avoid_water[cannot_fly_swim_mask]
    avoidance_vec[cannot_fly_walk_mask] += LAND_PUSH * avoid_land[cannot_fly_walk_mask]
    
    def _compute(orgs, index, pos, neighs, width, length, avoidance_arr):
        # pull out “my” values once
        my = orgs[index]
        my_diet = my['diet_type']
        my_cam = my['camouflage']
        my_att = my['attack']
        my_def = my['defense']
        my_spc = my['species']
        my_fly = my['fly']
        my_pack = pack_flag[index]
        my_speed = speed[index]
        if my_diet == 'Photo':
            my_def = 0
            my_att = 0
            move_vec = np.zeros(2, dtype=np.float32)
            return move_vec

        # make neighbors a NumPy array of ints
        neighs = np.asarray(neighs, dtype=int)

        # 1) camouflage filter
        mask_valid = (neighs != index) & (vision[neighs] >= my_cam)
        valid = neighs[mask_valid]

        # allocate movement accumulator
        move_vec = np.zeros(2, dtype=np.float32)

        # — social steering (non-pack) —
        if my_fly:
            pool = valid[fly_flag[valid]]
        else:
            pool = valid

        # assume `pool` is already valid subset
        my_net_pool    = my_att - defense[pool]
        their_net_pool = attack[pool] - my_def

        host_mask = their_net_pool > my_net_pool
        prey_mask = my_net_pool    > their_net_pool

        hostiles = pool[host_mask]
        prey     = pool[prey_mask]

        if hostiles.size > 0:
            move_vec += (pos - coords[hostiles]).mean(axis=0)
        if prey.size > 0:
            move_vec += (coords[prey] - pos).mean(axis=0)

        # crowd repulsion
        CROWD_PUSH = 0.001 * my_speed
        same_mask = species[valid] == my_spc
        same = valid[same_mask]
        if same.size > 0:
            repulse = np.mean(pos - coords[same], axis=0)
            move_vec += CROWD_PUSH * repulse
        
        #print(move_vec, 'pre move av array 3333')
        move_vec += avoidance_arr[index]
        #print(move_vec, 'post normal move av array 4444')
        # normalize & scale
        norm = np.linalg.norm(move_vec)
        step = (move_vec / norm) * \
            my_speed if norm > 0 else np.zeros(2, np.float32)

        new = pos + step
        new[0] = np.clip(new[0], 0, width - 1)
        new[1] = np.clip(new[1], 0, length - 1)
        return new

    return np.array(
        [
            _compute(
                organisms, 
                i, 
                coords[i], 
                neighs[i], 
                width, 
                length,
                avoidance_vec
            ) 
            for i in range(organisms.shape[0])
        ], 
        dtype=np.float32
    )

def pack_movement_compute(
    organisms: np.ndarray,
    coords: np.ndarray, 
    neighs: np.ndarray, 
    width:  int, 
    length: int,
    avoid_land: np.ndarray, 
    avoid_water: np.ndarray
    ):
    (
    diet_type, vision, attack, defense, pack_flag, 
    species, fly_flag, swim_flag, walk_flag, speed
    ) = grab_move_arrays(organisms)

    avoidance_vec = np.zeros((organisms.shape[0], 2), dtype=np.float32)

    cannot_fly_swim_mask = (~fly_flag & ~swim_flag)
    cannot_fly_walk_mask = (~fly_flag & ~walk_flag)
    
    # === Apply Avoidance Logic Based on Masks ===
    # Only apply the terrain avoidance where the mask is True
    avoidance_vec[cannot_fly_swim_mask] += WATER_PUSH * avoid_water[cannot_fly_swim_mask]
    avoidance_vec[cannot_fly_walk_mask] += LAND_PUSH * avoid_land[cannot_fly_walk_mask]

    def move_pack_behavior(orgs, index, pos, neighs, width, length, avoidance_arr):
        my = orgs[index]
        my_diet = my['diet_type']
        my_cam = my['camouflage']
        my_att = my['attack']
        my_def = my['defense']
        my_spc = my['species']
        my_fly = my['fly']
        my_pack = pack_flag[index]
        my_speed = speed[index]
        if my_diet == 'Photo':
            my_def = 0
            my_att = 0
            move_vec = np.zeros(2, dtype=np.float32)
            return move_vec

        neighs = np.asarray(neighs, dtype=int)
        mask_valid = (neighs != index) & (vision[neighs] >= my_cam)
        valid = neighs[mask_valid]

        # 2) pack_mates if pack_behavior array isn’t empty
        if pack_flag.shape[0] > 0:
            pack_mates = valid[pack_flag[valid]]

        # — behavioral overrides (pack) —
        if my_pack:

            # 1) compute net strengths against each neighbor in `valid`
            non_pack_mask = ~pack_flag[valid]       # True for neighbors that are NOT pack mates

            my_net    = my_att - defense[valid]     # our attack minus their defense
            their_net = attack[valid] - my_def      # their attack minus our defense

            # now require non-pack AND the appropriate net comparison
            host_mask = non_pack_mask & (their_net > my_net)     # if their net > our net → hostile
            prey_mask = non_pack_mask & (my_net    > their_net)  # if our net > their net → prey


            hostiles = valid[host_mask]
            if hostiles.size > 0:
                center = coords[hostiles].mean(axis=0)
                move_vec += (pos - center)
            else:
                prey = valid[prey_mask]
                if prey.size > 0:
                    center = coords[prey].mean(axis=0)
                    move_vec += (center - pos)
                else:
                    # c) cohesion + gentle separation
                    if pack_mates.size > 0:
                        center = coords[pack_mates].mean(axis=0)
                        move_vec += (center - pos)

                        dists = coords[pack_mates] - pos
                        norms = np.linalg.norm(dists, axis=1)
                        close = norms < SEPARATION_RADIUS
                        if close.any():
                            repulse = -np.mean(dists[close], axis=0)
                            move_vec += repulse * SEPARATION_WEIGHT

            #print(move_vec, 'pre pack move arr 1111')
            move_vec += avoidance_arr[index]
            #print(move_vec, 'post array of pack move 2222')

            # normalize & scale by speed
            norm = np.linalg.norm(move_vec)
            step = (move_vec / norm) * \
                my_speed if norm > 0 else np.zeros(2, np.float32)

            new = pos + step
            new[0] = np.clip(new[0], 0, width - 1)
            new[1] = np.clip(new[1], 0, length - 1)
            return new
    
    return np.array(
        [
            move_pack_behavior(
                organisms[pack_flag], 
                i, 
                coords[i], 
                neighs[i], 
                width, 
                length,
                avoidance_vec
            ) 
            for i in range(organisms.shape[0])
        ], 
        dtype=np.float32
    )
