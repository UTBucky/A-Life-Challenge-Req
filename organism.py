import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import map_coordinates
import random
from array_ops import *
from collections import defaultdict
from lineage_tracker import *

# Terrain avoidance constants
WATER_PUSH = 5.0
LAND_PUSH = 5.0

#Ecosystem energy constant
ECO_ENERGY_CONST = 100

#Reproduction constant
REPRODUCTION_PROXIMITY_CONST = 15

class Organisms:
    """
    Represents all organisms in an environment.
    Keeps track of all organism's statistics.
    """
    ORGANISM_CLASS = np.dtype([
        # species label
        ('species',           np.str_,   15),

        # — MorphologicalGenes (size, camouflage, defense, attack, vision) —
        ('size',              np.float32),
        ('camouflage',        np.float32),
        ('defense',           np.float32),
        ('attack',            np.float32),
        ('vision',            np.float32),

        # — MetabolicGenes (metabolism_rate, nutrient_efficiency, diet_type) —
        ('metabolism_rate',   np.float32),
        ('nutrient_efficiency', np.float32),
        ('diet_type',         np.str_,   15),

        # — ReproductionGenes (fertility_rate, offspring_count, reproduction_type) —
        ('fertility_rate',    np.float32),
        ('offspring_count',   np.int32),
        ('reproduction_type', np.str_,   15),

        # — BehavioralGenes (pack_behavior, symbiotic) —
        ('pack_behavior',     np.bool_),
        ('symbiotic',         np.bool_),

        # — LocomotionGenes (swim, walk, fly, speed) —
        ('swim',              np.bool_),
        ('walk',              np.bool_),
        ('fly',               np.bool_),
        ('speed',             np.float32),

        # — Simulation bookkeeping —
        ('energy',            np.float32),
        ('x_pos',             np.float32),
        ('y_pos',             np.float32),
            # — Lineage tracking —
        ('p_id',                  np.int32),
        ('c_id',                  np.int32),
        ('generation',            np.int32),
    ])

    def __init__(self, env: object, mutation_rate, O_CLASS = ORGANISM_CLASS):
        """
        Initialize an organism object.

        :param env: 2D Simulation Environment object
        """

        #Organisms and Neighbor ds
        self._organism_dtype = O_CLASS
        self._pos_tree = None
        self._organisms = np.zeros((0,), dtype=self._organism_dtype)
        
        #Environment
        self._env = env
        self._width = env.get_width()
        self._length = env.get_length()
        
        #Genes
        self._mutation_rate = mutation_rate
        self._gene_pool = None
        
        # Lineage
        self._lineage_tracker = LineageTracker()
        self._speciation_dict = {}
        self._next_id = 0
        self._founders = None
        self._lineage_map: dict[int, list[int]] = defaultdict(list)
        # Garbage collection optimization - scratch buffers
        self._dirs = np.array([[1,0],[-1,0],[0,1],[0,-1]], np.float32)


    def load_genes(self, gene_pool):
        """
        Loads the gene pool dictionary.
        """
        self._gene_pool = gene_pool

    # Get methods
    def get_genes(self) -> dict:
        """
        Returns the gene pool dictionary.
        """
        return self._gene_pool

    def get_organisms(self) -> np.ndarray:
        """
        Returns the numpy array of organism_dtypes.
        """
        return self._organisms

    def get_speciation_dict(self) -> dict:
        """
        Returns the dictionary of speciation events.
        """
        return self._speciation_dict

    def get_lineage_tracker(self) -> LineageTracker:
        """
        Returns the LineageTracker object associated with tracking
        lineages of the associated Organisms object.
        """
        return self._lineage_tracker

    # Set methods
    def set_organisms(self, new_organisms):
        """
        Changes the organisms array within the Organisms object
        """
        self._organisms = new_organisms

    def build_spatial_index(self):
        """
        Build or rebuild the KD-Tree index over organism positions.
        Call this once per tick (after any moves/spawns) to enable fast
        radius or nearest-neighbor queries via self._pos_tree.
        """
        # if we have any organisms, stack their x/y into an (N,2) array…
        if self._organisms.shape[0] > 0:
            coords = np.stack(
                (self._organisms['x_pos'], self._organisms['y_pos']),
                axis=1
            )
            # cKDTree is much faster for large N
            self._pos_tree = cKDTree(coords)
        else:
            # no points → no tree
            self._pos_tree = None

    def _generate_ids(self, count: int) -> np.ndarray:
        """
        Allocate count new unique IDs in one go and bump the counter.
        """
        ids = np.arange(self._next_id,
                        self._next_id + count,
                        dtype=np.int64)
        self._next_id += count
        return ids

    def reproduce(self):
        """
        Trigger reproduction for all eligible organisms.

        Parents with sufficient energy and safe proximity spawn one offspring each.
        Offspring inherit traits, may randomly mutate, and are placed near their parent.

        Effects:
        - Deducts reproduction cost from parents’ energy
        - Initializes offspring IDs and lineage
        - Records new species when first encountered
        - Updates environment birth count
        """
        # Find which organisms are able to reproduce and determine costs
        parents, reproduction_costs, parent_mask = self._determine_valid_parents()

        # No valid parents means we stop right here
        if not np.any(parent_mask):
            return 

        # Single offspring per parent, randomized offset
        num_parents = parents.shape[0]
        offset = self.sample_circular_offsets(num_parents, REPRODUCTION_PROXIMITY_CONST)
        offspring = np.empty(num_parents, dtype=self._organism_dtype)

        # Inherit parent traits  
        copy_parent_fields(parents, offspring)
        
        # Chance of mutating traits based on a mask
        flip_mask = (np.random.rand(num_parents) < 0.01).astype(bool)
        if flip_mask.any():
            names = random_name_generation(flip_mask.sum())
            offspring['species'][flip_mask] = names
            mutate_offspring(offspring, flip_mask, self._gene_pool, flip_mask.sum())

        # Assign energy and deduct cost from parents
        offspring['energy'] = reproduction_costs
        self._organisms['energy'][parent_mask] -= reproduction_costs

        # Position offspring within bounds
        p_coords = np.stack((parents['x_pos'], parents['y_pos']), axis=1).astype(np.float32)
        raw_positions = p_coords + offset
        offspring['x_pos'] = np.clip(raw_positions[:, 0], 0, self._width - 1)
        offspring['y_pos'] = np.clip(raw_positions[:, 1], 0, self._length - 1)

        # Assign unique IDs
        self.increment_p_id_and_c_id(offspring, num_parents, parents)
        
        # Record new species first occurrences
        self._log_speciation(offspring,flip_mask)

        # Finalize births
        self._env.add_births(offspring.shape[0])
        self._organisms = np.concatenate((self._organisms, offspring))

    def _log_speciation(self, offspring, flip_mask):
        """
        Announces when a speciation event occurs and records the new species.
        The generation that the new species arose is also documented.
        """
        gen = self._env.get_generation()
        for sp, c_id, p_id in zip(
            offspring['species'][flip_mask],
            offspring['c_id'][flip_mask],
            offspring['p_id'][flip_mask]
        ):
            if sp not in self._speciation_dict:
                self._speciation_dict[sp] = [int(c_id), int(p_id), gen]
                print(f"New species added: {sp} (gen {gen})")

    def _determine_valid_parents(self
        ) -> Tuple[
            np.ndarray, 
            np.ndarray, 
            np.ndarray
        ]:
        """
        Determines which organisms are eligible to reproduce.
        - Eligible organisms cannot have another organism within 7.5 units of them
            -This will be changed to a global constant later 
        - Must have greater energy than size * fertility rate * 10
            -Can also modify the base reproducing threshhold to a global constant
        Returns:
        --------
        - np.ndarray: Array of organisms that will reproduce
        - np.ndarray: Associated reproduction costs for each reproducing organism
        - np.ndarray: Boolean mask identifying which organisms are eligible for reproduction
        """
        orgs = self._organisms
        coords = np.stack((orgs['x_pos'], orgs['y_pos']), axis=1)
        tree = self._pos_tree  # cKDTree built on coords

        # Compute distance to nearest other organism
        dists, idxs = tree.query(coords, k=2)
        nearest_dist = dists[:, 1]
        nearest_idx  = idxs[:, 1]

        #Keep this for later, it's turned off right now
        same_species = orgs['species'] == orgs['species'][nearest_idx]

        # only apply the proximity rule when it's the same species:
        # — if same_species AND too close ⇒ unsafe (False)
        # — otherwise ⇒ safe (True)
        # NOT CURRENTLY APPLIED, NEED TO ADD   | (~same_species)
        safe_mask = (nearest_dist >= REPRODUCTION_PROXIMITY_CONST)

        # Identify parents with enough energy
        energy = orgs['energy']
        cost = orgs['fertility_rate'] * orgs['size'] * 10
        reproducing = energy > cost
        parent_mask = reproducing & safe_mask
        if not np.any(parent_mask):
            return (
                np.zeros(0, dtype=self._organism_dtype), 
                np.zeros(0, dtype=np.float32), 
                np.zeros(0, dtype=bool)
            )
        return orgs[parent_mask], cost[parent_mask], parent_mask

    def sample_circular_offsets(self, num_offspring, reproduction_proximity_constant):
        """
        Spawn children further away than the constant, but only so much as twice
        as far away as the reproduction proximity constant.
        """
        R_inner = reproduction_proximity_constant
        R_outer = 2 * reproduction_proximity_constant

        # 1) Sample squared‐radii uniformly so area is uniform
        r2 = np.random.uniform(R_inner**2, R_outer**2, size=num_offspring)
        r  = np.sqrt(r2)

        # 2) Sample angles uniformly from 0 to 2π
        theta = np.random.uniform(0, 2*np.pi, size=num_offspring)

        # 3) Convert to Cartesian offsets
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        return np.stack((x, y), axis=1)  # shape (num_offspring, 2)

    def spawn_initial_organisms(self, 
        number_of_organisms:    int,
        randomize:              bool = False
        ) -> int:
        """
        Spawns the initial organisms in the simulation.
        Organism stats can be randomized if desired.
        Updates the birth counter in the environment.

        :param number_of_organisms: Number of organisms to spawn (int)
        :param randomize:  Request to randomize stats of spawned organisms (bool)
        :returns: how many organisms were actually placed (int)
        """
        # --- get environment info ---
        env_width = self._width
        env_length = self._length
        env_terrain = self._env.get_terrain()
        n = number_of_organisms


        if randomize:
            (
                # — Morphological traits —
                species_arr,
                size_arr,
                camouflage_arr,
                defense_arr,
                attack_arr,
                vision_arr,

                # — Metabolic parameters —
                metabolism_rate_arr,
                nutrient_efficiency_arr,
                diet_type_arr,

                # — Reproduction settings —
                fertility_rate_arr,
                offspring_count_arr,
                reproduction_type_arr,

                # — Social behaviors —
                pack_behavior_arr,
                symbiotic_arr,

                # — Locomotion capabilities —
                swim_arr,
                walk_arr,
                fly_arr,
                speed_arr,

                # — Energy state —
                energy_arr,
            ) = initialize_random_traits(n, self._gene_pool)
        else:
            (
                # — Morphological traits —
                species_arr,
                size_arr,
                camouflage_arr,
                defense_arr,
                attack_arr,

                # — Sensory —
                vision_arr,

                # — Metabolic parameters —
                metabolism_rate_arr,
                nutrient_efficiency_arr,
                diet_type_arr,

                # — Reproduction settings —
                fertility_rate_arr,
                offspring_count_arr,
                reproduction_type_arr,

                # — Social behaviors —
                pack_behavior_arr,
                symbiotic_arr,

                # — Locomotion capabilities —
                swim_arr,
                walk_arr,
                fly_arr,
                speed_arr,

                # — Energy state —
                energy_arr,
            ) = initialize_default_traits(n, self._gene_pool)

        # — Pick random positions and filter to valid terrain cells —
        # — Truncate all related arrays to the number of valid spots —
        positions, valid_count = calculate_valid_founder_terrain(
            env_terrain,
            swim_arr,
            walk_arr,
            fly_arr,
            env_width,
            n,
        )


        # — Pack truncated data into the structured offspring array —
        # e.g., offspring = np.empty(valid_count, dtype=self._organism_dtype)
        spawned = np.zeros(valid_count, dtype=self._organism_dtype)
        copy_valid_count(
            spawned,
            valid_count,

            # morphological traits
            species_arr,
            size_arr,
            camouflage_arr,
            defense_arr,
            attack_arr,
            vision_arr,

            # metabolic characteristics
            metabolism_rate_arr,
            nutrient_efficiency_arr,
            diet_type_arr,

            # reproduction parameters
            fertility_rate_arr,
            offspring_count_arr,
            reproduction_type_arr,

            # behaviors
            pack_behavior_arr,
            symbiotic_arr,

            # movement capabilities
            swim_arr,
            walk_arr,
            fly_arr,
            speed_arr,

            # state
            energy_arr,
        )
        # positions is int32 → cast to float32 for x_pos/y_pos
        positions_f = positions.astype(np.float32)
        spawned['x_pos']             = positions_f[:, 0]
        spawned['y_pos']             = positions_f[:, 1]
        
        # DO NOT INITIALIZE THIS WITHIN SIGNITURE OF BELOW DO IT HERE
        empty = np.empty((0), dtype=self._organism_dtype)
        self.increment_p_id_and_c_id(spawned, valid_count, empty)
    
        # --- append to full array and update births ---
        self._organisms = np.concatenate((self._organisms, spawned))
        self._founders = self._organisms.copy()
        self._env.add_births(valid_count)
        return valid_count

    def apply_terrain_penalties(self):
        """
        Deduct energy from organisms that are out of their compatible terrain.

        - Retrieves terrain height map from the environment.
        - Builds masks for “swim-only” (must be in water) and “walk-only” (must be on land).
        - Flags any swim-only organisms on land or walk-only organisms in water.
        - Applies an energy penalty proportional to their metabolism rate.
        """
        terrain = self._env.get_terrain()
        ix = self._organisms['x_pos'].astype(np.int32)
        iy = self._organisms['y_pos'].astype(np.int32)
        land_mask = terrain[iy, ix] >= 0

        orgs = self._organisms
        swim_only = orgs['swim'] & ~orgs['walk'] & ~orgs['fly']
        walk_only = orgs['walk'] & ~orgs['swim'] & ~orgs['fly']

        # swim-only on land, or walk-only in water
        penalty = (swim_only & land_mask) | (walk_only & ~land_mask)

        # subtract 0.1 * metabolism_rate for each violation
        orgs['energy'][penalty] -= 15 * orgs['metabolism_rate'][penalty]

    def compute_terrain_avoidance(self, coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        N = coords.shape[0]
        w, h = self._env.get_terrain().shape[1], self._env.get_terrain().shape[0]
        terrain = self._env.get_terrain()
        # 9‐neighbor offsets
        offsets = np.array([[-0.5, -0.5], [-0.5, 0.0], [-0.5, 0.5],
                            [ 0.0, -0.5], [ 0.0, 0.0], [ 0.0, 0.5],
                            [ 0.5, -0.5], [ 0.5, 0.0], [ 0.5, 0.5]])

        # sample positions & clip
        patches = coords[:, None, :] + offsets[None, :, :]
        patches[...,0] = np.clip(patches[...,0], 0, w-1)
        patches[...,1] = np.clip(patches[...,1], 0, h-1)

        # sample terrain to build masks
        tv = map_coordinates(terrain,
                             [patches[...,1].ravel(), patches[...,0].ravel()],
                             order=1).reshape(N,9)
        water_mask = (tv < 0)        # (N,9)
        land_mask  = ~water_mask     # (N,9)

        # build raw direction vectors
        deltas = patches - coords[:,None,:]    # (N,9,2)
        raw_water = -(deltas * water_mask[...,None]).sum(axis=1)  # (N,2)
        raw_land  = -(deltas * land_mask[...,None]).sum(axis=1)   # (N,2)


        dist_to_water = distance_transform_edt(~water_mask)
        dist_to_land  = distance_transform_edt( land_mask)

        # normalize directions (avoid div0)
        def unit(v):
            norm = np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
            return v / norm

        u_water = unit(raw_water)
        u_land  = unit(raw_land)

        # sample distance‐to‐forbidden‐terrain
        d_water = map_coordinates(dist_to_water, [coords[:,1], coords[:,0]], order=1)
        d_land  = map_coordinates(dist_to_land,  [coords[:,1], coords[:,0]], order=1)

        # magnitude scaling: asymptotic to F_max
        F_max = 1000.0
        K     = 500.0   # so K / 0.5 == 1000
        eps   = 1e-3

        mag_water = np.minimum(F_max, K / (d_water + eps))  # (N,)
        mag_land  = np.minimum(F_max, K / (d_land  + eps))  # (N,)

        # final forces
        avoid_water = u_water * mag_water[:,None]
        avoid_land  = u_land  * mag_land [:,None]

        return avoid_land, avoid_water

    def move(self):
        """
        Advance each organism one time step by computing and applying its movement vector.

        - Applies terrain penalties before movement.
        - Uses a spatial index to find neighbors within vision radius.
        - Applies species-specific behavior (photosynthesis, pack cohesion, predation).
        - Avoids terrain mismatches (land vs. water) when moving.
        - Normalizes and scales movement by each organism’s speed.
        - Clips new positions to world bounds and deducts energy based on distance moved.
        """
        orgs = self._organisms
        N = orgs.shape[0]
        if N == 0:
            return

        self.apply_terrain_penalties()
        coords, neigh_lists = get_coords_and_neighbors(orgs, self._pos_tree)
        avoid_land, avoid_water = self.compute_terrain_avoidance(coords)

        # Calculate new positions - currently passes alot of memory outside the class
        new_pos = movement_compute(
            orgs, 
            coords, 
            neigh_lists, 
            self._width, 
            self._length, 
            avoid_land, 
            avoid_water)
        self.pay_energy_costs(orgs, new_pos, coords)
        self.build_spatial_index()

    def pay_energy_costs(self,
        organism_arr: np.ndarray,
        new_pos_arr: np.ndarray,
        old_coords_arr: np.ndarray,
    ):
        """
        Function modifies energy of organisms in place.
        - Energy cost is based on distance (not displacement yet)
        - Energy cost is based on metabolism_rate
        - Plants do not pay energy costs from movement
            - Right now plants also cannot move
        """
        non_photo = organism_arr['diet_type'] != 'Photo'
        photo = ~non_photo
        organism_arr['x_pos'][non_photo] = new_pos_arr[non_photo, 0]
        organism_arr['y_pos'][non_photo] = new_pos_arr[non_photo, 1]
        dists = np.linalg.norm(new_pos_arr[non_photo] - old_coords_arr[non_photo], axis=1)
        move_costs = 0.01 * dists * organism_arr['metabolism_rate'][non_photo]
        organism_arr['energy'][non_photo] -= move_costs
        organism_arr['energy'][photo] += ECO_ENERGY_CONST/(photo.shape[0])

    def resolve_attacks(self):
        """
        Vectorized one‐to‐one attacks on nearest neighbor within vision.
        Attackers gain energy equal to the damage they inflict.
        """
        orgs = self._organisms
        N = orgs.shape[0]
        if N < 2:
            return

        # --- 0) Extract flat arrays once ---
        coords  = np.stack((orgs['x_pos'], orgs['y_pos']), axis=1)  # (N, 2)
        att     = orgs['attack']                                    # (N,)
        deff    = orgs['defense']
        vision  = orgs['vision']
        pack    = orgs['pack_behavior']
        fly     = orgs['fly']
        swim    = orgs['swim']
        walk    = orgs['walk']
        x_pos   = orgs['x_pos']
        y_pos   = orgs['y_pos']
        energy  = orgs['energy']

        terrain = self._env.get_terrain()
        # --- 1) Ensure KD-Tree is fresh ---
        if self._pos_tree is None:
            self.build_spatial_index()

        # --- 2) Batch nearest-neighbor query (k=2) ---
        dists, idxs   = self._pos_tree.query(coords, k=2, workers=-1)
        nearest       = idxs[:,1]          # (N,)
        nearest_dist  = dists[:,1]

        # --- 3) Filter to those within vision ---
        can_see   = nearest_dist <= vision
        attackers = np.nonzero(can_see)[0]
        if attackers.size == 0:
            return

        i = attackers                 # attacker indices
        j = nearest[attackers]        # corresponding defender indices

        # --- 4) Terrain/fly/swim/walk restrictions ---
        i, j = self._terrain_restrictions(
            i, j,
            fly, swim, walk,
            x_pos, y_pos,
            terrain,
)
        # —–––––– diet‐type restrictions –––––— #
        i, j = self._diet_restrictions(i,j, orgs['diet_type'])

        if i.size == 0:
            return

        # --- Apply damage and energy gain in batch ---
        host, prey, my_net, their_net = self._classify_engagement(i,j,att,deff,pack)
        self._apply_damage(i,j,host,prey,my_net,their_net,energy)

    def _terrain_restrictions(
            self,
            i:        np.ndarray,
            j:        np.ndarray,
            fly_arr:  np.ndarray,
            swim_arr: np.ndarray,
            walk_arr: np.ndarray,
            x_pos:    np.ndarray,
            y_pos:    np.ndarray,
            terrain:  np.ndarray
        ) -> Tuple[
        np.ndarray, 
        np.ndarray
        ]:
        """
        :Parameters:
        - i : Attackers : numpy array
        - j : Defenders : numpy array
        - fly_arr : boolean mask of orgs with flying property
        - swim_arr : boolean mask of orgs with swimming property
        - walk_arr : boolean mask of orgs with walking property
        - x_pos : Array of x positions
        - y_pos : Array of y positions
        - terrain : f32 mask of terrain values
        
        :Returns:
        - Tuple of filtered np.ndarray in a tuple of 
        (attacker, nearest_attacker) pairs after 
        applying water/land/fly/swim/walk rules.
        """
        # pull just the columns we need:
        xj = x_pos[j].astype(int)
        yj = y_pos[j].astype(int)
        tiles = terrain[yj, xj]

        invalid = np.zeros_like(i, dtype=bool)
        invalid |= (~fly_arr[i]  & fly_arr[j])
        invalid |= (~swim_arr[i] & swim_arr[j] & (tiles < 0))
        invalid |= ( swim_arr[i] & ~walk_arr[i] & (tiles >= 0))
        invalid |= ( swim_arr[i] & ~fly_arr[i]  & fly_arr[j] & (tiles < 0))

        keep = ~invalid
        return i[keep], j[keep]

    def _diet_restrictions(self, 
        i: np.ndarray, 
        j: np.ndarray, 
        diet: np.ndarray
        )-> Tuple[
            np.ndarray,
            np.ndarray
        ]:
        """
        :Parameters:
        - i : Attackers : numpy array
        - j : Defenders : numpy array
        - diet : numpy array with diet types of attacking
        and defending organisms
        
        : Returns: Tuple of (i,j)
        - i : Attacker interactions filtered by diet type : numpy array
        - j : Defender interactions filtered by diet type : numpy array
        """
        dt_i, dt_j = diet[i], diet[j]
        invalid = np.zeros_like(i, dtype=bool)

        invalid |= (dt_i == 'Photo')
        invalid |= ( (dt_i == 'Herb') & ~np.isin(dt_j, ['Photo','Parasite']) )
        invalid |= ( (dt_i == 'Carn') &  (dt_j == 'Photo') )

        keep = ~invalid
        return i[keep], j[keep]

    def _classify_engagement(self,
        i:    np.ndarray,
        j:    np.ndarray,
        att:  np.ndarray,
        deff: np.ndarray,
        pack: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
    Classifies engagements between organisms based on attack and defense values, 
    considering pack behavior.

    :param i: Indices of attackers (np.ndarray)
    :param j: Indices of defenders (np.ndarray)
    :param att: Attack values for organisms (np.ndarray)
    :param deff: Defense values for organisms (np.ndarray)
    :param pack: Pack behavior indicators (np.ndarray)
    
    :returns:
    - np.ndarray: Mask for host engagements
    - np.ndarray: Mask for prey engagements
    - np.ndarray: Net attack values for attackers
    - np.ndarray: Net attack values for defenders
    """
        my_net    = att[i] - deff[j]
        their_net = att[j] - deff[i]

        if bool(pack.any()):
            non_pack = pack[i] & ~pack[j]
            host = (their_net > my_net) & non_pack
            prey = (my_net > their_net) & non_pack
        else:
            host =  their_net > my_net
            prey =  my_net > their_net

        # only positive‐damage
        host &= (their_net > 0)
        prey &= (my_net > 0)
        return host, prey, my_net, their_net


    def _apply_damage(self, 
        i:         np.ndarray, 
        j:         np.ndarray, 
        host:      np.ndarray, 
        prey:      np.ndarray, 
        my_net:    np.ndarray, 
        their_net: np.ndarray, 
        energy:    np.ndarray
        ):
        """
        Applies damage to organisms based on host and prey interactions.

        :param i: Indices of attackers (np.ndarray)
        :param j: Indices of defenders (np.ndarray)
        :param host: Mask for host engagements (np.ndarray)
        :param prey: Mask for prey engagements (np.ndarray)
        :param my_net: Net attack values for attackers (np.ndarray)
        :param their_net: Net attack values for defenders (np.ndarray)
        :param energy: Energy values for all organisms (np.ndarray)

        :returns: None (modifies energy array in place)
        """
        # Hostiles: j attacked i, damage = their_net
        if host.any():
            idx_i = i[host]
            idx_j = j[host]
            dmg   = their_net[host]
            energy[idx_i] -= 20 * dmg
            energy[idx_j] += 20 * dmg

        # Prey: i attacked j, damage = my_net
        if prey.any():
            idx_i = i[prey]
            idx_j = j[prey]
            dmg   = my_net[prey]
            energy[idx_j] -= 20 * dmg
            energy[idx_i] += 20 * dmg

    def kill_border(self, margin: float = 0.01):
        """
        Instantly kills (sets energy to 0 and removes) all organisms
        within `margin` fraction of the environment border.
        """
        orgs = self._organisms
        N = orgs.shape[0]
        if N == 0:
            return

        # world dimensions
        w, h = self._width, self._length
        # positions
        x = orgs['x_pos']
        y = orgs['y_pos']

        # mask of who’s too close to any edge
        border_mask = (
            (x <  margin * w) |
            (x > (1-margin) * w ) |
            (y <  margin * h) |
            (y > (1-margin) * h )
        )

        if not np.any(border_mask):
            return

        # set them to zero energy so remove_dead() will catch them
        orgs['energy'][border_mask] = -1000.0

    def remove_dead(self):
        """
        Removes dead organisms from the environment
        """

        # Retrieves which organisms are dead and updates death counter
        dead_mask = (self._organisms['energy'] <= 0)
        self._env.add_deaths(np.count_nonzero(dead_mask))
        # The dead are removed from the organisms array
        survivors = self._organisms[~dead_mask]
        self._organisms = survivors

        return

    def increment_p_id_and_c_id(self, 
        c_org_arr:   np.ndarray,  
        num_spawned: int,
        p_org_arr:   np.ndarray
        ):
        """
        Increment id's for reproduction and spawning founders.
        May be used for lineage later.
        """
        #TODO: Consider not passing the whole org array in but only the id fields
        start = self._next_id
        c_org_arr['c_id'] = np.arange(start, start + num_spawned, dtype=np.int32)
        self._next_id += num_spawned
        #
        # Founders are their own parents
        #
        gen = self._env.get_generation()
        if p_org_arr.size:
            c_org_arr['p_id'] = p_org_arr['c_id']
            self._lineage_tracker.track_lineage(p_org_arr,c_org_arr,gen)
        else:
            c_org_arr['p_id'] = c_org_arr['c_id'] = 0
            self._next_id = 1

    def apply_meteor_damage(self, x: float, y: float, radius: float, base_damage: float = None):
        """
        Instantly kills any organism within `radius` of (x, y) by setting energy to 0.
        """
        if self._organisms.shape[0] == 0:
            return

        coords = np.stack((self._organisms['x_pos'], self._organisms['y_pos']), axis=1)

        if self._pos_tree is None:
            self.build_spatial_index()

        indices = self._pos_tree.query_ball_point([x, y], radius)
        if not indices:
            return

        # Set energy of all in radius to 0 — instant death
        self._organisms['energy'][indices] = 0
        print(f"[Meteor] Killed {len(indices)} organisms.")

def random_name_generation(
    num_to_gen:     int,
    min_syllables:  int = 2,
    max_syllables:  int = 4
) -> np.ndarray:
    """
    Generate `num_to_gen` random species names and return them as a NumPy array.
    """
    syllables = [
        'ar', 'en', 'ex', 'ul', 'ra', 'zo', 'ka', 'mi',
        'to', 'lu', 'qui', 'fa', 'ne', 'si', 'ta', 'or',
        'an', 'el', 'is', 'ur', 'in', 'ox', 'al', 'om'
    ]
    names = []
    for _ in range(num_to_gen):
        count = random.randint(min_syllables, max_syllables)
        name = ''.join(random.choice(syllables) for _ in range(count)).capitalize()
        names.append(name)
    return np.array(names)