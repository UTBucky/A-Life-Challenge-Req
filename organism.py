import numpy as np
from scipy.spatial import cKDTree
import random
from array_ops import *
from collections import defaultdict
from lineage_tracker import *

# Terrain avoidance constants
WATER_PUSH = 5.0
LAND_PUSH = 5.0

# Pack behavior constants
SEPARATION_WEIGHT = 10
SEPARATION_RADIUS = 5


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

    def __init__(self, env: object, O_CLASS = ORGANISM_CLASS):
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
        self._mutation_rate = 0.05
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
        offset = np.random.uniform(-20, 20, size=(num_parents, 2))
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
        - Numpy array of organisms that will reproduce
        - the associated costs of reproduction specific to reproducing parents
        - A mask of values identifying which organisms will reproduce
        """
        orgs = self._organisms
        coords = np.stack((orgs['x_pos'], orgs['y_pos']), axis=1)
        tree = self._pos_tree  # cKDTree built on coords

        # Compute distance to nearest other organism
        dists, _ = tree.query(coords, k=2)
        nearest_dist = dists[:, 1]
        safe_mask = nearest_dist >= 7.5

        # Identify parents with enough energy
        energy = orgs['energy']
        cost = orgs['fertility_rate'] * orgs['size'] * 10
        reproducing = energy > cost
        parent_mask = reproducing & safe_mask
        if not np.any(parent_mask):
            return np.zeros(0, dtype=self._organism_dtype), np.zeros(0, dtype=np.float32), np.zeros(0, dtype=bool)

        return orgs[parent_mask], cost[parent_mask], parent_mask
    
    def spawn_initial_organisms(self, 
        number_of_organisms: int,
        randomize: bool = False
        ) -> int:
        """
        Spawns the initial organisms in the simulation.
        Organism stats can be randomized if desired.
        Updates the birth counter in the environment.

        :param number_of_organisms: Number of organisms to spawn
        :param randomize: Request to randomize stats of spawned organisms
        :returns: how many organisms were actually placed
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
        orgs['energy'][penalty] -= 0.1 * orgs['metabolism_rate'][penalty]

    def compute_terrain_avoidance(self, 
        coords: np.ndarray
        ) -> Tuple[
            np.ndarray,
            np.ndarray
        ]:
        """
        Given an (N,2) array of positions, compute per‐organism
        avoidance vectors for water and land based on the 4‐neighborhood.
        
        Returns:
            avoid_land  : np.ndarray of shape (N,2)
            avoid_water : np.ndarray of shape (N,2)
        """
        N = coords.shape[0]
        terrain = self._env.get_terrain()
        width, length = self._width, self._length
        dirs = self._dirs  # shape (4,2)

        # generate neighbor samples: shape (N,4,2)
        samples = coords[:, None, :] + dirs[None, :, :]

        # map to integer grid indices
        ix = samples[..., 0].astype(int)
        iy = samples[..., 1].astype(int)

        # mask out‐of‐bounds
        valid = (
            (ix >= 0) & (ix < width) &
            (iy >= 0) & (iy < length)
        )  # shape (N,4)

        # look up terrain, fill invalid with nan
        tiles = np.full((N, 4), np.nan, dtype=np.float32)
        tiles[valid] = terrain[iy[valid], ix[valid]]

        # water‐avoidance: push opposite dirs wherever terrain<0
        mask_water = tiles < 0
        avoid_water = - (dirs[None, :, :] * mask_water[..., None]).sum(axis=1)

        # land‐avoidance: push opposite dirs wherever terrain>=0
        mask_land = tiles >= 0
        avoid_land = - (dirs[None, :, :] * mask_land[..., None]).sum(axis=1)

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

        width, length = self._width, self._length
        coords = np.stack((orgs['x_pos'], orgs['y_pos']), axis=1)

        vision_radii = orgs['vision']
        neigh_lists = self._pos_tree.query_ball_point(coords, vision_radii, workers=-1)

        avoid_land, avoid_water = self.compute_terrain_avoidance(coords)

        # —––––––– grab items once, outside of any per-organism loop –––––––—
        diet_type = orgs['diet_type']
        vision = orgs['vision']
        attack = orgs['attack']
        defense = orgs['defense']
        pack_flag = orgs['pack_behavior']
        species = orgs['species']
        fly_flag = orgs['fly']
        swim_flag = orgs['swim']
        walk_flag = orgs['walk']
        speed = orgs['speed']

        def _compute(i, pos, neighs, width, length):
            # pull out “my” values once
            my = orgs[i]
            my_diet = my['diet_type']
            my_cam = my['camouflage']
            my_att = my['attack']
            my_def = my['defense']
            my_spc = my['species']
            my_pack = pack_flag[i]
            my_fly = fly_flag[i]
            my_swim = swim_flag[i]
            my_walk = walk_flag[i]
            my_speed = speed[i]
            if my_diet == 'Photo':
                my['energy'] += 0.25
                my_def = 0
                my_att = 0
                move_vec = np.zeros(2, dtype=np.float32)

                return move_vec

            # make neighbors a NumPy array of ints
            neighs = np.asarray(neighs, dtype=int)

            # 1) camouflage filter
            mask_valid = (neighs != i) & (vision[neighs] >= my_cam)
            valid = neighs[mask_valid]

            # 2) pack_mates if pack_behavior array isn’t empty
            if pack_flag.shape[0] > 0:
                pack_mates = valid[pack_flag[valid]]

            # allocate movement accumulator
            move_vec = np.zeros(2, dtype=np.float32)

            # — behavioral overrides (pack) —
            if my_pack:
                steer = np.zeros(2, dtype=np.float32)

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
                    steer += (pos - center)
                else:
                    prey = valid[prey_mask]
                    if prey.size > 0:
                        center = coords[prey].mean(axis=0)
                        steer += (center - pos)
                    else:
                        # c) cohesion + gentle separation
                        if pack_mates.size > 0:
                            center = coords[pack_mates].mean(axis=0)
                            steer += (center - pos)

                            dists = coords[pack_mates] - pos
                            norms = np.linalg.norm(dists, axis=1)
                            close = norms < SEPARATION_RADIUS
                            if close.any():
                                repulse = -np.mean(dists[close], axis=0)
                                steer += repulse * SEPARATION_WEIGHT

                # terrain avoidance
                if not my_swim:
                    steer += WATER_PUSH * avoid_water[i]
                if not my_walk:
                    steer += LAND_PUSH * avoid_land[i]

                # normalize & scale by speed
                norm = np.linalg.norm(steer)
                step = (steer / norm) * \
                    my_speed if norm > 0 else np.zeros(2, np.float32)

                new = pos + step
                new[0] = np.clip(new[0], 0, width - 1)
                new[1] = np.clip(new[1], 0, length - 1)
                return new

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
            CROWD_PUSH = 0.5 * my_speed
            same_mask = species[valid] == my_spc
            same = valid[same_mask]
            if same.size > 0:
                repulse = np.mean(pos - coords[same], axis=0)
                move_vec += CROWD_PUSH * repulse

            # terrain avoidance

            if not my_swim:
                move_vec += WATER_PUSH * avoid_water[i]
            if not my_walk:
                move_vec += LAND_PUSH * avoid_land[i]

            # normalize & scale
            norm = np.linalg.norm(move_vec)
            step = (move_vec / norm) * \
                my_speed if norm > 0 else np.zeros(2, np.float32)

            new = pos + step
            new[0] = np.clip(new[0], 0, width - 1)
            new[1] = np.clip(new[1], 0, length - 1)
            return new
        old_coords = coords
        # map across all organisms

        new_pos = np.array([
            _compute(i, coords[i], neigh_lists[i], width, length)
            for i in range(N)
        ], dtype=np.float32)


        non_photo = orgs['diet_type'] != 'Photo'
        orgs['x_pos'][non_photo] = new_pos[non_photo, 0]
        orgs['y_pos'][non_photo] = new_pos[non_photo, 1]
        dists = np.linalg.norm(new_pos[non_photo] - old_coords[non_photo], axis=1)
        move_costs = 0.01 * dists * orgs['metabolism_rate'][non_photo]
        orgs['energy'][non_photo] -= move_costs

        self.build_spatial_index()

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
        coords = np.stack((orgs['x_pos'], orgs['y_pos']), axis=1)  # (N,2)
        att    = orgs['attack']      # (N,)
        deff   = orgs['defense']
        vision = orgs['vision']
        pack   = orgs['pack_behavior']
        fly = orgs['fly']
        swim = orgs['swim']
        walk = orgs['walk']
        x_pos = orgs['x_pos']
        y_pos = orgs['y_pos']
        terrain = self._env.get_terrain()
        energy = orgs['energy']

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
        i: np.ndarray,
        j: np.ndarray,
        fly_arr: np.ndarray,
        swim_arr: np.ndarray,
        walk_arr: np.ndarray,
        x_pos: np.ndarray,
        y_pos: np.ndarray,
        terrain: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns filtered (i, j) pairs after applying water/land/fly/swim/walk rules.
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

    def _diet_restrictions(self, i, j, diet):
        dt_i, dt_j = diet[i], diet[j]
        invalid = np.zeros_like(i, dtype=bool)

        invalid |= (dt_i == 'Photo')
        invalid |= ( (dt_i == 'Herb') & ~np.isin(dt_j, ['Photo','Parasite']) )
        invalid |= ( (dt_i == 'Carn') &  (dt_j == 'Photo') )

        keep = ~invalid
        return i[keep], j[keep]

    def _classify_engagement(self, i, j, att, deff, pack):
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

    def _apply_damage(self, i, j, host, prey, my_net, their_net, energy):
        # Hostiles: j attacked i, damage = their_net
        if host.any():
            idx_i = i[host]
            idx_j = j[host]
            dmg   = their_net[host]
            energy[idx_i] -= dmg
            energy[idx_j] += dmg

        # Prey: i attacked j, damage = my_net
        if prey.any():
            idx_i = i[prey]
            idx_j = j[prey]
            dmg   = my_net[prey]
            energy[idx_j] -= dmg
            energy[idx_i] += dmg

    def kill_border(self, margin: float = 0.03):
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
        orgs['energy'][border_mask] = 0.0

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
        c_org_arr:np.ndarray,  
        num_spawned:int,
        p_org_arr:np.ndarray):
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

def random_name_generation(
    num_to_gen: int,
    min_syllables: int = 2,
    max_syllables: int = 4
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