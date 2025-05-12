import numpy as np
from scipy.spatial import cKDTree
import random
from array_ops import *

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
        self._species_count = {}
        self._next_id = 0
        # Garbage collection optimization - scratch buffers
        self._dirs = np.array([[1,0],[-1,0],[0,1],[0,-1]], np.float32)

    def load_genes(self, gene_pool):
        self._gene_pool = gene_pool

    # Get methods
    def get_organisms(self):
        return self._organisms

    def get_species_count(self):
        return self._species_count

    # Set methods
    def set_organisms(self, new_organisms):
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
        Causes all organisms that can to reproduce.
        Spawns offspring near the parent
        """
        orgs = self._organisms
        coords        = np.stack((orgs['x_pos'], orgs['y_pos']), axis=1)
        tree          = self._pos_tree  # a cKDTree built on coords

        # 2) Vectorized: find each organism’s *distance* to its nearest *other* neighbor
        #    - query k=2: idx 0 is self (dist=0), idx 1 is true nearest neighbor
        dists, idxs = tree.query(coords, k=2)     # dists.shape == (N,2)
        nearest_dist = dists[:, 1]                # shape (N,)
        # 3) Build a boolean mask of “safe” organisms
        too_close   = 7.5
        safe_mask   = nearest_dist >= too_close   # True if OK to interact

        
        # Obtains an array of all reproducing organisms
        reproducing = (self._organisms['energy']
                    >
                    (self._organisms['fertility_rate']
                    *
                    self._organisms['size'])*10)
        
        
        if not np.any(reproducing):
            return
        parent_mask = reproducing & safe_mask
        parents = self._organisms[parent_mask]
        parent_reproduction_costs = (10*
            self._organisms['fertility_rate'][parent_mask]
            * self._organisms['size'][parent_mask]
        )

        # TODO: Implement number of children, currently just one offspring
        # Put children randomly nearby
        offset = np.random.uniform(-20, 20, size=(parents.shape[0], 2))
        offspring = np.empty((parents.shape[0],), dtype=self._organism_dtype)


        # Create offspring simulation bookkeeping
        copy_parent_fields(parents, offspring)

        # --- Mutate mutated spawns ---
        # Use a bool mask with a % mutation chance to mutate
        # TODO: make this value adjustable?
        flip_mask = (np.random.rand(offspring.shape[0]) < 0.01).astype(bool)
        m = flip_mask.sum()
        species_arr = random_name_generation(m)
        offspring['species'][flip_mask] = species_arr
        
        
        mutate_offspring(offspring,flip_mask,self._gene_pool,m)


        offspring['energy'] = parent_reproduction_costs
        self._organisms['energy'][parent_mask] -= parent_reproduction_costs
        width, length = self._width, self._length
        raw_x = parents['x_pos'] + offset[:, 0]
        raw_y = parents['y_pos'] + offset[:, 1]

        #initialize id based on number that was produced
        self.increment_p_id_and_c_id(offspring,offspring.shape[0],parents)

        offspring['x_pos'] = np.clip(raw_x, 0, width  - 1)
        offspring['y_pos'] = np.clip(raw_y, 0, length - 1)

        # Speciation and lineage tracking
        # Do this after it's declared valid in the environment
        gen = self._env.get_generation()
        
        species_arr = offspring['species'][flip_mask]
        c_id_arr    = offspring['c_id'][flip_mask]
        p_id_arr    = offspring['p_id'][flip_mask]

        for species, c_id, p_id in zip(species_arr, c_id_arr, p_id_arr):
            if species not in self._species_count:
                # store as [c_id, p_id, generation]
                self._species_count[species] = [int(c_id), int(p_id), gen]
                print(f"New species added: {species} ({gen})")

        self._env.add_births(offspring.shape[0])
        self._organisms = np.concatenate((self._organisms, offspring))

    def spawn_initial_organisms(self, number_of_organisms: int,
                                randomize: bool = False) -> int:
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
        self._env.add_births(valid_count)
        return valid_count

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
        if p_org_arr['x_pos'].any():
            c_org_arr['p_id'] = p_org_arr['c_id']
        else:
            c_org_arr['p_id'] = c_org_arr['c_id']

    def apply_terrain_penalties(self):
        """"""
        terrain = self._env.get_terrain()
        ix = self._organisms['x_pos'].astype(np.int32)
        iy = self._organisms['y_pos'].astype(np.int32)
        land_mask = terrain[iy, ix] >= 0

        # Penalize energy for out-of-terrain conditions
        orgs = self._organisms
        swim_only = orgs['swim'] & ~orgs['walk'] & ~orgs['fly']
        walk_only = orgs['walk'] & ~orgs['swim'] & ~orgs['fly']

        # swim-only on land, or walk-only in water
        penalty = (swim_only & land_mask) | \
            (walk_only & ~land_mask)

        # subtract 5 energy per violation (they die via remove_dead when energy ≤ 0)
        orgs['energy'][penalty] -= 0.1 * orgs['metabolism_rate'][penalty]

    def compute_terrain_avoidance(self, coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
        orgs = self._organisms
        N = orgs.shape[0]
        if N == 0:
            return

        self.apply_terrain_penalties()

        terrain = self._env.get_terrain()
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
                SEPARATION_WEIGHT = 10
                SEPARATION_RADIUS = 5

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
                WATER_PUSH = 5.0
                LAND_PUSH = 5.0
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
            WATER_PUSH = 5.0
            LAND_PUSH = 5.0
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
        dt     = orgs['diet_type']
        fly    = orgs['fly']
        swim   = orgs['swim']
        walk   = orgs['walk']
        energy = orgs['energy']
        terrain = self._env.get_terrain()

        # only use pack logic if any pack_behavior is True
        use_pack = bool(pack.any())

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
        ix = orgs['x_pos'][j].astype(int)
        iy = orgs['y_pos'][j].astype(int)
        tiles = terrain[iy, ix]       # (M,)

        invalid = np.zeros_like(i, dtype=bool)
        invalid |= (~fly[i] & fly[j])
        invalid |= (~swim[i] & swim[j] & (tiles < 0))
        invalid |= ( swim[i] & ~walk[i] & (tiles >= 0))
        invalid |= ( swim[i] & ~fly[i]  & fly[j] & (tiles < 0))

        # —–––––– diet‐type restrictions –––––— #
        dt = orgs['diet_type']
        dt_i = dt[i]  # attacker diets
        dt_j = dt[j]  # defender diets

        # 1) Photo attackers can never attack anyone:
        invalid |= (dt_i == 'Photo')

        # 2) Herb attackers can only attack Photo or Parasite, so anything else is invalid:
        invalid |= ((dt_i == 'Herb') & ~np.isin(dt_j, ['Photo', 'Parasite']))

        # 3) Carn attackers cannot attack Photo:
        invalid |= ((dt_i == 'Carn') &  (dt_j == 'Photo'))

        # Omni and Parasite attackers have no extra restrictions, so we leave them out.

        valid_mask = ~invalid
        i = i[valid_mask]
        j = j[valid_mask]
        if i.size == 0:
            return

        # --- 5) Compute net attack values ---
        my_net    = att[i] - deff[j]    # (K,)
        their_net = att[j] - deff[i]

        # --- 6) Classify host vs prey ---
        if use_pack:
            non_pack = pack[i] & ~pack[j]
            host = (their_net > my_net) & non_pack
            prey = (my_net    > their_net) & non_pack
        else:
            host =  their_net > my_net
            prey =  my_net    > their_net

        # only positive damage engagements
        host &= (their_net > 0)
        prey &= (my_net     > 0)

        # --- 7) Apply damage and energy gain in batch ---
        # Hostiles: neighbor j attacked i
        if np.any(host):
            idx_i = i[host]           # defenders hit
            idx_j = j[host]           # attackers
            dmg   = their_net[host]
            energy[idx_i] -= dmg      # defender loses
            energy[idx_j] += dmg      # attacker gains

        # Prey: attacker i hit neighbor j
        if np.any(prey):
            idx_i = i[prey]           # attackers
            idx_j = j[prey]           # defenders hit
            dmg   = my_net[prey]
            energy[idx_j] -= dmg      # defender loses
            energy[idx_i] += dmg      # attacker gains
        
        #prep for reproduction
        self.build_spatial_index()

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

    def lineage(self):
        """
        Build a dict mapping each parent → list of its children.
        """
        c_id_arr = self._organisms['c_id']
        p_id_arr = self._organisms['p_id']
        
        # 1) sort births by parent ID
        order = np.argsort(p_id_arr)
        p_sorted = p_id_arr[order]
        c_sorted = c_id_arr[order]
        
        # 2) find where each parent’s block of children starts, and how many
        parents, start_idxs, counts = np.unique(
            p_sorted,
            return_index=True,
            return_counts=True       # <-- was misspelled `return_coutns`
        )
        parents = parents.tolist()
        start_idxs = start_idxs.tolist()
        counts = counts.tolist()
        # 3) build the dict parent → [child, child, …]
        children = {
            parent: c_sorted[start : start + cnt].tolist()
            for parent, start, cnt in zip(parents, start_idxs, counts)
        }
        
        return children
    
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