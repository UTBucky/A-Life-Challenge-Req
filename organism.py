import numpy as np
from scipy.spatial import cKDTree


class Organisms:
    """
    Represents all organisms in an environment.
    Keeps track of all organism's statistics.
    """

    def __init__(self, env: object):
        """
        Initialize an organism object.

        :param env: 2D Simulation Environment object
        """

        self._organism_dtype = np.dtype([
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
        ])

        self._pos_tree = None
        self._organisms = np.zeros((0,), dtype=self._organism_dtype)
        self._env = env
        self._width = env.get_width()
        self._length = env.get_length()
        self._mutation_rate = 0.05
        # TODO: Load genes from json file
        self._gene_pool = None
        
        self._ancestry = {}
        self._species_count = {}
        
        # Garbage collection optimization - scratch buffers
        self._dirs = np.array([[1,0],[-1,0],[0,1],[0,-1]], np.float32)

    def load_genes(self, gene_pool):
        self._gene_pool = gene_pool

    # Get methods
    def get_organisms(self):
        return self._organisms

    def get_ancestries(self):
        return self._ancestry

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

    def reproduce(self):
        """
        Causes all organisms that can to reproduce.
        Spawns offspring near the parent
        """

        # Obtains an array of all reproducing organisms
        reproducing = (self._organisms['energy']
                    >
                    (self._organisms['fertility_rate']
                    *
                    self._organisms['size'])*10)
        if not np.any(reproducing):
            return

        parents = self._organisms[reproducing]
        parent_reproduction_costs = (10*
            self._organisms['fertility_rate'][reproducing]
            * self._organisms['size'][reproducing]
        )

        # TODO: Implement number of children, currently just one offspring
        # Put children randomly nearby
        offset = np.random.uniform(-20, 20, size=(parents.shape[0], 2))
        offspring = np.zeros((parents.shape[0],), dtype=self._organism_dtype)



        # Create offspring simulation bookkeeping
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

        
        # --- Mutate mutated spawns ---
        flip_mask = (np.random.rand(offspring.shape[0]) < 0.01).astype(bool)
        m = flip_mask.sum()
        offspring['size'][flip_mask]               = np.random.uniform(
                                                        low=self._gene_pool['size'][0],
                                                        high=self._gene_pool['size'][1],
                                                        size=m
                                                    ).astype(np.float32)

        offspring['camouflage'][flip_mask]         = np.random.uniform(
                                                        low=self._gene_pool['camouflage'][0],
                                                        high=self._gene_pool['camouflage'][1],
                                                        size=m
                                                    ).astype(np.float32)

        offspring['defense'][flip_mask]            = np.random.uniform(
                                                        low=self._gene_pool['defense'][0],
                                                        high=self._gene_pool['defense'][1],
                                                        size=m
                                                    ).astype(np.float32)

        offspring['attack'][flip_mask]             = np.random.uniform(
                                                        low=self._gene_pool['attack'][0],
                                                        high=self._gene_pool['attack'][1],
                                                        size=m
                                                    ).astype(np.float32)

        offspring['vision'][flip_mask]             = np.random.uniform(
                                                        low=self._gene_pool['vision'][0],
                                                        high=self._gene_pool['vision'][1],
                                                        size=m
                                                    ).astype(np.float32)

        offspring['metabolism_rate'][flip_mask]    = np.random.uniform(
                                                        low=self._gene_pool['metabolism_rate'][0],
                                                        high=self._gene_pool['metabolism_rate'][1],
                                                        size=m
                                                    ).astype(np.float32)

        offspring['nutrient_efficiency'][flip_mask]= np.random.uniform(
                                                        low=self._gene_pool['nutrient_efficiency'][0],
                                                        high=self._gene_pool['nutrient_efficiency'][1],
                                                        size=m
                                                    ).astype(np.float32)

        offspring['diet_type'][flip_mask]          = np.random.choice(
                                                        self._gene_pool['diet_type'],
                                                        size=m
                                                    ).astype(np.str_)

        offspring['fertility_rate'][flip_mask]     = np.random.uniform(
                                                        low=self._gene_pool['fertility_rate'][0],
                                                        high=self._gene_pool['fertility_rate'][1],
                                                        size=m
                                                    ).astype(np.float32)

        offspring['offspring_count'][flip_mask]    = np.random.randint(
                                                        self._gene_pool['offspring_count'][0],
                                                        self._gene_pool['offspring_count'][1] + 1,
                                                        size=m
                                                    ).astype(np.int32)

        offspring['reproduction_type'][flip_mask]  = np.random.choice(
                                                        self._gene_pool['reproduction_type'],
                                                        size=m
                                                    ).astype(np.str_)

        offspring['pack_behavior'][flip_mask]      = np.random.choice(
                                                        self._gene_pool['pack_behavior'],
                                                        size=m
                                                    ).astype(np.bool_)

        offspring['symbiotic'][flip_mask]          = np.random.choice(
                                                        self._gene_pool['symbiotic'],
                                                        size=m
                                                    ).astype(np.bool_)

        offspring['swim'][flip_mask]               = np.random.choice(
                                                        self._gene_pool['swim'],
                                                        size=m
                                                    ).astype(np.bool_)

        offspring['walk'][flip_mask]               = np.random.choice(
                                                        self._gene_pool['walk'],
                                                        size=m
                                                    ).astype(np.bool_)

        offspring['fly'][flip_mask]                = np.random.choice(
                                                        self._gene_pool['fly'],
                                                        size=m
                                                    ).astype(np.bool_)

        offspring['speed'][flip_mask]              = np.random.uniform(
                                                        low=self._gene_pool['speed'][0],
                                                        high=self._gene_pool['speed'][1],
                                                        size=m
                                                    ).astype(np.float32)

        offspring['energy'] = parent_reproduction_costs
        self._organisms['energy'][reproducing] -= parent_reproduction_costs
        width, length = self._width, self._length
        raw_x = parents['x_pos'] + offset[:, 0]
        raw_y = parents['y_pos'] + offset[:, 1]


        offspring['x_pos'] = np.clip(raw_x, 0, width  - 1)
        offspring['y_pos'] = np.clip(raw_y, 0, length - 1)

        # # TODO: Possible to enhance this?
        # # Handles speciation and lineage tracking
        # for i in range(offspring.shape[0]):
        #     child = offspring['species'][i]
        #     parent = parents['species'][i]

        #     if parent == child:
        #         self._species_count[parent] += 1

        #     else:
        #         self._species_count[child] = 1
        #         self._ancestry[child] = self._ancestry[parent].copy()
        #         self._ancestry[child].append(parent)

        self._env.add_births(offspring.shape[0])
        self._organisms = np.concatenate((self._organisms, offspring))






    def spawn_initial_organisms(self, number_of_organisms: int,
                                randomize: bool = True) -> int:
        """
        Spawns the initial organisms in the simulation.
        Organism stats can be randomized if desired.
        Updates the birth counter in the environment.

        :param number_of_organisms: Number of organisms to spawn
        :param randomize: Request to randomize stats of spawned organisms
        :returns: how many organisms were actually placed
        """
        import numpy as np

        # --- get environment info ---
        env_width = self._width
        env_length = self._length
        env_terrain = self._env.get_terrain()
        n = number_of_organisms

        np.str_ = np.str_
        if randomize:
            # species label
            species_arr = np.full((n,), "ORG", dtype=np.str_)
            #
            # — MorphologicalGenes —
            size_arr        = np.random.uniform(
                low=self._gene_pool['size'][0],
                high=self._gene_pool['size'][1],
                size=(n,)
            ).astype(np.float32)

            camouflage_arr  = np.random.uniform(
                low=self._gene_pool['camouflage'][0],
                high=self._gene_pool['camouflage'][1],
                size=(n,)
            ).astype(np.float32)

            defense_arr     = np.random.uniform(
                low=self._gene_pool['defense'][0],
                high=self._gene_pool['defense'][1],
                size=(n,)
            ).astype(np.float32)

            attack_arr      = np.random.uniform(
                low=self._gene_pool['attack'][0],
                high=self._gene_pool['attack'][1],
                size=(n,)
            ).astype(np.float32)

            vision_arr      = np.random.uniform(
                low=self._gene_pool['vision'][0],
                high=self._gene_pool['vision'][1],
                size=(n,)
            ).astype(np.float32)
            #
            # — MetabolicGenes —
            metabolism_rate_arr = np.random.uniform(
                low=self._gene_pool['metabolism_rate'][0],
                high=self._gene_pool['metabolism_rate'][1],
                size=(n,)
            ).astype(np.float32)

            nutrient_efficiency_arr = np.random.uniform(
                low=self._gene_pool['nutrient_efficiency'][0],
                high=self._gene_pool['nutrient_efficiency'][1],
                size=(n,)
            ).astype(np.float32)

            diet_type_arr = np.random.choice(self._gene_pool['diet_type'], size=n).astype(np.str_)
            #
            # — ReproductionGenes —
            fertility_rate_arr = np.random.uniform(
                low=self._gene_pool['fertility_rate'][0],
                high=self._gene_pool['fertility_rate'][1],
                size=(n,)
            ).astype(np.float32)

            offspring_count_arr = np.random.randint(
                self._gene_pool['offspring_count'][0],
                self._gene_pool['offspring_count'][1] + 1,
                size=(n,)
            ).astype(np.int32)

            reproduction_type_arr = np.random.choice(
                self._gene_pool['reproduction_type'],
                size=n
            ).astype(np.str_)

            pack_behavior_arr = np.random.choice(
                self._gene_pool['pack_behavior'],
                size=n
            ).astype(np.bool_)

            symbiotic_arr = np.random.choice(
                self._gene_pool['symbiotic'],
                size=n
            ).astype(np.bool_)
            # — LocomotionGenes —
            swim_arr = np.random.choice(self._gene_pool['swim'], size=n).astype(np.bool_)
            walk_arr = np.random.choice(self._gene_pool['walk'], size=n).astype(np.bool_)
            fly_arr  = np.random.choice(self._gene_pool['fly'],  size=n).astype(np.bool_)

            speed_arr = np.random.uniform(
                low=self._gene_pool['speed'][0],
                high=self._gene_pool['speed'][1],
                size=(n,)
            ).astype(np.float32)
            #
            # — Simulation bookkeeping —
            energy_arr = np.random.uniform(
                low=10,
                high=30,
                size=(n,)
            ).astype(np.float32)

        else:
            species_arr = np.full((n,), "ORG", dtype=np.str_)
            size_arr = np.full((n,), 1.0, dtype=np.float32)
            camouflage_arr = np.zeros((n,), dtype=np.float32)
            defense_arr = np.zeros((n,), dtype=np.float32)
            attack_arr = np.zeros((n,), dtype=np.float32)
            #
            # or based on env scale
            vision_arr = np.full((n,), 15, dtype=np.float32)
            metabolism_rate_arr = np.full((n,), 1.0, dtype=np.float32)
            nutrient_efficiency_arr = np.full((n,), 1.0, dtype=np.float32)
            diet_type_arr = np.full((n,), self._gene_pool['diet_type'][0], dtype=np.str_)

            fertility_rate_arr = np.full((n,), 0.1, dtype=np.float32)
            offspring_count_arr = np.full((n,), 1, dtype=np.int32)
            reproduction_type_arr = np.full((n,), self._gene_pool['reproduction_type'][0], dtype=np.str_)

            pack_behavior_arr = np.full((n,), False, dtype=np.bool_)
            symbiotic_arr = np.full((n,), False, dtype=np.bool_)

            swim_arr = np.full((n,), False, dtype=np.bool_)
            walk_arr = np.full((n,), True,  dtype=np.bool_)
            fly_arr = np.full((n,), False, dtype=np.bool_)
            speed_arr = np.full((n,), 1.0,  dtype=np.float32)
            energy_arr = np.full((n,), 20, dtype=np.float32)

        # --- pick random positions and filter to valid land cells ---
        positions = np.random.randint(0, env_width, size=(n, 2)).astype(np.int32)


        # 2) Lookup terrain at each candidate
        ix = positions[:, 0]
        iy = positions[:, 1]
        terrain_values = env_terrain[iy, ix]

        # 3) Build locomotion-only masks
        swim_only = swim_arr & ~walk_arr & ~fly_arr
        walk_only = walk_arr & ~swim_arr & ~fly_arr
        # flyers: just fly_arr
        
        
        # 4) Filter out invalid positions
        valid_fly   = positions[ fly_arr ]
        valid_swim  = positions[ swim_only &  (terrain_values <  0) ]
        valid_walk  = positions[ walk_only & (terrain_values >= 0) ]

        # 5) Concatenate all valid positions, count them
        positions = np.concatenate((valid_fly, valid_swim, valid_walk), axis=0)
        valid_count = positions.shape[0]
        
        # --- truncate all arrays to the number of valid spots ---
        # --- pack into structured array ---
        spawned = np.zeros(valid_count, dtype=self._organism_dtype)
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
        spawned['energy']             = energy_arr[:valid_count]
        # positions is int32 → cast to float32 for x_pos/y_pos
        positions_f = positions.astype(np.float32)
        spawned['x_pos']             = positions_f[:, 0]
        spawned['y_pos']             = positions_f[:, 1]

        
        # --- append to full array and update births ---
        self._organisms = np.concatenate((self._organisms, spawned))
        self._env.add_births(valid_count)

        return valid_count

    def move(self):
        orgs = self._organisms
        N = orgs.shape[0]
        if N == 0:
            return

        terrain = self._env.get_terrain()
        width, length = self._width, self._length
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

        coords = np.stack((orgs['x_pos'], orgs['y_pos']), axis=1)
        vision_radii = orgs['vision']
        neigh_lists = self._pos_tree.query_ball_point(coords, vision_radii)


        # precompute once per tick, outside the per‐organism loop:
        # Wipes buffer for movement
        dirs = self._dirs

        # coords: (N,2) array of current positions
        samples = coords[:, None, :] + dirs[None, :, :]  # shape (N,4,2)

        # floor to grid‐indices
        ix = samples[..., 0].astype(int)  # (N,4)
        iy = samples[..., 1].astype(int)  # (N,4)

        # mask out‐of‐bounds
        valid = (
            (ix >= 0) & (ix < width) &
            (iy >= 0) & (iy < length)
        )  # (N,4)

        # lookup terrain values, fill invalid with a safe default
        tiles = np.full((N, 4), np.nan, dtype=np.float32)
        tiles[valid] = terrain[iy[valid], ix[valid]]

        # for water-avoidance:
        mask_water = tiles < 0     # (N,4)
        avoid_water = - (dirs[None, :, :] * mask_water[..., None]).sum(axis=1)
        # for land-avoidance:
        mask_land = tiles >= 0     # (N,4)
        avoid_land = - (dirs[None, :, :] * mask_land[..., None]).sum(axis=1)

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

        def _compute(i, pos, neighs):
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

                my['energy'] += 1
                
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
                dist = np.linalg.norm(new - pos)
                # use the per‐organism metabolism_rate
                my['energy'] -= dist * 0.01* my['metabolism_rate']
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
            dist = np.linalg.norm(new - pos)
            my['energy'] -= dist * 0.01 * my['metabolism_rate']

            return new
        old_coords = coords
        # map across all organisms

        new_pos = np.array([
            _compute(i, coords[i], neigh_lists[i])
            for i in range(N)
        ], dtype=np.float32)

        orgs['x_pos'], orgs['y_pos'] = new_pos[:, 0], new_pos[:, 1]
        distances = np.linalg.norm(new_pos - old_coords, axis=1)
        move_costs = 0.01* distances * orgs['metabolism_rate']
        orgs['energy'] -= move_costs

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

    def kill_border(self, margin: float = 0.05):
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
            (x > (1-margin) * w) |
            (y <  margin * h) |
            (y > (1-margin) * h)
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
