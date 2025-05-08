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
        # TODO: Load genes from json file
        self._gene_pool = None

    def load_genes(self, gene_pool):
        self._gene_pool = gene_pool

    # Get methods
    def get_organisms(self):
        return self._organisms

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
        import numpy as np

        # --- get environment info ---
        env_width = self._env.get_width()
        env_length = self._env.get_length()
        env_terrain = self._env.get_terrain()
        grid_size = env_width

        # --- build raw parameter arrays ---
        n = number_of_organisms
        # string dtype shorthand
        str15 = np.dtype('U15')

        if randomize:
            # species label
            species_arr = np.full((n,), "ORG", dtype=str15)
            #
            # — MorphologicalGenes —
            size_arr = np.random.rand(n).astype(np.float32)
            camouflage_arr = np.random.rand(n).astype(np.float32)
            defense_arr = np.random.rand(n).astype(np.float32)
            attack_arr = np.random.rand(n).astype(np.float32)
            vision_arr = np.random.rand(n).astype(np.float32)
            #
            # — MetabolicGenes —
            metabolism_rate_arr = np.random.rand(n).astype(np.float32)
            nutrient_efficiency_arr = np.random.rand(n).astype(np.float32)
            diet_type_arr = np.full((n,), "heterotroph", dtype=str15)
            #
            # — ReproductionGenes —
            fertility_rate_arr = np.random.rand(n).astype(np.float32)
            offspring_count_arr = np.random.randint(
                1, 5, size=(n,)).astype(np.int32)
            reproduction_type_arr = np.full((n,), "asexual", dtype=str15)
            #
            # — BehavioralGenes —
            pack_behavior_arr = np.random.choice(
                [False, True], size=(n,)).astype(np.bool_)
            symbiotic_arr = np.random.choice(
                [False, True], size=(n,)).astype(np.bool_)
            # — LocomotionGenes —
            swim_arr = np.random.choice(
                [False, True], size=(n,)).astype(np.bool_)
            walk_arr = np.random.choice(
                [False, True], size=(n,)).astype(np.bool_)
            fly_arr = np.random.choice(
                [False, True], size=(n,)).astype(np.bool_)
            speed_arr = np.random.uniform(
                0.1, 5.0, size=(n,)).astype(np.float32)
            #
            # — Simulation bookkeeping —
            energy_arr = np.random.uniform(
                10, 20, size=(n,)).astype(np.float32)
        else:
            species_arr = np.full((n,), "ORG", dtype=str15)
            size_arr = np.full((n,), 1.0, dtype=np.float32)
            camouflage_arr = np.zeros((n,), dtype=np.float32)
            defense_arr = np.zeros((n,), dtype=np.float32)
            attack_arr = np.zeros((n,), dtype=np.float32)
            #
            # or based on env scale
            vision_arr = np.full((n,), 15, dtype=np.float32)
            metabolism_rate_arr = np.full((n,), 1.0, dtype=np.float32)
            nutrient_efficiency_arr = np.full((n,), 1.0, dtype=np.float32)
            diet_type_arr = np.full((n,), "heterotroph", dtype=str15)

            fertility_rate_arr = np.full((n,), 0.1, dtype=np.float32)
            offspring_count_arr = np.full((n,), 1, dtype=np.int32)
            reproduction_type_arr = np.full((n,), "asexual", dtype=str15)

            pack_behavior_arr = np.full((n,), False, dtype=np.bool_)
            symbiotic_arr = np.full((n,), False, dtype=np.bool_)

            swim_arr = np.full((n,), False, dtype=np.bool_)
            walk_arr = np.full((n,), True,  dtype=np.bool_)
            fly_arr = np.full((n,), False, dtype=np.bool_)
            speed_arr = np.full((n,), 1.0,  dtype=np.float32)

            energy_arr = np.full((n,), 20, dtype=np.float32)

        # --- pick random positions and filter to valid land cells ---
        positions = (
            np.random.randint(0, grid_size, size=(n, 2))
            .astype(np.float32)
        )
        # bounds check
        mask = (
            (positions[:, 0] >= 0) & (positions[:, 0] < env_width) &
            (positions[:, 1] >= 0) & (positions[:, 1] < env_length)
        )
        positions = positions[mask]

        # land check
        ix = positions[:, 0].astype(np.int32)
        iy = positions[:, 1].astype(np.int32)
        terrain_values = env_terrain[iy, ix]

        swim_only = swim_arr & ~walk_arr & ~fly_arr
        walk_only = walk_arr & ~swim_arr & ~fly_arr
        # Flyers can be anywhere
        valid_fly_positions = positions[fly_arr]
        valid_swim_positions = positions[swim_only & (
            terrain_values < 0)]  # Swimmers need water
        valid_walk_positions = positions[walk_only & (
            terrain_values >= 0)]  # Walkers need land
        positions = np.concatenate(
            (valid_fly_positions, valid_swim_positions, valid_walk_positions), axis=0)

        valid_count = positions.shape[0]

        # --- truncate all arrays to the number of valid spots ---
        # --- pack into structured array ---
        spawned = np.zeros((valid_count,), dtype=self._organism_dtype)
        spawned['species'] = species_arr[:valid_count]
        spawned['size'] = size_arr[:valid_count]
        spawned['camouflage'] = camouflage_arr[:valid_count]
        spawned['defense'] = defense_arr[:valid_count]
        spawned['attack'] = attack_arr[:valid_count]
        spawned['vision'] = vision_arr[:valid_count]
        spawned['metabolism_rate'] = metabolism_rate_arr[:valid_count]
        spawned['nutrient_efficiency'] = nutrient_efficiency_arr[:valid_count]
        spawned['diet_type'] = diet_type_arr[:valid_count]
        spawned['fertility_rate'] = fertility_rate_arr[:valid_count]
        spawned['offspring_count'] = offspring_count_arr[:valid_count]
        spawned['reproduction_type'] = reproduction_type_arr[:valid_count]
        spawned['pack_behavior'] = pack_behavior_arr[:valid_count]
        spawned['symbiotic'] = symbiotic_arr[:valid_count]
        spawned['swim'] = swim_arr[:valid_count]
        spawned['walk'] = walk_arr[:valid_count]
        spawned['fly'] = fly_arr[:valid_count]
        spawned['speed'] = speed_arr[:valid_count]
        spawned['energy'] = energy_arr[:valid_count]
        spawned['x_pos'] = positions[:, 0]
        spawned['y_pos'] = positions[:, 1]

        # --- append to full array and update births ---
        self._organisms = np.concatenate((self._organisms, spawned))
        self._env.add_births(valid_count)

        return valid_count
    # TODO: Implement mutation and
    #       eventually different sexual reproduction types

    def reproduce(self, arg1, arg2):
        pass

    def move(self):
        orgs = self._organisms
        N = orgs.shape[0]
        if N == 0:
            return

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
        orgs['energy'][penalty] -= 5

        coords = np.stack((orgs['x_pos'], orgs['y_pos']), axis=1)
        vision_radii = orgs['vision']
        neigh_lists = self._pos_tree.query_ball_point(coords, vision_radii)

        width, length = self._env.get_width(), self._env.get_length()
        terrain = self._env.get_terrain()

        # precompute once per tick, outside the per‐organism loop:
        dirs = np.array([[1, 0],
                        [-1, 0],
                        [0, 1],
                        [0, -1]], dtype=np.float32)   # (4,2)

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
            my_cam = my['camouflage']
            my_att = my['attack']
            my_def = my['defense']
            my_spc = my['species']
            my_pack = pack_flag[i]
            my_fly = fly_flag[i]
            my_swim = swim_flag[i]
            my_walk = walk_flag[i]
            my_speed = speed[i]

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

        # map across all organisms
        new_pos = np.array([
            _compute(i, coords[i], neigh_lists[i])
            for i in range(N)
        ], dtype=np.float32)

        orgs['x_pos'], orgs['y_pos'] = new_pos[:, 0], new_pos[:, 1]
        self.build_spatial_index()

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
