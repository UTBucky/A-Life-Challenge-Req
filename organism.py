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

        # --- 1) get environment info ---
        env_width = self._env.get_width()
        env_length = self._env.get_length()
        env_terrain = self._env.get_terrain()
        grid_size = env_width  # TODO: handle non‐square later

        # --- 2) build raw parameter arrays ---
        n = number_of_organisms
        # string dtype shorthand
        str15 = np.dtype('U15')

        if randomize:
            # species label
            species_arr = np.full((n,), "ORG", dtype=str15)
            # — MorphologicalGenes —
            size_arr = np.random.rand(n).astype(np.float32)
            camouflage_arr = np.random.rand(n).astype(np.float32)
            defense_arr = np.random.rand(n).astype(np.float32)
            attack_arr = np.random.rand(n).astype(np.float32)
            vision_arr = np.random.rand(n).astype(np.float32)
            # — MetabolicGenes —
            metabolism_rate_arr = np.random.rand(n).astype(np.float32)
            nutrient_efficiency_arr = np.random.rand(n).astype(np.float32)
            diet_type_arr = np.full((n,), "heterotroph", dtype=str15)
            # — ReproductionGenes —
            fertility_rate_arr = np.random.rand(n).astype(np.float32)
            offspring_count_arr = np.random.randint(
                1, 5, size=(n,)).astype(np.int32)
            reproduction_type_arr = np.full((n,), "asexual", dtype=str15)
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
            # — Simulation bookkeeping —
            energy_arr = np.random.uniform(
                10, 20, size=(n,)).astype(np.float32)
        else:
            species_arr = np.full((n,), "ORG", dtype=str15)
            size_arr = np.full((n,), 1.0, dtype=np.float32)
            camouflage_arr = np.zeros((n,), dtype=np.float32)
            defense_arr = np.zeros((n,), dtype=np.float32)
            attack_arr = np.zeros((n,), dtype=np.float32)
            vision_arr = np.full((n,), 15, dtype=np.float32)  # or based on env scale
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

        # --- 3) pick random positions and filter to valid land cells ---
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
        land_mask = env_terrain[iy, ix] >= 0
        positions = positions[land_mask]

        valid_count = positions.shape[0]

        # --- 4) truncate all arrays to the number of valid spots ---
        def _trim(arr):
            return arr[:valid_count]

        species_arr = _trim(species_arr)
        size_arr = _trim(size_arr)
        camouflage_arr = _trim(camouflage_arr)
        defense_arr = _trim(defense_arr)
        attack_arr = _trim(attack_arr)
        vision_arr = _trim(vision_arr)
        metabolism_rate_arr = _trim(metabolism_rate_arr)
        nutrient_efficiency_arr = _trim(nutrient_efficiency_arr)
        diet_type_arr = _trim(diet_type_arr)
        fertility_rate_arr = _trim(fertility_rate_arr)
        offspring_count_arr = _trim(offspring_count_arr)
        reproduction_type_arr = _trim(reproduction_type_arr)
        pack_behavior_arr = _trim(pack_behavior_arr)
        symbiotic_arr = _trim(symbiotic_arr)
        swim_arr = _trim(swim_arr)
        walk_arr = _trim(walk_arr)
        fly_arr = _trim(fly_arr)
        speed_arr = _trim(speed_arr)
        energy_arr = _trim(energy_arr)

        # --- 5) pack into structured array ---
        spawned = np.zeros((valid_count,), dtype=self._organism_dtype)
        spawned['species'] = species_arr
        spawned['size'] = size_arr
        spawned['camouflage'] = camouflage_arr
        spawned['defense'] = defense_arr
        spawned['attack'] = attack_arr
        spawned['vision'] = vision_arr
        spawned['metabolism_rate'] = metabolism_rate_arr
        spawned['nutrient_efficiency'] = nutrient_efficiency_arr
        spawned['diet_type'] = diet_type_arr
        spawned['fertility_rate'] = fertility_rate_arr
        spawned['offspring_count'] = offspring_count_arr
        spawned['reproduction_type'] = reproduction_type_arr
        spawned['pack_behavior'] = pack_behavior_arr
        spawned['symbiotic'] = symbiotic_arr
        spawned['swim'] = swim_arr
        spawned['walk'] = walk_arr
        spawned['fly'] = fly_arr
        spawned['speed'] = speed_arr
        spawned['energy'] = energy_arr
        spawned['x_pos'] = positions[:, 0]
        spawned['y_pos'] = positions[:, 1]

        # --- 6) append to full array and update births ---
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

        coords = np.stack((orgs['x_pos'], orgs['y_pos']), axis=1)
        vision_radii = orgs['vision']
        neigh_lists = self._pos_tree.query_ball_point(coords, vision_radii)

        width, length = self._env.get_width(), self._env.get_length()
        terrain = self._env.get_terrain()
        dirs = np.array([[1,0],[-1,0],[0,1],[0,-1]], np.float32)

        # precompute once per tick, outside the per‐organism loop:
        dirs = np.array([[ 1, 0],
                        [-1, 0],
                        [ 0, 1],
                        [ 0,-1]], dtype=np.float32)   # (4,2)

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

        # Now avoid_water[i] is your 2D vector to push organism i away from adjacent water,
        # and similarly avoid_land[i] for land.

        def _compute(i, pos, neighs):
            my = orgs[i]
            # camouflage filter
            valid = [j for j in neighs if j!=i and orgs['vision'][j]>=my['camouflage']]

            # behavioral overrides
            if my['pack_behavior']:
                return self.move_pack_behavior(i, pos, valid)
            if my['symbiotic']:
                return self.move_symbiotic(i, pos, valid)
            
            WATER_PUSH = 5.0
            LAND_PUSH  = 5.0
            # build movement vector
            move_vec = np.zeros(2, dtype=np.float32)
            if not my['swim']:
                move_vec += WATER_PUSH * avoid_water[i]
            if not my['walk']:
                move_vec += LAND_PUSH * avoid_land[i]

            # social steering
            pool = [j for j in valid if (orgs['fly'][j] if my['fly'] else True)]
            hostiles = [j for j in pool if orgs['attack'][j] > my['defense']]
            prey     = [j for j in pool if my['attack'] > orgs['defense'][j]]
            if hostiles:
                move_vec += (pos - coords[hostiles]).mean(axis=0)
            if prey:
                move_vec += (coords[prey] - pos).mean(axis=0)
            
            CROWD_PUSH = 0.5 * my['speed']
            same = [j for j in valid if orgs['species'][j] == my['species']]
            if same:
                # average repulsion vector away from the group
                repulse = np.mean(pos - coords[same], axis=0)
                move_vec += CROWD_PUSH * repulse
            
            # normalize & scale
            norm = np.linalg.norm(move_vec)
            step = (move_vec/norm)*my['speed'] if norm>0 else np.zeros(2,np.float32)

            new = pos + step
            new[0] = np.clip(new[0], 0, width-1)
            new[1] = np.clip(new[1], 0, length-1)
            return new

        # map across all organisms
        new_pos = np.array([
            _compute(i, coords[i], neigh_lists[i])
            for i in range(N)
        ], dtype=np.float32)

        orgs['x_pos'], orgs['y_pos'] = new_pos[:,0], new_pos[:,1]
        self.build_spatial_index()


        

    # TODO: Cleanup since organisms eat other organisms
    # Once we deal with speciation, organisms will eat plantlike organisms
    # as an example
    def consume_organism(self):
        """
        """
        pass

    # TODO: Add method for organizim decision making
    def take_action(self):
        pass

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
