import numpy as np


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
            ('species', np.str_, 15),
            ('size', np.float32),
            ('speed', np.float32),
            ('max_age', np.float32),
            ('energy_capacity', np.float32),
            ('move_eff', np.float32),
            ('reproduction_eff', np.float32),
            ('min_temp_tol', np.float32),
            ('max_temp_tol', np.float32),
            ('energy_prod', np.str_, 15),
            ('move_aff', np.str_, 15),
            ('energy', np.float32),
            ('x_pos', np.float32),
            ('y_pos', np.float32),
        ])

        self._organisms = np.zeros((0,), dtype=self.organism_dtype)
        self._env = env
        # TODO: Load genes from json file
        self._gene_pool = None

    # Get methods
    def get_organisms(self):
        return self._organisms

    # TODO: Split logic into smaller functions for readability
    def spawn_initial_organisms(self, number_of_organisms: int,
                                randomize: bool = False) -> int:
        """
        Spawns the initial organisms in the simulation.
        Organism stats can be randomized if desired.
        Updates the birth counter in the environment.

        :param number_of_organisms: Number of organisms to spawn
        :param randomize: Request to randomize stats of spawned organisms
        """

        # Unpack important values from args
        env_width = self._env.get_width()
        env_length = self._env.get_length()
        env_terrain = self._env.get_terrain()
        # TODO: Change if environment can be NxM too
        grid_size = env_width

        # Use gene pool to randomize starting organisms if requested
        if randomize:

            # TODO: Randomize the rest of the genes
            speeds = np.random.randint(1, 5,
                                       size=(number_of_organisms,
                                             ).astype(np.float32))

        # All initial organisms start with the same stats
        # TODO: Use gene pool to create default values, currently hard coded
        else:
            species = np.full((number_of_organisms,), "ORG", dtype=np.str_)
            sizes = np.full((number_of_organisms,), 1, dtype=np.float32)
            speeds = np.full((number_of_organisms,), 1, dtype=np.float32)
            max_ages = np.full((number_of_organisms,), 5, dtype=np.float32)
            energy_capacities = np.full((number_of_organisms,),
                                        1.0, dtype=np.float32)
            move_efficiencies = np.full((number_of_organisms,),
                                        0.01, dtype=np.float32)
            reproduction_efficiencies = np.full((number_of_organisms,),
                                                0.1, dtype=np.float32)
            min_temp_tols = np.full((number_of_organisms,),
                                    2, dtype=np.float32)
            max_temp_tols = np.full((number_of_organisms,),
                                    2, dtype=np.float32)
            energy_productions = np.full((number_of_organisms,),
                                         "heterotroph", dtype=np.str_)
            move_affordances = np.full((number_of_organisms,),
                                       "terrestrial", dtype=np.str_)
            energies = np.full((number_of_organisms,), 0.5, dtype=np.float32)

        # Randomize starting positions
        positions = np.random.randint(0, grid_size,
                                      size=(number_of_organisms, 2)
                                      ).astype(np.float32)

        # Clip positions that are out of bound
        positions = positions[
            (positions[:, 0] >= 0) & (positions[:, 0] < env_width) &
            (positions[:, 1] >= 0) & (positions[:, 1] < env_length)
        ]

        # TODO: Ensure that the number requested is spawned rather than
        #       just valid positions
        # Take rows and columns of the positions and verify those on land
        ix = positions[:, 0].astype(np.int32)
        iy = positions[:, 1].astype(np.int32)
        land_filter = env_terrain[iy, ix] >= 0
        positions = positions[land_filter]
        valid_count = positions.shape[0]

        # Cut stat arrays to match valid count of organism spawn positions
        species = species[:valid_count]
        speeds = speeds[:valid_count]
        sizes = sizes[:valid_count]
        max_ages = max_ages[:valid_count]
        energy_capacities = energy_capacities[:valid_count]
        move_efficiencies = move_efficiencies[:valid_count]
        reproduction_efficiencies = reproduction_efficiencies[:valid_count]
        min_temp_tols = min_temp_tols[:valid_count]
        max_temp_tols = max_temp_tols[:valid_count]
        energy_productions = energy_productions[:valid_count]
        move_affordances = move_affordances[:valid_count]
        energies = energies[:valid_count]

        # Create array of spawned organisms
        spawned_orgs = np.zeros((valid_count,), dtype=self.organism_dtype)
        spawned_orgs['species'] = species
        spawned_orgs['size'] = sizes
        spawned_orgs['speed'] = speeds
        spawned_orgs['max_age'] = max_ages
        spawned_orgs['energy_capacity'] = energy_capacities
        spawned_orgs['move_eff'] = move_efficiencies
        spawned_orgs['reproduction_eff'] = reproduction_efficiencies
        spawned_orgs['min_temp_tol'] = min_temp_tols
        spawned_orgs['max_temp_tol'] = max_temp_tols
        spawned_orgs['energy_prod'] = energy_productions
        spawned_orgs['move_aff'] = move_affordances
        spawned_orgs['energy'] = energies
        spawned_orgs['x_pos'] = positions[:, 0]
        spawned_orgs['y_pos'] = positions[:, 1]

        # Add new data to existing organisms array
        self._organisms = np.concatenate((self._organisms, spawned_orgs))
        self._env.add_births(self._organisms.shape[0])

    # TODO: Implement mutation and
    #       eventually different sexual reproduction types
    def reproduce(self):
        """
        Causes all organisms that can to reproduce.
        Spawns offspring near the parent
        """

        # Obtains an array of all reproducing organisms
        reproducing = (self._organisms['energy'] >
                       self._organisms['reproduction_eff']
                       * self._organisms['energy_capacity'])

        if np.any(reproducing):

            parents = self._organisms[reproducing]
            parent_reproduction_costs = (self._organisms['reproduction_eff']
                                         [reproducing]
                                         *
                                         self._organisms['energy_capacity']
                                         [reproducing])

            # Put children randomly nearby
            offset = np.random.uniform(-2, 2, size=(parents.shape[0], 2))
            offspring = np.zeros((parents.shape[0],),
                                 dtype=self._organism_dtype)
            offspring['x_pos'] = parents['x_pos'] + offset[:, 0]
            offspring['y_pos'] = parents['y_pos'] + offset[:, 1]

            # Create offspring stats
            offspring['energy'] = parent_reproduction_costs
            self._organisms['energy'][reproducing] -= parent_reproduction_costs
            # TODO: Implement way to mutate offspring genes
            self._organisms = np.concatenate((self.organisms, offspring))

            self._env.add_births(offspring.shape[0])

    # TODO: Add cost to organism movement based on the movement efficiency
    def move_org(self):
        """
        Moves all organisms.
        """
        alive = (self._organisms['energy'] > 0)
        speed = self._organisms['speed'][alive][:, None]
        jitter_shape = (alive.sum(), 2)
        self.move_jitter = np.random.uniform(-1, 1, size=jitter_shape) * speed
        self.new_positions = np.stack(
            (
                self.organisms['x_pos'][alive], self._organisms['y_pos'][alive]
                ), axis=1
            ) + self.move_jitter

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
