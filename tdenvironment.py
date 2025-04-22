from noise import pnoise2
from scipy.ndimage import zoom, gaussian_gradient_magnitude
import numpy as np


class TDEnvironment:
    """
    TDEnvironment(width, height):
        - Stores width/height
        - Terrain is stored in an np array of floats
        - Contains generation counter
        - Initializes organism_dtype
            - stores unpacked organism object
            - also has organism object contained
        - self.organisms is an array with organism_dtype,
        allows for vectorized operations using numpy
        - food gradient is initialized on class creation
        and keeps one gradient as baseline for replenishment
        - energy is copied from food gradient created, this one
        becomes depleted over time

    """
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.terrain = np.zeros((height, width), dtype=np.float32)
        self._generation = 0
        self._total_births = 0
        self._total_deaths = 0
        self.organism_dtype = np.dtype([
            ('x', np.float32),
            ('y', np.float32),
            ('speed', np.float32),
            ('energy', np.float32),
            ('alive', np.bool_),
            ('org_ref', object)
        ])
        self.organisms = np.zeros((0,), dtype=self.organism_dtype)
        # Food energy gradient
        self.food_baseline = generate_fractal_terrain(width, height, seed=42)
        self.food_energy = np.copy(self.food_baseline)

    def set_terrain(self, terrain_mask: np.ndarray):
        """
        Applies terrain_mask to the environment and saves it
        """
        if terrain_mask.shape != self.terrain.shape:
            raise ValueError('Your terrain mask is wrong!!!!!')
        if not np.array_equal(self.terrain, terrain_mask):
            self.terrain[...] = terrain_mask.astype(np.float32)

    def replenish_food(self, replenish_rate=0.01):
        """
        Smoothly shift the food gradient towards baseline.
        - Because the gradient is a vector this method
        is extremely efficient since it uses matrix
        broadcasting
        """
        diff = self.food_baseline - self.food_energy
        self.food_energy += diff * replenish_rate

    # Organisms consume food at their position
    def consume_food(self, consumption_rate=0.1):
        """
        1 Argument, consumption rate
        Organisms consume food at their current location
        - organism energy is updated based on what they consume
        - *we need to change this to be based on internal organism
        attributes*
        """
        ix = self.organisms['x'].astype(np.int32)
        iy = self.organisms['y'].astype(np.int32)
        # Consume food and transfer energy to organisms
        energy_available = self.food_energy[iy, ix]
        consumed_energy = np.minimum(energy_available, consumption_rate)

        # Update organism energy
        self.organisms['energy'] += consumed_energy

        # Temporarily reduce food gradient
        self.food_energy[iy, ix] -= consumed_energy

    def add_organisms(self, positions, speeds=None, org_refs=None):
        """
        Takes a list of positions and converts them to np array
        - Filters for being in bound/on land/non-colliding
        - Adds organism with position/speed/energy
        """
        # Using numpy backend, directly interact with the
        # backend python list object and change it to a
        # numpy array
        # confirm the shape is correct before continuing
        positions = np.asarray(positions, dtype=np.float32)
        if positions.ndim != 2 or positions.shape[1] != 2:
            return

        # Clip positions that are out of bound, return if no valid positions
        positions = positions[
            (positions[:, 0] >= 0) & (positions[:, 0] < self.width) &
            (positions[:, 1] >= 0) & (positions[:, 1] < self.height)
        ]
        if positions.size == 0:
            return

        # Take rows and columns of the positions and verify those on land
        ix = positions[:, 0].astype(np.int32)
        iy = positions[:, 1].astype(np.int32)
        land_filter = self.terrain[iy, ix] >= 0
        positions = positions[land_filter]
        if positions.size == 0:
            return
        count = positions.shape[0]

        # If no input speed, default to 1.0 speed
        if speeds is None:
            speeds = np.full((count,), 1.0, dtype=np.float32)
        # If no organism given default, initialize obj to none
        if org_refs is None:
            org_refs = [None] * count

        # Get speeds and org_refs for packing new data
        speeds = speeds[:count]
        org_refs = org_refs[:count]

        # Make array for new data
        new_data = np.zeros((count,), dtype=self.organism_dtype)
        new_data['x'] = positions[:, 0]
        new_data['y'] = positions[:, 1]
        new_data['speed'] = speeds
        new_data['energy'] = 10.0
        new_data['alive'] = True
        new_data['org_ref'] = org_refs

        # Add new data to existing organisms array
        self.organisms = np.concatenate((self.organisms, new_data))
        self._total_births += self.organisms.shape[0]

    def move_org(self):
        """
        Environment method
        - Broadcasts over organism array to get speed
        and alive organisms
        - Currently uses jittering, will use gradient
        ascent later
        """
        alive = self.organisms['alive']
        speed = self.organisms['speed'][alive][:, None]
        jitter_shape = (alive.sum(), 2)
        self.move_jitter = np.random.uniform(-1, 1, size=jitter_shape) * speed
        self.new_positions = np.stack(
            (
                self.organisms['x'][alive], self.organisms['y'][alive]
                ), axis=1
            ) + self.move_jitter

    def inbounds(self):
        """
        Environment method
        - Clips in bound positions
        - To reduce overhead also creates land mask for use
        in verify_org_position
        """
        self.new_positions[:, 0] = np.clip(
            self.new_positions[:, 0], 0, self.width - 1
            )
        self.new_positions[:, 1] = np.clip(
            self.new_positions[:, 1], 0, self.height - 1
            )
        ix = self.new_positions[:, 0].astype(np.int32)
        iy = self.new_positions[:, 1].astype(np.int32)
        self.land_mask = self.terrain[iy, ix] >= 0

    def verify_org_position(self):
        """
        Environment method
        - verifies position of alive organisms
        """
        # Get indicies of alive organisms
        idx = np.flatnonzero(self.organisms['alive'])
        # Valid are the organisms on land
        valid = self.land_mask
        # 'alive' is a bool array, cast the 'valid' matrix
        # to soft remove organisms not on land
        self.organisms['alive'][idx] = valid
        self.organisms['x'][idx[valid]] = self.new_positions[valid][:, 0]
        self.organisms['y'][idx[valid]] = self.new_positions[valid][:, 1]

    def soft_remove_dead(self):
        """
        Alive organisms are those that have energy
        """
        pre = self.organisms['alive'].copy()
        self.organisms['alive'] &= self.organisms['energy'] > 0
        self._total_deaths += np.count_nonzero(pre & ~self.organisms['alive'])

    def compact(self):
        """
        Environment method
        - Assists in memory management
        - Because we use concatenate to stack arrays our arrays grow
        - As organisms die we remove them from the environment
        every 100 generations to improve performace of the array
        """
        self.organisms = self.organisms[self.organisms['alive']]

    def reproduce(self):
        """
        Environment method
        - Handles organism reproduction
        - confirms live organisms, currently does not track
        parent organisms but can be changed
        - creates new offspring all at once
        """
        # Only live organisms with energy greater than 20 can reproduce
        reproducing = (
            self.organisms['energy'] > 20.0
            ) & self.organisms['alive']
        if not np.any(reproducing):
            return
        parents = self.organisms[reproducing]
        # Put children randomly nearby
        offset = np.random.uniform(-2, 2, size=(parents.shape[0], 2))
        offspring = np.zeros((parents.shape[0],), dtype=self.organism_dtype)
        offspring['x'] = parents['x'] + offset[:, 0]
        offspring['y'] = parents['y'] + offset[:, 1]
        offspring['speed'] = parents['speed']
        # We can change this to make it more realistic later
        offspring['energy'] = 10.0  # Fresh energy
        offspring['alive'] = True
        offspring['org_ref'] = parents['org_ref']  # Optional: clone reference
        self.organisms['energy'][reproducing] *= 0.5
        self.organisms = np.concatenate((self.organisms, offspring))
        self._total_births += offspring.shape[0]

    def get_generation(self):
        """
        Returns encapsulated generation data member
        """
        return self._generation

    def get_total_births(self):
        """
        Returns encapsulated generation data member
        """
        return self._total_births

    def get_total_deaths(self):
        """
        Returns encapsulated generation data member
        """
        return self._total_deaths

    def step(self):
        """
        Simulates one timestep of the environment:
        - Organisms move randomly within terrain.
        - Consume energy each step.
        - Reproduce if energy is high enough.
        - Die if energy is depleted.
        """
        if self.organisms.shape[0] == 0:
            return
        self.move_org()
        self.inbounds()
        self.verify_org_position()
        alive_idx = np.flatnonzero(self.organisms['alive'])
        displacement = np.abs(
            self.move_jitter[:alive_idx.shape[0]]
            ).sum(axis=1)
        self.organisms['energy'][alive_idx] -= 0.05 * displacement
        # Consume food from gradient
        self.consume_food()

        # Natural food replenishment
        self.replenish_food()
        self.soft_remove_dead()
        self.reproduce()
        self._generation += 1
        if self._generation % 100 == 0:
            self.compact()


def generate_fractal_terrain(
    width,
    height,
    num_octaves=4,
    base_res=10,  # slightly coarser base resolution
    persistence=0.45,  # smoother transitions between octaves
    steepness_damping=0.4,  # increase slope damping to encourage flatness
    erosion_passes=4,  # slightly more erosion passes
    erosion_strength=0.015,  # increased erosion removes more material
    seed=None
):
    """
    Creates Terrain
    - fractal perlin noise generation with the noise library
    - Slope based simple gradient erosion
    - Not too important to understand, terrain generation
    methods came from a very useful online blogpost
    """
    terrain = np.zeros((height, width), dtype=np.float32)
    damping_mask = np.ones((height, width), dtype=np.float32)
    rng = np.random.default_rng(seed)
    actual_seed = seed if seed is not None else rng.integers(0, 1_000_000)
    print("[Terrain Gen] Seed used:", actual_seed)

    # Fractal Perlin Noise Generation
    for i in range(num_octaves):
        res = base_res * (2 ** i)
        amplitude = persistence ** i
        grid_shape = (height // res + 2, width // res + 2)
        noise = np.array(
            [
                [
                    pnoise2(
                        x / res,
                        y / res,
                        octaves=1,
                        repeatx=width,
                        repeaty=height,
                        base=actual_seed,
                    )
                    for x in range(grid_shape[1])
                ]
                for y in range(grid_shape[0])
            ],
            dtype=np.float32,
        )
        zoom_factors = (height / grid_shape[0], width / grid_shape[1])
        layer = zoom(noise, zoom_factors, order=3)
        layer = (layer - 0.5) * 2

        if i > 0:
            slope = gaussian_gradient_magnitude(terrain, sigma=1)
            attenuation = np.exp(-steepness_damping * slope)
            damping_mask = np.minimum(damping_mask, attenuation)

        terrain += layer * amplitude * damping_mask

    terrain -= terrain.mean()
    terrain /= np.abs(terrain).max()

    # Erosion Simulation (Simple Slope-Based)
    for _ in range(erosion_passes):
        grad_x = np.gradient(terrain, axis=1)
        grad_y = np.gradient(terrain, axis=0)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        erosion = erosion_strength * gradient_magnitude
        terrain -= erosion

    terrain -= terrain.mean()
    terrain /= np.abs(terrain).max()
    terrain = np.sign(terrain) * np.power(np.abs(terrain), 1.5)

    return terrain
