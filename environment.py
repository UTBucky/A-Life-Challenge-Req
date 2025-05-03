from noise import pnoise2
from scipy.ndimage import zoom, gaussian_gradient_magnitude
import numpy as np
from organism import Organisms


class Environment:
    """
    2D Simulation Environment
    """

    def __init__(self, width: int, length: int):
        """
        Initializes the 2D environment with set size,
        terrain, and organisms.

        :param width: Width of the environment
        :param length: Length of the environment
        """
        self._width = width
        self._length = length
        self._terrain = np.zeros((length, width), dtype=np.float32)
        self._generation = 0
        self._total_births = 0
        self._total_deaths = 0
        self._organisms = Organisms(self)

    # Get methods
    def get_width(self):
        return self._width

    def get_length(self):
        return self._length

    def get_terrain(self):
        return self._terrain

    def get_organisms(self):
        return self._organisms

    def get_total_births(self):
        return self._total_births

    def get_total_deaths(self):
        return self._total_deaths

    def get_generation(self):
        return self._generation

    # Set methods
    def set_terrain(self, terrain_mask: np.ndarray):
        """
        Applies terrain_mask to the environment
        """
        if terrain_mask.shape != self._terrain.shape:
            raise ValueError('Your terrain mask is wrong!!!!!')

        if not np.array_equal(self._terrain, terrain_mask):
            self._terrain[...] = terrain_mask.astype(np.float32)

    # Other methods
    def add_births(self, new_births):
        self._total_births += new_births

    def add_deaths(self, new_deaths):
        self._total_deaths += new_deaths

    # TODO: Cleanup and add parameters for clarity
    #       add masks for different tile types such as water
    def inbounds(self, new_positions):
        """
        Clips in bound positions
        returns land mask
        """

        # Clips the new organism positions to be inside the environment bounds
        new_positions[:, 0] = np.clip(
            new_positions[:, 0], 0, self._width - 1
            )
        new_positions[:, 1] = np.clip(
            new_positions[:, 1], 0, self._length - 1
            )

        # Sets the cleaned positions
        ix = new_positions[:, 0].astype(np.int32)
        iy = new_positions[:, 1].astype(np.int32)

        terrain_mask = self._terrain[iy, ix] >= 0

        return terrain_mask

    def step(self):
        """
        Steps one generation forward in the simulation.
        """

        organisms = self._organisms.get_organisms()

        # Only steps while organisms present TODO: May change as env developes
        if organisms.shape[0] != 0:

            # Organisms take an action
            # TODO: Implement action decision making,
            #       only moves currently
            self._organisms.move()

            # TODO: Could this be moved to an org method?
            self._organisms.remove_dead()

            self._generation += 1


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
