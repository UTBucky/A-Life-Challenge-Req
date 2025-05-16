from noise import pnoise2
from scipy.ndimage import zoom, gaussian_gradient_magnitude
import numpy as np
from organism import Organisms
from io import StringIO
from Bio import Phylo

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
        if width <= 0 or length <= 0:
            raise ValueError("Width and length must be greater than zero.")
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

        self._terrain[:] = terrain_mask.astype(np.float32)

    # Other methods
    def add_births(self, new_births):
        self._total_births += new_births

    def add_deaths(self, new_deaths):
        self._total_deaths += new_deaths

    def step(self):
        """
        Steps one generation forward in the simulation.
        """
        if self._generation % 50 == 0 and self._generation > 0:
            tree = Phylo.read((StringIO(self._organisms.get_lineage_tracker().full_forest_newick())), "newick")
            Phylo.write(tree, "my_tree.nwk", "newick")
        self._organisms.build_spatial_index()
        self._organisms.move()
        self._organisms.resolve_attacks()
        self._organisms.reproduce()
        self._organisms.kill_border()
        self._organisms.remove_dead()
        self._organisms.get_organisms()['energy'] -= 0.0001
        self._generation += 1

def generate_fractal_terrain(
    width,
    height,
    num_octaves=4,
    base_res=10,
    persistence=0.45,
    steepness_damping=0.4,
    erosion_passes=4,
    erosion_strength=0.015,
    seed=None
):
    """
    Fractal terrain, inspired by Inigo Quilez
    https://www.youtube.com/watch?v=gsJHzBTPG0Y&t=104s
    
    and the following resources from
    Copyright Inigo Quilez, 2016 - https://iquilezles.org/
    https://iquilezles.org/articles/morenoise/
    https://www.shadertoy.com/view/MdX3Rr
    It utilizes similar methodologies as described in Inigo's blog but does not
    outright copy code snippets as it is in a different language
    Generative AI was used in the process of development and commenting
    """
    if width <= 0 or height <= 0:
        raise ValueError("Width and height must be greater than zero.")

    # Allocate main terrain array, initialized to zero elevation
    terrain = np.zeros((height, width), dtype=np.float32)
    # Create a damping mask to progressively limit slope contributions
    damping_mask = np.ones_like(terrain, dtype=np.float32)

    # Determine maximum grid dimensions for Perlin noise generation
    max_gh = height // base_res + 2
    max_gw = width  // base_res + 2
    # Pre-allocate noise grid for various octaves
    noise_grid = np.empty((max_gh, max_gw), dtype=np.float32)
    # Layer buffer used for upsampled noise contribution
    layer = np.empty_like(terrain, dtype=np.float32)
    # Buffers for gradient-based erosion calculations
    grad_x = np.empty_like(terrain)
    grad_y = np.empty_like(terrain)
    slope  = np.empty_like(terrain)

    # Initialize random number generator and determine actual seed value
    rng = np.random.default_rng(seed)
    actual_seed = seed if seed is not None else int(rng.integers(0, 1_000_000))
    print("[Terrain Gen] Seed used:", actual_seed)


    # --- FRACTAL PERLIN NOISE GENERATION ---
    for i in range(num_octaves):
        # Compute resolution and amplitude for this octave
        res = base_res * (2 ** i)
        amp = persistence ** i

        # Calculate grid dimensions based on resolution
        gh = height // res + 2
        gw = width  // res + 2

        # Fill the noise grid at this resolution
        # Each cell uses 2D Perlin noise with single-octave detail
        for y in range(gh):
            for x in range(gw):
                noise_grid[y, x] = pnoise2(
                    x / res, y / res,
                    octaves=1,
                    repeatx=width,
                    repeaty=height,
                    base=actual_seed
                )

        # Upsample the coarse noise grid to full terrain size (bilinear interpolation)
        zoom_factors = (height / gh, width / gw)
        zoom(noise_grid[:gh, :gw], zoom_factors, order=1, output=layer)

        # Normalize noise values from [0,1] to [-1,1]
        layer = (layer - 0.5) * 2.0

        # After the first octave, compute terrain slope and update damping mask
        if i > 0:
            # Compute magnitude of gradient (slope) from current terrain
            gaussian_gradient_magnitude(terrain, sigma=1, output=slope)
            # Dampen contributions in steep regions according to damping factor
            np.minimum(
                damping_mask,
                np.exp(-steepness_damping * slope),
                out=damping_mask
            )

        # Accumulate the weighted and dampened noise layer into terrain
        terrain += layer * amp * damping_mask

    # Normalize terrain height to zero mean and unit max amplitude
    mean_val = terrain.mean()
    terrain -= mean_val
    max_val = np.abs(terrain).max()
    terrain /= max_val

    # --- SLOPE-BASED EROSION PASSES ---
    for _ in range(erosion_passes):
        # Compute horizontal and vertical gradients
        grad_x[:] = np.gradient(terrain, axis=1)
        grad_y[:] = np.gradient(terrain, axis=0)

        # Calculate slope magnitude using vector norm
        np.hypot(grad_x, grad_y, out=slope)

        # Remove material proportional to slope and erosion strength
        terrain -= erosion_strength * slope

    # Final height remapping to emphasize features
    terrain = np.sign(terrain) * np.abs(terrain) ** 1.5
    # One more normalization to stabilize range
    terrain -= terrain.mean()
    terrain /= np.abs(terrain).max()

    return terrain