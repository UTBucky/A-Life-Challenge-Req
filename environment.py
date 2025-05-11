from noise import pnoise2
from scipy.ndimage import zoom, gaussian_gradient_magnitude
import numpy as np
from organism import Organisms
from collections import defaultdict
from scipy.cluster.hierarchy import DisjointSet

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
        if self._generation % 100 == 0 and self._generation > 0:
            tree = build_newick_trees(self._organisms.get_species_count(), self._organisms.get_disjointset())
            print(tree)
            export_newick_to_file(tree)
            
        self._organisms.build_spatial_index()
        self._organisms.move()
        self._organisms.resolve_attacks()
        self._organisms.reproduce()
        # TODO: Could this be moved to an org method?
        self._organisms.kill_border()
        self._organisms.remove_dead()
        self._organisms.get_organisms()['energy'] -= 0.01
        self._generation += 1
        
def export_newick_to_file(newick_list, output_path='output.nwk'):
    """
    Writes a list of Newick strings to the specified file, one per line.
    """
    with open(output_path, 'w') as out_file:
        for tree in newick_list:
            out_file.write(tree + '\n')
    print(f"Wrote {len(newick_list)} trees to {output_path}")

def build_newick_trees(species_source, dsu, include_clusters=True):
    """
    Construct Newick-formatted trees, clustering by DSU sets.
    Named species are used as cluster roots; other PIDs attach below.
    Omit branch lengths for unknown PIDs.

    Args:
      species_source: dict mapping species_name -> (pid, parent, gen)
      dsu: DisjointSet instance containing all organism IDs
      include_clusters: if True, one tree per DSU cluster; if False, one per forest root.

    Returns:
      List of Newick strings.
    """
    # Extract species_dict from dict or class
    if hasattr(species_source, 'items'):
        species_dict = species_source
    elif hasattr(species_source, '_species_count'):
        species_dict = species_source._species_count
    else:
        raise TypeError("species_source must be a dict or have attribute '_species_count'")

    # Step 1: raw metadata from named species
    id_to_meta = {}
    for name, (pid, parent, gen) in species_dict.items():
        id_to_meta[pid] = {'species': name, 'parent': parent, 'gen': gen}
    named_pids = set(id_to_meta)

    # Step 2: attach DSU-only PIDs under named cluster root
    if include_clusters:
        all_ds_pids = set().union(*dsu.subsets())
    else:
        all_ds_pids = set(id_to_meta.keys())
    for pid in all_ds_pids:
        if pid not in id_to_meta:
            # find any named representative in same cluster
            rep = next((n for n in named_pids if int(dsu[n]) == int(dsu[pid])), None)
            if rep is None:
                continue
            parent = rep
            gen = id_to_meta[rep]['gen'] + 1
            id_to_meta[pid] = {'species': str(pid), 'parent': parent, 'gen': gen}

    # Build adjacency
    children = defaultdict(list)
    for pid, meta in id_to_meta.items():
        children[meta['parent']].append(pid)

    # Recursive Newick
    def to_newick(pid, visited=None):
        if visited is None:
            visited = set()
        if pid in visited:
            return ''
        visited.add(pid)

        meta = id_to_meta[pid]
        gen = meta['gen']
        subs = []
        for c in children.get(pid, []):
            if c == pid:
                continue
            length = id_to_meta[c]['gen'] - gen
            subtree = to_newick(c, visited.copy())
            if not subtree:
                continue
            # omit branch length for unknown (numeric) children
            if c not in named_pids:
                subs.append(f"{subtree}")
            else:
                subs.append(f"{subtree}:{length}")
        label = meta['species']
        if subs:
            return f"({','.join(subs)}){label}"
        else:
            return label

    # Generate output
    newick_list = []
    if include_clusters:
        # for each DSU set, pick named root
        for clade in dsu.subsets():
            root = next((n for n in named_pids if n in clade), None)
            if root is not None:
                newick_list.append(to_newick(root) + ';')
    else:
        # forest: roots whose parent isn't in map
        roots = [pid for pid, m in id_to_meta.items()
                 if m['parent'] not in id_to_meta or m['parent'] == pid]
        for root in roots:
            newick_list.append(to_newick(root) + ';')

    return newick_list


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