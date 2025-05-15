import pytest
import numpy as np
from organism import Organisms, random_name_generation
from scipy.spatial import cKDTree

# --- Helpers ------------------------------------------------------------

class DummyEnvForReproduce:
    def __init__(self, width=50, length=50, generation=1):
        self._width = width
        self._length = length
        self._births = 0
        self._generation = generation

    def get_width(self):
        return self._width

    def get_length(self):
        return self._length

    def get_terrain(self):
        # flat terrain of zeros (land)
        return np.zeros((self._length, self._width), dtype=np.float32)

    def get_generation(self):
        return self._generation

    def add_births(self, n):
        self._births += n

    @property
    def births(self):
        return self._births


def make_parents(env, coords, energies, fert_rates, sizes, parent_ids):
    """
    Create a structured array of parents suitable for reproduce().
    coords: list of (x,y) tuples
    energies, fert_rates, sizes: lists of floats
    parent_ids: list of ints for initial c_id and p_id
    """
    dtype = Organisms.ORGANISM_CLASS
    n = len(coords)
    parents = np.zeros(n, dtype=dtype)
    # positions
    xs, ys = zip(*coords)
    parents['x_pos'] = xs
    parents['y_pos'] = ys
    # reproduction parameters
    parents['energy'] = energies
    parents['fertility_rate'] = fert_rates
    parents['size'] = sizes
    # set initial lineage IDs
    parents['c_id'] = parent_ids
    parents['p_id'] = parent_ids
    return parents

# Minimal gene pool for loading (values unused when no mutation)
MINIMAL_GENE_POOL = {
    'size': (1.0, 1.0),
    'camouflage': (0.0, 0.0),
    'defense': (0.0, 0.0),
    'attack': (0.0, 0.0),
    'vision': (0.0, 0.0),
    'metabolism_rate': (1.0, 1.0),
    'nutrient_efficiency': (1.0, 1.0),
    'diet_type': ['Herb'],
    'fertility_rate': (0.1, 0.1),
    'offspring_count': (1, 1),
    'reproduction_type': ['A'],
    'pack_behavior': [False],
    'symbiotic': [False],
    'swim': [False],
    'walk': [False],
    'fly': [False],
    'speed': (1.0, 1.0),
}

# --- Tests --------------------------------------------------------------

def test_reproduce_happy_path(monkeypatch):
    env = DummyEnvForReproduce()
    orgs = Organisms(env)
    # two parents far apart so safe_mask=True
    parents = make_parents(
        env,
        coords=[(0.0, 0.0), (20.0, 0.0)],
        energies=[5.0, 10.0],
        fert_rates=[0.1, 0.2],  # cost = [1.0, 2.0]
        sizes=[1.0, 1.0],
        parent_ids=[5, 6],
    )
    orgs.set_organisms(parents)
    orgs.load_genes(MINIMAL_GENE_POOL)
    orgs.build_spatial_index()

    # Force no mutation: rand >=0.01 always false, and zero offsets
    monkeypatch.setattr(np.random, 'rand',   lambda size: np.ones(size))
    monkeypatch.setattr(np.random, 'uniform', lambda *args, **kwargs: np.zeros((2, 2)))

    orgs.reproduce()

    all_orgs = orgs.get_organisms()
    # Expect 2 original + 2 offspring
    assert all_orgs.shape[0] == 4

    # Parents energies reduced by cost
    # original parents are at indices 0 and 1
    expected_parent_energies = [5.0 - 1.0, 10.0 - 2.0]
    assert np.allclose(all_orgs['energy'][:2], expected_parent_energies)

    # Offspring energies equal reproduction_costs
    # offspring at indices 2 and 3
    assert np.allclose(all_orgs['energy'][2:], [1.0, 2.0])

    # Offspring IDs are sequential from 0
    assert all_orgs['c_id'][2:].tolist() == [0, 1]
    # Offspring p_id equals parents' initial c_id
    assert all_orgs['p_id'][2:].tolist() == [5, 6]

    # Environment recorded 2 births
    assert env.births == 2

    # No new species added in species_count (no mutation)
    assert orgs.get_speciation_dict() == {}


def test_reproduce_no_offspring_when_energy_not_sufficient(monkeypatch):
    env = DummyEnvForReproduce()
    orgs = Organisms(env)
    # two parents far apart but energy == cost -> no reproduction
    parents = make_parents(
        env,
        coords=[(0.0, 0.0), (20.0, 0.0)],
        energies=[1.0, 2.0],
        fert_rates=[0.1, 0.2],  # cost = [1.0, 2.0]
        sizes=[1.0, 1.0],
        parent_ids=[0, 1],
    )
    orgs.set_organisms(parents)
    orgs.load_genes(MINIMAL_GENE_POOL)
    orgs.build_spatial_index()

    # even without mutation, energy > cost is False
    orgs.reproduce()

    # no new organisms appended
    assert orgs.get_organisms().shape[0] == 2
    assert env.births == 0

def test_determine_valid_parents():
    """
    Test for _determine_valid_parents to ensure:
    1. Only organisms with sufficient energy and safe distance are selected.
    2. Costs are calculated correctly.
    3. Masks are generated correctly.
    """
    env = DummyEnvForReproduce()
    orgs = Organisms(env)

    # Create parents:
    # - (0, 0) and (20, 0) are far enough apart (> 7.5 units)
    # - (1, 1) is too close to (0, 0), should be filtered out
    parents = make_parents(
        env,
        coords=[(0.0, 0.0), (20.0, 0.0), (30.0, 30.0)],
        energies=[20.0, 15.0, 5.0],
        fert_rates=[0.1, 0.2, 0.5],  # Costs: [1.0, 2.0, 5.0]
        sizes=[1.0, 1.0, 1.0],
        parent_ids=[0, 1, 2]
    )

    # Initialize the organisms and spatial index
    orgs.set_organisms(parents)
    orgs.load_genes(MINIMAL_GENE_POOL)
    orgs.build_spatial_index()

    # Execute the _determine_valid_parents() function
    result_parents, result_costs, result_mask = orgs._determine_valid_parents()

    # Expected Results
    expected_x_pos = [0.0, 20.0]
    expected_y_pos = [0.0, 0.0]
    expected_costs = [1.0, 2.0]
    expected_mask = [True, True, False]

    # --- Assertions ---
    # Length check
    assert len(result_parents) == 2, f"Expected 2 parents, got {len(result_parents)}"
    
    # Position validation
    assert np.allclose(result_parents['x_pos'], expected_x_pos), f"X positions mismatch: {result_parents['x_pos']}"
    assert np.allclose(result_parents['y_pos'], expected_y_pos), f"Y positions mismatch: {result_parents['y_pos']}"
    
    # Cost validation
    assert np.allclose(result_costs, expected_costs), f"Reproduction costs mismatch: {result_costs}"
    
    # Mask validation
    assert np.array_equal(result_mask, expected_mask), f"Parent mask mismatch: {result_mask}"

    # 5Ensure types are correct (no float64 errors)
    assert result_parents.dtype == orgs._organism_dtype, "Parent dtype mismatch"
    assert result_costs.dtype == np.float32, "Cost dtype mismatch"
    assert result_mask.dtype == bool, "Mask dtype mismatch"

def test_reproduce_no_offspring_when_too_close(monkeypatch):
    env = DummyEnvForReproduce()
    orgs = Organisms(env)
    # two parents too close (<7.5)
    parents = make_parents(
        env,
        coords=[(0.0, 0.0), (5.0, 0.0)],
        energies=[10.0, 10.0],
        fert_rates=[0.1, 0.1],
        sizes=[1.0, 1.0],
        parent_ids=[0, 1],
    )
    orgs.set_organisms(parents)
    orgs.load_genes(MINIMAL_GENE_POOL)
    orgs.build_spatial_index()

    monkeypatch.setattr(np.random, 'rand', lambda size: np.ones(size))
    monkeypatch.setattr(np.random, 'uniform', lambda *args, **kwargs: np.zeros((2, 2)))

    orgs.reproduce()

    assert orgs.get_organisms().shape[0] == 2
    assert env.births == 0


def test_reproduce_without_spatial_index_raises():
    env = DummyEnvForReproduce()
    orgs = Organisms(env)
    parents = make_parents(
        env,
        coords=[(0.0, 0.0), (20.0, 0.0)],
        energies=[10.0, 10.0],
        fert_rates=[0.1, 0.1],
        sizes=[1.0, 1.0],
        parent_ids=[0, 1],
    )
    orgs.set_organisms(parents)
    # do NOT call build_spatial_index()

    with pytest.raises(AttributeError):
        orgs.reproduce()


def test_reproduce_without_gene_pool_raises(monkeypatch):
    env = DummyEnvForReproduce()
    orgs = Organisms(env)

    # Build one parent that *is* eligible to reproduce:
    parents = make_parents(
        env,
        coords=[(0.0, 0.0)],
        energies=[10.0],       # plenty of energy
        fert_rates=[0.1],      # cost = 1.0
        sizes=[1.0],
        parent_ids=[0],
    )
    orgs.set_organisms(parents)
    # Intentionally *do not* call orgs.load_genes(...)
    orgs.build_spatial_index()

    # Force reproduction path:
    monkeypatch.setattr(np.random, 'rand', lambda size: np.zeros(size))       # so flip_mask=True
    monkeypatch.setattr(np.random, 'uniform', lambda *args, **kw: np.zeros((1,2)))

    with pytest.raises(TypeError):
        orgs.reproduce()
