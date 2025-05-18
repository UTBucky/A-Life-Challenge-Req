import pytest
import numpy as np

from array_ops import (
    initialize_default_traits,
    initialize_random_traits,
    calculate_valid_founder_terrain,
    copy_valid_count,
    copy_parent_fields,
    mutate_offspring,
)
from load_genes import load_genes_from_file
from organism import Organisms

# Pull in your canonical dtype for structuring offspring/parents arrays
ORGANISM_CLASS = Organisms.ORGANISM_CLASS


# -- Fixtures ------------------------------------------------------------

@pytest.fixture
def minimal_gene_pool():
    # Only the keys used by initialize_default_traits() and mutate_offspring()
    return {
        'diet_type':          ['Herb', 'Omni', 'Carn', 'Photo', 'Parasite'],
        'reproduction_type':  ['Sexual', 'Asexual'],
        'size':               (0.9, 1.1),
        'camouflage':         (30.0, 60.0),
        'defense':            (3.0, 6.0),
        'attack':             (3.0, 6.0),
        'vision':             (30.0, 70.0),
        'metabolism_rate':    (0.5, 1.5),
        'nutrient_efficiency':(0.5, 1.6),
        'fertility_rate':     (0.9, 1.0),
        'offspring_count':    (1, 10),
        'pack_behavior':      [False, True],
        'symbiotic':          [False, True],
        'swim':               [False, True],
        'walk':               [False, True],
        'fly':                [False, True],
        'speed':              (1.0, 10.0),
    }


# -- initialize_default_traits tests ------------------------------------

def test_initialize_default_traits_valid_minimum(minimal_gene_pool):
    # n >= 1 should work
    n = 1
    traits = initialize_default_traits(n, minimal_gene_pool)
    # should return exactly 19 arrays
    assert len(traits) == 19

    species_arr, size_arr, camouflage_arr, defense_arr, attack_arr, \
    vision_arr, metabolism_rate_arr, nutrient_efficiency_arr, diet_type_arr, \
    fertility_rate_arr, offspring_count_arr, reproduction_type_arr, \
    pack_behavior_arr, symbiotic_arr, swim_arr, walk_arr, fly_arr, \
    speed_arr, energy_arr = traits

    # all arrays length == n and correct dtype
    assert species_arr.shape == (1,) and species_arr.dtype.kind == 'U'
    assert size_arr.shape == (1,)   and size_arr.dtype == np.float32
    assert diet_type_arr[0] == minimal_gene_pool['diet_type'][0]
    assert reproduction_type_arr[0] == minimal_gene_pool['reproduction_type'][0]

    # check the unique default that walk_arr==True
    assert bool(swim_arr[0]) == True
    # default energy should be exactly 20 per code
    assert energy_arr[0] == 10.0


def test_initialize_default_traits_invalid_n_raises(minimal_gene_pool):
    with pytest.raises(ValueError):
        initialize_default_traits(0, minimal_gene_pool)


# -- initialize_random_traits tests -------------------------------------

def test_initialize_random_traits_valid(minimal_gene_pool):
    n = 100
    traits = initialize_random_traits(n, minimal_gene_pool)
    assert len(traits) == 19

    # unpack just a few to spot-check
    _, size_arr, _, _, attack_arr, \
    vision_arr, _, _, diet_type_arr, \
    _, offspring_count_arr, reproduction_type_arr, \
    pack_behavior_arr, _, swim_arr, walk_arr, fly_arr, \
    speed_arr, energy_arr = traits

    # shapes & types
    assert size_arr.shape == (n,) and size_arr.dtype == np.float32
    assert diet_type_arr.shape == (n,) and diet_type_arr.dtype.kind == 'U'
    assert offspring_count_arr.dtype == np.int32

    # numeric ranges
    assert size_arr.min() >= minimal_gene_pool['size'][0]
    assert size_arr.max() <= minimal_gene_pool['size'][1]
    assert attack_arr.min() >= minimal_gene_pool['attack'][0]
    assert attack_arr.max() <= minimal_gene_pool['attack'][1]
    assert speed_arr.min() >= minimal_gene_pool['speed'][0]
    assert speed_arr.max() <= minimal_gene_pool['speed'][1]
    assert energy_arr.min() >= 10.0
    assert energy_arr.max() <= 30.0

    # categorical subsets
    assert set(diet_type_arr).issubset(set(minimal_gene_pool['diet_type']))
    assert set(reproduction_type_arr).issubset(set(minimal_gene_pool['reproduction_type']))


def test_initialize_random_traits_invalid_n_raises(minimal_gene_pool):
    with pytest.raises(ValueError):
        initialize_random_traits(0, minimal_gene_pool)


def test_initialize_random_traits_missing_key_raises(minimal_gene_pool):
    gp = minimal_gene_pool.copy()
    gp.pop('size')
    with pytest.raises(KeyError):
        initialize_random_traits(1, gp)


# -- calculate_valid_founder_terrain tests ------------------------------

def test_calculate_valid_founder_terrain_returns_empty_when_no_valid():
    # all water, and no swimmer/walker/flyer
    env_terrain = np.full((200, 200), -1, dtype=int)
    n = 50
    swim = np.zeros(n, dtype=bool)
    walk = np.zeros(n, dtype=bool)
    fly  = np.zeros(n, dtype=bool)

    positions, valid_count = calculate_valid_founder_terrain(
        env_terrain, swim, walk, fly, env_width=200, n=n
    )
    assert valid_count == 0
    assert positions.shape == (0, 2)


def test_calculate_valid_founder_terrain_env_width_too_small_raises():
    env_terrain = np.zeros((300, 300), dtype=int)
    swim = walk = fly = np.ones(10, dtype=bool)
    # env_width < 200 is disallowed
    with pytest.raises(ValueError):
        calculate_valid_founder_terrain(env_terrain, swim, walk, fly, env_width=100, n=10)


# -- copy_valid_count tests ---------------------------------------------

def test_copy_valid_count_truncates_and_copies_defaults(minimal_gene_pool):
    # start with default traits for n=5
    n = 5
    traits = initialize_default_traits(n, minimal_gene_pool)
    # we only want first 3
    valid_count = 3

    spawned = np.zeros(valid_count, dtype=ORGANISM_CLASS)
    # call copy_valid_count
    out = copy_valid_count(spawned, valid_count, *traits)

    # species should match first valid_count entries of species_arr
    species_arr = traits[0]
    assert np.array_equal(out['species'], species_arr[:valid_count])

    # other numeric fields likewise
    assert np.all(out['size'] == traits[1][:valid_count])
    assert np.all(out['camouflage'] == traits[2][:valid_count])


# -- copy_parent_fields tests ------------------------------------------

def test_copy_parent_fields_full_copy():
    # create a small parents array with distinct values
    parents = np.zeros(3, dtype=ORGANISM_CLASS)
    parents['size'] = [0.9, 1.0, 1.1]
    parents['attack'] = [3.3, 4.4, 5.5]
    parents['diet_type'] = ['Herb', 'Carn', 'Photo']

    offspring = np.zeros_like(parents)
    copy_parent_fields(parents, offspring)

    for name in ORGANISM_CLASS.names:
        # allow x_pos,y_pos,p_id,c_id to differ later
        if name in ('x_pos','y_pos','p_id','c_id','generation'):
            continue
        assert np.array_equal(offspring[name], parents[name]), f"{name} not copied"


# -- mutate_offspring tests ---------------------------------------------
