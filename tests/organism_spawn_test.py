import pytest
import numpy as np
import organism
from organism import Organisms
from array_ops import initialize_default_traits

# —————————————————————————————————————————————————————————————————————————————
# Dummy environment for spawning
# —————————————————————————————————————————————————————————————————————————————
class DummyEnvSpawn:
    def __init__(self, width=20, length=10):
        self._width = width
        self._length = length
        self._births = 0

    def get_width(self):
        return self._width

    def get_length(self):
        return self._length

    def get_terrain(self):
        # treat everything as walkable land
        return np.zeros((self._length, self._width), dtype=np.float32)

    def add_births(self, n):
        self._births += n

    @property
    def births(self):
        return self._births


# A simple flat gene-pool dict for testing
FLAT_GP = {
    'size':               (1.0, 1.0),
    'camouflage':         (0.0, 0.0),
    'defense':            (0.0, 0.0),
    'attack':             (0.0, 0.0),
    'vision':             (0.0, 0.0),
    'metabolism_rate':    (1.0, 1.0),
    'nutrient_efficiency':(1.0, 1.0),
    'diet_type':          ['Herb'],
    'fertility_rate':     (0.1, 0.1),
    'offspring_count':    (1, 1),
    'reproduction_type':  ['A'],
    'pack_behavior':      [False],
    'symbiotic':          [False],
    'swim':               [False, True],
    'walk':               [True],
    'fly':                [False],
    'speed':              (1.0, 1.0),
}


# —————————————————————————————————————————————————————————————————————————————
# 1. Zero organisms (n = 0) raise ValueError
# —————————————————————————————————————————————————————————————————————————————
def test_spawn_zero_raises_value_error():
    env = DummyEnvSpawn()
    orgs = Organisms(env)
    orgs.load_genes(FLAT_GP)
    with pytest.raises(ValueError):
        orgs.spawn_initial_organisms(0, randomize=False)


# —————————————————————————————————————————————————————————————————————————————
# 2. randomize=False happy path
# —————————————————————————————————————————————————————————————————————————————
def test_spawn_default_happy_path(monkeypatch):
    env = DummyEnvSpawn(width=300, length=300)
    orgs = Organisms(env)
    orgs.load_genes(FLAT_GP)

    n = 3
    # stub initialize_default_traits in organism module
    default_traits = initialize_default_traits(n, FLAT_GP)
    monkeypatch.setattr(organism, 'initialize_default_traits',
                        lambda nn, gp: default_traits)
    # stub calculate_valid_founder_terrain in organism module
    fake_positions = np.array([[2,2],[4,4],[6,6]], dtype=np.int32)
    monkeypatch.setattr(organism, 'calculate_valid_founder_terrain',
                        lambda terrain, s, w, f, width, nn: (fake_positions, n))
    # stub copy_valid_count
    def fake_copy(spawned, vc, *arrays):
        # stamp species so we can detect it
        spawned['species'] = arrays[0][:vc]
        return spawned
    monkeypatch.setattr(organism, 'copy_valid_count', fake_copy)
    # stub increment_p_id_and_c_id to no-op
    monkeypatch.setattr(Organisms, 'increment_p_id_and_c_id', lambda self, *args: None)

    cnt = orgs.spawn_initial_organisms(n, randomize=False)
    assert cnt == n
    assert env.births == n

    out = orgs.get_organisms()
    # positions in x_pos/y_pos
    assert np.all(out['x_pos'] == fake_positions[:,0].astype(np.float32))
    assert np.all(out['y_pos'] == fake_positions[:,1].astype(np.float32))
    # p_id/c_id default from no-op stub: zeros
    assert list(out['c_id']) == [0]*n
    assert list(out['p_id']) == [0]*n


# —————————————————————————————————————————————————————————————————————————————
# 3. randomize=True happy path
# —————————————————————————————————————————————————————————————————————————————
def test_spawn_randomized_happy_path(monkeypatch):
    env = DummyEnvSpawn(width=300, length=300)
    orgs = Organisms(env)
    orgs.load_genes(FLAT_GP)

    n = 4
    # stub initialize_random_traits in organism module
    random_traits = initialize_default_traits(n, FLAT_GP)
    monkeypatch.setattr(organism, 'initialize_random_traits',
                        lambda nn, gp: random_traits)
    # same stubs for terrain, copy, and ID
    fake_positions = np.stack([np.arange(n), np.arange(n)], axis=1).astype(np.int32)
    monkeypatch.setattr(organism, 'calculate_valid_founder_terrain',
                        lambda terrain, s, w, f, width, nn: (fake_positions, n))
    monkeypatch.setattr(organism, 'copy_valid_count', lambda spawned, vc, *a: spawned)
    monkeypatch.setattr(Organisms, 'increment_p_id_and_c_id', lambda self, *args: None)

    cnt = orgs.spawn_initial_organisms(n, randomize=True)
    assert cnt == n
    assert env.births == n

    out = orgs.get_organisms()
    assert out.shape[0] == n


# —————————————————————————————————————————————————————————————————————————————
# 4. randomize=True without gene_pool should raise TypeError
# —————————————————————————————————————————————————————————————————————————————
def test_spawn_randomize_true_without_gene_pool_raises():
    env = DummyEnvSpawn(width=300, length=300)
    orgs = Organisms(env)
    # skip orgs.load_genes
    with pytest.raises(TypeError):
        orgs.spawn_initial_organisms(1, randomize=True)
