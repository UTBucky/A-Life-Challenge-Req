import pytest
import numpy as np
from scipy.spatial import cKDTree

from organism import Organisms, random_name_generation

# A minimal stub Environment for testing
class DummyEnv:
    def __init__(self, width=10, length=5):
        self._width = width
        self._length = length
        self._births = 0
        self._deaths = 0

    def get_width(self):
        return self._width

    def get_length(self):
        return self._length

    def get_terrain(self):
        # flat terrain of zeros
        return np.zeros((self._length, self._width), dtype=np.float32)

    def get_generation(self):
        return 0

    def add_births(self, n):
        self._births += n

    def add_deaths(self, n):
        self._deaths += n


def test_init_and_defaults():
    env = DummyEnv(width=7, length=3)
    orgs = Organisms(env)

    # initial organisms array is empty
    arr = orgs.get_organisms()
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (0,)

    # species count starts empty
    assert orgs.get_species_count() == {}

    # private state initialized
    assert orgs._next_id == 0
    # no spatial index when empty
    orgs.build_spatial_index()
    assert orgs._pos_tree is None


def test_load_genes_and_access():
    env = DummyEnv()
    orgs = Organisms(env)
    fake_pool = {'size': (0.5, 1.5)}
    orgs.load_genes(fake_pool)
    # gene pool should be stored directly
    assert orgs._gene_pool is fake_pool


def test_set_and_get_organisms_roundtrip():
    env = DummyEnv(width=4, length=4)
    orgs = Organisms(env)
    dtype = orgs._organism_dtype

    # Create a small structured array
    data = np.zeros(3, dtype=dtype)
    # assign distinct positions
    data['x_pos'] = [0.0, 1.0, 2.0]
    data['y_pos'] = [1.0, 2.0, 3.0]

    orgs.set_organisms(data)
    out = orgs.get_organisms()
    # should be identical object/contents
    assert np.array_equal(out, data)


def test_build_spatial_index_with_points():
    env = DummyEnv(width=5, length=5)
    orgs = Organisms(env)
    dtype = orgs._organism_dtype

    # two points at (0,0) and (4,4)
    pts = np.zeros(2, dtype=dtype)
    pts['x_pos'] = [0.0, 4.0]
    pts['y_pos'] = [0.0, 4.0]
    orgs.set_organisms(pts)

    # build the KD-tree
    orgs.build_spatial_index()
    tree = orgs._pos_tree
    assert isinstance(tree, cKDTree)

    # query exactly locates each point
    dists, idxs = tree.query([[0.0, 0.0], [4.0, 4.0]], k=1)
    assert list(idxs) == [0, 1]


def test_generate_ids_sequences_and_increment():
    env = DummyEnv()
    orgs = Organisms(env)

    # first allocation
    ids1 = orgs._generate_ids(3)
    assert isinstance(ids1, np.ndarray)
    assert ids1.dtype == np.int64
    assert ids1.tolist() == [0, 1, 2]
    assert orgs._next_id == 3

    # next allocation continues
    ids2 = orgs._generate_ids(2)
    assert ids2.tolist() == [3, 4]
    assert orgs._next_id == 5


def test_random_name_generation_length_and_format(monkeypatch):
    # monkey-patch random.randint and random.choice for determinism
    import random
    monkeypatch.setattr(random, 'randint', lambda a, b: 2)
    monkeypatch.setattr(random, 'choice', lambda seq: 'xy')

    names = random_name_generation(5, min_syllables=2, max_syllables=2)
    # should return exactly 5 names
    assert isinstance(names, np.ndarray)
    assert names.shape == (5,)
    # each name should be capitalized and length == 4 (2 syllables * 2 chars)
    assert all(name.isalpha() for name in names)
    assert all(name[0].isupper() for name in names)
    assert all(len(name) == 4 for name in names)

