import pytest
import numpy as np
from scipy.spatial import cKDTree
import organism
from organism import Organisms

# ---------------------------
# Helper: Dummy Environment
# ---------------------------
class DummyEnv:
    def __init__(self, width=10, length=10, terrain=None):
        self._width = width
        self._length = length
        if terrain is None:
            # default flat terrain
            terrain = np.zeros((length, width), dtype=np.float32)
        self._terrain = terrain

    def get_width(self):
        return self._width

    def get_length(self):
        return self._length

    def get_terrain(self):
        return self._terrain

    def add_births(self, n):
        pass  # unused here

    def get_generation(self):
        return 0  # unused here

# ---------------------------
# Tests for increment_p_id_and_c_id
# ---------------------------

def test_increment_ids_for_founders():
    env = DummyEnv()
    orgs = Organisms(env)
    n = 3
    dtype = orgs.ORGANISM_CLASS

    # Create child array with zeros
    children = np.zeros(n, dtype=dtype)
    # empty parent array
    parents = np.zeros(0, dtype=dtype)

    # next_id starts at 0
    assert orgs._next_id == 0
    orgs.increment_p_id_and_c_id(children, n, parents)

    # c_id assigned 0,1,2
    assert children['c_id'].tolist() == [0, 1, 2]
    # p_id == c_id for founders
    assert children['p_id'].tolist() == [0, 1, 2]
    # next_id updated
    assert orgs._next_id == 3

def test_increment_ids_with_parents():
    env = DummyEnv()
    orgs = Organisms(env)
    n = 2
    dtype = orgs.ORGANISM_CLASS

    # Prepare children
    children = np.zeros(n, dtype=dtype)
    # Prepare parents with non-zero x_pos so any() is True
    parents = np.zeros(n, dtype=dtype)
    parents['x_pos'] = [5.0, 10.0]
    # Assign parent c_id
    parents['c_id'] = [42, 99]

    orgs._next_id = 5
    orgs.increment_p_id_and_c_id(children, n, parents)

    # c_id assigned 5,6
    assert children['c_id'].tolist() == [5, 6]
    # p_id copied from parents.c_id
    assert children['p_id'].tolist() == [42, 99]
    # next_id updated to 7
    assert orgs._next_id == 7

# ---------------------------
# Tests for apply_terrain_penalties
# ---------------------------

def test_apply_terrain_penalties_swim_and_walk():
    # 2x2 terrain: top-left land(>=0), others water(<0)
    terrain = np.array([[0, -1],
                        [-1, -1]], dtype=np.float32)
    env = DummyEnv(width=2, length=2, terrain=terrain)
    orgs = Organisms(env)
    dtype = orgs.ORGANISM_CLASS

    # Create 4 organisms at each cell
    arr = np.zeros(4, dtype=dtype)
    # positions: TL, TR, BL, BR
    arr['x_pos'] = [0, 1, 0, 1]
    arr['y_pos'] = [0, 0, 1, 1]
    # set swim_only at TL, walk_only at BR
    arr['swim'] = [True, False, False, False]
    arr['walk'] = [False, False, False, True]
    arr['fly']  = [False, False, False, False]
    # metabolism_rate = 2.0 for all
    arr['metabolism_rate'] = 2.0
    # initial energy = 10 for all
    arr['energy'] = 10.0

    orgs.set_organisms(arr)
    orgs.apply_terrain_penalties()

    energies = orgs.get_organisms()['energy']
    # TL: swim-only on land => penalty 0.1*2=0.2 => 9.8
    assert pytest.approx(energies[0], rel=1e-6) == 9.8
    # TR: not swim_only or walk_only => unchanged 10
    assert energies[1] == 10.0
    # BL: neither swim_only nor walk_only => unchanged
    assert energies[2] == 10.0
    # BR: walk-only in water => penalty => 9.8
    assert pytest.approx(energies[3], rel=1e-6) == 9.8

# ---------------------------
# Tests for compute_terrain_avoidance
# ---------------------------

def test_compute_terrain_avoidance_flat_terrain():
    # flat terrain: all zeros (land)
    terrain = np.zeros((5,5), dtype=np.float32)
    env = DummyEnv(width=5, length=5, terrain=terrain)
    orgs = Organisms(env)

    # coords: interior (2,2) and boundary (0,4)
    coords = np.array([[2,2], [0,4]], dtype=int)
    avoid_land, avoid_water = orgs.compute_terrain_avoidance(coords)

    # water‐avoidance should be zero everywhere
    assert np.allclose(avoid_water, 0.0)

    # interior point has all 4 neighbors land → no avoidance
    assert np.allclose(avoid_land[0], [0.0, 0.0])

    # boundary point (0,4) has valid land neighbors at dirs [1,0] and [0,-1],
    # so avoid_land = -([1,0] + [0,-1]) = [-1, +1]
    assert np.allclose(avoid_land[1], [-1.0, 1.0])

def test_compute_terrain_avoidance_mixed_terrain():
    # terrain: center cell land, neighbors consistent
    terrain = np.array([
        [ -1, -1, -1 ],
        [ -1,  0, -1 ],
        [ -1, -1, -1 ],
    ], dtype=np.float32)
    env = DummyEnv(width=3, length=3, terrain=terrain)
    orgs = Organisms(env)

    coords = np.array([[1,1]], dtype=int)
    avoid_land, avoid_water = orgs.compute_terrain_avoidance(coords)

    # all neighbors are water => avoid_water = -(sum of dirs) = -[0,0] => [0,0]
    assert np.allclose(avoid_water[0], [0.0, 0.0])
    # no land neighbors => avoid_land = 0
    assert np.allclose(avoid_land[0], [0.0, 0.0])

# ---------------------------
# Tests for move()
# ---------------------------

class DummyTree:
    def __init__(self, neighbors):
        self._neighbors = neighbors

    def query_ball_point(self, coords, radii, workers):
        # return same neighbors for all coords
        N = coords.shape[0]
        return [self._neighbors for _ in range(N)]

def test_move_empty_does_nothing():
    env = DummyEnv()
    orgs = Organisms(env)
    # empty organisms
    orgs.set_organisms(np.zeros((0,), dtype=orgs.ORGANISM_CLASS))
    # should not raise
    orgs.move()
    assert orgs.get_organisms().shape == (0,)

def test_move_photo_increases_energy_and_no_position_change(monkeypatch):
    env = DummyEnv()
    orgs = Organisms(env)
    dtype = orgs.ORGANISM_CLASS

    # one Photo organism at (5,5)
    o = np.zeros(1, dtype=dtype)
    o['x_pos'] = 5.0
    o['y_pos'] = 5.0
    o['energy'] = 1.0
    o['vision'] = 1.0
    o['diet_type'] = 'Photo'
    o['attack'] = 0.0
    o['defense'] = 0.0
    o['pack_behavior'] = False
    o['fly'] = False
    o['swim'] = False
    o['walk'] = False
    o['speed'] = 1.0

    orgs.set_organisms(o)
    # stub spatial index to return only itself
    orgs._pos_tree = DummyTree([0])
    # stub compute_terrain_avoidance to zeros
    monkeypatch.setattr(organism.Organisms, 'compute_terrain_avoidance',
                        lambda self, coords: (np.zeros_like(coords, float), np.zeros_like(coords, float)))
    # stub apply penalties to no-op
    monkeypatch.setattr(organism.Organisms, 'apply_terrain_penalties', lambda self: None)

    orgs.move()

    out = orgs.get_organisms()[0]
    # Energy increased by 0.25
    assert pytest.approx(out['energy'], rel=1e-6) == 1.25
    # Position unchanged
    assert out['x_pos'] == 5.0
    assert out['y_pos'] == 5.0
