# tests/test_organism_attacks.py

import pytest
import numpy as np
from scipy.spatial import cKDTree
import organism
from organism import Organisms

# ---------------------------
# Dummy Environment
# ---------------------------
class DummyEnv:
    def __init__(self, width=10, length=10, terrain=None):
        self._width = width
        self._length = length
        if terrain is None:
            terrain = np.zeros((length, width), dtype=np.float32)
        self._terrain = terrain

    def get_width(self):
        return self._width

    def get_length(self):
        return self._length

    def get_terrain(self):
        return self._terrain

    def add_births(self, n): pass
    def get_generation(self): return 0

# ---------------------------
# Tests for _terrain_restrictions
# ---------------------------
def test_terrain_restrictions_basic():
    # Two pairs: i attacks j at positions over land and water
    terrain = np.array([[0, -1],
                        [1,  0]], dtype=np.float32)
    # indices
    i = np.array([0, 1, 2, 3], dtype=int)
    j = np.array([1, 0, 2, 3], dtype=int)
    # capabilities arrays
    fly  = np.array([False, True, False, False])
    swim = np.array([True, False, True, False])
    walk = np.array([False, True, False, True])
    # positions corresponding to indices
    x_pos = np.array([0,1,1,0], dtype=np.float32)
    y_pos = np.array([0,1,0,1], dtype=np.float32)

    env = DummyEnv(2,2, terrain)
    orgs = Organisms(env)

    keep_i, keep_j = orgs._terrain_restrictions(
        i, j, fly, swim, walk, x_pos, y_pos, terrain
    )

    # manual calculation: only pairs at positions 1,2,3 are valid
    assert set(keep_i.tolist()) == {1, 2, 3}
    assert set(keep_j.tolist()) == {0, 2, 3}

# ---------------------------
# Tests for _diet_restrictions
# ---------------------------
def test_diet_restrictions():
    # dt_i and dt_j arrays
    diet = np.array(['Carn','Herb','Photo','Omni'])
    i = np.array([0,1,2,3], dtype=int)
    j = np.array([1,2,3,0], dtype=int)

    env = DummyEnv()
    orgs = Organisms(env)

    ki, kj = orgs._diet_restrictions(i,j,diet)
    # invalid cases:
    # i=0 Carn and j=1 Herb -> Herb!='Photo' -> no invalid (Carn only invalid if dt_j==Photo)
    # but dt_j=Herb, so keep
    # i=1 Herb & ~isin(dt_j,['Photo','Parasite']), dt_j=Photo -> ok
    # i=2 Photo invalid always -> remove
    # i=3 Omni -> no invalid -> keep
    assert 2 not in ki  # Photo attacker removed
    # other indices remain
    assert set(ki.tolist()) == {0,1,3}
    assert set(kj.tolist()) == {1,2,0}

# ---------------------------
# Tests for _classify_engagement
# ---------------------------
def test_classify_engagement_no_pack():
    # simple two pairs
    i = np.array([0,1], dtype=int)
    j = np.array([1,0], dtype=int)
    att = np.array([5.0, 2.0])
    deff= np.array([1.0, 1.0])
    pack= np.array([False, False])

    env = DummyEnv()
    orgs = Organisms(env)

    host, prey, my_net, their_net = orgs._classify_engagement(i,j,att,deff,pack)
    # my_net = [5-1,2-1]=[4,1], their_net = [2-1,5-1]=[1,4]
    assert np.allclose(my_net, [4,1])
    assert np.allclose(their_net, [1,4])
    # host: their_net>my_net -> [1>4 False,4>1 True] -> [False,True] & their_net>0
    assert host.tolist() == [False, True]
    # prey: my_net>their_net -> [4>1 True,1>4 False] -> [True,False]
    assert prey.tolist() == [True, False]

def test_classify_engagement_with_pack():
    i = np.array([0,1], dtype=int)
    j = np.array([1,0], dtype=int)
    att = np.array([5.0, 2.0])
    deff= np.array([1.0, 1.0])
    pack= np.array([True, False])
    env = DummyEnv()
    orgs = Organisms(env)

    host, prey, _, _ = orgs._classify_engagement(i,j,att,deff,pack)
    # only non_pack = pack[i]&~pack[j] -> [True&True,False&?] -> [True,False]
    # host: their_net>my_net at idx0? their_net[0]=1,my_net=4->False
    assert host.tolist() == [False, False]
    # prey: my_net>their_net at idx0:True -> [True,False]
    assert prey.tolist() == [True, False]

# ---------------------------
# Tests for _apply_damage
# ---------------------------
def test_apply_damage_host_and_prey():
    # build simple energy array
    energy = np.array([10.0, 10.0, 10.0])
    i = np.array([0,1,2], dtype=int)
    j = np.array([1,2,0], dtype=int)
    # host mask: only index0 True, prey mask: only index1 True
    host = np.array([True, False, False])
    prey = np.array([False, True, False])
    my_net = np.array([0.0, 3.0, 0.0])
    their_net = np.array([2.0, 0.0, 0.0])

    # expected:
    # host at idx0: energy[0]-=their_net[0]=10-2=8, energy[1]+=2 ->12
    # prey at idx1: energy[2]-=my_net[1]=10-3=7, energy[1]+=3 ->15
    env = DummyEnv()
    orgs = Organisms(env)
    orgs._apply_damage(i,j,host,prey,my_net,their_net,energy)

    assert pytest.approx(energy[0]) == 8.0
    assert pytest.approx(energy[1]) == 15.0
    assert pytest.approx(energy[2]) == 7.0

# ---------------------------
# Tests for resolve_attacks
# ---------------------------
def test_resolve_attacks_no_action_for_N_lt_2():
    env = DummyEnv()
    orgs = Organisms(env)
    # single organism
    o = np.zeros(1, dtype=orgs.ORGANISM_CLASS)
    o['energy'] = 5.0
    orgs.set_organisms(o)
    # should not error or change energy
    orgs.resolve_attacks()
    assert orgs.get_organisms()['energy'][0] == 5.0

def test_resolve_attacks_simple_predation(monkeypatch):
    # two organisms close enough to attack, no restrictions or packs
    env = DummyEnv()
    orgs = Organisms(env)
    dtype = orgs.ORGANISM_CLASS

    # create 2 organisms at same spot
    arr = np.zeros(2, dtype=dtype)
    arr['x_pos'] = [1.0, 1.0]
    arr['y_pos'] = [1.0, 1.0]
    arr['energy'] = [10.0, 10.0]
    arr['attack'] = [5.0, 2.0]
    arr['defense']= [1.0, 1.0]
    arr['vision'] = [5.0, 5.0]
    arr['pack_behavior'] = [False, False]
    arr['fly'] = [False, False]
    arr['swim']= [False, False]
    arr['walk']= [False, False]
    arr['diet_type'] = ['Carn','Herb']

    orgs.set_organisms(arr)
    orgs.build_spatial_index()

    # stub internal restrictions to identity
    monkeypatch.setattr(organism.Organisms, '_terrain_restrictions', lambda self,i,j,*a: (i,j))
    monkeypatch.setattr(organism.Organisms, '_diet_restrictions', lambda self,i,j,d: (i,j))

    orgs.resolve_attacks()
    out = orgs.get_organisms()['energy']
    # Only the "host" branch is applied in current implementation:
    # organism 0 gains their_net[1]=4 → 10+4=14
    # organism 1 loses their_net[1]=4 → 10-4=6
    assert pytest.approx(out[0], rel=1e-6) == 14.0
    assert pytest.approx(out[1], rel=1e-6) == 6.0
