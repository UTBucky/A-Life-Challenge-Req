# tests/test_environment.py

import pytest
import numpy as np

from environment import Environment, generate_fractal_terrain
from organism import Organisms

# --- Helpers / Mocks ----------------------------------------------------

class DummyOrgs:
    """
    Dummy Organisms-like object to stand in for env._organisms
    and record calls to its methods.
    """
    def __init__(self):
        # record of method calls
        self.calls = []
        # single‐organism structured array with an 'energy' field
        self.organisms = np.zeros(1, dtype=[('energy', np.float32)])
        self.organisms['energy'][0] = 1.0

    def lineage(self):
        self.calls.append('lineage')
        return 'dummy_lineage'

    def build_spatial_index(self):
        self.calls.append('build_spatial_index')

    def move(self):
        self.calls.append('move')

    def resolve_attacks(self):
        self.calls.append('resolve_attacks')

    def reproduce(self):
        self.calls.append('reproduce')

    def kill_border(self):
        self.calls.append('kill_border')

    def remove_dead(self):
        self.calls.append('remove_dead')

    def get_organisms(self):
        # return the same array so that in‐place energy updates persist
        return self.organisms


# --- Environment class tests -------------------------------------------

def test_init_and_getters_normal():
    env = Environment(width=10, length=20)
    # basic getters
    assert env.get_width() == 10
    assert env.get_length() == 20
    terrain = env.get_terrain()
    assert isinstance(terrain, np.ndarray)
    assert terrain.shape == (20, 10)
    # initial counters
    assert env.get_generation() == 0
    assert env.get_total_births() == 0
    assert env.get_total_deaths() == 0
    # organisms is an Organisms instance
    assert isinstance(env.get_organisms(), Organisms)

def test_init_invalid_dimensions_raises():
    # width or length zero or negative should error (np.zeros will raise)
    for w, l in [(0, 5), (5, 0), (-1, 5), (5, -1)]:
        with pytest.raises(ValueError):
            Environment(width=w, length=l)

def test_set_terrain_normal_and_invalid():
    env = Environment(5, 5)
    # normal: mask with exactly same shape
    mask = np.random.randn(5, 5).astype(np.float32)
    env.set_terrain(mask)
    # terrain should reflect mask exactly
    assert np.all(env.get_terrain() == mask)

    # invalid: wrong shape
    bad_mask = np.zeros((4, 5), dtype=np.float32)
    with pytest.raises(ValueError) as exc:
        env.set_terrain(bad_mask)
    assert "terrain mask is wrong" in str(exc.value)

def test_add_births_and_deaths_accumulate():
    env = Environment(5, 5)
    env.add_births(7)
    env.add_deaths(3)
    assert env.get_total_births() == 7
    assert env.get_total_deaths() == 3
    # further calls accumulate
    env.add_births(2)
    env.add_deaths(5)
    assert env.get_total_births() == 9
    assert env.get_total_deaths() == 8

def test_step_sequence_and_generation_increment(capsys):
    env = Environment(5, 5)
    dummy = DummyOrgs()
    env._organisms = dummy

    # generation=0 -> 0%50==0 -> lineage() printed
    env.step()
    # first 7 calls should include lineage + all six methods
    assert dummy.calls[:7] == [
        'lineage',
        'build_spatial_index',
        'move',
        'resolve_attacks',
        'reproduce',
        'kill_border',
        'remove_dead'
    ]
    # energy decreased by exactly 0.01
    remaining = dummy.organisms['energy'][0]
    assert pytest.approx(remaining, rel=1e-6) == 0.99
    # generation incremented
    assert env.get_generation() == 1

    # capture stdout to verify lineage was printed
    # (lineage() returns a string, printed by Environment.step)
    captured = capsys.readouterr()
    assert "dummy_lineage" in captured.out

    # next step: generation=1 -> 1%50!=0 -> no lineage print
    dummy.calls.clear()
    env.step()
    assert 'lineage' not in dummy.calls
    # generation increments again
    assert env.get_generation() == 2


# --- generate_fractal_terrain tests ------------------------------------

def test_generate_fractal_terrain_invalid_dimensions():
    # width and height must both be > 0
    for w, h in [(0, 10), (10, 0), (-5, 5), (5, -5)]:
        with pytest.raises(ValueError) as exc:
            generate_fractal_terrain(w, h)
        assert "greater than zero" in str(exc.value)