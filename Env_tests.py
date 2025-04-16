import pytest
import numpy as np
from grid_environment import GridEnvironment

class DummyOrganism:
    def __init__(self, position, speed):
        self.position = np.array(position, dtype=int)
        self.speed = speed

    def move(self, env):
        # Does not move — useful for step testing
        pass

@pytest.fixture
def env():
    return GridEnvironment(size=10)

def test_environment_initialization(env):
    assert env.get_size() == 10
    assert env.get_generation() == 0
    assert isinstance(env.terrain, np.ndarray)
    assert isinstance(env.occupancy, np.ndarray)
    assert env.terrain.shape == (10, 10)
    assert env.occupancy.shape == (10, 10)

def test_add_organism_valid(env):
    org = DummyOrganism(position=(2, 3), speed=1)
    added = env.add_organism(org)
    assert added is True
    assert env.occupancy[2, 3]
    assert org in env.get_organisms()

def test_add_organism_out_of_bounds(env):
    org = DummyOrganism(position=(20, 20), speed=1)
    added = env.add_organism(org)
    assert added is False

def test_add_organism_occupied(env):
    org1 = DummyOrganism(position=(2, 2), speed=1)
    org2 = DummyOrganism(position=(2, 2), speed=2)
    assert env.add_organism(org1) is True
    assert env.add_organism(org2) is False  # Same spot

def test_in_bounds(env):
    assert env.in_bounds(0, 0) is True
    assert env.in_bounds(9, 9) is True
    assert env.in_bounds(10, 10) is False
    assert env.in_bounds(-1, 0) is False

def test_step_increments_generation(env):
    org = DummyOrganism(position=(3, 3), speed=2)
    env.add_organism(org)
    assert env.get_generation() == 0
    env.step()
    assert env.get_generation() == 1

def test_step_does_not_crash_on_move(env):
    class Walker(DummyOrganism):
        def move(self, env):
            # Force move right
            r, c = self.position
            if env.in_bounds(r, c + 1):
                self.position = np.array([r, c + 1])

    org = Walker(position=(5, 5), speed=2)
    env.add_organism(org)
    env.step()
    r, c = org.position
    assert (r, c) in [(5, 5), (5, 6)]  # Moved or blocked
    assert env.occupancy[r, c]