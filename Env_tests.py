import pytest
import numpy as np
from organism import Organism
from grid_environment import GridEnvironment

org1 = Organism("ORG1", [1, 1, 1, 1, 1], 0, (0, 0))
org2 = Organism("ORG2", [1, 1, 2, 1, 1], 0, (0, 0))


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
    org1.set_position((2, 3))
    added = env.add_organism(org1)
    assert added is True
    assert env.occupancy[2, 3]
    assert org1 in env.get_organisms()


def test_add_organism_out_of_bounds(env):
    org1.set_position((20, 20))
    added = env.add_organism(org1)
    assert added is False


def test_add_organism_occupied(env):
    org1.set_position((2, 2))
    org2.set_position((2, 2))
    assert env.add_organism(org1) is True
    assert env.add_organism(org2) is False  # Same spot


def test_in_bounds(env):
    assert env.in_bounds(0, 0) is True
    assert env.in_bounds(9, 9) is True
    assert env.in_bounds(10, 10) is False
    assert env.in_bounds(-1, 0) is False


def test_step_increments_generation(env):
    org2.set_position((3, 3))
    env.add_organism(org2)
    assert env.get_generation() == 0
    env.step()
    assert env.get_generation() == 1


def test_step_does_not_crash_on_move(env):
    class Walker(Organism):
        def move(self, env):
            # Force move right
            r, c = self._position
            if env.in_bounds(r, c + 1):
                self._position = np.array([r, c + 1])

    org = Walker("WALKER", [1, 1, 1, 1, 1], 0, position=(5, 5))
    env.add_organism(org)
    env.step()
    r, c = org.get_position()
    assert (r, c) in [(5, 5), (5, 6)]  # Moved or blocked
    assert env.occupancy[r, c]
