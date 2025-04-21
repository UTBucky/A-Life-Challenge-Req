
import numpy as np
import os
import traceback
import pygame
import pytest
from tdenvironment import TDEnvironment, generate_fractal_terrain
from organism import Organism
from viewer2dp import Viewer2D
os.environ["SDL_VIDEODRIVER"] = "dummy"  # Headless testing for PyGame


# Fixtures
@pytest.fixture
def basic_env():
    env = TDEnvironment(100, 100)
    env.set_terrain(np.zeros((100, 100), dtype=np.float32))
    return env


@pytest.fixture
def viewer_env():
    pygame.init()
    env = TDEnvironment(10, 10)
    terrain = np.zeros((10, 10), dtype=np.float32)
    terrain[2:5, 2:5] = -0.5  # Water patch
    env.set_terrain(terrain)
    env.add_organisms(np.array([[5, 5]], dtype=np.float32))
    return Viewer2D(env, window_size=(300, 300), sidebar_width=50)


# TDEnvironment Tests
def test_add_and_step_organisms(basic_env):
    positions = np.array([[10, 10], [20, 20]], dtype=np.float32)
    speeds = np.array([1.0, 1.0], dtype=np.float32)
    org_refs = [Organism("Test1", [1, 1, 1, 1, 1], 0, (10, 10)),
                Organism("Test2", [1, 1, 1, 1, 1], 0, (20, 20))]
    basic_env.add_organisms(positions, speeds=speeds, org_refs=org_refs)
    assert basic_env.organisms.shape[0] == 2
    basic_env.step()
    assert np.all(basic_env.organisms['energy'] <= 10.0)  # energy reduces


def test_soft_remove_dead(basic_env):
    # Create two organisms: one alive, one to be marked dead
    positions = np.array([[10, 10], [20, 20]], dtype=np.float32)
    speeds = np.array([1.0, 1.0], dtype=np.float32)
    org_refs = [
        Organism("Dead", [1, 1, 1, 1, 1], 0, (10, 10)),
        Organism("Alive", [1, 1, 1, 1, 1], 0, (20, 20))
    ]

    # Add both organisms to the environment
    basic_env.add_organisms(positions, speeds=speeds, org_refs=org_refs)

    # Manually reduce energy of first to simulate death
    basic_env.organisms['energy'][0] = -1.0

    # Apply soft deletion logic
    basic_env.soft_remove_dead()

    # Only one organism should be alive
    alive = basic_env.organisms[basic_env.organisms['alive']]
    assert len(alive) == 1
    assert alive['org_ref'][0].get_name() == "Alive"


def test_generate_terrain():
    terrain = generate_fractal_terrain(100, 100)
    assert terrain.shape == (100, 100)
    assert np.all((terrain >= -1.0) & (terrain <= 1.0))


# Organism Tests
def test_organism_attributes():
    genome = [4, 10, 3, 1, 5]
    position = (5, 5)
    org = Organism("TestOrg", genome, 2, position)
    assert org.get_name() == "TestOrg"
    assert org.get_size() == 4
    assert org.get_position().tolist() == list(position)


def test_organism_move_and_die():
    genome = [4, 5, 1, 1, 5]
    org = Organism("TestOrg", genome, 5, (10, 10))

    class DummyEnv:
        def in_bounds(self, x, y): return 0 <= x < 100 and 0 <= y < 100
    result = org.move(DummyEnv())
    assert result in ("move", "die", "reproduce")


# Viewer2D Tests
def test_draw_methods_run(viewer_env):
    try:
        viewer_env.draw_terrain()
        viewer_env.draw_organisms()
        viewer_env.draw_sidebar()
        viewer_env.draw_generation_stat()
        viewer_env.draw_total_population_stat()
        assert isinstance(viewer_env.handle_events(), bool)
    except Exception as e:
        traceback.print_exc()
        pytest.fail(
            f"Rendering methods failed with error: {type(e).__name__}: {e}"
            )
