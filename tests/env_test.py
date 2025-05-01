
import numpy as np
import os
import traceback
import pygame
import pytest
from environment import Environment, generate_fractal_terrain
from viewer2dp import Viewer2D
os.environ["SDL_VIDEODRIVER"] = "dummy"  # Headless testing for PyGame


# Fixtures
@pytest.fixture
def basic_env():
    env = Environment(100, 100)
    env.set_terrain(np.zeros((100, 100), dtype=np.float32))
    return env


@pytest.fixture
def viewer_env():
    pygame.init()
    env = Environment(10, 10)
    terrain = np.zeros((10, 10), dtype=np.float32)
    terrain[2:5, 2:5] = -0.5  # Water patch
    env.set_terrain(terrain)
    env.get_organisms().spawn_initial_organisms(1)
    return Viewer2D(env, window_size=(300, 300), sidebar_width=50)


# Environment Tests
def test_spawn_default_organisms(basic_env):
    organisms = basic_env.get_organisms()
    organisms.spawn_initial_organisms(2)
    assert organisms.get_organisms().shape[0] == 2


def test_organisms_step(basic_env):
    basic_env.get_organisms().spawn_initial_organisms(2)
    organisms = basic_env.get_organisms().get_organisms()
    energies_before = organisms['energy']
    basic_env.step()
    energies_after = organisms['energy']
    assert np.all(energies_after <= energies_before)  # energy reduces


def test_organism_death(basic_env):
    # Add both organisms to the environment
    basic_env.get_organisms().spawn_initial_organisms(2)
    organisms = basic_env.get_organisms().get_organisms()

    # Manually reduce energy of first to simulate death
    organisms['energy'][0] = -1.0

    # Apply death logic
    basic_env.step()

    # Only one organism should be alive
    organisms = basic_env.get_organisms().get_organisms()
    assert len(organisms) == 1

# TODO: Test death from invalid movement tile i.e. water


def test_generate_terrain():
    terrain = generate_fractal_terrain(100, 100)
    assert terrain.shape == (100, 100)
    assert np.all((terrain >= -1.0) & (terrain <= 1.0))


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
