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
    print("Basic environment initialized.")
    return env


@pytest.fixture
def viewer_env():
    pygame.init()
    env = Environment(10, 10)
    terrain = np.zeros((10, 10), dtype=np.float32)
    terrain[2:5, 2:5] = -0.5  # Water patch
    env.set_terrain(terrain)
    if env.get_organisms() is not None:
        env.get_organisms().spawn_initial_organisms(1)
    return Viewer2D(env, window_size=(300, 300), sidebar_width=50)


# Environment Tests
def test_spawn_default_organisms(basic_env):
    organisms = basic_env.get_organisms()
    assert organisms is not None, "Organisms object is None."
    organisms.spawn_initial_organisms(2)
    array = organisms.get_organisms()
    assert array is not None, "Organisms array is None."
    assert array.shape[0] == 2


def test_organisms_step(basic_env):
    basic_env.get_organisms().spawn_initial_organisms(2)
    organisms = basic_env.get_organisms().get_organisms()
    assert organisms is not None, "Organisms array is None."
    print("Organisms before step:", organisms)
    print("Energies before step:", organisms['energy'])
    energies_before = organisms['energy']
    basic_env.step()
    energies_after = organisms['energy']
    print("Energies after step:", energies_after)
    assert np.all(energies_after <= energies_before)  # energy reduces


def test_organism_death(basic_env):
    basic_env.get_organisms().spawn_initial_organisms(2)
    organisms = basic_env.get_organisms().get_organisms()
    assert organisms is not None, "Organisms array is None."
    assert organisms.size > 0  # Ensure not empty
    organisms['energy'][0] = -1.0
    basic_env.step()
    organisms = basic_env.get_organisms().get_organisms()
    assert organisms is not None, "Organisms array is None after step."
    assert len(organisms) == 1


def test_generate_terrain():
    terrain = generate_fractal_terrain(100, 100)
    assert terrain.shape == (100, 100)
    assert np.all((terrain >= -1.0) & (terrain <= 1.0))


def test_environment_initialization(basic_env):
    assert basic_env.get_width() == 100
    assert basic_env.get_length() == 100
    assert basic_env.get_total_births() == 0
    assert basic_env.get_total_deaths() == 0


def test_add_births(basic_env):
    basic_env.add_births(5)
    assert basic_env.get_total_births() == 5


def test_add_deaths(basic_env):
    basic_env.add_deaths(3)
    assert basic_env.get_total_deaths() == 3


def test_set_terrain(basic_env):
    terrain = np.ones((100, 100), dtype=np.float32)
    basic_env.set_terrain(terrain)
    assert np.array_equal(basic_env.get_terrain(), terrain)


def test_invalid_terrain_shape(basic_env):
    terrain = np.ones((50, 50), dtype=np.float32)
    with pytest.raises(ValueError):
        basic_env.set_terrain(terrain)


def test_environment_step_increases_generation(basic_env):
    basic_env.step()
    assert basic_env.get_generation() == 1


def test_generate_fractal_terrain_edge_cases():
    with pytest.raises(ValueError):
        generate_fractal_terrain(-1, 100)
    with pytest.raises(ValueError):
        generate_fractal_terrain(100, -1)
    with pytest.raises(ValueError, match="Cannot generate terrain with zero dimensions"):
        generate_fractal_terrain(0, 0)


# Viewer2D Tests
@pytest.mark.skipif(os.environ.get('SDL_VIDEODRIVER') == 'dummy', reason="Headless environment")
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
        pytest.fail(f"Rendering methods failed with error: {type(e).__name__}: {e}")
