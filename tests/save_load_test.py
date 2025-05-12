import pickle
import tempfile
import io
from button import Button, create_save_button, create_load_button
import numpy as np


class DummyViewer:
    def __init__(self, environment, window_size=(100, 100)):
        self.environment = environment
        self.window_size = window_size
        self.timestep = 0

# Dummy environment class
class DummyEnv:
    def __init__(self, name):
        self.name = name
        self.width = 100
        self.height = 100
        self._organisms = DummyOrgs(self)


class DummyOrgs:
    def __init__(self, env):
        self.env = env
        self._organism_dtype = np.dtype([
            ('x_pos', np.float32),
            ('y_pos', np.float32),
        ])

        self._organisms = np.zeros((0,), dtype=self._organism_dtype)

    def dummy_spawn_initial_organisms(self, number_of_organisms: int) -> int:
        """
        Spawns initial organisms with randomized genome and stats.
        """
        env_width = self.env.width

        # Random positions in the environment
        positions = np.random.randint(0, env_width, size=(number_of_organisms, 2)).astype(np.float32)

        # === Create structured organism array ===
        orgs = np.zeros(number_of_organisms, dtype=self._organism_dtype)
        orgs['x_pos'] = positions[:, 0]
        orgs['y_pos'] = positions[:, 1]

        # Store arrays
        self._organisms = np.concatenate((self._organisms, orgs))

def test_save_and_load_simulation_memory():
    env = DummyEnv("TestEnv")
    timestep = 123
    buffer = io.BytesIO()

    # Simulate saving to memory
    pickle.dump({'env': env, 'timestep': timestep}, buffer)

    # Simulate loading from memory
    buffer.seek(0)
    data = pickle.load(buffer)

    assert data['env'].name == env.name
    assert data['timestep'] == timestep

def test_save_and_load_simulation_file():
    """Tests that save and load correctly restore variable values"""
    button = Button()
    env = DummyEnv("TestEnv")
    timestep = 123

    # Use a temporary file to save/load
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        button.save_simulation(tmp.name, env, timestep)
        loaded_env, loaded_timestep = button.load_simulation(tmp.name)

    assert loaded_env.name == env.name
    assert loaded_timestep == timestep

def test_save_and_load_viewer_env_after_reset():
    """Initializes a viewer with environment attribute. Saves initial state then
    resets with new values, tests that load function correctly restores initial state."""
    env = DummyEnv("TestEnv_One")
    viewer = DummyViewer(env)
    viewer.timestep = 10
    save_btn = create_save_button(None, None)
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        # Save
        save_path = tmp.name
        save_btn.save_simulation(save_path, viewer.environment, viewer.timestep)

    env = DummyEnv("TestEnv_Two")
    viewer = DummyViewer(env)
    viewer.timestep = 15

    load_btn = create_load_button(None, None)
    viewer.env, viewer.timestep = load_btn.load_simulation(save_path)

    assert viewer.env.name == "TestEnv_One"
    assert viewer.timestep == 10
