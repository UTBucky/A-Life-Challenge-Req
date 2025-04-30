from tdenvironment import TDEnvironment
from tdenvironment import generate_fractal_terrain
from viewer2dp import Viewer2D
from organism import Organism
import numpy as np


# Variables for setup/testing
GRID_SIZE = 1000         # Determines size of environment
NUM_ORGANISMS = 10000   # Attempt organism creation this many times


def main():
    # Initialize environment
    env = TDEnvironment(GRID_SIZE, GRID_SIZE)

    # Generate terrain and apply terrain mask
    raw_terrain = generate_fractal_terrain(GRID_SIZE, GRID_SIZE, seed=200)
    env.set_terrain(raw_terrain)

    # Vectorized organism creation
    max_attempts = int(NUM_ORGANISMS)
    rand_positions = np.random.randint(
        0, GRID_SIZE, size=(max_attempts, 2)
        ).astype(np.float32)
    rand_speeds = np.random.randint(
        1, 5, size=(max_attempts,)
        ).astype(np.float32)

    # Pre-make Organism references
    org_refs = [Organism("ORG1", [1, 1, int(speed), 1, 1], 0, tuple(pos))
                for speed, pos in zip(rand_speeds, rand_positions)]

    # Batch insert
    env.add_organisms(rand_positions, speeds=rand_speeds, org_refs=org_refs)

    # Initialize PyGame visualization
    viewer = Viewer2D(env)

    # Run PyGame methods - could probably be combined into an "execute" method
    # in grid_viewer, here for now to make easier to see what is running
    running = True
    while running:
        running = viewer.handle_events()            # checks for events (quit, mouseclick, etc)
        if viewer.is_running():
            # Need to call step on env attached to the viewer if loading a saved state
            viewer.get_env().step()                       # Progresses simulation 1 gen
            viewer.draw_screen()                    # Renders environment


if __name__ == "__main__":
    main()
