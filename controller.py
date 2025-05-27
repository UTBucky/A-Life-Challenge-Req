from environment import Environment
from environment import generate_fractal_terrain
from viewer2dp import Viewer2D
from load_genes import load_genes_from_file
from hazard import Meteor


# Variables for setup/testing
GRID_SIZE = 1000         # Determines size of environment
NUM_ORGANISMS = 100    # Attempt organism creation this many times


def main():
    # Initialize environment
    env = Environment(GRID_SIZE, GRID_SIZE)

    # Generate terrain and apply terrain mask
    raw_terrain = generate_fractal_terrain(GRID_SIZE, GRID_SIZE, seed=200)
    env.set_terrain(raw_terrain)

    # Spawn initial organisms
    # TODO: Implement choice to randomize initial organisms
    gene_pool = load_genes_from_file()
    env.get_organisms().load_genes(gene_pool)
    number_of_organisms = int(NUM_ORGANISMS)
    env.get_organisms().spawn_initial_organisms(number_of_organisms, True)

    # Initialize meteor hazard with random location and passes it to env
    meteor = Meteor()
    terrain = env.get_terrain()
    meteor.determine_random_location(env.get_width(), env.get_length(), terrain)
    env.set_meteor(meteor)

    # Initialize PyGame visualization
    viewer = Viewer2D(env)

    # Run PyGame methods - could probably be combined into an "execute" method
    # in grid_viewer, here for now to make easier to see what is running
    running = True
    while running:
        # checks for events (quit, mouseclick, etc)
        running = viewer.handle_events()
        if viewer.is_running():
            # Need to call step on env attached to the viewer if loading a saved state
            viewer.get_env().step()                       # Progresses simulation 1 gen
            viewer.draw_screen()                    # Renders environment

if __name__ == "__main__":
    main()
