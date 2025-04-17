# Controller for running simulation through PyGame

from grid_environment import GridEnvironment
from grid_viewer import GridViewer
from placeholder_organism import DummyOrganism
import numpy as np
import pygame


# Variables for setup/testing
GRID_SIZE = 50          # Determines size of environment
NO_OF_ORGANISMS = 50
FPS = 10                # Controls rate of display


def main():
    # Initialize environment
    env = GridEnvironment(GRID_SIZE)

    # Add dummy organisms - maybe add this as a method inside grid_environment
    # to initially populate?
    for _ in range(50):
        while True:
            r, c = np.random.randint(0, 50, size=2)
            speed = np.random.randint(1, 5)
            if env.add_organism(DummyOrganism(position=(r, c), speed=speed)):
                break

    # Initialize PyGame visualization
    viewer = GridViewer(env)

    # Run PyGame methods - This could probably be combined into an "execute" method
    # in grid_viewer, but leaving here for now to make easier to see what is running
    running = True
    while running:
        running = viewer.handle_events()        # Right now just checks for pygame quit event
        env.step()                              # Progresses simulation 1 turn (generation)
        viewer.draw_screen()                    # Displays visual of environment
        viewer.draw_organisms()                 # Displays each organism based on current position
        viewer.draw_sidebar()                   # Displays sidebar
        viewer.draw_generation_stat()           # Creates & displays generation no. text
        viewer.draw_total_population_stat()     # Creates and displays population text
        pygame.display.flip()                   # Updates content of entire display to screen
        viewer.clock.tick(FPS)                  # Controls tick rate, allowing program to run slower/faster


if __name__ == "__main__":
    main()
