# A-Life Challenege
# Zhou, Prudent, Hagan
# Preliminary rendering

import pygame
import numpy as np
from button import create_stop_start_button, create_save_button, create_load_button, create_skip_button


class Viewer2D:
    """
    Plane-based viewer (continuous 2D coordinates) with sidebar stats.
    """
    def __init__(
            self,
            environment,
            window_size=(1000, 800),
            sidebar_width=200
            ):
        """
        Stores an a-life environment to render, window size and sidebar are
        default (1000,800)/200 respectively
        """
        self.env = environment

        # Draw window based on controls rather than environment size
        self.sidebar_width = sidebar_width
        self.window_size = window_size
        pygame.init()
        # Choose what to display
        self.screen = pygame.display.set_mode(window_size)

        # Window caption, at the very very tippy top, might change to title
        pygame.display.set_caption("A-Life 2D Viewer")

        # pygames var for rendering, keeps track of time
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 16)
        self.timestep = 0

        # Scale factors for rendering based on env size vs screen size
        self.main_area = (window_size[0] - sidebar_width, window_size[1])
        self.scale_x = self.main_area[0] / self.env.get_width()
        self.scale_y = self.main_area[1] / self.env.get_length()

        # Flag for running state, used by start/stop button
        self._running = True

        # Creates reference to button objects for use in draw/handle_event functions
        self._stop_start_button = create_stop_start_button(self.screen, self.font, self._running)
        self._save_button = create_save_button(self.screen, self.font)
        self._load_button = create_load_button(self.screen, self.font)
        self._skip_button = create_skip_button(self.screen, self.font)

    def get_env(self):
        return self.env

    def is_running(self):
        """Returns current run state (True or False)"""
        return self._running

    def draw_screen(self):
        """Method that renders the environment"""
        self.screen.fill((0, 0, 0))  # Clear background

        self.draw_terrain()
        self.draw_organisms()
        self.draw_sidebar()
        self.draw_generation_stat()
        self.draw_total_population_stat()
        self.draw_additional_stats()

        # Draws all buttons
        self._stop_start_button.draw_button()
        self._save_button.draw_button()
        self._load_button.draw_button()
        self._skip_button.draw_button()

        pygame.display.flip()
        self.clock.tick(5)
        self.timestep += 1

    def draw_terrain(self):
        """
        Renders terrain as a full-surface image in the main viewing area.
        Land (terrain == 0) is solid green.
        Water (terrain < 0) is rendered as a blue
        gradient based on depth (-1 to 0).
        """
        terrain_surface = pygame.Surface((self.env.get_width(),
                                          self.env.get_length()))
        terrain = self.env.get_terrain()
        rgb = np.zeros((terrain.shape[0], terrain.shape[1], 3), dtype=np.uint8)

        # Land - Green
        land_mask = terrain >= 0.0
        rgb[land_mask] = np.asarray(
            [34, 139, 34], dtype=np.uint8
            )

        # Water gradient
        water_mask = terrain < 0.0
        # Set only B for water no R/G
        blue_values = 255 * (1 + terrain[water_mask])
        blue_values[blue_values >= 255] = 0
        blue_values = blue_values.astype(np.uint8)
        rgb[water_mask, 0] = 0       # Red
        rgb[water_mask, 1] = 0       # Green
        rgb[water_mask, 2] = blue_values  # Blue gradient

        # Display
        pygame.surfarray.blit_array(terrain_surface, rgb.swapaxes(0, 1))
        terrain_surface = pygame.transform.scale(
            terrain_surface, self.main_area
            )
        self.screen.blit(terrain_surface, (self.sidebar_width, 0))

    def draw_organisms(self):
        """
        Renders all organisms as colored dots depending on energy level.
        Only renders those marked as alive.
        """
        alive = self.env.get_organisms().get_organisms()

        for org in alive:
            x = int(org['x_pos'] * self.scale_x) + self.sidebar_width
            y = int(org['y_pos'] * self.scale_y)
            energy = org['energy']

            if energy < 5:
                color = (255, 0, 0)
            elif energy < 15:
                color = (255, 255, 0)
            else:
                color = (0, 255, 0)

            pygame.draw.circle(self.screen, color, (x, y), 3)

    def draw_additional_stats(self):
        """
        Display births, deaths, and average energy of the alive population.
        """
        alive = self.env.get_organisms().get_organisms()
        births = self.env.get_total_births()
        deaths = self.env.get_total_deaths()
        avg_energy = np.mean(alive['energy']) if len(alive) > 0 else 0

        birth_text = self.font.render(
            f"Births: {births}", True, (255, 255, 255)
            )
        death_text = self.font.render(
            f"Deaths: {deaths}", True, (255, 255, 255)
            )
        energy_text = self.font.render(
            f"Avg Energy: {avg_energy:.2f}", True, (255, 255, 255)
            )

        self.screen.blit(birth_text, (10, 50))
        self.screen.blit(death_text, (10, 70))
        self.screen.blit(energy_text, (10, 90))

    def draw_sidebar(self):
        """
        Sidebar background box.
        """
        pygame.draw.rect(
            self.screen, (30, 30, 30),
            pygame.Rect(0, 0, self.sidebar_width, self.window_size[1])
            )

    def draw_generation_stat(self):
        """
        Display generation counter
        """
        gen_text = self.font.render(
            f"Generation: {self.timestep}", True, (255, 255, 255)
            )
        self.screen.blit(gen_text, (10, 10))

    def draw_total_population_stat(self):
        """
        Display total population counter
        """
        live_count = np.count_nonzero(self.env.get_organisms().get_organisms())
        pop_text = self.font.render(
            f"Population: {live_count}",
            True,
            (255, 255, 255)
        )
        self.screen.blit(pop_text, (10, 30))

    def handle_events(self):
        """
        Pygames method for interactability
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            # Button mouse click events (Stop/start, save, load, skip)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if self._stop_start_button.get_rectangle().collidepoint(event.pos):         # Start/stop
                    self._running = not self._running

                if self._save_button.get_rectangle().collidepoint(event.pos):               # Save simulation
                    self._save_button.save_simulation_prompt(self.env, self.timestep)

                if self._load_button.get_rectangle().collidepoint(event.pos):               # Load simulation
                    saved_env, saved_timestep = self._load_button.load_simulation_prompt()
                    if saved_env is not None:
                        self.env = saved_env
                        self.timestep = saved_timestep
        return True
