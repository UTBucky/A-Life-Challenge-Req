# A-Life Challenege
# Zhou, Prudent, Hagan
# Preliminary rendering

import pygame
import numpy as np


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
        self.scale_x = self.main_area[0] / self.env.width
        self.scale_y = self.main_area[1] / self.env.height

    def draw_screen(self):
        """Method that renders the environment"""
        self.screen.fill((0, 0, 0))  # Clear background

        self.draw_terrain()
        self.draw_organisms()
        self.draw_sidebar()
        self.draw_generation_stat()
        self.draw_total_population_stat()
        self.draw_additional_stats()

        pygame.display.flip()
        self.clock.tick(60)
        self.timestep += 1

    def draw_terrain(self):
        """
        Renders terrain as a full-surface image in the main viewing area.
        Land (terrain == 0) is solid green.
        Water (terrain < 0) is rendered as a blue
        gradient based on depth (-1 to 0).
        """
        terrain_surface = pygame.Surface((self.env.width, self.env.height))
        terrain = self.env.terrain
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
        alive = self.env.organisms[self.env.organisms['alive']]

        for org in alive:
            x = int(org['x'] * self.scale_x) + self.sidebar_width
            y = int(org['y'] * self.scale_y)
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
        alive = self.env.organisms[self.env.organisms['alive']]
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
        live_count = np.count_nonzero(self.env.organisms['alive'])
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
        return True
