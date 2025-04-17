# A-Life Challenege
# Zhou, Prudent, Hagan
# Preliminary rendering

import pygame


class GridViewer:
    """
    Pygames rendering for nxn environment
    """
    def __init__(self, environment, tile_size=10, sidebar_width=200):
        """
        Stores an a-life environment to render, tile size and sidebar are
        default 10/200 respectively
        """
        self.env = environment
        self.env_size = environment.get_size()
        self.tile_size = tile_size
        self.sidebar_width = sidebar_width

        # make the windows size
        self.window_size = (
            environment.get_size() * tile_size + sidebar_width,
            environment.get_size() * tile_size
            )
        pygame.init()

        # choose what to display
        self.screen = pygame.display.set_mode(self.window_size)

        # Window caption, at the very very tippy top, might change to title
        pygame.display.set_caption("A-Life Grid Viewer")

        # pygames var for rendering, keeps track of time
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 20)

    def draw_screen(self):
        """
        Render the environment
        """
        self.screen.fill((0, 0, 0))  # Clear background
        # --- Draw Grid Environment ---
        for r in range(self.env_size):
            for c in range(self.env_size):
                color = (0, 105, 148)
                pygame.draw.rect(
                    self.screen, color,
                    pygame.Rect(c * self.tile_size + self.sidebar_width,
                                r * self.tile_size,
                                self.tile_size,
                                self.tile_size
                                )
                )

    def draw_organisms(self):
        """
        Render organisms
        """
        for org in self.env.get_organisms():
            r, c = org.position
            pygame.draw.rect(
                self.screen, (255, 255, 0),
                pygame.Rect(c * self.tile_size + self.sidebar_width + 2,
                            r * self.tile_size + 2,
                            self.tile_size - 4,
                            self.tile_size - 4
                            )
            )

    def draw_sidebar(self):
        """
        Render sidebar
        """
        # Draw the sidebar
        pygame.draw.rect(
            self.screen, (30, 30, 30),
            pygame.Rect(0, 0, self.sidebar_width, self.env_size * self.tile_size)
            )

    def draw_generation_stat(self):
        """
        Display generation counter
        """
        gen_text = self.font.render(
            f"Generation: {self.env.get_generation()}", True, (255, 255, 255)
            )
        self.screen.blit(gen_text, (10, 10))

    def draw_total_population_stat(self):
        """
        Display total population counter
        """
        gen_text = self.font.render(
            f"Population: {len(self.env.get_organisms())}", True, (255, 255, 255)
        )
        self.screen.blit(gen_text, (10, 30))

    def handle_events(self):
        """
        Pygames method for interactability
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True
