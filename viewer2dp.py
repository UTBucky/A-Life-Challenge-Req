# A-Life Challenege
# Zhou, Prudent, Hagan
# Preliminary rendering

import pygame
import numpy as np
import hashlib
from button import create_stop_start_button, create_save_button, create_load_button, create_skip_button, \
    create_hazard_button


class Viewer2D:
    """
    Plane-based viewer (continuous 2D coordinates) with sidebar stats.
    """

    def __init__(
            self,
            environment,
            window_size=(1600, 900),
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
        self._stop_start_button = create_stop_start_button(
            self.screen, self.font, self._running)
        self._save_button = create_save_button(self.screen, self.font)
        self._load_button = create_load_button(self.screen, self.font)
        self._skip_button = create_skip_button(self.screen, self.font)
        self._hazard_button = create_hazard_button(self.screen, self.font)
        self._meteor_struck = False
        self._species_colors = {}

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
        self._hazard_button.draw_button()

        if self._meteor_struck:             # Checks for hazard button click
            self.draw_meteor()

        pygame.display.flip()

        self.clock.tick(10)
        self.timestep += 1

    def draw_terrain(self):
        """
        Renders terrain as a full-surface image in the main viewing area.
        Land (terrain == 0) is solid green.
        Water (terrain < 0) is rendered as a blue
        gradient based on depth (-1 to 0).
        """
        w, h = self.env.get_width(), self.env.get_length()
        terrain_surface = pygame.Surface((w, h))
        terrain = self.env.get_terrain()

        rgb = np.zeros((terrain.shape[0], terrain.shape[1], 3), dtype=np.uint8)

        # Land - Green
        flat_mask     = terrain == 0.0
        water_mask    = terrain < 0.0
        mountain_mask = terrain > 0.0

        rgb[flat_mask] = np.array([34, 139, 34], dtype=np.uint8)

        # Water gradient
        depth = terrain[water_mask]           # negative values in [-1, 0)
        blue = (255 * (1 + depth)).clip(0, 255).astype(np.uint8)
        rgb[water_mask, 0] = 0
        rgb[water_mask, 1] = 0
        rgb[water_mask, 2] = blue

        # Mountain graident
        if mountain_mask.any():
            heights = terrain[mountain_mask]            # in (0, max_h]
            max_h   = float(terrain.max())              # assume > 0 since mask.any()
            norm_h  = (heights / max_h).clip(0.0, 1.0)   # range [0,1]

            # base color (same green) → peak color (white)
            base_rgb  = np.array([34, 139, 34], dtype=np.float32)
            brightness = 1.0 - 0.8 * norm_h   # at peak, brightness = 0.2
            # apply to the base green:
            darker = (base_rgb[None, :] * brightness[:, None]).clip(0,255)
            rgb[mountain_mask] = darker.astype(np.uint8)

        
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
        Energy → color mapping:
        e < 5   → red
        5 ≤ e < 10  → red-orange
        10 ≤ e < 20 → orange-yellow
        20 ≤ e < 40 → yellow
        40 ≤ e < 80 → white
        e ≥ 80      → white
        """
        alive = self.env.get_organisms().get_organisms()

        for org in alive:
            x = int(org['x_pos'] * self.scale_x) + self.sidebar_width
            y = int(org['y_pos'] * self.scale_y)
            species = org['species']
            if isinstance(species, bytes):
                species = species.decode()
            color = self._generate_species_color(species)

            # pygame.draw.circle(self.screen, color, (x, y), 3)
            diet = org['diet_type'].decode() if isinstance(org['diet_type'], bytes) else org['diet_type']

            if diet == "Herb":
                # Yellow hexagon
                r = 4  # radius of hexagon
                points = [
                    (x + r * np.cos(np.pi / 3 * i), y + r * np.sin(np.pi / 3 * i))
                    for i in range(6)
                ]
                pygame.draw.polygon(self.screen, color, points)

            elif diet == "Omni":
                # White diamond
                points = [(x, y - 5), (x - 5, y), (x, y + 5), (x + 5, y)]
                pygame.draw.polygon(self.screen, color, points)

            elif diet == "Carn":
                # Red square
                pygame.draw.rect(self.screen, color, pygame.Rect(x - 3, y - 3, 6, 6))

            elif diet == "Photo":
                # Green circle
                pygame.draw.circle(self.screen, color, (x, y), 4)

            elif diet == "Parasite":
                # Purple X shape (cross)
                pygame.draw.line(self.screen, color, (x - 3, y - 3), (x + 3, y + 3), 2)
                pygame.draw.line(self.screen, color, (x - 3, y + 3), (x + 3, y - 3), 2)

            else:
                # Default fallback shape
                pygame.draw.circle(self.screen, (0, 0, 0), (x, y), 3)

    def draw_additional_stats(self):
        """
        Display births, deaths, and average energy of the alive population.
        """
        orgs = self.env.get_organisms().get_organisms()
        alive_mask = (orgs['energy'] >= 0)
        
        births = self.env.get_total_births()
        deaths = self.env.get_total_deaths()
        energies = orgs['energy'][alive_mask]
        avg_energy = np.mean(energies) if energies.size > 0 else 0

        y = 50
        for label, val in [("Births", births),("deaths", deaths), ("Avg_Energy", f"{avg_energy:.2f}")]:
            txt = self.font.render(f"{label}: {val}", True, (255, 255, 255))
            self.screen.blit(txt, (10,y))
            y += 20
        
        species_array = orgs['species'][alive_mask]
        if species_array.size > 0:
            uniq, counts = np.unique(species_array, return_counts=True)
            
            order = np.argsort(-counts)
            y += 10
            header = self.font.render("Live Species:", True, (255,255,0))
            self.screen.blit(header, (10,y))
            y += 20
            
            for idx in order:
                sp    = uniq[idx]
                cnt   = counts[idx]
                line  = self.font.render(f"{sp}: {cnt}", True, (200,200,200))
                self.screen.blit(line, (20,y))
                y += 20
        
        
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

    def draw_meteor(self):
        """Renders the meteor on-screen"""
        meteor = self.env.get_meteor()
        if meteor is None or meteor.get_x_pos() is None or meteor.get_y_pos() is None:
            return
        x = int(meteor.get_x_pos() * self.scale_x) + self.sidebar_width
        y = int(meteor.get_y_pos() * self.scale_y)
        radius = int(meteor.get_radius() * ((self.scale_x + self.scale_y) / 2))

        # Draw jagged rocky appearance using polygon
        jagged_points = []
        segments = 50
        angle_step = 2 * np.pi / segments
        for i in range(segments):
            angle = i * angle_step
            # Add a little noise to the edge
            noise = np.random.uniform(0.8, 1.2)
            r = radius * noise
            px = int(x + r * np.cos(angle))
            py = int(y + r * np.sin(angle))
            jagged_points.append((px, py))

        # Outer rocky edge - dark gray outline
        pygame.draw.polygon(self.screen, (80, 80, 80), jagged_points)

        # Inner fill - solid gray circle
        pygame.draw.circle(self.screen, (169, 169, 169), (x, y), int(radius * 0.6))

    def apply_meteor_effect(self):
        """Calls the apply_meteor_damage method from Organisms, using base damage
        and meteor location to determine affected organisms."""
        meteor = self.env.get_meteor()
        organisms = self.env.get_organisms()
        organisms.apply_meteor_damage(
            x=meteor.get_x_pos(),
            y=meteor.get_y_pos(),
            radius=meteor.get_radius(),
            base_damage=meteor.get_base_damage()
        )

    def _generate_species_color(self, species_name):
        """Generates a unique color for each species name. Uses hashlib to make colors
        deterministic across all simulations. Avoids very dark/black colors."""
        if species_name not in self._species_colors:
            h = int(hashlib.md5(species_name.encode()).hexdigest()[:6], 16)
            color = ((h >> 16) & 255, (h >> 8) & 255, h & 255)
            self._species_colors[species_name] = color
        return self._species_colors[species_name]

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
                    self._save_button.save_simulation_prompt(
                        self.env, self.timestep)

                if self._load_button.get_rectangle().collidepoint(event.pos):               # Load simulation
                    saved_env, saved_timestep = self._load_button.load_simulation_prompt()
                    if saved_env is not None:
                        self.env = saved_env
                        self.timestep = saved_timestep

                if self._hazard_button.get_rectangle().collidepoint(event.pos):             # Create environment hazard
                    self._meteor_struck = True
                    self.apply_meteor_effect()
                
                # TODO: Add the following button for creating a phylogenetic tree.
                # tree = Phylo.read((StringIO(self.env.get_organisms().get_lineage_tracker().full_forest_newick())), "newick")
                # Phylo.write(tree, "my_tree.nwk", "newick")
        return True
