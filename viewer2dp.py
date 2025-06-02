# A-Life Challenege
# Zhou, Prudent, Hagan
# Preliminary rendering

import pygame
import numpy as np
import hashlib
import math
from button import *
from tk_user_made_species import run_popup
from pygame_slider import Slider



class Viewer2D:
    """
    Plane-based viewer (continuous 2D coordinates) with sidebar stats.
    """

    def __init__(
            self,
            environment,
            sidebar_width=200
    ):
        """
        Stores an a-life environment to render, window size and sidebar are
        default (1000,800)/200 respectively
        """
        self.env = environment

        # Draw window based on controls rather than environment size
        self.sidebar_width = sidebar_width
        pygame.init()
        window = pygame.display.Info()
        window_size = (window.current_w - 200, window.current_h - 200)
        self.window_size = (window_size[0], window_size[1])

        self.screen = pygame.display.set_mode(self.window_size)

        # Window caption, at the very very tippy top, might change to title
        pygame.display.set_caption("A-Life 2D Viewer")

        # pygames var for rendering, keeps track of time
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("segoeui", 14, bold=True)
        self.timestep = 0

        # Scale factors for rendering based on env size vs screen size
        self.main_area = (window_size[0] - 2*sidebar_width, window_size[1])
        self.scale_x = self.main_area[0] / self.env.get_width()
        self.scale_y = self.main_area[1] / self.env.get_length()

        # Flag for running state, used by start/stop buttona
        self._running = True

        # Creates reference to button objects for use in draw/handle_event functions
        slider_x = self.window_size[0] - self.sidebar_width + (self.sidebar_width - 20) // 2  # Adjust width if needed
        slider_y = 600  # Or set just below your last button
        self.slider = Slider(x=slider_x, y=slider_y, height=100, min_val=1, max_val=60, start_val=30)

        self.tick_rate = 5
        
        x_offset = self.window_size[0] - self.sidebar_width + (self.sidebar_width - BUTTON_WIDTH) // 2
        initial_text = "PAUSE" if self._running else "START"
        initial_color = (200, 50, 50) if self._running else (50, 200, 50)
        self._stop_start_button = create_stop_start_button(
            self.screen, self.font, text=initial_text, color=initial_color, x_offset=x_offset)
        self._save_button = create_save_button(self.screen, self.font, x_offset)
        self._load_button = create_load_button(self.screen, self.font, x_offset)
        self._skip_button = create_skip_button(self.screen, self.font, x_offset)
        self._hazard_button = create_hazard_button(self.screen, self.font, x_offset)
        self._custom_organism_button = create_custom_organism_button(self.screen, self.font, x_offset)
        self._radioactive_button = create_radioactive_button(self.screen, self.font, x_offset)
        self._drought_button = create_drought_button(self.screen, self.font, x_offset)
        self._flood_button = create_flood_button(self.screen, self.font, x_offset)
        self._print_tree_button = create_make_tree_button(self.screen, self.font, x_offset)
        self._meteor_struck = False
        self._species_colors = {}
        self._ring_radius = 1
        self._center = (window_size[0] / 2, window_size[1] / 2)
        self._radioactive_started = False
        self._radioactive = False
        self._flood = False
        self._drought = False
        self._notification_time = 0

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
        self.draw_right_sidebar()
        self.draw_generation_stat()
        self.draw_total_population_stat()
        self.draw_additional_stats()

        # Draws all buttons
        self._stop_start_button.draw_button()
        self._save_button.draw_button()
        self._load_button.draw_button()
        self._skip_button.draw_button()
        self._print_tree_button.draw_button()
        self._hazard_button.draw_button()
        self._custom_organism_button.draw_button()
        self.slider.draw(self.screen)
        self.tick_rate = self.slider.get_value()
        self._radioactive_button.draw_button()
        self._drought_button.draw_button()
        self._flood_button.draw_button()
     
        # Meteor check
        if self._meteor_struck:
            meteor = self.env.get_meteor()
            if meteor:
                meteor.draw(self.screen, self.scale_x, self.scale_y, self.sidebar_width)

        if self._radioactive_started:
            self.env.get_organisms().apply_radioactive_wave()
            self._radioactive_started = False

        if self._radioactive:
            self.draw_radioactive_wave()

        if self._notification_time > 5:
            self._notification_time = 0
            self._flood = False
            self._drought = False

        if self._flood:
            self.draw_flood()
            self._notification_time += 1
        
        if self._drought:
            self.draw_drought()
            self._notification_time += 1

        pygame.display.flip()

        self.clock.tick(self.tick_rate)
        if self._running:
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
        for label, val in [("Births", births),("Deaths", deaths), ("Avg. Energy", f"{avg_energy:.2f}")]:
            txt = self.font.render(f"{label}: {val}", True, (255, 255, 255))
            self.screen.blit(txt, (10,y))
            y += 20

        species_array = orgs['species'][alive_mask]
        diet_array = orgs['diet_type'][alive_mask]
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
                diet  = diet_array[idx]
                
                # Ensure string
                sp_str = sp.decode() if isinstance(sp, bytes) else sp
                color = self._generate_species_color(sp_str)

                # Draw color box
                box_rect = pygame.Rect(20, y + 4, 12, 12)  # small box (x, y, w, h)
                pygame.draw.rect(self.screen, color, box_rect)

                # Draw text next to the box
                line = self.font.render(f"{sp_str}: {cnt} | {diet}", True, (200, 200, 200))
                self.screen.blit(line, (40, y))
                y += 20
        
        
    def draw_sidebar(self):
        """
        Sidebar background box.
        """
        pygame.draw.rect(
            self.screen, (30, 30, 30),
            pygame.Rect(0, 0, self.sidebar_width, self.window_size[1])
        )

    def draw_right_sidebar(self):
        """
        Draws the right sidebar
        """
        
        pygame.draw.rect(
            self.screen, (30, 30, 30),
            pygame.Rect(self.window_size[0] - self.sidebar_width, 0, self.sidebar_width, self.window_size[1])
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
    
    def draw_radioactive_wave(self):
        """Renders the radioactive wave on-screen"""

        color_1 = (144, 238, 144)
        color_2 = (34, 139, 34)
        color_3 = (0, 100, 0)

        radius_2 = max(self._ring_radius - 1, 0)
        radius_3 = max(self._ring_radius - 2, 0)

        pygame.draw.circle(self.screen, color_1, self._center, self._ring_radius, 4)
        pygame.draw.circle(self.screen, color_2, self._center, radius_2, 4)
        pygame.draw.circle(self.screen, color_3, self._center, radius_3, 4)

        self._ring_radius += 20

        if self._ring_radius >= min(self.main_area) - self.sidebar_width:
            self._radioactive_started = True
            self._radioactive = False
            self._ring_radius = 1

    def draw_flood(self):
        """Renders the flood on-screen"""
        
        color = (100, 149, 237)
        
        pygame.draw.circle(self.screen, color, self._center, 40)
        
        apex = (self._center[0], self._center[1] - 100)
        base_left = (self._center[0] - 37, self._center[1] - 15)
        base_right = (self._center[0] + 37, self._center[1] - 15)
        points = [apex, base_left, base_right]
        pygame.draw.polygon(self.screen, color, points)

    def draw_drought(self):
        """Renders the drought on-screen"""
        
        sun_color = (255, 223, 0)
        ray_color = (255, 200, 0)
        num_rays = 8
        ray_length = 50

        pygame.draw.circle(self.screen, sun_color, self._center, 40)

        for i in range(num_rays):
            angle = i * (360 / num_rays)
            radian_angle = math.radians(angle)
            point_1 = (
                self._center[0] + 40 * math.cos(radian_angle),
                self._center[1] + 40 * math.sin(radian_angle)
            )

            point_2 = (
                self._center[0] + (40 + ray_length) * math.cos(radian_angle),
                self._center[1] + (40 + ray_length) * math.sin(radian_angle)
            )

            pygame.draw.line(self.screen, ray_color, point_1, point_2, 3)

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

    def update_button_hover(self):
        mouse_pos = pygame.mouse.get_pos()
        for btn in [
            self._stop_start_button, self._save_button, self._load_button,
            self._skip_button, self._hazard_button, self._custom_organism_button
        ]:
            btn.check_hover(mouse_pos)

    def handle_events(self):
        """
        Pygames method for interactability
        """
        self.update_button_hover()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            # Button mouse click events (Stop/start, save, load, skip)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if self._stop_start_button.get_rectangle().collidepoint(event.pos):
                    self._running = not self._running
                    if self._running:
                        self._stop_start_button.set_text("PAUSE")
                        self._stop_start_button.set_color((200, 50, 50))  # red
                    else:
                        self._stop_start_button.set_text("START")
                        self._stop_start_button.set_color((50, 200, 50))  # green

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
                    if self.env.get_meteor().get_landed():
                        self.apply_meteor_effect()

                if self._custom_organism_button.get_rectangle().collidepoint(event.pos):
                    self._running = False  # Pause the sim
                    gene_dict, count = run_popup()
                    if gene_dict:
                        self.env.get_organisms().spawn_initial_organisms(number_of_organisms=count, user_genes=gene_dict)
                    self._running = True
        
                if self._radioactive_button.get_rectangle().collidepoint(event.pos): 
                    self._radioactive = True
                
                if self._drought_button.get_rectangle().collidepoint(event.pos):
                    self.env.drought()
                    self._drought = True
                    self._flood = False

                if self._flood_button.get_rectangle().collidepoint(event.pos):
                    self.env.flood()
                    self._flood = True
                    self._drought = False

                if self._print_tree_button.get_rectangle().collidepoint(event.pos):
                    self._print_tree_button.print_phylo_tree(self.env)
                
                if self._skip_button.get_rectangle().collidepoint(event.pos):
                    self.skip_frames(50)

            # Clock tick rate slider
            self.slider.handle_event(event)

        return True
    
    def skip_frames(self, input:int):
        for i in range (input):
            self.env.step()