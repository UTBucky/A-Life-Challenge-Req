# Class for creating buttons, contains functions to create standard required buttons
import pygame
import pickle
from tkinter import filedialog

BUTTON_WIDTH = 150
BUTTON_HEIGHT = 35
BUTTON_X = 10

class Button:
    """
    Base class for creating buttons

    Parameters:
        rectangle -- a pygame rectangle object
        text -- a text string or function to display text on button
        screen -- a pygame display object
        text_offset_coords -- x and y offset of text on button, tuple with 2 values for x and y
        color -- the color of the button, a tuple with 3 values for RGB
    """

    def __init__(self,
                 rectangle=pygame.Rect(10, 600, 100, 50),
                 text="Insert text",
                 screen=None,
                 color=(200, 0, 0),
                 font=None
                 ):
        self._rectangle = rectangle
        self._text = text
        self._screen = screen
        self._color = color
        self._font = font
        self._hovered = False

    def get_font(self):
        return self._font

    def get_rectangle(self):
        return self._rectangle

    def check_hover(self, mouse_pos):
        self._hovered = self._rectangle.collidepoint(mouse_pos)

    def set_text(self, new_text):
        self._text = new_text

    def set_color(self, new_color):
        self._color = new_color

    def draw_button(self):
        btn_text = self._font.render(self._text, True, (255, 255, 255))
        color = tuple(min(255, c + 30) if self._hovered else c for c in self._color)
        # Draw white border
        border_rect = self._rectangle.inflate(4, 4)
        pygame.draw.rect(self._screen, (255, 255, 255), border_rect, border_radius=10)
        # Draw inner colored button
        pygame.draw.rect(self._screen, color, self._rectangle, border_radius=8)

        text_rect = btn_text.get_rect(center=self._rectangle.center)
        self._screen.blit(btn_text, text_rect)

    def save_simulation_prompt(self, env, timestep):
        """Opens file explorer and allows user to name save file and set location"""
        file_path = filedialog.asksaveasfilename(
            defaultextension='.pkl',
            filetypes=[('Pickle Files', '*.pkl'), ("All files", "*.*")],
            title='Save Simulation',
        )
        if file_path:
            self.save_simulation(file_path, env, timestep)

    def save_simulation(self, filename, env, timestep):
        """Uses pickle to serialize environment and viewer into a binary file"""
        with open(filename, "wb") as f:
            pickle.dump({'env': env, 'timestep': timestep}, f)

    def load_simulation_prompt(self):
        """Opens file explorer and allows user to select a specific save file"""
        file_path = filedialog.askopenfilename(
            filetypes=[('Pickle Files', '*.pkl'), ("All files", "*.*")],
            title='Load Simulation',
        )
        if file_path:
            return self.load_simulation(file_path)
        return None, None   # Invalid file return

    def load_simulation(self, filename):
        """Uses pick deserialize and load environment and viewer object states"""
        with open(filename, "rb") as f:
            data = pickle.load(f)
        return data['env'], data['timestep']


# Portable functions to create the required button styles
def create_stop_start_button(screen, font, text="START", color=(50, 200, 50), x_offset=0):
    return Button(pygame.Rect(x_offset, 50, BUTTON_WIDTH, BUTTON_HEIGHT),
                  text, screen, color=color, font=font)

def create_save_button(screen, font, x_offset=0):
    return Button(pygame.Rect(x_offset, 100, BUTTON_WIDTH, BUTTON_HEIGHT),
                  "SAVE", screen, color=(52, 52, 236), font=font)

def create_load_button(screen, font, x_offset=0):
    return Button(pygame.Rect(x_offset, 150, BUTTON_WIDTH, BUTTON_HEIGHT),
                  "LOAD", screen, color=(52, 157, 236), font=font)

def create_skip_button(screen, font, x_offset=0):
    return Button(pygame.Rect(x_offset, 200, BUTTON_WIDTH, BUTTON_HEIGHT),
                  "SKIP", screen, color=(40, 40, 40), font=font)

def create_custom_organism_button(screen, font, x_offset=0):
    return Button(pygame.Rect(x_offset, 250, BUTTON_WIDTH, BUTTON_HEIGHT),
                  "CUSTOM ORGANISM", screen, color=(146, 38, 162), font=font)

def create_hazard_button(screen, font, x_offset=0):
    return Button(pygame.Rect(x_offset, 300, BUTTON_WIDTH, BUTTON_HEIGHT),
                  "METEOR", screen, color=(139, 0, 0), font=font)

def create_radioactive_button(screen, font, x_offset=0):
    """
    Draws a button with text
    Returns rectangle object for mouse click check
    """
    return Button(pygame.Rect(x_offset, 350, BUTTON_WIDTH, BUTTON_HEIGHT),
                    "RADIOACTIVE WAVE", screen, color=(139, 0, 0), font=font)

def create_drought_button(screen, font, x_offset=0):
    """
    Draws a button with text
    Returns rectangle object for mouse click check
    """
    return Button(pygame.Rect(x_offset, 400, BUTTON_WIDTH, BUTTON_HEIGHT),
                    "DROUGHT", screen, color=(139, 0, 0), font=font)

def create_flood_button(screen, font, x_offset=0):
    """
    Draws a button with text
    Returns rectangle object for mouse click check
    """
    return Button(pygame.Rect(x_offset, 450, BUTTON_WIDTH, BUTTON_HEIGHT),
                    "FLOOD", screen, color=(139, 0, 0), font=font)
