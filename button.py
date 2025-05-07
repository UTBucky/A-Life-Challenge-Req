# Class for creating buttons, contains functions to create standard required buttons
import pygame
import pickle
import tkinter as tk
from tkinter import filedialog

tk.Tk().withdraw()


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
                 text_offset_x=35,
                 text_offset_y=8,
                 color=(200, 0, 0),
                 font=None
                 ):
        self._rectangle = rectangle
        self._text = text
        self._screen = screen
        self._text_offset_x = text_offset_x
        self._text_offset_y = text_offset_y
        self._color = color
        self._font = font

    def get_font(self):
        return self._font

    def get_rectangle(self):
        return self._rectangle

    def draw_button(self):
        btn_text = self._font.render(self._text, True, (255, 255, 255))
        pygame.draw.rect(self._screen, self._color, self._rectangle)
        self._screen.blit(
            btn_text,
                         (self._rectangle.x + self._text_offset_x,
                          self._rectangle.y + self._text_offset_y)
        )

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
def create_stop_start_button(screen, font, running):
    """
    Draws a single button with that shows start/stop depending on run state
    Returns rectangle object for mouse click check
    """
    stop_start_button = Button(pygame.Rect(10, 400, 100, 35),
                               "Pause" if running else "Start",
                               screen,
                               color=(89, 236, 52),
                               font=font
                               )

    return stop_start_button


def create_save_button(screen, font):
    """
    Draws a button with 'Save' text
    Returns rectangle object for mouse click check
    """
    save_button = Button(pygame.Rect(10, 450, 100, 35),
                         "Save",
                         screen,
                         color=(52, 52, 236),
                         font=font
                         )

    return save_button


def create_load_button(screen, font):
    """
    Draws a button with 'Load' text
    Returns rectangle object for mouse click check
    """
    load_button = Button(pygame.Rect(10, 500, 100, 35),
                         "Load",
                         screen,
                         color=(52, 157, 236),
                         font=font
                         )

    return load_button


def create_skip_button(screen, font):
    """
    Draws a button with 'Skip' text
    Returns rectangle object for mouse click check
    """
    skip_button = Button(pygame.Rect(10, 550, 100, 35),
                         "Skip",
                         screen,
                         color=(146, 38, 162),
                         font=font
                         )

    return skip_button
