# Class for creating buttons, contains functions to create standard required buttons
import pygame


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
                 rectangle = pygame.Rect(10, 600, 100, 50),
                 text = "Insert text",
                 screen = None,
                 text_offset_x = 35,
                 text_offset_y = 8,
                 color = (200, 0, 0),
                 font = None
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



def create_stop_start_button(screen,font, running):
    """
    Draws a single button with that shows start/stop depending on run state
    Returns rectangle object for mouse click check
    """
    stop_start_button = Button(pygame.Rect(10, 400, 100, 35),
                          "Pause" if running else "Start",
                          screen,
                          color = (89, 236, 52),
                          font = font
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
                         color = (52, 52, 236),
                         font = font
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
                         color = (52, 157, 236),
                         font = font
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
                         color = (146, 38, 162),
                         font = font
                         )

    return skip_button


