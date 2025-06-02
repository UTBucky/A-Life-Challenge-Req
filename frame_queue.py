from environment import Environment
from environment import generate_fractal_terrain
from viewer2dp import Viewer2D
from load_genes import load_genes_from_file
import queue

class Producer:
    
    def __init__ (self, environment, terrain):
        self._producer_env = environment
        self._length = environment.get_length()
        self._width = environment.get_length()
        self._producer_terrain = terrain
        self._queue = queue.Queue(maxsize=25)
        
    def fill_queue(self):
        """
        Step the environment and enqueue frames until the internal queue is full. 
        Each frame 
        """
        while not self._queue.full():
            self._producer_env.step()
            
            organsisms = self._producer_env.get_organisms().get_organisms()
            births = self._producer_env.get_total_births()
            deaths = self._producer_env.get_total_deaths()
            generation = self._producer_env.get_generation()
            
            frame = Frame(
                    self._width,
                    self._length,
                    self._producer_terrain,
                    organsisms,
                    births,
                    deaths,
                    generation,
                    )
            self._queue.put(frame)
    
    def get_queue(self):
        return self._queue.get()
            
class Frame:
    
    def __init__ (self, 
                  width, 
                  length, 
                  terrain, 
                  organisms, 
                  births, 
                  deaths, 
                  generation
        ):
        self._frame_width = width
        self._frame_length = length
        self._frame_terrain = terrain
        self._frame_organisms = organisms
        self._frame_births = births
        self._frame_deaths = deaths
        self._frame_generation = generation
    
    def get_frame_width(self):
        return self._frame_width

    def get_frame_length(self):
        return self._frame_length

    def get_frame_terrain(self):
        return self._frame_terrain

    def get_frame_organisms(self):
        return self._frame_organisms

    def get_frame_births(self):
        return self._frame_births

    def get_frame_deaths(self):
        return self._frame_deaths

    def get_frame_generation(self):
        return self._frame_generation