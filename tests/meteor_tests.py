import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from organism import Organisms
from hazard import Meteor
from environment import Environment

def test_meteor_kills_organisms():
    env = Environment(800, 800)

    dtype = Organisms.ORGANISM_CLASS
    orgs = np.zeros(2, dtype=dtype)

    orgs[0]['x_pos'], orgs[0]['y_pos'], orgs[0]['energy'] = 100.0, 100.0, 50.0  # in blast radius
    orgs[1]['x_pos'], orgs[1]['y_pos'], orgs[1]['energy'] = 500.0, 500.0, 50.0  # out of range

    organisms = Organisms(env)
    organisms.set_organisms(orgs)
    env._organisms = organisms

    meteor = Meteor(radius=100, base_damage=9999)
    meteor._x_pos, meteor._y_pos = 100.0, 100.0
    env.set_meteor(meteor)

    organisms.apply_meteor_damage(
        x=meteor.get_x_pos(),
        y=meteor.get_y_pos(),
        radius=meteor.get_radius(),
        base_damage=meteor.get_base_damage()
    )

    energy = organisms.get_organisms()['energy']
    assert energy[0] <= 0  # should be dead
    assert energy[1] > 0   # should still be alive

