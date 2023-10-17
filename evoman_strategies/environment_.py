import os
from collections.abc import Iterable
from copy import copy

import numpy as np
from evoman.environment import Environment

default_params = {
    "experiment_name": "specialist",
    "multiplemode": "no",
    "enemies": "2",
    "loadplayer": "yes",
    "loadplayer": "yes",
    "loadenemy": "yes",
    "level": 2,
    "playermode": "ai",
    "enemymode": "static",
    "speed": "fastest",
    "inputscoded": "no",
    "randomini": "no",
    "sound": "off",
    "contacthurt": "player",
    "logs": "off",
    "savelogs": "yes",
    "clockprec": "low",
    "timeexpire": 3000,
    "overturetime": 100,
    "solutions": None,
    "fullscreen": False,
    "player_controller": None,
    "enemy_controller": None,
    "use_joystick": False,
    "visuals": False,
}


class Environment_(Environment):
    def __init__(self, headless=True, **kwargs):
        params = copy(default_params)
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        enemies = kwargs.get("enemies", "2")
        if not isinstance(enemies, str):
            if not isinstance(enemies, Iterable):
                enemies = str(enemies)
            else:
                enemies = "".join(enemies)

        enemies.replace(",", "")

        if len(enemies) > 1:
            kwargs["multiplemode"] = "yes"

        kwargs["enemies"] = enemies
        params.update(kwargs)

        super().__init__(**params)

    def simulation(self, pcont):
        f, *_ = self.play(pcont=pcont)
        return f

    def return_gain(self, pcont):
        _, plife, elife, _ = self.play(pcont=pcont)
        return plife - elife

    def evaluate(self, pop):
        return np.array(list(map(lambda y: self.simulation(y), pop)))
