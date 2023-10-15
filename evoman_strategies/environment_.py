import os
from collections.abc import Iterable

import numpy as np
from evoman.environment import Environment


class Environment_(Environment):
    def __init__(
        self,
        experiment_name="specialist",
        multiplemode="no",
        enemies="2",
        loadplayer="yes",
        loadenemy="yes",
        level=2,
        playermode="ai",
        enemymode="static",
        speed="fastest",
        inputscoded="no",
        randomini="no",
        sound="off",
        contacthurt="player",
        logs="off",
        savelogs="yes",
        clockprec="low",
        timeexpire=3000,
        overturetime=100,
        solutions=None,
        fullscreen=False,
        player_controller=None,
        enemy_controller=None,
        use_joystick=False,
        visuals=False,
        headless=True,
    ):
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        if not isinstance(enemies, str):
            if not isinstance(enemies, Iterable):
                enemies = str(enemies)
            else:
                enemies = "".join(enemies)

        enemies.replace(",", "")

        if len(enemies) > 1:
            multiplemode = "yes"

        super().__init__(
            experiment_name,
            multiplemode,
            enemies,
            loadplayer,
            loadenemy,
            level,
            playermode,
            enemymode,
            speed,
            inputscoded,
            randomini,
            sound,
            contacthurt,
            logs,
            savelogs,
            clockprec,
            timeexpire,
            overturetime,
            solutions,
            fullscreen,
            player_controller,
            enemy_controller,
            use_joystick,
            visuals,
        )

    def simulation(self, pcont):
        f, *_ = self.play(pcont=pcont)
        return f

    def return_gain(self, pcont):
        _, plife, elife, _ = self.play(pcont=pcont)
        return plife - elife

    def evaluate(self, pop):
        return np.array(list(map(lambda y: self.simulation(y), pop)))
