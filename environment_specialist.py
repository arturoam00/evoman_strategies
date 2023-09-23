import numpy as np

from evoman.environment import Environment


class EnvironmentSpecialist(Environment):
    def __init__(
        self,
        experiment_name="specialist",
        multiplemode="no",
        enemies=[2],
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
    ):
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

    def evaluate(self, pop):
        return np.array(list(map(lambda y: self.simulation(y), pop)))
