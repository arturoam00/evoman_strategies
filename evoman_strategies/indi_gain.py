import os

import hydra
import numpy as np
from demo_controller import player_controller
from environment_ import Environment_


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    enemies = "12345678"
    games = np.zeros(len(enemies))
    pcont = np.loadtxt(os.path.join(cfg.agent.best_folder, "best.txt"))

    for i, enemy in enumerate(enemies):
        env = Environment_(
            enemies=enemy, player_controller=player_controller(cfg.nn.n_hidden)
        )
        games[i] = env.return_gain(pcont=pcont)

    print(f"Games: {games}; average individual gain: {np.sum(games) / len(games)}")

    return


if __name__ == "__main__":
    main()
