import os

import hydra
import numpy as np
from demo_controller import player_controller
from environment_ import Environment_
from hydra.utils import instantiate


@hydra.main(version_base=None, config_path="conf", config_name="train")
def train(cfg):
    # initialize environment
    env = Environment_(
        experiment_name=cfg.environment.name,
        enemies=cfg.environment.enemies,
        player_controller=player_controller(cfg.nn.n_hidden),
    )

    # initialize evolution objects
    evo = instantiate(cfg._evolution, env=env)

    # run evolution
    # this runs ALL n_gens generations without stopping
    print(f"Running {cfg.n_gens} generations for {str(evo)} ...")
    evo.run_all(n_gens=cfg.n_gens)

    # set current and last best individuals
    # if not os.path.exists(cfg.agent.best_folder):
    # os.makedirs(cfg.agent.best_folder)

    # best_path = os.path.join(cfg.agent.best_folder, "best.txt")
    # if not os.path.exists(best_path):
    # last_best = evo.pop[1]  # random reference solution if none previously saved
    # else:
    # last_best = np.loadtxt(best_path)
    # this_best = evo.fittest_individual

    # enemies = "12345678"
    # games_new = np.zeros(len(enemies))
    # games_last = np.zeros(len(enemies))

    ## make them play
    # for i, enemy in enumerate(enemies):
    # env = Environment_(
    # enemies=enemy, player_controller=player_controller(cfg.nn.n_hidden)
    # )
    # games_new[i] = env.return_gain(pcont=this_best)
    # games_last[i] = env.return_gain(pcont=last_best)

    # FOUND = False
    ## check for improvements and if any, save new best individual
    # if np.sum(games_new > 0) == np.sum(games_last > 0):
    # if np.sum(games_new) > np.sum(games_last):
    # FOUND = True

    # elif np.sum(games_new > 0) > np.sum(games_last > 0):
    # FOUND = True

    # if FOUND:
    # print("New best individual found! Saving results ...")
    # np.savetxt(best_path, this_best)

    return evo.max_fitness


if __name__ == "__main__":
    train()
