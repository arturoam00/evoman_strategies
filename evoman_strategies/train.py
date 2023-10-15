import hydra
import numpy as np
from demo_controller import player_controller
from environment_ import Environment_
from hydra.utils import instantiate


@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg):
    # initialize environment
    env = Environment_(
        experiment_name=cfg.environment.name,
        enemies=cfg.environment.enemies,
        player_controller=player_controller(cfg.nn.n_hidden),
    )

    # initialize evolution objects
    evo = instantiate(cfg.train.evo, env=env)

    for i in evo.run_simulation(n_gens=cfg.train.n_gens):
        print(f"Running single simulation for {str(evo)}. Generation #{evo.gen} ...")

    enemies = "12345678"
    games = np.zeros(len(enemies))
    for i, enemy in enumerate(enemies):
        env = Environment_(
            enemies=enemy, player_controller=player_controller(cfg.nn.n_hidden)
        )
        games[i] = env.return_gain(pcont=evo.pop[0])

    print(f"Results of best individual: {games}")

    return len([a for a in games if a > 0])


if __name__ == "__main__":
    train()
