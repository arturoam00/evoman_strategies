import json

import numpy as np
from base_evolution import BaseEvolution
from demo_controller import player_controller
from environment_ import Environment_


def main():
    with open("config.json", "r") as f:
        cfg = json.load(f)

    *_, enemies, pop_size, n_gens, upper, lower = cfg.values()

    n_hidden = 10  # neural network hidden layers

    # initializes environment
    env = Environment_(
        experiment_name="specialist",
        enemies=[enemies],
        player_controller=player_controller(n_hidden),
    )

    # initializes evolution object
    evo = BaseEvolution(env=env, pop_size=pop_size, lower=lower, upper=upper)

    for _ in evo.run_simulation(n_gens=n_gens):
        print(np.mean(evo.fit_pop))


if __name__ == "__main__":
    main()
