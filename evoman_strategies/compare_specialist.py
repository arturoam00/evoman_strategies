import json

import numpy as np

# from base_evolution import BaseEvolution
from data_manager import DataManager
from demo_controller import player_controller
from ea1 import EA1
from ea2 import EA2
from environment_ import Environment_


def main():
    # load configuration object
    with open("config.json", "r") as f:
        cfg = json.load(f)

    id_1, id_2, enemies, pop_size, n_gens, upper, lower = cfg.values()

    n_hidden = 10  # neural network hidden layers
    n_sim = 10  # number or simulations

    # initializes environment
    env = Environment_(
        experiment_name="specialist",
        enemies=enemies,
        player_controller=player_controller(n_hidden),
    )

    # initializes evolutions objects
    evo1 = EA1(env=env, pop_size=pop_size, lower=lower, upper=upper)
    evo2 = EA2(env=env, pop_size=pop_size, lower=lower, upper=upper)

    # initialize data managers for the two algorithms
    dm1 = DataManager(id=id_1, n_sim=n_sim, n_gens=n_gens)
    dm2 = DataManager(id=id_2, n_sim=n_sim, n_gens=n_gens)

    dm = {evo1: dm1, evo2: dm2}

    # for each evolutionary algorithm, run n_sim independent simulations
    for evo in [evo1, evo2]:
        for sim in range(n_sim):
            print(
                f"Running simulation #{sim} for e.a. {str(evo)} against enemy {enemies}..."
            )
            for _ in evo.run_simulation(n_gens=n_gens):
                dm[evo].store_single_run(evo.gen, evo.pop, evo.fit_pop)
            else:
                evo.restore()

    for d in [dm1, dm2]:
        individual_gain = np.zeros(len(d.best_guys))

        # best guy analysis (testing individual gain)
        for i in range(len(d.best_guys)):
            individual_gain[i] = env.return_gain(d.best_guys[i])

        # save final results
        d.store_individual_gain(individual_gain)
        d.save_results(enemy=enemies)


if __name__ == "__main__":
    main()
