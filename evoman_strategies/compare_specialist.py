import json

import numpy as np
from base_evolution import BaseEvolution
from data_manager import DataManager
from demo_controller import player_controller
from environment_ import Environment_


class EvolutionSpecialist(BaseEvolution):
    def __init__(self, env, pop_size=100, lower=-1, upper=1) -> None:
        super().__init__(env, pop_size, lower, upper)

    def mutate(self, x, prob=0.1):
        for i in range(len(x)):
            if prob > np.random.uniform():
                x[i] += np.random.normal(0, 1)
                x[i] = self._check_limits(x[i])
        return x


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
    evo1 = BaseEvolution(env=env, pop_size=pop_size, lower=lower, upper=upper)
    evo2 = EvolutionSpecialist(env=env, pop_size=pop_size, lower=lower, upper=upper)

    # initialize data managers for the two algorithms
    dm1 = DataManager(id=id_1, n_sim=n_sim, n_gens=n_gens)
    dm2 = DataManager(id=id_2, n_sim=n_sim, n_gens=n_gens)

    dm = {evo1: dm1, evo2: dm2}

    # for each evolutionary algorithm, run n_sim independent simulations
    for evo in [evo1, evo2]:
        for _ in range(n_sim):
            for _ in evo.run_simulation(n_gens=n_gens):
                dm[evo].store_single_run(evo.gen, evo.pop, evo.fit_pop)
            else:
                evo.restore()

    n_runs = 5
    for d in [dm1, dm2]:
        individual_gain = np.zeros(n_runs)

        # best guy analysis (testing individual gain)
        for i in range(n_runs):
            individual_gain[i] = env.return_gain(d.best)

        # save final results
        d.store_individual_gain(individual_gain)
        d.save_results(enemy=enemies)


if __name__ == "__main__":
    main()
