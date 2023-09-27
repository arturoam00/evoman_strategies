import os
from collections import namedtuple

import numpy as np

Solution = namedtuple("Solution", ["fitness", "weights"])


class DataManager:
    """Helps to save evolutionary data across many independent simulations
    After n_sim simulations (each one with n_gen generations), one would
    have:
        - DataManager.mean_fits: mean of average fitness value for each generation
        - DataManager.max_fits: mean of maximum fitness value for each generation
        - DataManager.best: weights of the best individual across all simulations
    """

    def __init__(self, id, n_sim, n_gens) -> None:
        self.id = id
        self.n_sim = n_sim
        self.n_gens = n_gens
        # temporary data collectors
        self._mean_fits = np.zeros((self.n_gens, self.n_sim))
        self._max_fits = np.zeros((self.n_gens, self.n_sim))
        self._best_guys = []
        # actual data collectors after all simulations
        self.mean_fits = np.zeros(n_gens)
        self.max_fits = np.zeros(n_gens)
        self.best = None
        self.individual_gain = None

        self.sim_counter = 0

    def calculate_averages(self):
        for i in range(self.n_gens):
            self.mean_fits[i] = np.mean(self._mean_fits[i, :])
            self.max_fits[i] = np.mean(self._max_fits[i, :])

    def calculate_best(self):
        self.best = sorted(self._best_guys, key=lambda x: x.fitness, reverse=True)[
            0
        ].weights
        return

    def store_single_run(self, n_gen, pop, fit_pop):
        mean_fit = np.mean(fit_pop)
        max_fit = max(fit_pop)

        self._mean_fits[n_gen][self.sim_counter] = mean_fit
        self._max_fits[n_gen][self.sim_counter] = max_fit

        if n_gen == self.n_gens - 1 and self.sim_counter == self.n_sim - 1:
            self._best_guys.append(
                Solution(fitness=max_fit, weights=pop[np.argsort(-fit_pop)][0])
            )
            self.calculate_averages()
            self.calculate_best()

        elif n_gen == self.n_gens - 1:
            self._best_guys.append(
                Solution(fitness=max_fit, weights=pop[np.argsort(-fit_pop)][0])
            )
            self.sim_counter += 1
        return

    def store_individual_gain(self, arr):
        self.individual_gain = arr

    def save_results(self, folder="data", enemy=""):
        arr_dict = {
            "mean_fits": self.mean_fits,
            "max_fits": self.max_fits,
            "best": self.best,
            "individual_gain": self.individual_gain,
        }
        if not os.path.exists(folder) and not os.path.isdir(folder):
            os.mkdir(folder)

        np.save(os.path.join(folder, f"{self.id}_enemy{enemy}.npy"), arr_dict)
