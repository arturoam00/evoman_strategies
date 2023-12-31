import os

import numpy as np


class DataManager:
    def __init__(self, id, n_sim, n_gens) -> None:
        self.id = id
        self.n_sim = n_sim
        self.n_gens = n_gens
        # temporary data collectors
        self._mean_fits = np.zeros((self.n_gens, self.n_sim))
        self._max_fits = np.zeros((self.n_gens, self.n_sim))
        # actual data collectors after all simulations
        self.best_guys = [None] * n_sim
        self.mean_fits = np.zeros(n_gens)
        self.max_fits = np.zeros(n_gens)
        self.individual_gain = None  # is this necessary?

        self.sim_counter = 0

    def calculate_averages(self):
        for i in range(self.n_gens):
            self.mean_fits[i] = np.mean(self._mean_fits[i, :])
            self.max_fits[i] = np.mean(self._max_fits[i, :])
        return

    def store_single_run(self, n_gen, pop, fit_pop):
        self._mean_fits[n_gen][self.sim_counter] = np.mean(fit_pop)
        self._max_fits[n_gen][self.sim_counter] = np.max(fit_pop)

        if n_gen == self.n_gens - 1 and self.sim_counter == self.n_sim - 1:
            self.best_guys[self.sim_counter] = pop[np.argsort(-fit_pop)][0]
            self.calculate_averages()

        elif n_gen == self.n_gens - 1:
            self.best_guys[self.sim_counter] = pop[np.argsort(-fit_pop)][0]
            self.sim_counter += 1
        return

    # TODO: this is too rigid, something like `save column` would be more convenient
    def store_individual_gain(self, a):
        self.individual_gain = a
        return

    # TODO: this should be more flexible, .csv, .json, ...
    def save_results(self, folder, enemy):
        arr_dict = {
            "mean_fits": self.mean_fits,
            "max_fits": self.max_fits,
            "individual_gain": self.individual_gain,
            "best_guys": self.best_guys,
        }
        if not os.path.exists(folder) and not os.path.isdir(folder):
            os.makedirs(folder)

        np.save(os.path.join(folder, f"{self.id}_enemies{enemy}.npy"), arr_dict)
        return
