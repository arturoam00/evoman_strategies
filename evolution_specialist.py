import os

import numpy as np

from demo_controller import player_controller
from environment_specialist import EnvironmentSpecialist


class EvolutionSpecialistBase:
    def __init__(
        self,
        experiment_name,
        pop_size=100,
        lower=-1,
        upper=1,
        n_hidden=10,
        pcont=None,  # if provided, remember to specify the right number of hidden layers in `n_hidden`
        headless=True,
    ) -> None:
        self.pop_size = pop_size
        self.lower = lower
        self.upper = upper

        # this has to be done BEFORE initializing any environment
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        self.env = EnvironmentSpecialist(experiment_name=experiment_name, enemies=[2])
        if not pcont:
            pcont = player_controller(n_hidden)
        self.n_vars = (self.env.get_num_sensors() + 1) * n_hidden + (n_hidden + 1) * 5
        self.pop = np.random.uniform(
            self.lower, self.upper, (self.pop_size, self.n_vars)
        )
        self.fit_pop = None

    def _check_limits(self, x):
        if x < self.lower:
            return self.lower
        elif x > self.upper:
            return self.upper
        return x

    @staticmethod
    def norm(x):
        try:
            x -= min(x)  # remove negative values
            return x / sum(x)
        except ZeroDivisionError:
            return x

    def select_parents(self, prop=0.5):
        if self.fit_pop is None:
            self.fit_pop = self.env.evaluate(self.pop)

        fps = EvolutionSpecialistBase.norm(self.fit_pop)
        parents = np.random.choice(
            np.arange(self.pop_size),
            size=int(self.pop_size * prop),
            p=fps,
            replace=False,
        )
        return self.pop[parents]

    def mutate(self, x):
        return x

    def calculate_offspring(self):
        n_parents = np.size(self.parents, 0)
        offspring = np.zeros((n_parents, self.n_vars))

        for i in range(0, n_parents - 1, 2):
            # select two parents out of all possible parents
            p1 = self.parents[np.random.randint(0, n_parents)]
            p2 = self.parents[np.random.randint(0, n_parents)]

            # choose the recombination parameter ɑ
            alpha = np.random.uniform(0, 1)

            # recombine
            offspring[i] = alpha * p1 + (1 - alpha) * p2
            offspring[i + 1] = alpha * p2 + (1 - alpha) * p1

            # mutate
            offspring[i] = self.mutate(offspring[i])
            offspring[i + 1] = self.mutate(offspring[i + 1])

        return offspring

    def selection(self):
        fit_pop_total = np.concatenate(
            (self.fit_pop, self.env.evaluate(self.offspring))
        )
        total = np.vstack((self.pop, self.offspring))
        self.fit_pop = np.array(sorted(fit_pop_total, reverse=True)[: self.pop_size])
        return total[np.argsort(-fit_pop_total)][: self.pop_size]

    def run_simulation(self, n_gens=30):
        for gen in range(n_gens):
            self.gen = gen
            self.parents = self.select_parents()
            self.offspring = self.calculate_offspring()
            self.pop = self.selection()
            yield
