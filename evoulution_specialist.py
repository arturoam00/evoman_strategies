import os

import numpy as np

from demo_controller import player_controller
from environment_specialist import EnvironmentSpecialist


class EvolutionSpecialist:
    def __init__(
        self,
        experiment_name,
        pop_size=100,
        lower=-1,
        upper=1,
        n_hidden=10,
        headless=True,
        **kwargs,
    ) -> None:
        self.pop_size = pop_size
        self.lower = lower
        self.upper = upper

        # this has to be done BEFORE initializing any environment
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        self.env = EnvironmentSpecialist(experiment_name=experiment_name, enemies=[2])
        self.env.player_controller = player_controller(n_hidden)
        self.n_vars = (self.env.get_num_sensors() + 1) * n_hidden + (n_hidden + 1) * 5
        self.pop = np.random.uniform(
            self.lower, self.upper, (self.pop_size, self.n_vars)
        )
        self.fit_pop = None
        self.kwargs = kwargs

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

        fps = EvolutionSpecialist.norm(self.fit_pop)
        parents = np.random.choice(
            np.arange(self.pop_size),
            size=int(self.pop_size * prop),
            p=fps,
            replace=False,
        )
        return self.pop[parents]

    def nonuniform_mutation(self, x, **kwargs):
        try:
            mutation_prob = kwargs["mutation_prob"]
        except KeyError:
            print("Mutation probabiliy UNDEFINED")
            raise

        for i in range(len(x)):
            if mutation_prob >= np.random.uniform(0, 1):
                x[i] += np.random.normal(0, 1)
                x[i] = self._check_limits(x[i])
        return x

    def self_adaptive_mutation(self, x):
        raise NotImplementedError

    def no_mutation(self, x, **kwargs):
        return x

    def mutate(self, x, mutation_strategy="no_mutation", **kwargs):
        mutation = {
            "non_uniform": self.nonuniform_mutation,
            "self_adaptive": self.self_adaptive_mutation,
            "no_mutation": self.no_mutation,
        }
        return mutation[mutation_strategy](x, **kwargs)

    def calculate_offspring(self, **kwargs):
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
            offspring[i] = self.mutate(offspring[i], **kwargs)
            offspring[i + 1] = self.mutate(offspring[i + 1], **kwargs)

        return offspring

    def selection(self):
        fit_pop_total = np.concatenate(
            (self.fit_pop, self.env.evaluate(self.offspring))
        )
        total = np.vstack((self.pop, self.offspring))
        self.fit_pop = np.array(sorted(fit_pop_total, reverse=True)[: self.pop_size])
        return total[np.argsort(-fit_pop_total)][: self.pop_size]

    def run_simulation(self, n_gens=30):
        mutation_strategy = self.kwargs.get("mutation_strategy", "non_uniform")
        mutation_prob = self.kwargs.get("mutation_prob", 0.2)

        for gen in range(n_gens):
            self.parents = self.select_parents(prop=0.5)
            self.offspring = self.calculate_offspring(
                mutation_strategy=mutation_strategy, mutation_prob=mutation_prob
            )
            self.pop = self.selection()

            print(
                f"gen: {gen}; max_fit: {max(self.fit_pop)}; mean_fit: {np.mean(self.fit_pop)}; std_fit: {np.std(self.fit_pop)}"
            )
