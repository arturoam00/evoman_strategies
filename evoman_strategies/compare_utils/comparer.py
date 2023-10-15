from .data_manager import DataManager


class Comparer:
    def __init__(self, ea1, ea2) -> None:
        self.ea1 = ea1
        self.ea2 = ea2

    def compare(self, n_sim, n_gens):
        # initialize data managers for the two algorithms
        dm1 = DataManager(id=str(self.ea1).lower(), n_sim=n_sim, n_gens=n_gens)
        dm2 = DataManager(id=str(self.ea2).lower(), n_sim=n_sim, n_gens=n_gens)

        dm = {self.ea1: dm1, self.ea2: dm2}

        # for each evolutionary algorithm, run n_sim independent simulations
        for evo in [self.ea1, self.ea2]:
            for sim in range(n_sim):
                print(
                    f"Running simulation #{sim} for e.a. {str(evo)} against enemies {evo.env.enemies}..."
                )
                for _ in evo.run_simulation(n_gens=n_gens):
                    dm[evo].store_single_run(evo.gen, evo.pop, evo.fit_pop)
                evo.restore()

        return dm1, dm2
