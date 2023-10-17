from .data_manager import DataManager


class Simulator:
    def __init__(self, evolutions) -> None:
        self.evolutions = evolutions

    def run(self, n_sim, n_gens):
        # initialize data managers for the algorithms
        data_managers = []
        for evolution in self.evolutions:
            dm = DataManager(id=str(evolution).lower(), n_sim=n_sim, n_gens=n_gens)

            for sim in range(n_sim):
                print(
                    f"Running simulation #{sim} for e.a. {str(evolution)} against enemies {evolution.env.enemies}..."
                )
                for _ in evolution.run_simulation(n_gens=n_gens):
                    dm.store_single_run(evolution.gen, evolution.pop, evolution.fit_pop)
                evolution.restore()

        data_managers.append(dm)

        return data_managers
