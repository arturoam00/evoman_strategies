from .base_evolution import BaseEvolution


class Progressive(BaseEvolution):
    @property
    def mutation_ssize(self):
        alpha = self.params.get("alpha", 1)
        return alpha - 0.9 * (self.gen / self._total_gen) ** 2

    @property
    def offspring_prop(self):
        beta = self.params.get("beta", 0.3)
        return beta + (self.gen / self._total_gen) ** 2
