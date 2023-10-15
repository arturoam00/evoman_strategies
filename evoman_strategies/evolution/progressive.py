from .base_evolution import BaseEvolution


class Progressive(BaseEvolution):
    @property
    def mutation_prop(self):
        alpha = self.params.get("alpha", 0.5)
        return alpha - (self.gen / self._total_gen) ** 2

    @property
    def offspring_prop(self):
        beta = self.params.get("beta", 0.5)
        return beta + (self.gen / self._total_gen) ** 2
