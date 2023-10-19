from .base_evolution import BaseEvolution


class Progressive(BaseEvolution):
    @property
    def offspring_prop(self):
        return (
            self.params.lambda_0
            + self.params.a * (self.gen / self._total_gen) ** self.params.b
        )

    @property
    def mutation_ssize(self):
        min_sigma = 0.1
        return max(
            self.params.sigma_0
            - self.params.c * (self.gen / self._total_gen) ** self.params.d,
            min_sigma,
        )
