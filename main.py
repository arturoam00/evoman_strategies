from evoulution_specialist import EvolutionSpecialist


def main():
    evo = EvolutionSpecialist(
        experiment_name="specialist", pop_size=100, lower=-1, upper=1, n_hidden=10
    )
    evo.run_simulation(n_gens=100)


if __name__ == "__main__":
    main()
