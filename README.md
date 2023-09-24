This repository is intended to provide evolutionary strategies to to optimise the solutions for different scenarios of the video game playing framework [EvoMan](https://github.com/karinemiras/evoman_framework). 

## Get Started
Just clone the repository and run: 
```bash
$ ./setup.sh
```
to obtain the latest version of the framework. This will preserved just a static copy of the framework folder. If one wanted to update the framework, just run `setup.sh` again. 


Depending on your preferences, you can install the dependencies with `pipenv` or `pip`:
- **pipenv**
```bash
$ pipenv install
```
- **pip**
```bash
$ python3 -m pip install -r requirements.txt
```

Now you should be ready to go. In your activated virtual environment, try:
```bash
$ python3 main.py
```

## HowTo

One can run just the main file without modifying it:
```python
# main.py

from evoulution_specialist import EvolutionSpecialistBase


def main():
    evo = EvolutionSpecialistBase(
        experiment_name="specialist", pop_size=100, lower=-1, upper=1, n_hidden=10
    )

    for _ in evo.run_simulation(n_gens=100):
        print(max(evo.fit_pop)) # this will output maximum finess value of each generation


if __name__ == "__main__":
    main()
```

Or alternatively overwrite some of the base class methods to create a different evolutionary strategy. For example, if one wanted to modify the mutation (default is no mutation at all):
```python
# main.py

import numpy as np

from evoulution_specialist import EvolutionSpecialistBase


class EvolutionSpecialist(EvolutionSpecialistBase):
    def __init__(
        self,
        experiment_name,
        pop_size=100,
        lower=-1,
        upper=1,
        n_hidden=10,
        headless=True,
    ) -> None:
        super().__init__(experiment_name, pop_size, lower, upper, n_hidden, headless)

    def mutate(self, x, prob=0.2):
        for i in range(len(x)):
            if prob >= np.random.uniform(0, 1):
                x[i] += np.random.normal(0, 1)
                x[i] = self._check_limits(x[i])
        return x


def main():
    evo = EvolutionSpecialist(
        experiment_name="specialist", pop_size=100, lower=-1, upper=1, n_hidden=10
    )

    for _ in evo.run_simulation(n_gens=100):
        # do stuff 
        # `evo.fit_pop` and `evo.pop` are maybe the attributes you want to do stuff with
        # for example `max(evo.fit_pop)` or `np.std(evo.fit_pop)` will output the maximum 
        # value and the standard deviation of the fitness respectively


if __name__ == "__main__":
    main()

```