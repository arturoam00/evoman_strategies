This repository is intended to provide evolutionary strategies to to optimise the solutions for different scenarios of the video game playing framework [EvoMan](https://github.com/karinemiras/evoman_framework). 

## Get Started
Just clone the repository and run: 
```bash
$ ./setup.sh
```
to obtain the latest version of the framework. This will preserve just a static copy of the framework folder. If one wanted to update the framework, just run `setup.sh` again. 


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

The cleanest example of what one might want to do with this is in the file `main.py`. 

Some parameters are loaded from a configuration object `config.json` and a single evolutionary simulation is run using the base class `BaseEvolution`. The environment for the simulation is the one from __EvoMan__ with a couple of irrelevant modifications. For each generation in the simulation, one can access to the different variables of interest as shown.
```python
# main.py

import json

import numpy as np
from base_evolution import BaseEvolution
from demo_controller import player_controller
from environment_ import Environment_

with open("config.json", "r") as f:
    cfg = json.load(f)

*_, enemies, pop_size, n_gens, upper, lower = cfg.values()


def main():
    n_hidden = 10  # neural network hidden layers

    # initializes environment
    env = Environment_(
        experiment_name="specialist",
        enemies=[enemies],
        player_controller=player_controller(n_hidden),
    )

    # initializes evolution object
    evo = BaseEvolution(env=env, pop_size=pop_size, lower=lower, upper=upper)

    for _ in evo.run_simulation(n_gens=n_gens):
        print(np.mean(evo.fit_pop))
        # also `evo.gen`, `evo.pop` or `evo.offspring` can be accessed


if __name__ == "__main__":
    main()
```

The `BaseEvolution` class provides a starting point to build different evolutionary algorithms. It is possible to change just some evolutionary steps while keeping the default behaviour. For example:
```python
# adding mutation to default evolution

import numpy as np

from base_evolution import BaseEvolution

class MyEvolution(BaseEvolution):
    
    def mutate(self, x, prob=0.2):
        for i in range(len(x)):
            if prob >= np.random.uniform(0, 1):
                x[i] += np.random.normal(0, 1)
                x[i] = self._check_limits(x[i])
        return x

```

