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

Now you should be ready to go. Navigate to the `evoulutionary_strategies` folder and, in your activated virtual environment, try:
```bash
$ python3 main.py
```

## HowTo

The cleanest example of what one might want to do with this is in the file `main.py`. 

Some parameters are loaded from a configuration object `config.json` and a single evolutionary simulation is run using the base class `BaseEvolution`. The environment for the simulation is the one from __EvoMan__ with a couple of non important modifications. For each generation in the simulation, one can access to the different variables of interest as shown.
```python
# main.py

import json

import numpy as np
from base_evolution import BaseEvolution
from demo_controller import player_controller
from environment_ import Environment_


def main():
    with open("config.json", "r") as f:
        cfg = json.load(f)

    *_, enemies, pop_size, n_gens, upper, lower = cfg.values()

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
        # also `evo.gen`, `evo.pop` or `evo.parents` can be accessed


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

## Comparing two Evolutionary Algorithms

The file `compare_specialist.py` compares two Evolutionary Algorithms that train a specialist agent. 

Consider the following steps:

1. #### Configuration

The configuration object `config.json` will be read by `compare_specialist.py` to set some parameters during the simulations. One can change the file by hand or call `setup_config.py` with the desired parameters as follows:
```bash
$ python3 setup_config.py [id_1] [id_2] [enemies] [population-size] [number-of-generations] [upper-bound-weights] [lower-bound-weights]
```
an example would be:

```bash
$ python3 setup_config.py "EA1" "EA2" "2" 100
```
That will set the configuration so that the id for the first algorithm to compare is "EA1", the second is "EA2" and the population size is 100. The rest of the values will be the default values. __The algorithm ids will be used to name the output data files__.

2. #### Run simulations
By default, 10 **independent** simulations are run for each of the algorithms. The data from this simulations is averaged and store in output data files under the `data/` folder. To do that simply run:
```bash
$ python3 compare_specialist.py
```
After the simulations, the **best individual** for each simulation is saved and evaluated again against the same enemy. The results are also saved in the same data file (one for each algorithm).

3. #### Plot results

To plot the results obtained in the previous step run:
```bash
$ python3 plotting.py
```
This will output two plots in the `images/` folder:
 - A line plot presenting the averages over the 10 simulations of the mean fitness value and the maximum fitness value for each generation. 
 - A box plot presenting the results of the best individuals against the enemy (individual gain, i.e. player_life - enemy_life).


 Remember that both `compare_specialist.py` and `plotting.py` read the configuration object `config.json`. 


 ### Loop over enemies
 To do every step of the comparasion of two EAs for all enemies, run:
 ```bash
 $ ./loop.sh
 ```
 This will perform the three steps of the comparasion (configuration, run simulation and plotting) for each enemy.


 ### Run your own `compare_specialist.py`
 Here it is sketched out how one compare two arbitrary EAs using this framework. 

 ```python
 # compare_specialist.py

 class YourEA(BaseEvolution):
    ...

 class YourOtherEA(BaseEvolution):
    ...

def main():
    # load configuration ...
    with open("config.json") as f:
        ...

    # initialize important objects for simulations
    n_hidden = ... # hidden layers for the neural network
    n_sim = ... # number of simulations

    # initialize environment 
    env = Envrionment_(...)

    # initialize your evolution objects
    evo1 = YourEA(...)
    evo2 = YourOtherEA(...)

    # initialize data managers for each of your EAs
    dm1 = DataManager(...)
    dm2 = DataManager(...)

    # create a mapping between data managers and EAs
    dm = {evo1: dm1, evo2: dm2}

    # run the simulations
    for evo in [evo1, evo2]: # algorithms loop
        for sim in range(n_sim): # simulations loop
            for _ in evo.run_simulation(...): # generations loop
                dm[evo].store_single_run(evo.gen, evo.pop, evo.fit_pop)
            else:
                evo.restore() # the runs must be independent, so restore the EA

     for d in [dm1, dm2]:
        individual_gain = np.zeros(len(d.best_guys))

        # best guy analysis (testing individual gain)
        for i in range(len(d.best_guys)):
            individual_gain[i] = env.return_gain(d.best_guys[i])

        # save final results
        d.store_individual_gain(individual_gain)
        d.save_results(...)

    if __name__ == "__main__":
        main()
 ```
