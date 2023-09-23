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
$ python3 specialist.py
```