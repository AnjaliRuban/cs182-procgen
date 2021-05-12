# Training an Agent for Procgen Fruitbot

## Install
You can get miniconda from https://docs.conda.io/en/latest/miniconda.html if you don't have it, or install the dependencies from `environment.yml` manually.

```
git clone https://github.com/AnjaliRuban/cs182-procgen.git
conda env update --name cs182-procgen --file cs182-procgen/environment.yml
conda activate cs182-procgen
pip install -e cs182-procgen
```
## Quick Run
Train to match best runs:
```
python -m train --num_levels 200 --num_timesteps 5000000 --results_loc "exp/best_model" --save_frequency 60
```

Run evaluation on best model:
```
python -m train --num_levels 100 --num_timesteps 100000 --results_loc "exp/best_model/checkpoints/00600" --save_frequency 60 --start_level 200 --eval True
```
## Data Augmentation

## Featurization

## Entropy Scaling

## Value Functions
