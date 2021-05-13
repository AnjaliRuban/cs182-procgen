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
## Experiments
For more detailed experiments, you may run the above with any number of flags:
```
--env_name ENV_NAME   env to run on from procgen
--num_envs NUM_ENVS   number of environments run simultaneously
--distribution_mode {easy,hard,exploration,memory,extreme}
                      level difficulty
--num_levels NUM_LEVELS
                      number of levels to train/test on
--start_level START_LEVEL
                      start level (used to avoid testing on seen levels)
--num_timesteps NUM_TIMESTEPS
                      number of timesteps total to train/test on
--save_frequency SAVE_FREQUENCY
                      checkpoint frequency
--model_loc MODEL_LOC
                      location of pretrained model
--results_loc RESULTS_LOC
                      location of where to save current model/logs
--eval EVAL           if true, does not update model
--data_aug DATA_AUG   whether to apply data augmentation
--gray_p GRAY_P       p value for grayscale data augmentation
--value_fn {fc,gmm,lbmdp}
                      value function for ppo2 critic
--cnn_fn {impala_cnn,nature_cnn,impala_cnn_lstm,lstm}
                      cnn for featurization
--entropy_fn {constant,scaled}
                      function for entropy loss coefficient
--ent_coef ENT_COEF   coefficient applied to entropy loss
--ent_scalar ENT_SCALAR
                      coefficient applied within sigmoid to scaled entropy
                      coefficient
--seed SEED           seed for tensorflow
--gamma GAMMA         discount factor
--lam LAM             advantage discount factor
--lr LR               learning rate for Adam
--imp_h1 IMP_H1       impala cnn first hidden state
--imp_h2 IMP_H2       impala cnn second hidden state
--imp_h3 IMP_H3       impala cnn third hidden state
```

To see our full (messy) process, please go to https://github.com/nikmandava/cs182-final-project
