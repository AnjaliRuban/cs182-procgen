import os
import sys
import importlib
import tensorflow as tf
from baselines.common.models import build_impala_cnn, nature_cnn, impala_cnn_lstm, lstm
from procgen import ProcgenEnv
from ppo2.learn import learn
from baselines.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecFrameStack,
    VecNormalize
)
from baselines import logger
import argparse
import os

def main():

    parser = argparse.ArgumentParser(description='Process procgen training arguments.')
    parser.add_argument('--env_name', type=str, default='fruitbot', help='env to run on from procgen')
    parser.add_argument('--num_envs', type=int, default=64, help='number of environments run simultaneously')
    parser.add_argument('--distribution_mode', type=str, default='easy', choices=["easy", "hard", "exploration", "memory", "extreme"], help='level difficulty')
    parser.add_argument('--num_levels', type=int, default=0, help='number of levels to train/test on')
    parser.add_argument('--start_level', type=int, default=0, help='start level (used to avoid testing on seen levels)')
    parser.add_argument('--num_timesteps', type=int, default=0, help='number of timesteps total to train/test on')
    parser.add_argument('--save_frequency', type=int, default=0, help='checkpoint frequency')
    parser.add_argument('--model_loc', type=str, default=None, help='location of pretrained model')
    parser.add_argument('--results_loc', type=str, default=None, help='location of where to save current model/logs')

    parser.add_argument('--eval', type=bool, default=False, help='if true, does not update model')
    parser.add_argument('--data_aug', type=str, default='normal', help='whether to apply data augmentation')
    parser.add_argument('--gray_p', type=float, default=0.8, help='p value for grayscale data augmentation')

    parser.add_argument('--value_fn', type=str, default='fc', choices=['fc', 'gmm', 'lbmdp'], help='value function for ppo2 critic')
    parser.add_argument('--cnn_fn', type=str, default='impala_cnn', choices=['impala_cnn', 'nature_cnn', 'impala_cnn_lstm', 'lstm'], help='cnn for featurization')
    parser.add_argument('--entropy_fn', type=str, default='constant', choices=['constant', 'scaled'], help='function for entropy loss coefficient')


    parser.add_argument('--ent_coef', type=float, default=0.01, help='coefficient applied to entropy loss')
    parser.add_argument('--ent_scalar', type=float, default=1, help='coefficient applied within sigmoid to scaled entropy coefficient')
    parser.add_argument('--seed', type=int, default=None, help='seed for tensorflow')
    parser.add_argument('--gamma', type=float, default=0.999, help='discount factor')
    parser.add_argument('--lam', type=float, default=0.95, help='advantage discount factor')
    parser.add_argument('--lr',  type=float, default=5e-4, help='learning rate for Adam')
    parser.add_argument('--imp_h1', type=float, default=16, help='impala cnn first hidden state')
    parser.add_argument('--imp_h2', type=float, default=64, help='impala cnn second hidden state')
    parser.add_argument('--imp_h3', type=float, default=64, help='impala cnn third hidden state')


    args = parser.parse_args()

    logger.configure(dir=args.results_loc, format_strs=['csv', 'stdout'])
    logger.info("Creating Environment")
    venv = ProcgenEnv(num_envs=args.num_envs, env_name=args.env_name, num_levels=args.num_levels, start_level=args.start_level, distribution_mode=args.distribution_mode)
    venv = VecExtractDictObs(venv, 'rgb')
    venv = VecMonitor(
        venv=venv,
        filename=None,
        keep_buf=100,
    )
    venv = VecNormalize(venv=venv, ob=False)

    logger.info("Creating Tensorflow Session")
    config = tf.ConfigProto()
    sess = tf.Session(config=config)
    sess.__enter__()

    if args.cnn_fn == 'impala_cnn':
        conv_fn = lambda x: build_impala_cnn(x, depths=[args.imp_h1,args.imp_h2,args.imp_h3], emb_size=256)
    elif args.cnn_fn == 'nature_cnn':
        conv_fn = lambda x: nature_cnn(x)
    elif args.cnn_fn == 'impala_cnn_lstm':
        conv_fn = impala_cnn_lstm()
    elif args.cnn_fn == 'lstm':
        conv_fn = lstm()
    else:
        conv_fn = mlp()

    logger.info("Training")
    learn(
        network=conv_fn,
        env=venv,
        total_timesteps=args.num_timesteps,
        eval_env = None,
        seed=args.seed,
        nsteps=256,
        ent_coef=args.ent_coef,
        lr=args.lr,
        vf_coef=0.5,
        max_grad_norm=0.5,
        gamma=args.gamma,
        lam=args.lam,
        log_interval=args.save_frequency,
        nminibatches=4,
        noptepochs=3,
        cliprange=0.2,
        save_interval=0,
        load_path=args.model_loc,
        data_aug=args.data_aug,
        args=args,
    )

if __name__ == '__main__':
    main()
