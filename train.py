import os
import sys
import importlib
import tensorflow as tf
from baselines.common.models import build_impala_cnn
from baselines.common.models import nature_cnn
from procgen import ProcgenEnv
from .ppo2.learn import learn
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
    parser.add_argument('--env_name', type=str, default='fruitbot')
    parser.add_argument('--num_envs', type=int, default=64)
    parser.add_argument('--distribution_mode', type=str, default='easy', choices=["easy", "hard", "exploration", "memory", "extreme"])
    parser.add_argument('--num_levels', type=int, default=0)
    parser.add_argument('--start_level', type=int, default=0)
    parser.add_argument('--num_timesteps', type=int, default=0)
    parser.add_argument('--save_frequency', type=int, default=0)
    parser.add_argument('--model_loc', type=str, default="exp/model/")
    parser.add_argument('--results_loc', type=str, default="exp/model/")

    parser.add_argument('--eval', type=bool, default=False)
    parser.add_argument('--data_aug', type=str, default='normal')
    parser.add_argument('--gray_p', type=float, default=0.8)

    parser.add_argument('--value_fn', type=str, default='fc', choices=['fc', 'gmm', 'lbmdp'])
    parser.add_argument('--cnn_fn', type=str, default='impala_cnn', choices=['impala_cnn', 'nature_cnn', 'impala_cnn_lstm', 'lstm'])
    parser.add_argument('--entropy_fn', type=str, default='constant', choices=['constant', 'scaled'])


    parser.add_argument('--ent_coef', type=float, default=0.01)
    parser.add_argument('--ent_scalar', type=float, default=1)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--gamma', type=float, default=0.999)
    parser.add_argument('--lam', type=float, default=0.95)
    parser.add_argument('--lr',  type=float, default=5e-4)


    args = parser.parse_args()

    logger.configure(dir=args.result_loc, format_strs=['csv', 'stdout'])
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
        conv_fn = impala_cnn()
    elif args.cnn_fn == 'nature_cnn':
        conv_fn = cnn()
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
        seed=None,
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
        args=None,
    )

if __name__ == '__main__':
    main()
