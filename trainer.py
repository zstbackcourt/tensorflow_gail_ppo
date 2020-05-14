# -*- coding:utf-8 -*-
"""

@author: Weijie Shen
"""
import sys
import argparse
import gym
import ppo as ppo
import policies as policies


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', default='gail', type=str, help='ppo,gail')
    parser.add_argument('--env_name',default='Pendulum-v0',type=str,help='env name')
    parser.add_argument('--ent_coef',default=0.01,type=float,help='the coefficient of entropy')
    parser.add_argument('--vf_coef',default=0.1,type=float,help='the coefficient of value loss')
    parser.add_argument('--max_grad_norm',default=20,type=float,help='gradient norm clipping coefficient')
    parser.add_argument('--gamma',default=0.997,type=float,help='gamma for GAE,discounting factor')
    parser.add_argument('--lam',default=0.95,type=float,help='advantage estimation discounting factor')
    parser.add_argument('--log_interval', default=3, type=int, help='number of timesteps between logging events')
    parser.add_argument('--nminibatches',default=10,type=int,help='number of training minibatches per update. For recurrent policies')
    parser.add_argument('--noptepochs',default=3,type=int,help='number of training epochs per update')
    parser.add_argument('--cliprange',default=0.2,type=float,help='clipping range')
    parser.add_argument('--save_interval',default=20,type=int,help='number of timesteps between saving events')
    parser.add_argument('--bc_steps',default=0,type=int,help='behavior cloning steps')
    parser.add_argument('--epochs', default=3, type=int, help='epochs of each iteration.')
    parser.add_argument('--buffer_size', default=2000, type=int, help='buffer size')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--total_timesteps',default=2e6,type=float,help='total sample steps')
    parser.add_argument('--log_dir',default='./gail_log/',type=str,help='log dir')
    parser.add_argument('--save_path',default='./gail_models/',type=str,help='save path')
    parser.add_argument('--bc',default=False,type=bool,help='wether behavior cloning')
    FLAGS, _ = parser.parse_known_args()
    return FLAGS


if __name__ == "__main__":
    args = parse_args()
    env = gym.make(args.env_name)
    ppo.learn(policy=policies.MlpPolicy,
              env=env,
              buffer_size=args.buffer_size,
              nminibatches=args.nminibatches,
              total_timesteps=args.total_timesteps,
              ent_coef=args.ent_coef,
              lr=args.learning_rate,
              vf_coef=args.vf_coef,
              max_grad_norm=args.max_grad_norm,
              gamma=args.gamma,
              lam=args.lam,
              log_interval=args.log_interval,
              noptepochs=args.noptepochs,
              cliprange=args.cliprange,
              save_interval=args.save_interval,
              bc=args.bc,
              bc_steps=args.bc_steps,
              algo=args.algo,
              log_dir=args.log_dir,
              save_path=args.save_path)

