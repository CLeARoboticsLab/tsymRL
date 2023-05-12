import numpy as np
import torch
import argparse
import os
import math
import gym
import sys
import random
import time
import json
import dmc2gym
import copy
from logger import Logger


import utils
from supervised_pix2prop_nets import pix2prop

def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--domain_name', default='cartpole')
    # parser.add_argument('--task_name', default='two_pole_balance')
    parser.add_argument('--task_name', default='swingup')

    parser.add_argument('--image_size', default=84, type=int)
    parser.add_argument('--action_repeat', default=1, type=int)
    parser.add_argument('--frame_stack', default=3, type=int)
    parser.add_argument('--time_rev', default=False, type=bool)
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=200000, type=int)
    # train
    parser.add_argument('--agent', default='pix2prop', type=str)
    parser.add_argument('--init_steps', default=1000, type=int)
    parser.add_argument('--num_train_steps', default=10000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    # eval
    parser.add_argument('--eval_freq', default=400000, type=int)
    parser.add_argument('--num_eval_episodes', default=10, type=int)
    # critic
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--critic_beta', default=0.9, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float)
    parser.add_argument('--critic_target_update_freq', default=2, type=int)
    parser.add_argument('--critic_update_freq', default=1, type=int)
    # actor
    parser.add_argument('--actor_lr', default=1e-2, type=float)
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=1, type=int)
    # encoder/decoder
    parser.add_argument('--encoder_type', default='pixel', type=str)
    # parser.add_argument('--encoder_type', default='identity', type=str)
    parser.add_argument('--encoder_feature_dim', default=50, type=int)
    parser.add_argument('--encoder_lr', default=1e-3, type=float)
    parser.add_argument('--encoder_tau', default=0.05, type=float)
    parser.add_argument('--decoder_type', default='pixel', type=str)
    # parser.add_argument('--decoder_type', default='identity', type=str)

    parser.add_argument('--decoder_lr', default=1e-3, type=float)
    parser.add_argument('--decoder_update_freq', default=1, type=int)
    parser.add_argument('--decoder_latent_lambda', default=1e-6, type=float)
    parser.add_argument('--decoder_weight_lambda', default=1e-7, type=float)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--num_filters', default=32, type=int)
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    parser.add_argument('--alpha_beta', default=0.5, type=float)
    # misc
    parser.add_argument('--seed', default=2, type=int)
    parser.add_argument('--work_dir', default='./saved_buffers', type=str)
    parser.add_argument('--save_tb', default=True, action='store_true')
    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--save_buffer', default=False, action='store_true')
    parser.add_argument('--save_video', default=False, action='store_true')
    parser.add_argument('--gpu_choice', default=1, type=int)

    args = parser.parse_args()
    return args

def make_agent(obs_shape, prop_obs_shape, args, device):
    if args.agent == 'pix2prop':
        return pix2prop(
            obs_shape=obs_shape,
            prop_obs_shape=prop_obs_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            prop_state_lr=args.actor_lr,
            prop_state_beta=args.actor_beta,
            prop_state_update_freq=args.actor_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            decoder_type=args.decoder_type,
            decoder_lr=args.decoder_lr,
            decoder_update_freq=args.decoder_update_freq,
            decoder_latent_lambda=args.decoder_latent_lambda,
            decoder_weight_lambda=args.decoder_weight_lambda,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
        )
    else:
        assert 'agent is not supported: %s' % args.agent


def main():

    args = parse_args()

    torch.set_num_threads(6)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print('Using device ', device)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_choice)

    env = dmc2gym.make(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed,
        visualize_reward=False,
        from_pixels=(args.encoder_type == 'pixel'),
        height=args.image_size,
        width=args.image_size,
        frame_skip=args.action_repeat,
        time_rev = args.time_rev
    )

    if args.encoder_type == 'pixel':
        env = utils.FrameStack(env, k=args.frame_stack)

    dual_buffer = utils.DualReplayBuffer(
        pix_obs_shape=env.observation_space.shape,
        prop_obs_shape=env.state_space.shape,
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device
    )

    buffer_dir = utils.make_dir(os.path.join(args.work_dir, 'buffer'))
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))

    dual_buffer.load(buffer_dir)


    agent = make_agent(
        obs_shape=env.observation_space.shape,
        prop_obs_shape=env.state_space.shape[0],
        args=args,
        device=device
    )

    # Make logger
    L = Logger(os.path.join(args.work_dir, 'sup_learning'), use_tb=args.save_tb)

    # Num of epochs
    num_epochs = 30000
    agent.save(model_dir, 0)
    for step in range(num_epochs):
        agent.update(dual_buffer, L, step)

        # print("Epoch ", step)
    agent.save(model_dir, step)
if __name__ == '__main__':
    main()
