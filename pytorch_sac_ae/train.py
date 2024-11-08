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
# import gymnasium as gym
# from gymnasium.wrappers import FlattenObservation

import utils
from logger import Logger
from video import VideoRecorder

from sac_ae import SacAeAgent
# import custom_env

def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--domain_name', default='cartpole')
    # parser.add_argument('--task_name', default='two_pole_balance')
    # parser.add_argument('--task_name', default='swingup')
    parser.add_argument('--task_name', default='swingup')


    parser.add_argument('--image_size', default=84, type=int)
    parser.add_argument('--action_repeat', default=1, type=int)
    parser.add_argument('--frame_stack', default=3, type=int)
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=20000, type=int)
    # train
    parser.add_argument('--agent', default='sac_ae', type=str)
    parser.add_argument('--init_steps', default=1000, type=int)
    parser.add_argument('--num_train_steps', default=200000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    # eval
    parser.add_argument('--eval_freq', default=4000, type=int)
    parser.add_argument('--num_eval_episodes', default=1, type=int)
    # critic
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--critic_beta', default=0.9, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float)
    parser.add_argument('--critic_target_update_freq', default=2, type=int)
    parser.add_argument('--critic_update_freq', default=1, type=int)
    # actor
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=1, type=int)
    # encoder/decoder
    # parser.add_argument('--encoder_type', default='pixel', type=str)
    parser.add_argument('--encoder_type', default='identity', type=str)
    parser.add_argument('--encoder_feature_dim', default=50, type=int)
    parser.add_argument('--encoder_lr', default=1e-3, type=float)
    parser.add_argument('--encoder_tau', default=0.05, type=float)
    # parser.add_argument('--decoder_type', default='pixel', type=str)
    parser.add_argument('--decoder_type', default='identity', type=str)

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
    parser.add_argument('--seed', default=3, type=int)
    # parser.add_argument('--work_dir', default='./hand_log/gripper_sparse2', type=str)
    parser.add_argument('--work_dir', default='./multiact_cartpole_log/test', type=str)

    parser.add_argument('--save_tb', default=True, action='store_true')
    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--save_buffer', default=False, action='store_true')
    parser.add_argument('--save_video', default=False, action='store_true')
    parser.add_argument('--gpu_choice', default=0, type=int)
    parser.add_argument('--gym_env', default=False, action='store_true')
    parser.add_argument('--time_rev', default=False, type=bool)
    parser.add_argument('--percent_tsym', default=20, type=int)
    parser.add_argument('--percent_sampling', default='phase_in', type=str)
    parser.add_argument('--phase_percent', default=10, type=int)
    parser.add_argument('--prioritized_replay', default=False, action='store_true')

    # parser.add_argument('--reset_agent_aftersteps', default=False, action='store_true')

    args = parser.parse_args()
    return args

def conjugate_obs(obs, next_obs, args):

    # Pixel based obs we reverse the stacked frames
    if args.encoder_type == "pixel":
        # Make a view of the original obs and reshape into rgb array of size framestack
        conj_next_obs = obs.view().reshape(args.frame_stack, 3, args.image_size, args.image_size)
        # Reshape and flip ordering of frames
        conj_next_obs = np.flip(conj_next_obs, 0)
        # Reshape again so we have our frames stacked properly for replay buffer
        conj_next_obs = conj_next_obs.reshape(args.frame_stack * 3, args.image_size, args.image_size)

        # Make a view of the original obs and reshape into rgb array of size framestack
        conj_obs = next_obs.view().reshape(args.frame_stack, 3, args.image_size, args.image_size)
        # Reshape and flip ordering of frames
        conj_obs = np.flip(conj_obs, 0)
        # Reshape again so we have our frames stacked properly for replay buffer
        conj_obs = conj_obs.reshape(args.frame_stack * 3, args.image_size, args.image_size)
    elif args.task_name == 'balance' or args.task_name == 'swingup' or args.task_name == 'swingup_sparse':
        conj_obs = next_obs.copy()
        conj_next_obs = obs.copy()

        # # hard coded for cartpole for now
        for idx in [3,4]:
            conj_obs[idx] = conj_obs[idx] * -1
            conj_next_obs[idx] = conj_next_obs[idx] * -1
        # hard coded for pendulum for now
        # for idx in [2]:
        #     conj_obs[idx] = conj_obs[idx] * -1
        #     conj_next_obs[idx] = conj_next_obs[idx] * -1

    elif args.task_name == 'two_pole_balance' or args.task_name == 'two_poles':
        conj_obs = next_obs.copy()
        conj_next_obs = obs.copy()

        # hard coded for cartpole for now
        # for idx in [5, 6, 7]:
        # Hard coded for pendulum with two pends well powered
        for idx in [4,5]:
            conj_obs[idx] = conj_obs[idx] * -1
            conj_next_obs[idx] = conj_next_obs[idx] * -1

    elif args.task_name == 'three_poles':
        conj_obs = next_obs.copy()
        conj_next_obs = obs.copy()

        # hard coded for cartpole for now
        # for idx in [7, 8, 9, 10]:
        # Hard coded for pendulum with two pends well powered
        for idx in [6, 7, 8]:
            conj_obs[idx] = conj_obs[idx] * -1
            conj_next_obs[idx] = conj_next_obs[idx] * -1

    elif args.task_name == 'four_poles' or args.task_name == 'four_pole_balance':
        conj_obs = next_obs.copy()
        conj_next_obs = obs.copy()

        # hard coded for cartpole for now
        # for idx in [9, 10, 11, 12, 13]:
        # hard coded for actuated pendulum for now
        for idx in [8, 9, 10, 11]:
            conj_obs[idx] = conj_obs[idx] * -1
            conj_next_obs[idx] = conj_next_obs[idx] * -1


    elif args.task_name == 'five_poles':
        conj_obs = next_obs.copy()
        conj_next_obs = obs.copy()

        # hard coded for cartpole for now
        # for idx in [9, 10, 11, 12, 13]:
        # hard coded for actuated pendulum for now
        for idx in [10, 11, 12, 13, 14]:
            conj_obs[idx] = conj_obs[idx] * -1
            conj_next_obs[idx] = conj_next_obs[idx] * -1

    # Return our conjugate obs (starting observation) and our conj_next_obs (where we transition to in reverse time)
    elif args.task_name == 'run' and args.domain_name == 'humanoid':
        conj_obs = next_obs.copy()
        conj_next_obs = obs.copy()
        for idx in range(37, 67): # range doesn't include final number so real list is 37:66
            conj_obs[idx] = conj_obs[idx] * -1
            conj_next_obs[idx] = conj_next_obs[idx] * -1
    elif args.task_name == 'stand' or args.task_name == 'run' and args.domain_name == 'walker':
        conj_obs = next_obs.copy()
        conj_next_obs = obs.copy()
        for idx in range(15, 24): # range doesn't include final number so real list is 37:66
            conj_obs[idx] = conj_obs[idx] * -1
            conj_next_obs[idx] = conj_next_obs[idx] * -1

    elif args.task_name == 'lander':
        conj_obs = next_obs.copy()
        conj_next_obs = obs.copy()
        for idx in [2, 3, 5]: # range doesn't include final number so real list is 37:66
            conj_obs[idx] = conj_obs[idx] * -1
            conj_next_obs[idx] = conj_next_obs[idx] * -1
    else:
        print('Unknown task name ', args.task_name)
        print('Exiting....')
        return
    return conj_obs, conj_next_obs


def evaluate(env, agent, video, num_episodes, L, step, args):
    for i in range(num_episodes):
        if args.gym_env == True:
            obs, _ = env.reset()
        else:
            obs = env.reset()
        video.init(enabled=(i == 1))
        done = False
        episode_reward = 0
        # step_in_ep = 0
        while not done:

            with utils.eval_mode(agent):
                action = agent.select_action(obs)
            if args.gym_env == True:
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            else:
                obs, reward, done, _ = env.step(action)
                # step_in_ep += 1
                # print(obs)
                # print(step_in_ep)
            # if step_in_ep == 534 and i ==1:
            #     break

            video.record(env)
            episode_reward += reward

        video.save('%d.mp4' % step)
        L.log('eval/episode_reward', episode_reward, step)
    L.dump(step)


def make_agent(obs_shape, action_shape, args, device):
    if args.agent == 'sac_ae':
        return SacAeAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            critic_update_freq=args.critic_update_freq,
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
            prioritized_replay = args.prioritized_replay,
            total_timesteps=args.num_train_steps
        )
    else:
        assert 'agent is not supported: %s' % args.agent


def main():

    args = parse_args()
    print("Running with following configs")
    print("Time sym boolean is ", str(args.time_rev))
    print("Num of steps is ", str(args.num_train_steps))
    print("Work dir is ", str(args.work_dir))
    utils.set_seed_everywhere(args.seed)

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

    env.seed(args.seed)


    # config = {
    #     "observation": {
    #         "type": "Kinematics",
    #         # "vehicles_count": 50,
    #         "absolute": False,
    #         "duration": 100,
    #         "order": "sorted"
    #     },
    #     "action": {
    #         "type": "ContinuousAction"
    #     }
    # }

    # config = {
    #     "action": {
    #         "type": "ContinuousAction"
    #     }
    # }

    # env = gym.make('highway-fast-v0')
    # env = gym.make('two-way-v0')
    # env = gym.make('HandReach-v1')

    # env = gym.make('FetchReach-v3')
    # env.configure(config)


    # shorten episode length
    # max_episode_steps = (episode_length + args.action_repeat - 1) // args.action_repeat

    # env = gym.make( "LunarLander-v2", continuous = True, gravity = -10.0, enable_wind = True, wind_power = 15.0, turbulence_power = 1.5)
    # env = gym.make("custom_env/LunarLander-v2", continuous = True, gravity= -10, enable_wind= False, wind_power = 15.0, turbulence_power = 1.5, from_pixels = args.encoder_type == 'pixel', height = args.image_size, width = args.image_size, render_mode="rgb_array", kwargs=dict(frame_skip=args.action_repeat), max_episode_steps=max_episode_steps)

    # env.reset(seed=args.seed)
    # env.action_space.seed(args.seed)

    # stack several consecutive frames together
    if args.encoder_type == 'pixel':
        if args.gym_env == False:
            env = utils.FrameStack(env, k=args.frame_stack)
        else:
            from gymnasium.wrappers import FrameStack
            env = utils.ResizeObservation(env, (args.image_size, args.image_size))
            env = utils.GymnasiumFrameStack(env, args.frame_stack)
    else:
        if args.gym_env == True:
            env = FlattenObservation(env)

    utils.make_dir(args.work_dir)
    video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    buffer_dir = utils.make_dir(os.path.join(args.work_dir, 'buffer'))

    video = VideoRecorder(video_dir if args.save_video else None)
    video.enabled=True

    with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)


    torch.set_num_threads(6)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print('Using device ', device)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_choice)
    print("Using device number", torch.cuda.current_device())

    # the dmc2gym wrapper standardizes actions
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    # if args.time_rev:
    #     replay_buffer = utils.TsymReplayBuffer(
    #         obs_shape=env.observation_space.shape,
    #         action_shape=env.action_space.shape,
    #         capacity=args.replay_buffer_capacity,
    #         batch_size=args.batch_size,
    #         device=device,
    #         percent_tsym=args.percent_tsym,
    #         percent_sampling = args.percent_sampling,
    #         phase_percent = args.phase_percent
    #     )
    # else:
    if args.prioritized_replay:
        replay_buffer = utils.PrioritizedReplayBuffer(capacity=args.replay_buffer_capacity,
                batch_size=args.batch_size,
                device=device)
    else:
        replay_buffer = utils.ReplayBuffer(
                obs_shape=env.observation_space.shape,
                action_shape=env.action_space.shape,
                capacity=args.replay_buffer_capacity,
                batch_size=args.batch_size,
                device=device
            )

    # dual_buffer = utils.DualReplayBuffer(
    #     pix_obs_shape=env.observation_space.shape,
    #     prop_obs_shape=env.state_space.shape,
    #     capacity=args.replay_buffer_capacity,
    #     batch_size=args.batch_size,
    #     device=device
    # )

    agent = make_agent(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        args=args,
        device=device
    )

    L = Logger(args.work_dir, use_tb=args.save_tb)

    # Determine which states we need to do conjugate state transformation on
    if args.time_rev:
        if args.domain_name == 'cartpole':
            print("Running ", args.domain_name)
        elif args.domain_name == 'pendulum':
            print("Running ", args.domain_name)
        elif args.domain_name == 'varmass_pend':
            print("Running ", args.domain_name)
        elif args.domain_name == 'half_varmass_pend':
            print("Running ", args.domain_name)
        elif args.domain_name == 'acrobot':
            print("Running ", args.domain_name)
        elif args.domain_name == 'humanoid':
            print("Running ", args.domain_name)
        elif args.domain_name == 'walker':
            print("Running ", args.domain_name)
        elif args.domain_name == 'multiact_cartpole':
            print("Running ", args.domain_name)
        else:
            print("Unknown environment for now. Add a new tsymmetric environment for ", args.domain_name)
            return
        
    episode, episode_reward, done = 0, 0, True
    episode_rev_reward = 0
    start_time = time.time()
    iter_train_steps = iter(range(args.num_train_steps))
    for step in iter_train_steps:

        # if args.reset_agent_aftersteps == True:
        #     if step % 250000 == 0:
        #         # Reset the entire agent to test if we are dealing with primacy type bias
        #         agent = make_agent(
        #             obs_shape=env.observation_space.shape,
        #             action_shape=env.action_space.shape,
        #             args=args,
        #             device=device
        #         )

        # evaluate agent periodically
        if step % args.eval_freq == 0:
            L.log('eval/episode', episode, step)

            evaluate(env, agent, video, args.num_eval_episodes, L, step, args)

            if args.save_model:
                agent.save(model_dir, step)
            if args.save_buffer:
                replay_buffer.save(buffer_dir)
                # dual_buffer.save(buffer_dir)

        if done:
            if step > 0:
                L.log('train/duration', time.time() - start_time, step)
                start_time = time.time()
                L.dump(step)

            L.log('train/episode_reward', episode_reward, step)

            if args.time_rev:   
                L.log('train/episode_rev_reward', episode_rev_reward, step)
                episode_rev_reward = 0

            if args.gym_env == True:
                obs, _ = env.reset()
            else:
                obs = env.reset()

            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1

            L.log('train/episode', episode, step)
        # sample action for data collection
        if step < args.init_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.sample_action(obs)

        # run training update
        if step >= args.init_steps:
            num_updates = args.init_steps if step == args.init_steps else 1
            for _ in range(num_updates):
                agent.update(replay_buffer, L, step)

        # Modified this since the dmc2gym wrapper gives us internal state for free in the extras dict
        if args.gym_env == True:
            next_obs, reward, terminated, truncated, extra = env.step(action)
            done = terminated or truncated
        else:
            next_obs, reward, done, extra = env.step(action)

        # If time symmetric is being used then do the reverse time step
        if args.time_rev:

            # Find conjugate current and next observations
            conj_obs, conj_next_obs = conjugate_obs(obs, next_obs, args)

            # Assess the reward, done bool using simulation oracle - control suite tasks only have time based terminations
            if episode_step == 0:
                conj_done = True # Treat start of reverse time trajectory the same as a timeout
            else: 
                conj_done = False
            # if args.task_name == 'lander':
                # if next_obs[6] == 0.0 and next_obs[7] == 0.0:
            # replay_buffer.add_tsym(conj_obs, action, extra['rev_reward'], conj_next_obs, conj_done)
            replay_buffer.add(conj_obs, action, extra['rev_reward'], conj_next_obs, conj_done)

            episode_rev_reward += extra['rev_reward']


        # allow infinite bootstrap
        if args.gym_env == True:
            done_bool = 0 if truncated else float(terminated)
        else:
            done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(
                done
            )
        episode_reward += reward

        replay_buffer.add(obs, action, reward, next_obs, done_bool)

        # dual_buffer.add(obs, extra['prop_obs'])

        obs = next_obs
        episode_step += 1
    # dual_buffer.save(buffer_dir)


if __name__ == '__main__':
    main()
