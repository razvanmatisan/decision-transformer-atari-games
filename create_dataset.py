import csv
import logging
# make deterministic
from mingpt.utils import set_seed
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torch.utils.data import Dataset
from mingpt.model_atari import GPT, GPTConfig
from mingpt.trainer_atari import Trainer, TrainerConfig
from mingpt.utils import sample
from collections import deque, Counter
import random
import torch
import pickle
import blosc
import argparse
from fixed_replay_buffer import FixedReplayBuffer
import agc.dataset as ds
import cv2
import numpy as np
import math
from os import path, listdir
import numpy as np
from scipy import stats as st

####### 1. Montezuma's Revenge ################
def preprocess_state(state, resize_shape=(84, 84)):
    # Resize state
    state = cv2.resize(state, resize_shape)

    # Gray scale
    if len(state.shape) == 3:
        if state.shape[2] == 3:
            state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)

    # Check type is compatible
    if state.dtype != np.float32:
        state = state.astype(np.float32)

    # normalize
    if state.max() > 1:
        state *= 1. / 255.

    return state

def stack_observations_per_trajectory(dir_screens_t, len_curr_t, stack_size=4):
    observations = []
    
    frame_0 = preprocess_state(cv2.imread(path.join(dir_screens_t, str(0) + '.png'), cv2.IMREAD_GRAYSCALE))
    frame_1 = preprocess_state(cv2.imread(path.join(dir_screens_t, str(1) + '.png'), cv2.IMREAD_GRAYSCALE))
    frame_2 = preprocess_state(cv2.imread(path.join(dir_screens_t, str(2) + '.png'), cv2.IMREAD_GRAYSCALE))
    frame_3 = None

    for i in range(len_curr_t - 3):
        frame = preprocess_state(cv2.imread(path.join(dir_screens_t, str(i + 3) + '.png'), cv2.IMREAD_GRAYSCALE))
        frame_3 = np.array(frame)

        observations += [[frame_0, frame_1, frame_2, frame_3]]
        frame_0, frame_1, frame_2 = frame_1, frame_2, frame_3
    
    observations += [[frame_0, frame_1, frame_2, np.zeros((84, 84))]]
    observations += [[frame_1, frame_2, np.zeros((84, 84)), np.zeros((84, 84))]]
    observations += [[frame_2, np.zeros((84, 84)), np.zeros((84, 84)), np.zeros((84, 84))]]

    return observations

def create_dataset(dataset, num_steps):
    # dataset.trajectories returns the dictionary with all the trajs from the dataset
    indices_trajectories = np.array(list(dataset.trajectories["revenge"].keys()))
    obss = []
    actions = []
    returns = []
    done_idxs = []
    timesteps = []
    rtgs = []
  

    i = 0
    curr_done_idx = -1
    for t in indices_trajectories:
        rewards = []
        ret = 0

        dir_screens_t = path.join(path.join(dataset.screens_path, "revenge"), str(t))
        curr_trajectory = dataset.trajectories["revenge"][t]
        print(dir_screens_t)

        # Trajectory length
        len_curr_t = len(listdir(dir_screens_t))

        # 1. Get observations
        obs_t = stack_observations_per_trajectory(dir_screens_t, len_curr_t)
        obss += obs_t
        
        # 2. Done_idxs
        curr_done_idx += len(curr_trajectory)
        done_idxs += [curr_done_idx]

        # 3. Get actions, timesteps, return, rtgs
        for idx, traj in enumerate(curr_trajectory):
            timesteps += [traj["frame"]]
            actions += [traj["action"]]
            rewards += [traj["reward"]]
            ret += traj["reward"]
        
        returns += [ret]

        for idx, reward in enumerate(rewards):
            ret -= reward
            rtgs += [ret]

        i += 1
        print(f'Done [{i}/{len(indices_trajectories)}]. Number transitions: {len(obss)}')
        del rewards
        if len(obss) > num_steps:
            break
    
    returns += [0]

    actions = np.array(actions)
    returns = np.array(returns)
    done_idxs = np.array(done_idxs)
    rtgs = np.array(rtgs)
    timesteps = np.array(timesteps)
    
    print(len(obss))
    print(len(actions))
    print(len(returns))
    print(len(done_idxs))
    print(len(timesteps))
    print(len(rtgs))

    print(f"Max timestep: {max(timesteps)}")
    print(f"Max rtg: {max(returns)}")

    return obss, actions, returns, done_idxs, rtgs, timesteps




################## 2. Private Eye ############################
# def create_dataset(num_buffers, num_steps, game, data_dir_prefix, trajectories_per_buffer):
#     # -- load data from memory (make more efficient)
#     obss = []
#     actions = []
#     returns = [0]
#     done_idxs = []
#     stepwise_returns = []

#     transitions_per_buffer = np.zeros(50, dtype=int)
#     num_trajectories = 0
#     while len(obss) < num_steps:
#         buffer_num = np.random.choice(np.arange(50 - num_buffers, 50), 1)[0]
#         i = transitions_per_buffer[buffer_num]
#         print('loading from buffer %d which has %d already loaded' % (buffer_num, i))
#         frb = FixedReplayBuffer(
#             data_dir=data_dir_prefix + game + '/1/replay_logs',
#             replay_suffix=buffer_num,
#             observation_shape=(84, 84),
#             stack_size=4,
#             update_horizon=1,
#             gamma=0.99,
#             observation_dtype=np.uint8,
#             batch_size=32,
#             replay_capacity=100000)
#         if frb._loaded_buffers:
#             done = False
#             curr_num_transitions = len(obss)
#             trajectories_to_load = trajectories_per_buffer
#             while not done:
#                 states, ac, ret, next_states, next_action, next_reward, terminal, indices = frb.sample_transition_batch(batch_size=1, indices=[i])
#                 states = states.transpose((0, 3, 1, 2))[0] # (1, 84, 84, 4) --> (4, 84, 84)
#                 obss += [states]
#                 actions += [ac[0]]
#                 stepwise_returns += [ret[0]]
#                 if terminal[0]:
#                     done_idxs += [len(obss)]
#                     returns += [0]
#                     if trajectories_to_load == 0:
#                         done = True
#                     else:
#                         trajectories_to_load -= 1
#                 returns[-1] += ret[0]
#                 i += 1
#                 if i >= 100000:
#                     obss = obss[:curr_num_transitions]
#                     actions = actions[:curr_num_transitions]
#                     stepwise_returns = stepwise_returns[:curr_num_transitions]
#                     returns[-1] = 0
#                     i = transitions_per_buffer[buffer_num]
#                     done = True
#             num_trajectories += (trajectories_per_buffer - trajectories_to_load)
#             transitions_per_buffer[buffer_num] = i
#         print('this buffer has %d loaded transitions and there are now %d transitions total divided into %d trajectories' % (i, len(obss), num_trajectories))

#     actions = np.array(actions)
#     returns = np.array(returns)
#     stepwise_returns = np.array(stepwise_returns)
#     done_idxs = np.array(done_idxs)

#     # -- create reward-to-go dataset
#     start_index = 0
#     rtg = np.zeros_like(stepwise_returns)
#     for i in done_idxs:
#         i = int(i)
#         curr_traj_returns = stepwise_returns[start_index:i]
#         for j in range(i-1, start_index-1, -1): # start from i-1
#             rtg_j = curr_traj_returns[j-start_index:i-start_index]
#             rtg[j] = sum(rtg_j)
#             # print(f'start index: {start_index}')
#             # print(f'i: {i}, j: {j}')
#         start_index = i
#     print('max rtg is %d' % max(rtg))

#     # -- create timestep dataset
#     start_index = 0
#     timesteps = np.zeros(len(actions)+1, dtype=int)
#     for i in done_idxs:
#         i = int(i)
#         timesteps[start_index:i+1] = np.arange(i+1 - start_index)
#         start_index = i+1
#     print('max timestep is %d' % max(timesteps))

#     # print(f"Obss: {len(obss)}, {obss[0].shape}")
#     # print(f"Actions: {len(actions)}, {Counter(actions)}")
#     # print(f"Returns: {len(returns)}, {returns}")
#     # print(f"done_idxs: {len(done_idxs)}, {done_idxs}")
#     # print(f"Rtgs: {len(rtg)}, {rtg[:680]}")
#     # print(f"Timesteps: {len(timesteps)}, {timesteps}, {timesteps[5292], timesteps[5293]}")
#     return obss, actions, returns, done_idxs, rtg, timesteps
