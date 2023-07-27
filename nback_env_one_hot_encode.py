# N-back environment

import numpy as np
import gymnasium as gym
from gymnasium import Env, spaces, utils

class NBack(Env):
    
    # Examples
    # N = 2
    # step_count =        [ 0  1   2  3  4  5  6 ]
    # sequence =          [ a  b   c  d  a  d  a ] (except these are usually digits between 0-9)
    # correct actions =   [ ~  ~   0  0  0  1  1 ]
    # actions =           [ ~  ~   1  0  0  1  0 ]
    # reward_class =      [ ~  ~  FP TN TN TP FN ]
    # reward =            [ ~  ~  -1  0  0  1 -1 ]
    # Rewards input is structured as (TP, TN, FP, FN) (positive being matches)
    
    def __init__(self, N=2, num_trials=25, num_targets=None, rewards=(1, 1, -1, -1), num_obs=5, seed=2023):
        
        self.N = N
        self.num_trials = num_trials
        self.episode_length = num_trials + self.N
        self.num_targets = num_targets
        self.rewards = rewards
        self.num_obs = num_obs
        self.num_actions = 2
        np.random.seed(seed)

        # Check that parameters are legal
        assert(len(rewards) == 4)
        assert(num_targets is None or num_targets <= num_trials)

        # Define rewards, observation space and action space
        self.reward_range = (min(rewards), max(rewards))    # Range of rewards based on inputs
        # self.observation_space = spaces.Tuple([spaces.Discrete(10) for i in range(self.num_obs)])     # Tuple num_obs long with 10 possibilities
        self.observation_space = spaces.Box(low=0, high=9, shape=(num_obs, ))
        self.action_space = spaces.Discrete(self.num_actions)                        # 0 (No match) or 1 (Match)

    def reset(self, seed=None):

        # Seed RNG
        if seed is not None:
            np.random.seed(seed)

        # Generate sequence and correct actions
        self._generate_sequence()
        self._get_correct_actions()

        # Observation is first character
        self.step_count = 0
        
        # initialize
        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):

        # Calculate reward
        if self.step_count >= self.N:
            if (self.correct_actions[self.step_count - self.N]): # Match
                reward = self.rewards[0] if action else self.rewards[3] # TP if matched else FN
            else: # No match
                reward = self.rewards[2] if action else self.rewards[1] # FP if matches else TN
        else:
            reward = 0

        # Return next character or None

        self.step_count += 1
        observation = self._get_obs()
        info = self._get_info()

        if self.step_count < self.episode_length:
            return observation, reward, False, info
        else:
            return observation, reward, True, info

    def _generate_sequence(self):

        # Generate sequence of length self.episode_length (with correct number of targets)
        while True:
            self.sequence = np.random.randint(0, 9, size=(self.episode_length))
            if not self.num_targets or sum(self._get_correct_actions()) == self.num_targets:
                break


    def _get_obs(self):

        if self.step_count < self.num_obs:
            window = self.sequence[:self.step_count + 1]
            observation = np.pad(window, (self.num_obs - self.step_count -1, 0), mode='constant', constant_values=(0))
        elif self.step_count == self.episode_length:
            window = self.sequence[self.step_count + 1 - self.num_obs : self.step_count + 1]
            observation = np.pad(window, (0,1), mode='constant', constant_values=(0))
        else:
            window = self.sequence[self.step_count + 1 - self.num_obs : self.step_count + 1]
            observation = window
        observation = np.eye(10,dtype=int)[observation]
        return observation

    def _get_correct_actions(self):
        self.correct_actions = np.array([int(self.sequence[i] == self.sequence[i + self.N]) for i in range(self.num_trials)])
        return self.correct_actions
    
    def _get_info(self):
        info = {
            'step_count': self.step_count,
            }
        return info

    
