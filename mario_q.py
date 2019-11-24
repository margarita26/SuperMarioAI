import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

import torch
import random
import numpy as np
import pandas as pd

class MarioManager():
    '''
        Initialize the environment, class contains basic Open Gym AI operations
        along with deep_q operations done by pytorch
    '''
    def __init(self, device):
        self.device = device
        self.env = gym_super_mario_bros.make('SuperMarioBros-v0')
        self.env = JoypadSpace(env, RIGHT_ONLY)
        self.env.reset()
        self.current_screen = None
        self.done = False

    def reset(self):
        self.env.reset()
        self.current_screen = None

    def close(self):
        self.env.close()

    def render(self, mode = 'human'):
        return self.env.render(mode)

    def num_actions(self):
        return self.env.action_space.n

    def take_act(self, action):
        observation, reward, self.done, info = self.env.step(action.item()) #uses action.item
        return torch.tensor([reward], device = self.device)

    def is_starting(self):
        return self.current_screen is None

    def state(self):
        if self.is_starting() or self.done:
            self.current_screen = self.get_proccessed_screen()
            black_screen = torch.zeros_like(self.current_screen)
            return black_screen
        else:
            screen = self.current_screen
            next_screen = self.get_proccessed_screen()
            self.current_screen = next_screen
            return next_screen - screen
