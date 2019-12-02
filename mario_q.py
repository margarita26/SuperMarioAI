import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

import torch
import torchvision.transforms as t
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


'''
This class manages open gym ai environment and preproccess the image
'''
class MarioManager():
    '''
        Initialize the environment, class contains basic Open Gym AI operations
        along with screen proccessing operations done by pytorch
    '''
    def __init__(self, device):
        self.device = device
        self.env = JoypadSpace(gym_super_mario_bros.make('SuperMarioBros-v0'), RIGHT_ONLY)
        self.env.reset()
        self.current_screen = None
        self.done = False
        self.current_score = 0
        self.current_coins = 0
        self.x = -9999999
        self.coins = 0
        self.score = 0
        self.count_same_posn = 0

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
        #if new coins
        if self.coins != info['coins']:
            reward += int(info['coins']) - self.coins
            self.coins = int(info['coins'])
        #if ghost is killed
        if self.score != info['score']:
            reward += int(info['score']) - self.score
            self.score = int(info['score'])
        #checking for same position in the game, means he is stuck, kill him
        if self.x == info['x_pos']:
            self.count_same_posn += 1
        #if he moved after being stuck, give a reward
        elif self.count_same_posn > 0 and self.x != info['x_pos']:
            self.count_same_posn = 0
            reward += 15
        #else reset count to 0
        else:
            self.count_same_posn = 0
        # if reward == 0:
        #     reward -= 1
        #make negative reward even more negative

        #kill him after the first life to speed up training
        # if info['life'] < 2:
        #     self.done = True
        #check that he actually moved to the right
        if self.x < info['x_pos']:
            reward += 0
        #he didn't more right byt taking the action, pinalize
        else:
            reward -= 1
            
        if info['x_pos'] != 40:
            self.x = info['x_pos']
        return torch.tensor([reward], device = self.device)

    def return_count(self):
        return self.count_same_posn

    def return_posn(self):
        return self.x

    def is_starting(self):
        return self.current_screen is None

    def state(self):
        if self.is_starting() or self.done:
            self.current_screen = self.get_proccessed_screen()
            black_screen = torch.zeros_like(self.current_screen)
            return black_screen
        else:
            # screen = self.current_screen
            # next_screen = self.get_proccessed_screen()
            # self.current_screen = next_screen
            self.current_screen =  self.get_proccessed_screen()
            return self.current_screen

    def screen_height(self):
        return self.get_proccessed_screen().shape[2]

    def screen_width(self):
        return self.get_proccessed_screen().shape[3]

    def get_proccessed_screen(self):
        screen = self.render('rgb_array').transpose((2,0,1))
        screen = self.crop_screen(screen)
        return self.transform_screen_data(screen)

    def crop_screen(self, screen):
        screen_height = screen.shape[1]
        top = int(screen_height * 0.5)
        bottom = int(screen_height * 0.9)
        screen = screen[:,top:bottom, :]
        return screen

    def transform_screen_data(self, screen):
        screen = np.ascontiguousarray(screen, dtype = np.float32)/255
        screen = torch.from_numpy(screen)
        size = t.Compose(
        [t.ToPILImage(),
        t.Resize((15,40)),
        #t.Grayscale(num_output_channels=1),
        t.ToTensor()])

        return size(screen).unsqueeze(0).to(self.device)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Testing~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# em = MarioManager(device)
# em.reset()
#
# screen = em.state()
#
# screen = em.render('rgb_array')
# for i in range(155):
#     em.take_act(1)
# for i in range(5):
#     em.take_act(4)
# screen = em.get_proccessed_screen()
#
# #
# plt.figure()
# plt.imshow(screen.squeeze(0).permute(1, 2, 0).cpu(), interpolation='none')
# plt.title('proccessed screen')
# plt.show()
