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
        self.env = gym_super_mario_bros.make('SuperMarioBros-v0')
        #self.env = JoypadSpace(env, RIGHT_ONLY)
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
        reward += info['score'] + info['coins']
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
        size = t.Compose([t.ToPILImage(),t.Resize((40,90)),t.ToTensor()])

        return size(screen).unsqueeze(0).to(self.device)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Testing~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# em = MarioManager(device)
# em.reset()
#
# screen = em.state()
# print(screen)
# # screen = em.render('rgb_array')
# # screen = em.get_proccessed_screen()
#
# plt.figure()
# plt.imshow(screen.squeeze(0).permute(1,2,0), interpolation = 'none')
# plt.title('proccessed screen')
# plt.show()
