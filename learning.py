import math
import time
import torch
import torch.nn as nn
import torchvision.transforms as t
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import count
from mario_q import MarioManager
from helpers import Transition
from helpers import ReplayMemory
from helpers import QValues
from helpers import DQN
import torchvision.models as models

is_ipython = 'inline' in plt.get_backend()
if is_ipython: from IPython import display


class Learning():
    def __init__(self, policy, target, em, memory, optimizer):
        self.policy = policy
        self.target = target
        self.em = em
        self.memory = memory
        self.optimizer = optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 35
        self.num_episodes = 350
        self.gamma = 0.9
        self.eps_start = 0.99
        self.eps_decay = 0.99
        self.current_rate = self.eps_start
        self.weights_update = 10 #how freequently we want to update network weights
        self.episodes = []
        self.losses = []
        self.total_rewards = []
        self.total_runtime = 0
        self.num_actions = em.num_actions()
        self.training = 1

    def play(self):
        self.em.reset()
        for i in range(1000):
            state = self.em.state()
            action = self.target(state).argmax(dim = 1).to(self.device)
            self.em.take_act(action)
            self.render()
        self.em.close()

    def select_action(self, state):
        rand = random.random()

        if self.current_rate > rand:
            action = random.randrange(self.num_actions)
            return torch.tensor([action]).to(self.device)
        else:
            with torch.no_grad():
                #print(self.policy(state).argmax(dim = 1).to(self.device))
                return self.policy(state).argmax(dim = 1).to(self.device)

    def extract_tensors(self, transitions):
        batch = Transition(*zip(*transitions))
        t1 = torch.cat(batch.state)
        t2 = torch.cat(batch.action)
        t3 = torch.cat(batch.reward)
        t4 = torch.cat(batch.next_state)
        return (t1,t2,t3,t4)

    def plot_on_figure(self):
        plt.figure(2)
        plt.clf()
        durations_t = torch.tensor(self.episodes, dtype=torch.float)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())

    def learn(self):
    #~~~~~~~~~~~ training loop ~~~~~~~~~~~~
        for episode in range(self.num_episodes):
            self.em.reset()
            game_state = self.em.state()
            total_reward = 0
            current = time.time()

            #decay rate
            if self.current_rate > 0.1:
              self.current_rate *= self.eps_decay
            #loop for each episode
            for duration in count():
                action = self.select_action(game_state)
                #add to total reward of this game
                reward = self.em.take_act(action)
                total_reward += reward.item()
                #get next game state aka screen diff and add tuple to experiences
                next_game_state = self.em.state()
                #train
                if self.training%4 == 0:
                    self.memory.push(Transition(game_state, action, reward, next_game_state))
                    if self.memory.enough_for_sample(self.batch_size):
                        experiences = self.memory.sample(self.batch_size)
                        states, actions, rewards, next_states = self.extract_tensors(experiences)

                        current_qvalues = QValues.get_current(self.policy, states, actions)
                        next_qvalues = QValues.get_next(self.target, next_states)
                        target_qvalues = (next_qvalues * self.gamma) + rewards

                        loss = F.mse_loss(current_qvalues, target_qvalues.unsqueeze(1))

                        self.losses.append(loss)
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                game_state = next_game_state
                self.training += 1
                self.em.render()
                #episode is done, report
                if self.em.done or self.em.return_count() > 700:
                    diff = time.time() - current
                    self.total_runtime += diff
                    self.episodes.append(diff)
                    self.total_rewards.append(total_reward)
                    print('Episode:', episode, "Reward:", total_reward, "Time in game", diff)
                    print('x posn', self.em.return_posn())
                    break
            #update weights after 10 episodes
            if episode % self.weights_update == 0:
                self.target.load_state_dict(self.policy.state_dict())
        em.close()
