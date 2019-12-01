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

batch_size = 35
num_episodes = 10
gamma = 0.99
#~~~~~~~~~
eps_start = 0.99
eps_decay = 0.9 #0.989
current_rate = eps_start
#~~~~~~~~~~
weights_update = 10 #how freequently we want to update network weights
capacity = 1000000
learning_rate = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
em = MarioManager(device)
memory = ReplayMemory(capacity)

# policy = models.resnet50(pretrained=False)
# policy.fc = nn.Sequential(nn.Linear(2048, 512),
#                                  nn.ReLU(),
#                                  nn.Dropout(0.2),
#                                  nn.Linear(512, 5),
#                                  nn.LogSoftmax(dim=1))
# policy.to(device)
# target = models.resnet50(pretrained=False)
# target.fc = nn.Sequential(nn.Linear(2048, 512),
#                                  nn.ReLU(),
#                                  nn.Dropout(0.2),
#                                  nn.Linear(512, 5),
#                                  nn.LogSoftmax(dim=1))
# target.to(device)
#
# criterion = nn.NLLLoss()
policy = DQN(em.screen_height(), em.screen_width()).to(device)
target = DQN(em.screen_height(), em.screen_width()).to(device)
optimizer = optim.Adam(params = policy.parameters(), lr = learning_rate)
target.load_state_dict(policy.state_dict())
target.eval()

episodes = []
times_in_rand = 0
times_in_exploit = 0
#~~~~~~~helper functions to show on a plot
# def get_exploration_rate(state, current_step):
#     rate = self.end + (self.start - self.end) * \
#             math.exp(-1. * current_step * self.decay)

def select_action(state):
    global times_in_rand
    global times_in_exploit
    global current_rate

    rand = random.random()

    if current_rate > rand:
        times_in_rand +=1
        action = random.randrange(em.num_actions())
        return torch.tensor([action]).to(device)
    else:
        times_in_exploit +=1
        with torch.no_grad():
            return policy(state).argmax(dim = 1).to(device)

def extract_tensors(transitions):
    batch = Transition(*zip(*transitions))
    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)
    return (t1,t2,t3,t4)

def plot_on_figure(epis):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(epis, dtype=torch.float)
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

total_rewards = []
losses = []
total_runtime = 0
#~~~~~~~~~~~ training loop ~~~~~~~~~~~~
for episode in range(num_episodes):
    em.reset()
    game_state = em.state()
    times_in_exploit = 0
    times_in_rand = 0
    total_reward =0
    current = time.time()

    #decay rate

    if current_rate > 0.01:
      current_rate *= eps_decay
    #loop for each episode
    for duration in count():

        action = select_action(game_state)

        #add to total reward of this game
        reward = em.take_act(action)

        total_reward += reward.item()
        #get next game state aka screen diff and add tuple to experiences
        next_game_state = em.state()
        memory.push(Transition(game_state, action, reward, next_game_state))
        game_state = next_game_state
        #train

        if memory.enough_for_sample(batch_size):

            experiences = memory.sample(batch_size)
            states, actions, rewards, next_states = extract_tensors(experiences)

            current_qvalues = QValues.get_current(policy, states, actions)
            next_qvalues = QValues.get_next(target, next_states)
            target_qvalues = (next_qvalues * gamma) + rewards

            #loss = criterion(current_qvalues.long(), target_qvalues.unsqueeze(1).long())
            loss = F.mse_loss(current_qvalues, target_qvalues.unsqueeze(1))
            losses.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        em.render()
        #episode is done, report
        if em.done or em.return_count() > 700:
            print("In random times", times_in_rand, "Times in exploit", times_in_exploit)
            diff = time.time() - current
            total_runtime += diff
            episodes.append(diff)

            print('Episode:', episode, "Reward:", total_reward, "Time in game", diff)

            total_rewards.append(total_reward)
            break
    #update weights after 10 episodes
    if episode % weights_update == 0:
        target.load_state_dict(policy.state_dict())
# em.close()
plot_on_figure(total_rewards)
plot_on_figure(losses)
