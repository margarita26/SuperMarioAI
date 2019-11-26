import torch
import torchvision.transforms as t
import torch.optim as optim
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mario_q import MarioManager
from helpers import ReplayMemory
from helpers import DQN
from helpers import Transition

is_ipython = 'inline' in plt.get_backend()
if is_ipython: from IPython import display

batch_size = 200
gamma = 0.9
eps_start = 1
eps_end = 0.01
eps_decay = 0.001
t_update = 10 #how freequently we want to update network weights
capacity = 100000
learning_rate = 0.001
num_episodes = 500
current_step = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
em = MarioManager(device)
memory = ReplayMemory(capacity)

policy = DQN(em.screen_height(), em.screen_width()).to(device)
target = DQN(em.screen_height(), em.screen_width()).to(device)
target.load_state_dict(policy.state_dict())
target.eval()
optimzier = optim.Adam(params = policy.parameters(), lr = learning_rate)

#~~~~~~~~~~~ training loop ~~~~~~~~~~~~
episodes = []
for episode in range(num_episodes):
    em.reset()
    game_state = em.get_state()
    for duration in count():
        action = select_action(game_state, policy,device, em.num_actions())
        reward = em.take_act(action)
        next_game_state = em.get_state()
        memory.push(Transition(state, action, reward, next_game_state))
        game_state = next_game_state



#~~~~~~~helper functions to show on a plot
def get_exploration_rate(state, current_step):
    rate = self.end + (self.start - self.end) * \
            math.exp(-1. * current_step * self.decay)

def select_action(state,policy,device,num_actions):
    rate = get_exploration_rate(current_step)
    current_step +=1

    if rate > random.random():
        action = random.randrange(num_actions)
        return torch.tensor([action]).to(device)
    else:
        policy(state).argmax(dim = 1).to(device)

def extract_tensors(transitions):
    batch = Transition(*zip(*transitions))
    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batchn.next_state)
    return (t1,t2,t3,t4)

def plot_on_figure(values, average):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(values) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    plt.pause(0.001)  # pause a bit so that plots are updated
    print('Episode', len(values), 'Duration')
    if is_ipython:
        display.clear_output(wait=True)
