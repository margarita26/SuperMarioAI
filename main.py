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
num_episodes = 500
gamma = 0.9
#~~~~~~~~~
eps_start = 1
eps_end = 0.01
eps_decay = 0.001
current_rate = eps_start
#~~~~~~~~~~
weights_update = 10 #how freequently we want to update network weights
capacity = 100000
learning_rate = 0.001
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
        action = select_action(game_state)
        reward = em.take_act(action)
        next_game_state = em.state()
        memory.push(Transition(state, action, reward, next_game_state))
        game_state = next_game_state

        if memory.enough_for_sample(batch_size):
            experiences = memory.sample(batch_size)
            states, actions, rewards, next_states = extract_tensors(experiences)

            current_qvalues = get_current_qvalues(policy,states, actions)
            next_qvalues = get_next_qvalues(target,next_states)
            target_qvalues = (next_qvalues * gamma) + rewards
            loss = F.mse_loss(current_qvalues, target_qvalues.unsqueeze(1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if em.done:
            episodes.append(duration)
            plot_on_figure()
            print('Episode', episode, "Reward", reward, "Duration", duration)
            break
    if episode % weights_update == 0:
        target.load_state_dict(policy.state_dict())
em.close()

#~~~~~~~helper functions to show on a plot
# def get_exploration_rate(state, current_step):
#     rate = self.end + (self.start - self.end) * \
#             math.exp(-1. * current_step * self.decay)

def get_current_qvalues(policy_net, states, actions):
    return policy_net(states).gather(dim = 1, index = actions.unsqueeze(-1))

def get_next_qvalues(target_net, next_states):
    final_state_locations = next_states.flatten(start_dim=1) \
            .max(dim=1)[0].eq(0).type(torch.bool)
    non_final_state_locations = (final_state_locations == False)
    non_final_states = next_states[non_final_state_locations]
    batch_size = next_states.shape[0]
    values = torch.zeros(batch_size).to(device)
    values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0].detach()
    return values

def select_action(state):
    if current_rate > eps_end:
        current_rate += eps_decay

    if current_rate > random.random():
        action = random.randrange(em.num_actions())
        return torch.tensor([action]).to(device)
    else:
        with torch.no_grad():
            return policy(state).argmax(dim = 1).to(device)

def extract_tensors(transitions):
    batch = Transition(*zip(*transitions))
    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batchn.next_state)
    return (t1,t2,t3,t4)

def plot_on_figure():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episodes, dtype=torch.float)
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
