import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple

Transition = namedtuple('Transition',
                    ('state', 'action','reward','next_state'))

class QValues():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def get_current(policy_net, states, actions):
        return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))

    @staticmethod
    def get_next(target_net, next_states):
        final_state_locations = next_states.flatten(start_dim=1) \
            .max(dim=1)[0].eq(0).type(torch.bool)
        non_final_state_locations = (final_state_locations == False)
        non_final_states = next_states[non_final_state_locations]
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(QValues.device)
        values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0].detach()
        return values

#buffer storage for action states
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.posn = 0

    def push(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[posn] = transition
        self.posn = (self.posn+1)%self.capacity

    def sample(self, batch):
        return random.sample(self.memory, batch)

    def enough_for_sample(self, batch_size):
        return len(self.memory) >= batch_size

class DQN(nn.Module):
    def __init__(self, height, width):
        super().__init__()
        self.fc1 = nn.Linear(in_features = height*width*3, out_features=24)
        self.fc2 = nn.Linear(in_features = 24, out_features=32)
        self.out = nn.Linear(in_features=32, out_features=7)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, t):
        t = t.flatten(start_dim = 1)
        #activation functions
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        return t
