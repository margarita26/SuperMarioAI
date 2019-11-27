import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple

Transition = namedtuple('Transition',
                    ('state', 'action','reward','next_state'))

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

    def enough_for_sample(batch_size):
        return len(memory) >= batch_size

class DQN(nn.Module):
    def __init__(self, height, width):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(in_features = height*width*3, out_features=24)
        self.fc2 = nn.Linear(in_features = 24, out_features=32)
        self.out = nn.Linear(in_features=32, out_features=6)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, t):
        t = t.flatten(start_dim = 1)
        #activation functions
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        return t
