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
from mario_q import MarioManager
from helpers import Transition
from helpers import ReplayMemory
from helpers import DQN
from learning import Learning

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
em = MarioManager(device)
memory = ReplayMemory(1000000)
policy = DQN(em.screen_height(), em.screen_width()).to(device)
target = DQN(em.screen_height(), em.screen_width()).to(device)
optimizer = optim.Adam(params = policy.parameters(), lr = 0.001)
target.load_state_dict(policy.state_dict())
target.eval()

learning_agent = Learning(policy, target, em, memory, optimizer)
learning_agent.learn()
learning_agent.plot_on_figure()
