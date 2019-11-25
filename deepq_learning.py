import torch
import torchvision.transforms as t
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#our mario manager
import mario_q

batch_size = 10
gamma = 0.9
eps_start = 1
eps_end = 0.01
eps_decay = 0.001
t_update = 10
memory = 100000











#~~~~~~~helper functions to show on a plot
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
    if is_ipython:
        display.clear_output(wait=True)
