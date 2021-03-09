import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# sources
# https://towardsdatascience.com/plotting-probabilities-for-discrete-and-continuous-random-variables-353c5bb62336
def frequencies(values, d):
    frequencies = {}
    for d_i in range(d):
        frequencies[d_i] = 0

    for v in values:
        if v in frequencies:
            frequencies[v] += 1
        else:
            frequencies[v] = 1
    return frequencies


def probabilities(sample, freqs):
    probs = []
    for k, v in freqs.items():
        probs.append(round(v / len(sample), 1))
    return probs


def compute_crossentropyloss_manual(x, y0):
    torch.exp()
    """
    x is the vector of probabilities with shape (batch_size,C)
    y0 shape is the same (batch_size), whose entries are integers from 0 to C-1
    """
    loss = 0.
    n_batch, n_class = x.shape
    # print(n_class)
    for x1, y1 in zip(x, y0):
        class_index = int(y1.item())
        loss = loss + torch.log(torch.exp(x1[class_index]) / (torch.exp(x1).sum()))
    loss = - loss / n_batch
    return loss


# usage
# sample = [0,1,1,1,1,1,2,2,2,2]
# freqs = frequencies(sample)
# probs = probabilities(sample, freqs)
# x_axis = list(set(sample))plt.bar(x_axis, probs)

# sources:
# https://discuss.pytorch.org/t/how-could-i-create-a-module-with-learnable-parameters/28115
# https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
# https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html

class Histogram(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.W = nn.Parameter(torch.zeros(1, d))
        self.W.requires_grad = True

    def loss(self, x):
        batch_size = x.shape[0]
        mock_x = torch.repeat_interleave(self.W, repeats=batch_size, dim=0)
        loss = nn.CrossEntropyLoss()
        input = mock_x
        target = x
        output = loss(input, target)
        return output

    def get_distribution(self):
        softmax = nn.Softmax()
        distribution = softmax(self.W)
        distribution = distribution.cpu().detach().numpy()
        distribution = np.squeeze(distribution)
        return distribution
