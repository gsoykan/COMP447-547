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


# https://github.com/Rayhane-mamah/Tacotron-2/issues/155
# https://arxiv.org/pdf/1701.05517.pdf

class MixtureOfLogistics(nn.Module):
    def __init__(self, d, n_mix=4):
        super().__init__()
        self.d = d
        self.n_mix = 4
        # pi, m, s
        self.W = torch.zeros(n_mix, 3)
        self.W[:, 2] = torch.tensor(1 / n_mix)
        self.W[:, 0] = torch.tensor(1 / n_mix)
        self.W = nn.Parameter(self.W)
        self.W.requires_grad = True

    def forward(self, x):
        value = 0
        # pis = self.W[:, 1]
        # softmaxed_pis = nn.Softmax(pis)
        for i in range(self.n_mix):
            s_pi = self.W[i, 0]  # softmaxed_pis[i]
            m = self.W[i, 1]
            s = self.W[i, 2]
            active_x = torch.div(torch.subtract(x, m), s)
            sigm = nn.Sigmoid()
            sigmoid_result = sigm(active_x)
            value += s_pi * sigmoid_result
        # TODO: add clamping here
        return value

    # For the edge case of when $x = 0$,
    # we replace $x-0.5$ by $-\infty$,
    # and for $x = d-1$,
    # we replace $x+0.5$ by $\infty$.

    # https://github.com/Rayhane-mamah/Tacotron-2/blob/d13dbba16f0a434843916b5a8647a42fe34544f5/wavenet_vocoder/models/mixture.py#L44-L49

    def loss(self, x):
        #torch.autograd.set_detect_anomaly(True)
        batch_size = x.shape[0]
        bins = torch.tensor(list(range(0, self.d)))
        minus_inf = float('-inf')
        positive_inf = float('inf')
        term_a = bins - 0.5
        term_a[term_a == -0.5] = torch.tensor(minus_inf)
        term_b = bins + 0.5
        term_b[term_b >= (self.d - 0.5)] = torch.tensor(positive_inf)
        values = []
        # pis = self.W[:, 1]
        # softmaxed_pis = nn.Softmax(pis)
        for i in range(self.n_mix):
            s_pi = self.W[i, 0]  # softmaxed_pis[i]
            m = self.W[i, 1]
            s = self.W[i, 2]
            term_a_pre_res = torch.div(term_a - m, s)
            sigm_a = nn.Sigmoid()
            term_a_res = sigm_a(term_a_pre_res)
            term_b_pre_res = torch.div(term_b - m, s)
            sigm_b = nn.Sigmoid()
            term_b_res = sigm_b(term_b_pre_res)
            internal_res = term_a_res - term_b_res
            value = s_pi * internal_res
            values.append(value)
        stacked_values = torch.stack(values)
        summed_values = stacked_values.sum(dim=0)
        neg_probs = F.log_softmax(summed_values)
        neg_probs = torch.reshape(neg_probs, (1, self.d))
        neg_probs_for_batch = torch.repeat_interleave(neg_probs, repeats=batch_size, dim=0)
        nll_loss = nn.NLLLoss()
        total_loss = nll_loss(neg_probs_for_batch, x)
        return total_loss

    def get_distribution(self):
        """ YOUR CODE HERE """
