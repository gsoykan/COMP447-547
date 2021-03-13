import torch

if torch.cuda.is_available():
    use_cuda = True
else:
    use_cuda = False

device = torch.device("cuda:0" if use_cuda else "cpu")