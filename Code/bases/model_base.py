import torch
from torch import nn

"""
Inherit our class from the PyTorch core class `nn.Module`, which is used to create neural networks.
"""

# https://www.codenong.com/cs106322023/
class Network(nn.Module):
    def __init__(self, device=None):
        """
        Call the constructor of the parent class.
        """
        super().__init__()
        """
        If we use a GPU, we set the device attribute to 'cuda'; otherwise, we set it to 'cpu'.
        This will help us avoid checking for the availability of CUDA throughout the code.
        """
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        pass
