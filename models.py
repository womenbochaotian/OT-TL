import torch.nn as nn

class MyNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        return self.fc2(nn.functional.relu(self.fc1(x)))
