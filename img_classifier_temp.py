import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3) # specify the size (outer numbers) and amount (middle number) of filters
        self.pool = nn.MaxPool2d(2, 2) # specify pool size first number is size of pool, second is step size
        self.conv2 = nn.Conv2d(16, 8, 3) # new depth is amount of filters in previous conv layer
        self.fc1 = nn.Linear(54*54*8, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, 2) # final fc layer needs 19 outputs because we have 19 layers # ???

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 54*54*8) # flatten
        x = F.relu(self.fc1(x))    # fully connected, relu        
        x = F.relu(self.fc2(x))    
        x = self.fc3(x)     # output    
        return x