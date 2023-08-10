import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, input_channel=3, num_channel=16):
        super(SimpleNet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(input_channel, num_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(num_channel, num_channel*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channel*2),
            nn.ReLU(),
        )
       
        self.block3 = nn.Sequential(
            nn.Conv2d(num_channel*2, num_channel*4, kernel_size=3),
            nn.BatchNorm2d(num_channel*4),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(num_channel*4, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

        self.pool = nn.MaxPool2d(kernel_size=2)
        self.avgPool = nn.AvgPool2d(kernel_size=4)

    def forward(self, x):
        x = self.block1(x)
        x = self.pool(x)

        x = self.block2(x)
        x = self.pool(x)

        x = self.block3(x)
        x = self.avgPool(x)
        
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class SimpleMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=16):
        super(SimpleMLP, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
