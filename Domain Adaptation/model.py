import torch.nn as nn
from pytorch_revgrad import RevGrad

class MyNet(nn.Module):
    def __init__(self, in_channel=3, dim=16, num_classes=10):
        super(MyNet, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channel, dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(dim, dim*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim*2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(dim*2, dim*4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim*4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # class classifier
        self.classifier = nn.Sequential(
            nn.Linear(dim*4*3*3, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, num_classes)
        )

        # domain classifier
        self.domain_classifier = nn.Sequential(
            RevGrad(),
            nn.Linear(dim*4*3*3, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(3, 3))
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        output = self.classifier(x)
        domain = self.domain_classifier(x)
        return output, domain