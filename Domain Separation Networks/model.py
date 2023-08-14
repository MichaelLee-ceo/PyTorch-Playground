import torch.nn as nn
from pytorch_revgrad import RevGrad

''' CNN Block '''
class cnn_block(nn.Module):
    def __init__(self, dim=32, out_channel=64):
        super(cnn_block, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(dim, dim*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim*2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(dim*2, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
    def forward(self, x):
        return self.main(x)


class Shared_Encoder(nn.Module):
    def __init__(self, out_dim=64):
        super(Shared_Encoder, self).__init__()
        self.feature_extractor = cnn_block(out_channel=out_dim)
        self.fc = nn.Sequential(
            nn.Linear(out_dim*3*3, 100),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

class Private_Encoder(nn.Module):
    def __init__(self, out_dim=32):
        super(Private_Encoder, self).__init__()
        self.feature_extractor = cnn_block(out_channel=out_dim)
        self.fc = nn.Sequential(
            nn.Linear(out_dim*3*3, 100),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x =x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


class Shared_Decoder(nn.Module):
    def __init__(self,):
        super(Shared_Decoder, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(100, 128),
            nn.ReLU(inplace=True),
        )
        self.up_block1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.up_block2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.up_block3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.up_block4 = nn.Sequential(
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

        self.up = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(x.shape[0], x.shape[1], 1, 1)

        x = self.up_block1(x)
        # print("Decoder up_block1:", x.shape)
        x = self.up_block2(self.up(x))
        x = self.up_block3(self.up(x))
        x = self.up_block4(self.up(x))
        # print("Decoder up_block4:", x.shape)
        return x

''' Label Classifier '''
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(100, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 10),
        )

    def forward(self, x):
        x = self.fc(x)
        return x

''' Domain Classifier '''
class Domain_Classifier(nn.Module):
    def __init__(self):
        super(Domain_Classifier, self).__init__()
        self.fc = nn.Sequential(
            RevGrad(),
            nn.Linear(100, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 1),
        )

    def forward(self, x):
        x = self.fc(x)
        return x