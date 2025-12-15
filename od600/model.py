import torch.nn as nn
import torchvision.models as models

class RegressionModel(nn.Module):
    def __init__(self, pretrained=False, dropout_p=0.3):
        super().__init__()

        self.net = models.resnet18(weights=None)
        nfeat = self.net.fc.in_features
        self.net.fc = nn.Identity()

        self.reg = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(nfeat, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.net(x)
        return self.reg(x)