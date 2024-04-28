import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000) :
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) :
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x



model = AlexNet()

conv_params = sum(p.numel() for p in model.features.parameters() if p.requires_grad)
print('conv_params: ', conv_params)
linear_params = sum(p.numel() for p in model.classifier.parameters() if p.requires_grad)
print('linear_params: ', linear_params)
print('conv_params+linear_params: ', conv_params + linear_params)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('total_params: ', total_params)
print('conv_params: {}%'.format((conv_params/total_params)*100))
print('linear_params: {}%: '.format((linear_params/total_params)*100))