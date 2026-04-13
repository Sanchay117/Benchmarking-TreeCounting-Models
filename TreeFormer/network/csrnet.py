import torch
import torch.nn as nn
from torchvision.models import VGG16_Weights, vgg16


def _make_layers(cfg, in_channels=3, dilation=False):
    layers = []
    d_rate = 2 if dilation else 1
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class CSRNet(nn.Module):
    def __init__(self, load_frontend_pretrained=True):
        super().__init__()
        self.frontend_feat = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]

        self.frontend = _make_layers(self.frontend_feat)
        self.backend = _make_layers(self.backend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        self._init_weights()
        if load_frontend_pretrained:
            self._load_frontend_vgg16()

    def _load_frontend_vgg16(self):
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        frontend_state = self.frontend.state_dict()
        vgg_items = list(vgg.features.state_dict().items())

        # CSRNet frontend has the same first 13 conv blocks as VGG16 before pool4.
        for idx, (k, _) in enumerate(frontend_state.items()):
            frontend_state[k] = vgg_items[idx][1]
        self.frontend.load_state_dict(frontend_state)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x
