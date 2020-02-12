import torch
import torch.nn as nn

cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name, layer_cfg=None, num_classes=10):
        super(VGG, self).__init__()
        self.layer_cfg = layer_cfg
        self.cfg_index = 0
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512 if self.layer_cfg is None else self.layer_cfg[-1], num_classes)
        self._initialize_weights()

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                x = x if self.layer_cfg is None else self.layer_cfg[self.cfg_index]
                layers += [nn.Conv2d(in_channels,
                                     x,
                                     kernel_size=3,
                                     padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
                self.cfg_index += 1
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def test():

    model = VGG('vgg16')
    ckpt = torch.load('../pretrain/vgg16_cifar10.pt', map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])

    # print(model)
    for name, module in model.named_modules():

        if isinstance(module, nn.Conv2d):

            print(module.weight.size(1)*module.weight.size(2)*module.weight.size(3))



# test()