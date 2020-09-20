# # Code for "AMC: AutoML for Model Compression and Acceleration on Mobile Devices"
# # Yihui He*, Ji Lin*, Zhijian Liu, Hanrui Wang, Li-Jia Li, Song Han
# # {jilin, songhan}@mit.edu

# import torch.nn as nn
# import math


# def conv_bn(inp, oup, stride):
#     return nn.Sequential(
#         nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
#         nn.BatchNorm2d(oup),
#         nn.ReLU(inplace=True)
#     )


# def conv_dw(inp, oup, stride):
#     return nn.Sequential(
#         nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
#         nn.BatchNorm2d(inp),
#         nn.ReLU(inplace=True),

#         nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
#         nn.BatchNorm2d(oup),
#         nn.ReLU(inplace=True),
#     )


# class MobileNet(nn.Module):
#     def __init__(self, n_class,  profile='normal'):
#         super(MobileNet, self).__init__()

#         # original
#         if profile == 'normal':
#             in_planes = 32
#             cfg = [64, (128, 2), 128, (256, 2), 256, (512, 2), 512, 512, 512, 512, 512, (1024, 2), 1024]
#         elif profile == '0.3flops':
#             #3, 16, 40, 72, 64, 136, 136, 288, 288, 288, 288, 296, 304, 480, 208
#             in_planes = 16
#             cfg = [40, (72, 2), 64, (136, 2), 136, (288, 2), 288, 288, 288, 296, 304, (480, 2), 208]
#         # 0.5 AMC
#         elif profile == '0.5flops':
#             in_planes = 24
#             # 3, 24, 48, 88, 88, 176, 184, 360, 360, 360, 360, 360, 352, 792, 536
#             # cfg = [48, (88, 2), 88, (176, 2), 184, (360, 2), 360, 360, 360, 360, 352, (792, 2), 536]
#             #原生代码的cfg
#             # 3,24,48,96,80,192,200,328,352,368,360,328,400,736,752
            
#             cfg = [48, (96, 2), 80, (192, 2), 200, (328, 2), 352, 368, 360, 328, 400, (736, 2), 752]
#         elif profile == '0.7flops':
#             #3, 32, 56, 104, 104, 208, 216, 424, 424, 424, 416, 408, 408, 872, 880
#             in_planes = 32
#             cfg = [56, (104, 2), 104, (208, 2), 216, (424, 2), 424, 424, 416, 408, 408, (872, 2), 880]
#         else:
#             raise NotImplementedError

#         self.conv1 = conv_bn(3, in_planes, stride=2)

#         self.features = self._make_layers(in_planes, cfg, conv_dw)

#         self.classifier = nn.Sequential(
#             nn.Linear(cfg[-1], n_class),
#         )

#         self._initialize_weights()

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.features(x)
#         x = x.mean(3).mean(2)  # global average pooling
#         x = self.classifier(x)
#         return x

#     def _make_layers(self, in_planes, cfg, layer):
#         layers = []
#         for x in cfg:
#             out_planes = x if isinstance(x, int) else x[0]
#             stride = 1 if isinstance(x, int) else x[1]
#             layers.append(layer(in_planes, out_planes, stride))
#             in_planes = out_planes
#         return nn.Sequential(*layers)

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 n = m.weight.size(1)
#                 m.weight.data.normal_(0, 0.01)
#                 m.bias.data.zero_()


# Code for "AMC: AutoML for Model Compression and Acceleration on Mobile Devices"
# Yihui He*, Ji Lin*, Zhijian Liu, Hanrui Wang, Li-Jia Li, Song Han
# {jilin, songhan}@mit.edu

import torch.nn as nn
import math


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_dw(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )


class MobileNet(nn.Module):
    def __init__(self, n_class,  profile='normal'):
        super(MobileNet, self).__init__()

        # original
        if profile == 'normal':
            in_planes = 32
            cfg = [64, (128, 2), 128, (256, 2), 256, (512, 2), 512, 512, 512, 512, 512, (1024, 2), 1024]
        # 0.5 AMC
        elif profile == '0.5flops':
            in_planes = 24
            cfg = [48, (96, 2), 80, (192, 2), 200, (328, 2), 352, 368, 360, 328, 400, (736, 2), 752]
        else:
            raise NotImplementedError

        self.conv1 = conv_bn(3, in_planes, stride=2)

        self.features = self._make_layers(in_planes, cfg, conv_dw)

        self.classifier = nn.Sequential(
            nn.Linear(cfg[-1], n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.features(x)
        x = x.mean(3).mean(2)  # global average pooling

        x = self.classifier(x)
        return x

    def _make_layers(self, in_planes, cfg, layer):
        layers = []
        for x in cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(layer(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()



                