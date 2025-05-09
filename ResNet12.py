############################### The ResNet12 (with attention from Attention.py) architecture and its sub-module #####################################

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

        # Attention module 1.
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()


    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Attention module I.
        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

        self.sigmoid = nn.Sigmoid()

        # Attention module 2.
        self.ca = ChannelAttention(planes*4)
        self.sa = SpatialAttention()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # Attention module II.
        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):                         # 32, 64, 128, 256        # 512

    def __init__(self, block, layers, widths=[32, 64, 128, 256], feature_dim=512, num_classes=1000, projection=False, zero_init_residual=False, drop_rate=0, use_fc=False):
        super(ResNet, self).__init__()
        if drop_rate > 0:
            raise NotImplementedError()
        self.use_fc = use_fc
        self.projection = projection
        self.inplanes = 64       #64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, widths[0], layers[0])
        self.layer2 = self._make_layer(block, widths[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, widths[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, widths[3], layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if self.projection:
            self.proj1 = nn.Linear(widths[3]*block.expansion, feature_dim)

        if self.use_fc:
            self.fc = nn.Linear(widths[3] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, use_fc=False, cat=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        if cat and not self.training:
            # returns intermediate layers for multi-eval evaluation
            x2 = self.layer1(x)
            x2_avg = self.avgpool(x2).view(x.size(0), -1)
            x3 = self.layer2(x2)
            x3_avg = self.avgpool(x3).view(x.size(0), -1)
            x4 = self.layer3(x3)
            x4_avg = self.avgpool(x4).view(x.size(0), -1)
            x5 = self.layer4(x4)

            x5_avg = self.avgpool(x5).view(x.size(0), -1)

            x = torch.cat([x3_avg, x4_avg, x5_avg], axis=1)

            x1 = None
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)

            x = x.view(x.size(0), -1)

            x1 = self.fc(x) if use_fc else None

        if self.training and self.projection:
            x = self.proj1(x)

        return x

def resnet12(**kwargs):
    """Constructs a ResNet-12 model.
    """
   # print('\n>> Using custom ResNet12 architecture')  # 1, 1, 2, 1
    model = ResNet(BasicBlock, [1, 1, 2, 1], widths=[32, 64, 128, 256], **kwargs)
    return model
