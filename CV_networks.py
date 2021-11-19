import torch
import torch.nn as nn
from torch.nn.modules import padding
from torch.nn.modules.batchnorm import _BatchNorm # All neural network modules, nn.Linear, nn.Con2d, BatchNorm, Loss functions
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader # Give easier dataset management and creates mini batches
import torchvision.datasets as datasets # Has standard datasets we can import in a nice way
import torchvision.transforms as transforms 
########################################################LeNet architecture##################################################
# 1*32*32 Input -> (5*5), s=1,p=1 -> avg pool s=2,p=0 -> (5*5),s=1,p=1 -> avg pool s=2,p=0
# -> Conv 5*5 to 120 channels -> Linear 120 -> Linear 84 -> Linear 10

# class LeNet(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.relu = nn.ReLU()
#         self.pool = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2))
#         self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size=(5,5), stride = (1,1), padding=(0,0))
#         self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size=(5,5), stride = (1,1), padding=(0,0))
#         self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 120, kernel_size=(5,5), stride = (1,1), padding=(0,0))
#         self.linear1 = nn.Linear(120,84)
#         self.linear2 = nn.Linear(84,10)
    
#     def forward(self,x):
#         x = self.relu(self.conv1(x))
#         x = self.pool(x)
#         x = self.relu(self.conv2(x))
#         x = self.pool(x)
#         x = self.relu(self.conv3(x))  # num_examples * 120 * 1 * 1 -> num_examples * 120
#         x = x.reshape(x.shape[0],-1)
#         x = self.relu(self.linear1(x))
#         x = self.linear2(x)
#         return x

## Check model 
# x = torch.rand(64,1,32,32)
# model = LeNet()
# print(model(x).shape)
#############################################################################################################################

########################################################VGG architecture#####################################################
# VGG16 = [64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M']
# # Then flatten and 4096*4096*1000 Linear Layers
# class VGG_net(nn.Module):
#     def __init__(self, in_channels = 3, num_classes = 1000) -> None:
#         super().__init__()
#         self.in_channels = in_channels
#         self.con_layers = self.create_conv_layers(VGG16)
#         self.fcs = nn.Sequential(nn.Linear(512*7*7,4096), nn.ReLU(), nn.Dropout(p=0.5),nn.Linear(4096,4096),nn.ReLU(),nn.Dropout(p=0.5),nn.Linear(4096,num_classes))

#     def forward(self,x):
#         x = self.con_layers(x)
#         x = x.reshape(x.shape[0],-1)
#         x = self.fcs(x)
#         return x

#     def create_conv_layers(self,architecture):
#         layers = []
#         in_channels = self.in_channels

#         for x in architecture:
#             if type(x) == int:
#                 out_channels = x

#                 layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size = (3,3), stride = (1,1), padding = (1,1)), nn.BatchNorm2d(x), nn.ReLU()]
#                 in_channels = x
#             elif x == 'M':
#                 layers += [nn.MaxPool2d(kernel_size = (2,2), stride = (2,2))]
#         return nn.Sequential(*layers)

# ## check model
# model = VGG_net(in_channels = 3, num_classes = 1000)
# x = torch.randn(1,3,224,224)
# print(model(x).shape)
#################################################################################################################################

######################################################GoogLeNet##################################################################
# class GoogLeNet(nn.Module):
#     def __init__(self, aux_logits=True, num_classes=1000):
#         super(GoogLeNet, self).__init__()
#         assert aux_logits == True or aux_logits == False
#         self.aux_logits = aux_logits

#         # Write in_channels, etc, all explicit in self.conv1, rest will write to
#         # make everything as compact as possible, kernel_size=3 instead of (3,3)
#         self.conv1 = conv_block(
#             in_channels=3,
#             out_channels=64,
#             kernel_size=(7, 7),
#             stride=(2, 2),
#             padding=(3, 3),
#         )

#         self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.conv2 = conv_block(64, 192, kernel_size=3, stride=1, padding=1)
#         self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

#         # In this order: in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool
#         self.inception3a = Inception_block(192, 64, 96, 128, 16, 32, 32)
#         self.inception3b = Inception_block(256, 128, 128, 192, 32, 96, 64)
#         self.maxpool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)

#         self.inception4a = Inception_block(480, 192, 96, 208, 16, 48, 64)
#         self.inception4b = Inception_block(512, 160, 112, 224, 24, 64, 64)
#         self.inception4c = Inception_block(512, 128, 128, 256, 24, 64, 64)
#         self.inception4d = Inception_block(512, 112, 144, 288, 32, 64, 64)
#         self.inception4e = Inception_block(528, 256, 160, 320, 32, 128, 128)
#         self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

#         self.inception5a = Inception_block(832, 256, 160, 320, 32, 128, 128)
#         self.inception5b = Inception_block(832, 384, 192, 384, 48, 128, 128)

#         self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
#         self.dropout = nn.Dropout(p=0.4)
#         self.fc1 = nn.Linear(1024, num_classes)

#         if self.aux_logits:
#             self.aux1 = InceptionAux(512, num_classes)
#             self.aux2 = InceptionAux(528, num_classes)
#         else:
#             self.aux1 = self.aux2 = None

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.maxpool1(x)
#         x = self.conv2(x)
#         x = self.maxpool2(x)

#         x = self.inception3a(x)
#         x = self.inception3b(x)
#         x = self.maxpool3(x)

#         x = self.inception4a(x)

#         # Auxiliary Softmax classifier 1
#         if self.aux_logits and self.training:
#             aux1 = self.aux1(x)

#         x = self.inception4b(x)
#         x = self.inception4c(x)
#         x = self.inception4d(x)

#         # Auxiliary Softmax classifier 2
#         if self.aux_logits and self.training:
#             aux2 = self.aux2(x)

#         x = self.inception4e(x)
#         x = self.maxpool4(x)
#         x = self.inception5a(x)
#         x = self.inception5b(x)
#         x = self.avgpool(x)
#         x = x.reshape(x.shape[0], -1)
#         x = self.dropout(x)
#         x = self.fc1(x)

#         if self.aux_logits and self.training:
#             return aux1, aux2, x
#         else:
#             return x


# class Inception_block(nn.Module):
#     def __init__(
#         self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool
#     ):
#         super(Inception_block, self).__init__()
#         self.branch1 = conv_block(in_channels, out_1x1, kernel_size=(1, 1))

#         self.branch2 = nn.Sequential(
#             conv_block(in_channels, red_3x3, kernel_size=(1, 1)),
#             conv_block(red_3x3, out_3x3, kernel_size=(3, 3), padding=(1, 1)),
#         )

#         self.branch3 = nn.Sequential(
#             conv_block(in_channels, red_5x5, kernel_size=(1, 1)),
#             conv_block(red_5x5, out_5x5, kernel_size=(5, 5), padding=(2, 2)),
#         )

#         self.branch4 = nn.Sequential(
#             nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#             conv_block(in_channels, out_1x1pool, kernel_size=(1, 1)),
#         )

#     def forward(self, x):
#         return torch.cat(
#             [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1
#         )


# class InceptionAux(nn.Module):
#     def __init__(self, in_channels, num_classes):
#         super(InceptionAux, self).__init__()
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(p=0.7)
#         self.pool = nn.AvgPool2d(kernel_size=5, stride=3)
#         self.conv = conv_block(in_channels, 128, kernel_size=1)
#         self.fc1 = nn.Linear(2048, 1024)
#         self.fc2 = nn.Linear(1024, num_classes)

#     def forward(self, x):
#         x = self.pool(x)
#         x = self.conv(x)
#         x = x.reshape(x.shape[0], -1)
#         x = self.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)

#         return x


# class conv_block(nn.Module):
#     def __init__(self, in_channels, out_channels, **kwargs):
#         super(conv_block, self).__init__()
#         self.relu = nn.ReLU()
#         self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
#         self.batchnorm = nn.BatchNorm2d(out_channels)

#     def forward(self, x):
#         return self.relu(self.batchnorm(self.conv(x)))


# if __name__ == "__main__":
#     # N = 3 (Mini batch size)
#     x = torch.randn(3, 3, 224, 224)
#     model = GoogLeNet(aux_logits=True, num_classes=1000)
#     print(model(x)[2].shape)
#####################################################################################################

###########################################ResNet####################################################
class block(nn.Module):
    def __init__(
        self, in_channels, intermediate_channels, identity_downsample=None, stride=1
    ):
        super(block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(
            in_channels, intermediate_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Essentially the entire ResNet architecture are in these 4 lines below
        self.layer1 = self._make_layer(
            block, layers[0], intermediate_channels=64, stride=1
        )
        self.layer2 = self._make_layer(
            block, layers[1], intermediate_channels=128, stride=2
        )
        self.layer3 = self._make_layer(
            block, layers[2], intermediate_channels=256, stride=2
        )
        self.layer4 = self._make_layer(
            block, layers[3], intermediate_channels=512, stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes
        # we need to adapt the Identity (skip connection) so it will be able to be added
        # to the layer that's ahead
        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    intermediate_channels * 4,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(intermediate_channels * 4),
            )

        layers.append(
            block(self.in_channels, intermediate_channels, identity_downsample, stride)
        )

        # The expansion size is always 4 for ResNet 50,101,152
        self.in_channels = intermediate_channels * 4

        # For example for first resnet layer: 256 will be mapped to 64 as intermediate layer,
        # then finally back to 256. Hence no identity downsample is needed, since stride = 1,
        # and also same amount of channels.
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)


def ResNet50(img_channel=3, num_classes=1000):
    return ResNet(block, [3, 4, 6, 3], img_channel, num_classes)


def ResNet101(img_channel=3, num_classes=1000):
    return ResNet(block, [3, 4, 23, 3], img_channel, num_classes)


def ResNet152(img_channel=3, num_classes=1000):
    return ResNet(block, [3, 8, 36, 3], img_channel, num_classes)


def test():
    net = ResNet101(img_channel=3, num_classes=1000)
    y = net(torch.randn(4, 3, 224, 224)).to("cuda")
    print(y.size())

test()
#########################################################################################################################

#############################################################EfficientNet################################################
base_model = [
    # expand_ratio, channels, repeats, stride, kernel_size
    [1, 16, 1, 1, 3],
    [6, 24, 2, 2, 3],
    [6, 40, 2, 2, 5],
    [6, 80, 3, 2, 3],
    [6, 112, 3, 1, 5],
    [6, 192, 4, 2, 5],
    [6, 320, 1, 1, 3],
]

phi_values = {
    # tuple of: (phi_value, resolution, drop_rate)
    "b0": (0, 224, 0.2),  # alpha, beta, gamma, depth = alpha ** phi
    "b1": (0.5, 240, 0.2),
    "b2": (1, 260, 0.3),
    "b3": (2, 300, 0.3),
    "b4": (3, 380, 0.4),
    "b5": (4, 456, 0.4),
    "b6": (5, 528, 0.5),
    "b7": (6, 600, 0.5),
}

class CNNBlock(nn.Module):
    def __init__(
            self, in_channels, out_channels, kernel_size, stride, padding, groups=1
    ):
        super(CNNBlock, self).__init__()
        self.cnn = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU() # SiLU <-> Swish

    def forward(self, x):
        return self.silu(self.bn(self.cnn(x)))

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # C x H x W -> C x 1 x 1
            nn.Conv2d(in_channels, reduced_dim, 1),
            nn.SiLU(),
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)

class InvertedResidualBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            expand_ratio,
            reduction=4, # squeeze excitation
            survival_prob=0.8, # for stochastic depth
    ):
        super(InvertedResidualBlock, self).__init__()
        self.survival_prob = 0.8
        self.use_residual = in_channels == out_channels and stride == 1
        hidden_dim = in_channels * expand_ratio
        self.expand = in_channels != hidden_dim
        reduced_dim = int(in_channels / reduction)

        if self.expand:
            self.expand_conv = CNNBlock(
                in_channels, hidden_dim, kernel_size=3, stride=1, padding=1,
            )

        self.conv = nn.Sequential(
            CNNBlock(
                hidden_dim, hidden_dim, kernel_size, stride, padding, groups=hidden_dim,
            ),
            SqueezeExcitation(hidden_dim, reduced_dim),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def stochastic_depth(self, x):
        if not self.training:
            return x

        binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.survival_prob
        return torch.div(x, self.survival_prob) * binary_tensor

    def forward(self, inputs):
        x = self.expand_conv(inputs) if self.expand else inputs

        if self.use_residual:
            return self.stochastic_depth(self.conv(x)) + inputs
        else:
            return self.conv(x)


class EfficientNet(nn.Module):
    def __init__(self, version, num_classes):
        super(EfficientNet, self).__init__()
        width_factor, depth_factor, dropout_rate = self.calculate_factors(version)
        last_channels = ceil(1280 * width_factor)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.features = self.create_features(width_factor, depth_factor, last_channels)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(last_channels, num_classes),
        )

    def calculate_factors(self, version, alpha=1.2, beta=1.1):
        phi, res, drop_rate = phi_values[version]
        depth_factor = alpha ** phi
        width_factor = beta ** phi
        return width_factor, depth_factor, drop_rate

    def create_features(self, width_factor, depth_factor, last_channels):
        channels = int(32 * width_factor)
        features = [CNNBlock(3, channels, 3, stride=2, padding=1)]
        in_channels = channels

        for expand_ratio, channels, repeats, stride, kernel_size in base_model:
            out_channels = 4*ceil(int(channels*width_factor) / 4)
            layers_repeats = ceil(repeats * depth_factor)

            for layer in range(layers_repeats):
                features.append(
                    InvertedResidualBlock(
                        in_channels,
                        out_channels,
                        expand_ratio=expand_ratio,
                        stride = stride if layer == 0 else 1,
                        kernel_size=kernel_size,
                        padding=kernel_size//2, # if k=1:pad=0, k=3:pad=1, k=5:pad=2
                    )
                )
                in_channels = out_channels

        features.append(
            CNNBlock(in_channels, last_channels, kernel_size=1, stride=1, padding=0)
        )

        return nn.Sequential(*features)

    def forward(self, x):
        x = self.pool(self.features(x))
        return self.classifier(x.view(x.shape[0], -1))


def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    version = "b0"
    phi, res, drop_rate = phi_values[version]
    num_examples, num_classes = 4, 10
    x = torch.randn((num_examples, 3, res, res)).to(device)
    model = EfficientNet(
        version=version,
        num_classes=num_classes,
    ).to(device)

    print(model(x).shape) # (num_examples, num_classes)

test()



