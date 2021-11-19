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
class GoogLeNet(nn.Module):
    def __init__(self, aux_logits=True, num_classes=1000):
        super(GoogLeNet, self).__init__()
        assert aux_logits == True or aux_logits == False
        self.aux_logits = aux_logits

        # Write in_channels, etc, all explicit in self.conv1, rest will write to
        # make everything as compact as possible, kernel_size=3 instead of (3,3)
        self.conv1 = conv_block(
            in_channels=3,
            out_channels=64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
        )

        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = conv_block(64, 192, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # In this order: in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool
        self.inception3a = Inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)

        self.inception4a = Inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = Inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception_block(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(1024, num_classes)

        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)
        else:
            self.aux1 = self.aux2 = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)

        # Auxiliary Softmax classifier 1
        if self.aux_logits and self.training:
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)

        # Auxiliary Softmax classifier 2
        if self.aux_logits and self.training:
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc1(x)

        if self.aux_logits and self.training:
            return aux1, aux2, x
        else:
            return x


class Inception_block(nn.Module):
    def __init__(
        self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool
    ):
        super(Inception_block, self).__init__()
        self.branch1 = conv_block(in_channels, out_1x1, kernel_size=(1, 1))

        self.branch2 = nn.Sequential(
            conv_block(in_channels, red_3x3, kernel_size=(1, 1)),
            conv_block(red_3x3, out_3x3, kernel_size=(3, 3), padding=(1, 1)),
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, red_5x5, kernel_size=(1, 1)),
            conv_block(red_5x5, out_5x5, kernel_size=(5, 5), padding=(2, 2)),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            conv_block(in_channels, out_1x1pool, kernel_size=(1, 1)),
        )

    def forward(self, x):
        return torch.cat(
            [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1
        )


class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.7)
        self.pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = conv_block(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_block, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))


if __name__ == "__main__":
    # N = 3 (Mini batch size)
    x = torch.randn(3, 3, 224, 224)
    model = GoogLeNet(aux_logits=True, num_classes=1000)
    print(model(x)[2].shape)
