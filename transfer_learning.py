import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F #relu,tanh
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision
import sys

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Identity(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self,x):
        return x

# Load pretrain model & modify it
model = torchvision.models.vgg16(pretrained = True)

# freeze certain parameters
for param in model.parameters():
    param.requires_grad = False

# modify avgpool
model.avgpool = Identity()
#model.classifier = nn.Linear(512, 10)
model.classifier = nn.Sequential(nn.Linear(512,100), nn.ReLU(),nn.Linear(100,10))
# change specific layer
# model.classifier[0] = nn.Linear(512,10)
model.to(device)
print(model)

## experiment on the model
# model = NN(784,10)
# x = torch.randn(64, 784)
# print(model(x).shape)

# Create simple CNN
class CNN(nn.Module):
    def __init__(self, in_channels = 1, num_classes = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels = 8,kernel_size=(3,3), stride = (1,1), padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size = (2,2), stride = (2,2))
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 16,kernel_size=(3,3), stride = (1,1), padding=(1,1))
        self.fc1 = nn.Linear(16*7*7,num_classes)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fc1(x)
        return x

# # experiment on the model
# model = CNN()
# x = torch.randn(64, 1, 28,28)
# print(model(x).shape)

# function to save model
def save_checkpoint(state, filename = "my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

# load model
def load_checkpoint(checkpoint):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


# Hyperparameters
in_channels = 1
num_classes = 10
learning_rate = .001
batch_size = 1024
num_epoch = 2
load_model = False

# Load Data
train_dataset = datasets.MNIST(root = 'dataset/', train = True, transform = transforms.ToTensor(), download = True)
train_loader = DataLoader(dataset = train_dataset, batch_size=batch_size, shuffle = True)
test_dataset = datasets.MNIST(root = 'dataset/', train = False, transform = transforms.ToTensor(), download = True)
test_loader = DataLoader(dataset = test_dataset, batch_size=batch_size, shuffle = True)

# Initialize network
model = CNN().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

# Load model
if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"))

# Train Network
for epoch in range(num_epoch):

    # # save model
    # if epoch % 3 == 0:
    #     checkpoint = {'state_dict':model.state_dict(), 'optimizer': optimizer.state_dict()}
    #     save_checkpoint(checkpoint)

    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device = device)
        targets = targets.to(device=device)
        
        # # flatten the data
        # data = data.reshape(data.shape[0], -1)

        # forward
        scores = model(data)
        loss = criterion(scores,targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradeiend descent
        optimizer.step()
    print(f"Loss at epoch {epoch} was {loss:.5f}")
# Check accuracy on training & test to see how good our model
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on testing data")
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x,y in loader:
            x = x.to(device = device)
            y = y.to(device = device)
            #x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')

    model.train()
    return 

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)



