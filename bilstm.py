import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F #relu,tanh
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Create RNN
class BRNN(nn.Module):
    def __init__(self,input_size, hidden_size, num_layers,num_classes): 
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size,num_layers, batch_first = True, bidirectional = True)
        self.fc = nn.Linear(hidden_size*2, num_classes)
    
    def forward(self,x):
        h0 = torch.zeros(self.num_layers*2,x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers*2,x.size(0), self.hidden_size).to(device)

        # Forward Prop
        out,_ = self.lstm(x,(h0,c0))
        out = self.fc(out[:,-1,:])
        return out
        

# # experiment on the model
# model = CNN()
# x = torch.randn(64, 1, 28,28)
# print(model(x).shape)



# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 28
sequence_length = 28
num_layers = 2
hidden_size = 256
num_classes = 10
learning_rate = .001
batch_size = 64
num_epoch = 2

# Load Data
train_dataset = datasets.MNIST(root = 'dataset/', train = True, transform = transforms.ToTensor(), download = True)
train_loader = DataLoader(dataset = train_dataset, batch_size=batch_size, shuffle = True)
test_dataset = datasets.MNIST(root = 'dataset/', train = False, transform = transforms.ToTensor(), download = True)
test_loader = DataLoader(dataset = test_dataset, batch_size=batch_size, shuffle = True)

# Initialize network
model = BRNN(input_size, hidden_size, num_layers,num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

# Train Network
for epoch in range(num_epoch):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device = device).squeeze(1)  # 64*1*28*28 --> 64*28*28
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
            x = x.to(device = device).squeeze(1)
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
