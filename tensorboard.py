import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F #relu,tanh
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
#from torch.utils.tensorboard import SummaryWriter # to print to tensorboard

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

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
in_channels = 1
num_classes = 10
num_epoch = 4

# Load Data
train_dataset = datasets.MNIST(root = 'dataset/', train = True, transform = transforms.ToTensor(), download = False)

# test_dataset = datasets.MNIST(root = 'dataset/', train = False, transform = transforms.ToTensor(), download = False)
# test_loader = DataLoader(dataset = test_dataset, batch_size=batch_size, shuffle = True)

# Initialize network

# Loss and optimizer
criterion = nn.CrossEntropyLoss()

# tensorboard

batch_sizes = [2,64,1024]
learning_rates = [.1,.01,.001,.0001]
classes = ['0','1','2','3','4','5','6','7','8','9']
for batch_size in batch_sizes:
    for learning_rate in learning_rates:
        step = 0
        model = CNN().to(device)
        model.train()
        writer = SummaryWriter(f'runs/MNIST/MiniBatchSize {batch_size} LR {learning_rate}' )
        train_loader = DataLoader(dataset = train_dataset, batch_size=batch_size, shuffle = True)
        optimizer = optim.Adam(model.parameters(), lr = learning_rate)

        # Train Network
        for epoch in range(num_epoch):
            accuracies = []
            losses = []
            for batch_idx, (data, targets) in enumerate(train_loader):
                data = data.to(device = device)
                targets = targets.to(device=device)

                # forward
                scores = model(data)
                loss = criterion(scores,targets)
                losses.append(loss.item())
                # backward
                optimizer.zero_grad()
                loss.backward()

                # gradeiend descent
                optimizer.step()

                # Calculate 'running' training accuracy
                features = data.reshape(data.shape[0], -1)


                # see transformation in training
                img_grid = torchvision.utils.make_grid(data)
                writer.add_image('mnist_images', img_grid)
                # see how weight is changing
                writer.add_histogram('fc1', model.fc1.weight)

                _, predictions = scores.max(1)
                num_correct = (predictions == targets).sum()
                running_train_acc = float(num_correct)/float(data.shape[0])
                accuracies.append(running_train_acc)
                # tensorboard
                writer.add_scalar('Training loss', loss, global_step = step)
                writer.add_scalar('Training accuracy', running_train_acc,global_step = step)

                # tensorboard projection (PCA TSNE etc...)
                class_labels = [classes[label] for label in predictions]
                if batch_idx == 230:
                    writer.add_embedding(features, metadeta = class_labels, label_img = data, global_step = batch_idx)
                step += 1
            # hyper parameter plots
            writer.add_hparams({'lr':  learning_rate, 'bsize': batch_size}, {'accuracy': sum(accuracies)/len(accuracies), 'loss': sum(losses)/len(losses)})
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

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')

    model.train()
    return 

check_accuracy(train_loader, model)
# check_accuracy(test_loader, model)



