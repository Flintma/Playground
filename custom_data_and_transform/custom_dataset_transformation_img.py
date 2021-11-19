import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F #relu,tanh
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import pandas as pd
import torchvision
import sys
from torchvision.utils import save_image
#######################################################################
#####custom data structure####
# 1. image data
# 2. csv file with 1 column: image data name; 2 column: image data label

########################################################################

# import dataset
class CatsAndDogsDataset(Dateset):
    def __init__(self) -> None:
        super().__init__(self,csv_file,root_dir,transform = None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self,index):
        # return specific image along with its corresponding target
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index,0])
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index,1]))

        if self.transform:
            image = self.transform(image)
        return (image,y_label)


# Set device

# Create simple CNN

# Hyperparameters

# apply transformation to img
my_transforms = transforms.Compose([
    transforms.ToPILImage(), 
    transforms.resize((256,256)),
    transforms.RandomCrop((225,225)),
    transforms.ColorJitter(brightness=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    # in every channels, has to be set after ToTensor()
    transforms.Normalize(mean=[0,0,0],std=[1,1,1]) # this did nothing, have to first find mean and std, then input them here
    ])

# Load Data
dataset = CatsAndDogsDataset(csv_file = '.csv', root_dir = 'cats_dogs_resized', transform = my_transforms)
train_set,test_set = torch.utils.data.random_split(dataset,[20000,5000])
train_loader = DataLoader(dataset = train_set, batch_size = batch_size,shuffle = True)  
test_loader = DataLoader(dataset = test_set, batch_size = batch_size,shuffle = True)  

img_num = 0
for _ in range(10):
    for img, label in dataset:
        #save image
        save_image(img, 'img'+str(img_num)+'.png')
        img_num += 1

# Initialize network

# Loss and optimizer

# Load model

# Train Network

# Check accuracy on training & test to see how good our model