import torch
import torchvision.datasets as datasets
import os
from torch.utils.data import WeightedRandomSampler, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn

# Methods for dealing with imbalanced datsets:
# 1. Oversampling
# 2. Class weighting

#########################################example#########################################
# Image with different classes are stored in different folders with root_dir = "/dataset"
######################################################################################### 

## Oversampling
def get_loader(root_dir, batch_size):
    my_transforms = transforms.Compose(
        [
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ]
    )

    dataset = datasets.ImageFolder(root = root_dir, transform=my_transforms)

    class_weights = []
    for root, subdir,files in os.walk(root_dir):
        if len(files > 0):
            class_weights.append(1/len(files))
        
    # class_weights = [1,50]
    
    sample_weight = [0] * len(dataset)
    for idx, (data, label) in enumerate(dataset):
        class_weight = class_weights[label]
        sample_weights[idx] = class_weight

    sampler = WeightedRandomSampler(sample_weights, num_samples = len(sample_weight), replacement = True) # sample with replacement

    loader = DataLoader(dataset, batch_size=batch_size, sampler = sampler)
    return loader

def main():
    loader = get_loader(root_dir = "dataset", batch_size = 8)

    for data, labels in loader:
        print(labels)


if __name__ == '__main__':
    main()


## Class weighting method
# loss_fn = nn.CrossEntropyLoss(weight = torch.tensor([1,50]))