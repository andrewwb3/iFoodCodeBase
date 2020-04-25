import numpy as np
import pandas as pd
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import torch.nn.functional as F
from PIL import Image 
from torch.utils.data import Dataset

#Creates a list of all validation images
count = 0
val_names = [] 
for dirname, _, filenames in os.walk('/kaggle/input/ifood-data/val_set'):
    for filename in filenames:
        val_names.append(filename)
        count += 1
print(count)

#Creates a list of all training images
train_names = [] 
for dirname, _, filenames in os.walk('/kaggle/input/ifood-data/train_set'):
    for filename in filenames:
        # print(os.path.join(dirname, filename))
        train_names.append(filename)
        count += 1
print(count)

#custom classes need to work with the data in the ways we need to
class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

class MyDataset(Dataset):
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
    def __init__(self, 
                 image_path, 
                 image_names, 
                 label_path,  transform=None):
        
        self.image_path = image_path
        self.image_names = image_names
        self.transform = transform
        self.labels = pd.read_csv(label_path, 
                                  names=["img_name", "label"])
        
    def get_class_label(self, image_name):
        # your method here
        y = self.labels[self.labels["img_name"] == image_name].iloc[0]["label"]
        return y
        
    def __getitem__(self, index):
        path = self.image_path + "/" + self.image_names[index]
        x = Image.open(path)
        y = self.get_class_label(path.split('/')[-1])
        if self.transform is not None:
            x = self.transform(x)
        return x, y
    
    def __len__(self):
        return len(self.image_names)

#pre process all of the images using Transforms
train_transforms = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.768986940,.6641706 ,0.5923363),(0.18613161, 0.22524446, 0.23932885))
])

val_transforms = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.768986940,.6641706 ,0.5923363),(0.18613161, 0.22524446, 0.23932885))
])

test_transforms = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.768986940,.6641706 ,0.5923363),(0.18613161, 0.22524446, 0.23932885))
])    


#create the data sets
path = "/kaggle/input/ifood-data/"
train_data = MyDataset(path +"train_set/train_set", 
                       train_names, 
                       "/kaggle/input/ifood-rice/train_info.csv", 
                       transform= train_transforms)
val_data = MyDataset(path +"val_set/val_set", 
                       val_names, 
                       "/kaggle/input/ifood-rice/val_info.csv", 
                       transform= val_transforms)

test_data = ImageFolderWithPaths(path + "test_set", transform = test_transforms)

batch_size = 16

#create the data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1)

#necessary if running on Kaggle Kernel
!nvidia-smi

#set up CUDA
USE_GPU = True

dtype = torch.float32 # we will be using float throughout this tutorial

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Constant to control how frequently we print train loss
print_every = 11800

print('using device:', device)

#our accuracy and training functions
def check_accuracy(loader, model):  
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))

def train(model, optimizer, epochs):
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for e in range(epochs):
        
        for t, (x, y) in enumerate(train_loader):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            if t % 100 == 0:
                print(t, loss.item())
            if t % print_every == 0:
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
                check_accuracy(val_loader, model)
                print()

#use provided resnet 152 with pretrained weights
model = models.resnet152(pretrained=True)
#change the last fully connected layer of the provided model to support 251 classes
model.fc = nn.Linear(2048,251)

#From here we repeatedly change the learning rate, train the model, and create a kaggle output
optimizer = optim.SGD(model.parameters(), lr=.01, momentum=.9, nesterov=True)
train(model, optimizer, 2)
check_accuracy(val_loader, model)
model.eval()
model.to(device=device)
with open('submission1.txt', 'w') as file:
    file.write("img_name,label\n")
    with torch.no_grad():
        for x, y, path in test_loader:
            name = path[0][-15:]
            x = x.to(device=device, dtype=dtype)
            scores = model(x)
            out_labels = [int(x) for x in (torch.topk(scores, 3)[1][0])]
            file.write(name + "," + str(out_labels[0]) + " " + str(out_labels[1]) + " " + str(out_labels[2]) + "\n")
print("done writing")

optimizer = optim.SGD(model.parameters(), lr=.005, momentum=.9, nesterov=True)
train(model, optimizer, 1)
check_accuracy(val_loader, model)
model.eval()
model.to(device=device)
with open('submission2.txt', 'w') as file:
    file.write("img_name,label\n")
    with torch.no_grad():
        for x, y, path in test_loader:
            name = path[0][-15:]
            x = x.to(device=device, dtype=dtype)
            scores = model(x)
            out_labels = [int(x) for x in (torch.topk(scores, 3)[1][0])]
            file.write(name + "," + str(out_labels[0]) + " " + str(out_labels[1]) + " " + str(out_labels[2]) + "\n")
print("done writing")

optimizer = optim.SGD(model.parameters(), lr=.001, momentum=.9, nesterov=True)
train(model, optimizer, 2)
check_accuracy(val_loader, model)
model.eval()
model.to(device=device)
with open('submission3.txt', 'w') as file:
    file.write("img_name,label\n")
    with torch.no_grad():
        for x, y, path in test_loader:
            name = path[0][-15:]
            x = x.to(device=device, dtype=dtype)
            scores = model(x)
            out_labels = [int(x) for x in (torch.topk(scores, 3)[1][0])]
            file.write(name + "," + str(out_labels[0]) + " " + str(out_labels[1]) + " " + str(out_labels[2]) + "\n")
print("done writing")

optimizer = optim.SGD(model.parameters(), lr=.005, momentum=.9, nesterov=True)
train(model, optimizer, 2)
check_accuracy(val_loader, model)
model.eval()
model.to(device=device)
with open('submission4.txt', 'w') as file:
    file.write("img_name,label\n")
    with torch.no_grad():
        for x, y, path in test_loader:
            name = path[0][-15:]
            x = x.to(device=device, dtype=dtype)
            scores = model(x)
            out_labels = [int(x) for x in (torch.topk(scores, 3)[1][0])]
            file.write(name + "," + str(out_labels[0]) + " " + str(out_labels[1]) + " " + str(out_labels[2]) + "\n")
print("done writing")

optimizer = optim.SGD(model.parameters(), lr=.0001, momentum=.9, nesterov=True)
train(model, optimizer, 2)
check_accuracy(val_loader, model)
model.eval()
model.to(device=device)
with open('submission5.txt', 'w') as file:
    file.write("img_name,label\n")
    with torch.no_grad():
        for x, y, path in test_loader:
            name = path[0][-15:]
            x = x.to(device=device, dtype=dtype)
            scores = model(x)
            out_labels = [int(x) for x in (torch.topk(scores, 3)[1][0])]
            file.write(name + "," + str(out_labels[0]) + " " + str(out_labels[1]) + " " + str(out_labels[2]) + "\n")
print("done writing")

#save the model

torch.save(model, "/kaggle/working/resnet152_9epochs.pth")