# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 15:08:29 2019

@author: Albert Tran
"""

# %% -------------------------------------------------------------------------------------------------------------------
# File I/O
# ----------------------------------------------------------------------------------------------------------------------
'''
Point this to your own directory of images. My directory has the following structure:

DATA_DIR:
    - alice (photos of alice)
    - gina  (photos of gina)

'''
DATA_DIR = r'D:\Programming\pytorch_scrap\data\kubrick_faces' 


# %% -------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tqdm import tqdm


# %% -------------------------------------------------------------------------------------------------------------------
# Device Setup
# ----------------------------------------------------------------------------------------------------------------------
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('GPU Detected: ', torch.cuda.get_device_name())


# %% -------------------------------------------------------------------------------------------------------------------
# Data Setup
# ----------------------------------------------------------------------------------------------------------------------
IMGNET_SIZE = 224
IMGNET_MEAN = [0.485, 0.456, 0.406]
IMGNET_STD  = [0.229, 0.224, 0.225]


# Setting up the transforms
transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                transforms.Resize((IMGNET_SIZE, IMGNET_SIZE)),
                                transforms.Grayscale(3), # Grayscale
                                transforms.ToTensor(),
                                transforms.Normalize(IMGNET_MEAN, IMGNET_STD)])

dataset = torchvision.datasets.ImageFolder(DATA_DIR, transform=transform)

# Splitting the dataset into train and test
dataset_size = len(dataset)
train_size   = int(dataset_size * 0.8)
test_size    = dataset_size - train_size
trainset, testset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Setting up dataloaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader  = torch.utils.data.DataLoader(testset,  batch_size=64, shuffle=True)

# Names of classes
classes = dataset.classes



# %% -------------------------------------------------------------------------------------------------------------------
# Visualise a few images
# ----------------------------------------------------------------------------------------------------------------------
nrows = 3
ncols = 5
npics = nrows * ncols

indexes    = np.random.choice(range(len(trainset)), npics)
fig, axarr = plt.subplots(nrows, ncols)

for i, index in enumerate(indexes):
    image, label = trainset[index]
    curr_ax = axarr.ravel()[i]
    curr_ax.imshow(np.transpose(image, (1,2,0)))
    curr_ax.set_xticks([])
    curr_ax.set_yticks([])
    curr_ax.set_title(classes[label])






# %% -------------------------------------------------------------------------------------------------------------------
# Defining the neural network (pre-trained)
# ----------------------------------------------------------------------------------------------------------------------
net_tl = models.resnet18(pretrained=True)
net_tl.fc = nn.Linear(net_tl.fc.in_features, len(classes)) # .fc is the final layer!
net_tl.to(device)

# Freeze the first several layers
layers_frozen = 5
for index, child in enumerate(net_tl.children()):
    if index >= 0:
        break
    else:
        for param in child.parameters():
            param.requires_grad = False




# %% -------------------------------------------------------------------------------------------------------------------
# Function - Network Training
# ----------------------------------------------------------------------------------------------------------------------
def train_network(net, trainloader, criterion, optimizer, num_epochs):
    losses     = []
    
    for epoch in range(num_epochs):
        print('epoch: ', epoch)
        for images, labels in tqdm(trainloader, position=0, leave=True):
            # To GPU
            images = images.to(device)
            labels = labels.to(device)
            
            # Calculate predictions
            outputs = net(images)
            
            # Calculate the loss and calculate gradients
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Running statistics
            curr_loss = loss.item()
            losses.append(curr_loss)
            
    return losses
        



# %% -------------------------------------------------------------------------------------------------------------------
# Function - Network Testing
# ----------------------------------------------------------------------------------------------------------------------
def test_network(net, testloader, criterion):
    num_correct = 0
    counter     = 0
    losses      = []
    
    with torch.no_grad():
        for images, labels in tqdm(testloader, position=0, leave=True):
            # To GPU
            images = images.to(device)
            labels = labels.to(device)
            
            # Calculate predictions
            outputs = net(images)
            predicted = torch.max(outputs, 1)[1]
            loss = criterion(outputs, labels).item()
            
            # Keeping track of the number correct
            num_correct += (labels == predicted).sum().item()
            counter     += images.shape[0]
            losses.append(loss)
    
    accuracy = num_correct / counter
    print('Test Accuracy   : ', accuracy)
    print('Num Observations:', counter)
    print('Num Correct     :', num_correct)
    return losses





# %% -------------------------------------------------------------------------------------------------------------------
# Let's train the model
# ----------------------------------------------------------------------------------------------------------------------
# Initialize parameters
losses_train = []
losses_test  = []
best_loss    = np.inf
best_model   = None

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net_tl.parameters(), lr=0.0001, momentum=0.8)

# %% 
# Running the training
for i in range(15):
    # Train
    curr_losses_train = train_network(net_tl, trainloader, criterion, optimizer, 1)
    avg_loss_train    = np.mean(curr_losses_train)
    losses_train.append(avg_loss_train)

    # Test
    curr_losses_test = test_network(net_tl, testloader, criterion)
    avg_loss_test    = np.mean(curr_losses_test)
    losses_test.append(avg_loss_test)
    
    # Keep track of the best model
    if avg_loss_test <= best_loss:
        best_model = copy.deepcopy(net_tl)
        best_loss  = avg_loss_test
    

# Plotting the running loss
fig, ax = plt.subplots()
ax.plot(losses_train, label='training loss')
ax.plot(losses_test,  label='test loss')
ax.grid()
ax.legend()
ax.set_ylabel('Average Loss')
ax.set_xlabel('Epoch')


