#!/usr/bin/env python
# coding: utf-8

# In[25]:


#Author: Barış Büyüktaş
#Reference: https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import model
import numpy as np


# In[19]:


#rgb_mean = (0.4914, 0.4822, 0.4465)
#rgb_std = (0.2023, 0.1994, 0.2010)
transform = transforms.Compose(
    [transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

batch_size = 100


testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)


# In[30]:


net = model.Net()
net.load_state_dict(torch.load("model.pt"))


# In[31]:


correct = 0
total = 0
with torch.no_grad():
    for instance in testloader:
        batch_data, batch_labels = instance
        results,y = net(batch_data)
        _, predicted = torch.max(results.data, 1)
        total += batch_labels.size(0)
        correct += (predicted == batch_labels).sum().item()

print('Test Accuracy: '+str(100 * correct / total))


# In[ ]:




