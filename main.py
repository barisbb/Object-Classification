#!/usr/bin/env python
# coding: utf-8

# In[32]:


#Author: Barış Büyüktaş
#Reference: https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import model
import numpy as np


# In[55]:


#rgb_mean = (0.4914, 0.4822, 0.4465)
#rgb_std = (0.2023, 0.1994, 0.2010)
transform = transforms.Compose(
    [transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

batch_size = 100

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)


# In[56]:


net = model.Net()
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = torch.optim.Adam(net.parameters())
#optimizer = torch.optim.Adadelta(net.parameters())
#optimizer = torch.optim.Adagrad(net.parameters())


# In[57]:


losses=[]
accuracies=[]


# In[58]:


for epoch in range(40): 
    
    total_loss = 0.0
    correct = 0
    total = 0
    if(len(accuracies)>1):
        if(accuracies[-1]+0.1<accuracies[-2]):
            print("Stopped")
            break
    
    for i, instance in enumerate(trainloader, 0):
        
        batch_data, batch_labels = instance
       
        optimizer.zero_grad()

        results,y = net(batch_data)
        
        _, predicted = torch.max(results.data, 1)
        total += batch_labels.size(0)
        correct += (predicted == batch_labels).sum().item()
        
        loss = criterion(results, batch_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    accuracies.append(100 * correct / total)   
    losses.append(total_loss / 500)   
    print(total_loss / 500)
    print('Train Accuracy: '+str(100 * correct / total))
            

print('Finished Training')

torch.save(net.state_dict(), "model.pt")


# In[ ]:




features=np.zeros((50000,128))
all_labels=[]
a=0
for data in trainloader:
    images, labels = data
   
    features[a:a+batch_size,:]=net(images)[1].cpu().detach().numpy()
    a=a+batch_size
    for i in labels.data:
        all_labels.append(int(i.cpu().detach().numpy()))


tsne_results = TSNE(n_components=2).fit_transform(features)

colors=['b','g','r','c','m','y','black','darkorange','grey','brown']
fig, ax = plt.subplots()
for i in range(tsne_results.shape[0]):
    ax.scatter(tsne_results[i,0], tsne_results[i,1],s=0.2,color=colors[all_labels[i]])
ax.figure.savefig("scatter.png")


# In[60]:


"""
x = np.arange(1,41)
plt.plot(x, losses)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig('loss_figure.png')
"""


# In[62]:


"""
x = np.arange(1,11)

adadelta=np.array([44.242,56.044,60.246,63.046,65.176,66.59,67.892,68.532,69.18,69.822])
adagrad=np.array([45.406,55.074,58.458,60.568,62.218,63.178,64.038,65.122,65.414,66.308])
adam=np.array([44.704,57.184,61.136,63.52,65.622,66.75,67.94,68.82,69.454,69.872])
sgd=np.array([22.526,34.278,40.81,46.116,49.24,52.276,54.482,56.158,57.468,58.698])

plt.plot(x, adadelta,color='green',label="adadelta")
plt.plot(x, adagrad,color='blue',label="adagrad")
plt.plot(x, adam,color='red',label="adam")
plt.plot(x, sgd,color='cyan',label="sgd")
plt.legend()
plt.xlabel('epoch')
plt.ylabel('training accuracy (%)')
plt.savefig('optimizers.png')
"""


# In[ ]:




