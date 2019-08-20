import numpy as np
import torch
import torch.functional as F
import matplotlib.pyplot as plt
from torch import nn, optim
from torchvision import datasets, transforms
import pathlib
from time import time
from cv2 import cv2

class Net(nn.Module):

    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_sizes[0])
        self.layer2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.layer3 = nn.Linear(hidden_sizes[1], output_size)
        self.relu = nn.ReLU()
        self.lsm = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        x = self.lsm(x)
        return x

# Knowing the dataset
transform = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

# Download the database
mydir = pathlib.PurePath()
trainset = datasets.MNIST(mydir, download=True, train=True, transform=transform)
testset = datasets.MNIST(mydir, download=True, train=False, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

# Knowing the dataset better
dataiter = iter(trainloader)
images, labels = dataiter.next()

print(images.shape)
print(labels.shape)
print(labels[0])
plt.imshow(images[0].numpy().squeeze(), cmap='gray_r')
plt.show()

# Neural network
input_size = 784
hidden_sizes = [128, 64]
output_size = 10
model = Net(input_size, hidden_sizes, output_size)
print(model)

criterion = nn.NLLLoss()
# Training the model
optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
time0 = time()
epochs = 20
for epoch in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
    
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    else:
        print("Epoch {} - Training loss: {:.2}".format(epoch, running_loss/len(trainloader)))
print("\n The training time is {:.2} min".format((time()-time0)/60))

# Test accuracy
correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        for i in range(len(labels)):
            img = images[i].view(1, 784)
            output = model(img)
            probability = torch.exp(output)
            probability = probability[0].numpy()
            pred_label = np.argmax(probability)
            true_label = labels[i].numpy()
            if(true_label == pred_label):
              correct += 1
            total += 1

print("The number of images tested are {}".format(total))
print("\n The model accuracy is {:.2}%".format(correct/total))

filename = pathlib.Path(mydir, 'numpredic.pt')
torch.save(model.state_dict(), filename)
