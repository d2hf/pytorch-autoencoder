import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
import torch.optim as optim


train = datasets.MNIST("", train= True, download= True,
                       transform= transforms.Compose([transforms.ToTensor()]))

test = datasets.MNIST("", train= True, download= True,
                       transform= transforms.Compose([transforms.ToTensor()]))
                       
trainset = torch.utils.data.DataLoader(train, batch_size= 10, shuffle= True)
testset = torch.utils.data.DataLoader(test, batch_size= 10, shuffle= True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

EPOCHS = 10
net = Net()
net.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.00001)

from datetime import datetime

for epoch in range(EPOCHS):
  start = datetime.now()
  for data in trainset:
    # data to enter the Neural Network
    # y is not used in the AutoEncoder architecture
    X, y = data
    
    net.zero_grad()

    # predict
    output = net(X.to(device))
    loss = criterion(output, X.to(device))
    loss.backward()
    optimizer.step()
  
  end = datetime.now()
  print(f'loss: {loss} seconds: {end-start}')
