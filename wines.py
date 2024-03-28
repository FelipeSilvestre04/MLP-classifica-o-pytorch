from sklearn import datasets
from sklearn.preprocessing import StandardScaler 
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


features = [0, 9]

wine = datasets.load_wine()
data = wine.data[:, features]
targets = wine.target

scaler = StandardScaler()
data = scaler.fit_transform(data)

X = torch.FloatTensor(data).to(device)
Y = torch.LongTensor(targets).to(device)



input_size  = data.shape[1]
hidden_size = 32
output_size = len(wine.target_names)

net = nn.Sequential(nn.Linear(input_size, hidden_size),
                   nn.ReLU(),
                   nn.Linear(hidden_size, output_size),
                   nn.Softmax())
net = net.to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(net.parameters(), lr=1e-3)

for i in range(1000):
 #Forward
    pred = net(X)
    loss = criterion(pred, Y)

 #Backward
    loss.backward()
    optimizer.step()

def plot_boundary(X, y, model):
  x_min, x_max = X[:, 0].min()-0.1, X[:, 0].max()+0.1
  y_min, y_max = X[:, 1].min()-0.1, X[:, 1].max()+0.1
  
  spacing = min(x_max - x_min, y_max - y_min) / 100
  
  XX, YY = np.meshgrid(np.arange(x_min, x_max, spacing),
                       np.arange(y_min, y_max, spacing))
  
  data = np.hstack((XX.ravel().reshape(-1,1), 
                    YY.ravel().reshape(-1,1)))

  db_prob = model(torch.Tensor(data).to(device) )
  clf = np.argmax(db_prob.cpu().data.numpy(), axis=-1)
  
  Z = clf.reshape(XX.shape)
  
  plt.contourf(XX, YY, Z, cmap=plt.cm.brg, alpha=0.5)
  plt.scatter(X[:,0], X[:,1], c=y, edgecolors='k', s=25, cmap=plt.cm.brg)

plot_boundary(data, targets, net)

plt.scatter(data[:, 0], data[:, 1], c=targets, s=15, cmap=plt.cm.brg)
plt.show()