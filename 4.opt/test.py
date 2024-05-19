# %%
import torch
from torch import nn, optim, tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset, RandomSampler
from torchvision import datasets, transforms as T
import matplotlib.pyplot as plt
import numpy as np


# %%
def qwerties(A, B, s, n):
    x1 = tensor(A) + s * torch.randn((n, 2))
    x2 = tensor(B) + s * torch.randn((n, 2))
    c = ["r"] * n + ["b"] * n
    y = torch.vstack((torch.zeros(n)[:, None], torch.ones(n)[:, None])).float()
    return torch.vstack((x1, x2)), y, c


X, y, c = qwerties([1, 2], [2, 3], 0.5, 100)
plt.scatter(X[:, 0], X[:, 1], c=c)
plt.show()

# %%
import torch
from torch import nn, optim, tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset, RandomSampler
from torchvision import datasets, transforms as T
import matplotlib.pyplot as plt
import numpy as np


def createdatanet(X, y, momentum=0, batch_size=None, device="cuda"):
    if len(X) != len(y):
        raise ValueError("lengths unequal")
    if batch_size == None:
        batch_size = len(X)
    dataset = TensorDataset(X, y)
    sampler = RandomSampler(
        dataset,
        generator=torch.Generator().manual_seed(0),
    )
    loader = DataLoader(
        dataset,
        sampler=sampler,
        pin_memory=device == "cuda",
        batch_size=batch_size,
    )
    torch.manual_seed(42)
    net = nn.Sequential(
        nn.Linear(2, 256),
        nn.ReLU(),
        nn.Linear(256, 1),
    ).to(device)
    # net = nn.Linear(2, 1, bias=False).to(device)
    lossfun = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.00001, momentum=momentum)
    return net, lossfun, optimizer, loader


def printnet(net):
    for name, param in net.named_parameters():
        print(name, param)


# GD
net1, lossfun1, optimizer1, loader1 = createdatanet(X, y)
# GD w momentum
net2, lossfun2, optimizer2, loader2 = createdatanet(X, y, momentum=0.5)
# SGD
net3, lossfun3, optimizer3, loader3 = createdatanet(X, y, batch_size=50)
# GD w momentum
net4, lossfun4, optimizer4, loader4 = createdatanet(X, y, momentum=0.5, batch_size=50)


printnet(net1)
printnet(net2)
printnet(net3)
printnet(net4)


def comparenets(net1, net2):
    boo = tensor(True)
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        boo = (param1 == param2).all()
    print(boo)


def compareloaders(loader1, loader2):
    boo = tensor(True)
    indices = torch.zeros(len(loader1))
    for (X1, y1), (X2, y2) in zip(loader1, loader2):
        boo = boo and (X1 == X2).all() and (y1 == y2).all()
        torch.where(X1 == X)[0].unique()
    print(boo)
    print((indices.unique() == indices).all())


comparenets(net1, net2)
compareloaders(loader1, loader2)


def train(net, lossfun, optimizer, loader, numepochs=10, device="cuda"):
    xs = torch.zeros(1, 2)
    for _ in range(numepochs):
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            x_ = list(net2.parameters())[0].data.cpu()
            xs = torch.vstack((xs, x_))
            y_hat = net(X)
            loss = lossfun(y_hat, y)
            optimizer.zero_grad(True)
            loss.backward()
            optimizer.step()
    return xs


def plottrajectory(xs, c, label):
    plt.plot(xs[:, 0], xs[:, 1], c=c, label=label)


# %%
len(loader1), len(loader2), len(loader3), len(loader4)

# %%
# %%
xs1 = train(
    net1,
    lossfun1,
    optimizer1,
    loader1,
)
xs2 = train(
    net2,
    lossfun2,
    optimizer2,
    loader2,
)
xs3 = train(
    net3,
    lossfun3,
    optimizer3,
    loader3,
)
xs4 = train(
    net4,
    lossfun4,
    optimizer4,
    loader4,
)
plt.plot(xs1[:, 0], xs1[:, 1], c="r", label="GD")
plt.scatter(xs1[:, 0], xs1[:, 1], c="r")
plt.plot(xs2[:, 0], xs2[:, 1], c="g", label="GD w momentum")
plt.scatter(xs2[:, 0], xs2[:, 1], c="g")
plt.plot(xs3[:, 0], xs3[:, 1], c="b", label="SGD")
plt.scatter(xs3[:, 0], xs3[:, 1], c="b")
plt.plot(xs3[:, 0], xs3[:, 1], c="m", label="SGD w momentum")
plt.scatter(xs3[:, 0], xs3[:, 1], c="m")
plt.legend()
plt.show()
