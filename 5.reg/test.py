# %%
import torch
from torch import nn, tensor, optim
from torch.optim import Optimizer
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy


# %%
def createData(x, border=5):
    y = torch.zeros(x.shape).float()
    mask = x >= border
    indices = np.random.choice(x.shape[0], 3, replace=False)
    mask[indices] = ~mask[indices]
    y[mask] = 1.0
    c = ["r" if y_ == 1 else "b" for y_ in y]
    return y, c


# %%
torch.manual_seed(0)
x = torch.randint(0, 10, (100, 1)).float()
y, c = createData(x)
plt.scatter(x, y, c=c)
plt.show()

# %%
net1 = nn.Sequential(
    nn.Linear(1, 1),
    nn.Sigmoid(),
)
optimizer1 = optim.SGD(net1.parameters(), lr=0.01)
net2 = deepcopy(net1)
net3 = deepcopy(net1)
net4 = deepcopy(net1)

optimizer2 = deepcopy(optimizer1)
optimizer3 = deepcopy(optimizer1)
optimizer4 = deepcopy(optimizer1)
loss_fun = nn.BCELoss()


def print_params(net):
    for name, param in net.named_parameters():
        print(name, param)


def l2reg(net: nn.Module, bias=False):
    # regPenalty=tensor(0.0,requires_grad=True)
    regPenalty = 0
    for name, param in net.named_parameters():
        if not bias and name.endswith("weight"):
            regPenalty += (param.data**2).sum()
        elif bias:
            regPenalty += (param.data**2).sum()
    return regPenalty


l2reg(net1)


def train(
    net: nn.Module,
    optim: Optimizer,
    regbias: bool | None = None,
    donttrain=False,
    numepochs=10,
):
    losses = np.zeros(numepochs)
    net.train()
    for i in range(numepochs):
        y_hat = net(x)
        loss = loss_fun(y_hat, y)
        losses[i] = loss.item()
        if not donttrain:
            optim.zero_grad(True)
            if regbias is not None:
                regPenalty = l2reg(net, regbias)
                loss += regPenalty
            loss.backward()
            optim.step()
        print_params(net)
    return losses


# %%
names_list = ["datapts", "no reg", "l2-norm", "l2-norm with bias", "untrained net"]

print(names_list[1])
losses1 = train(net1, optimizer1)
print(names_list[2])
losses2 = train(net2, optimizer2, False)
print(names_list[3])
losses3 = train(net3, optimizer3, True)
print(names_list[4])
losses4 = train(net4, optimizer4, None, True)
plt.plot(losses1, c="g")
plt.plot(losses2, c="y")
plt.plot(losses3, c="k")
plt.plot(losses4, c="m")
plt.legend(names_list[1:])
plt.show()


# %%
def plotScatter(x, y, c, x_range=None, multiplier=100):
    if x_range is None:
        x_range = x
    y *= multiplier
    net1.eval()
    net2.eval()
    net3.eval()
    net4.eval()
    y1 = net1(x_range) * multiplier
    y2 = net2(x_range) * multiplier
    y3 = net3(x_range) * multiplier
    y4 = net4(x_range) * multiplier
    plt.scatter(x, y, c=c)
    plt.plot(x_range, y1.detach(), c="g")
    plt.plot(x_range, y2.detach(), c="y")
    plt.plot(x_range, y3.detach(), c="k")
    plt.plot(x_range, y4.detach(), c="m")
    plt.legend(names_list)
    plt.show()


# %%
plotScatter(
    x, y, c, torch.vstack((-x.unique()[:, None].flip(0), x.unique()[:, None][1:]))
)

# %%
torch.manual_seed(1)
test_length = 10
xtest = torch.arange(test_length).reshape((test_length, 1)).float()
ytest, ctest = createData(xtest)
plotScatter(
    xtest,
    ytest,
    ctest,
    torch.arange(-test_length, test_length).reshape((2 * test_length, 1)).float(),
)
