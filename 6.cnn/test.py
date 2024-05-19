# %%
import torch
from torch import nn, optim
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as T
from sklearn.metrics import accuracy_score

# %%
ROOT = "./data_for_test"
traindata = MNIST(ROOT, train=True, download=True)
traindata, valdata = random_split(traindata, [50000, 10000])
trainloader = DataLoader(traindata)
evalloader = DataLoader(valdata)

# %%
for X, y in trainloader:
    print(X.shape, y.shape)

# %%
for X, y in trainloader:
    print(X.shape, y.shape)

# %%
DEVICE = "cuda"

# setup training hyperparameters for the MLP
num_epochs = 10
batch_size = 50
learning_rate = 0.05
momentum = 0.9
linear_units = 30

mlp_model = nn.Sequential(
    nn.Linear(784, linear_units),
    nn.ReLU(),
    nn.Linear(linear_units, 10),
).to(DEVICE)
# create optimizer for the model
optimizer = optim.SGD(mlp_model.parameters(), lr=learning_rate, momentum=momentum)

# setup training hyperparameters
batch_size = 50
learning_rate = 0.01
# setup model hyperparameters
kernel_size = (4, 4)
stride = (2, 2)
padding = (1, 1)
n_filters_conv1 = 5
n_filters_conv2 = 10

conv_model = nn.Sequential(
    nn.Conv2d(1, n_filters_conv1, kernel_size, stride, padding),
    nn.ReLU(),
    nn.Conv2d(n_filters_conv1, n_filters_conv2, kernel_size, stride, padding),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(490, 10),
).to(DEVICE)
optimizer = optim.SGD(conv_model.parameters(), lr=learning_rate, momentum=momentum)

lossfun = nn.CrossEntropyLoss()


# %%
def train(
    model: nn.Module,
    loader: DataLoader,
    num_epochs: int,
    lossfun,
):
    model.train()
    for epoch in range(num_epochs):
        print("Epoch {} / {}:".format(epoch + 1, num_epochs))
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            y_hat = model(X)
            loss = lossfun(y_hat, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            acc = accuracy_score(y, y_hat)
            print("  Training Accuracy: {:.4f}".format(acc))
            print("  Training Cost: {:.4f}".format(loss))


def eval(
    model: nn.Module,
    loader: DataLoader,
    lossfun,
):
    model.eval()
    with torch.inference_mode():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            y_hat = model(X)
            loss = lossfun(y_hat, y)
            acc = accuracy_score(y, y_hat)

            print("  Validation Accuracy: {:.4f}".format(acc))
            print("  Validation Cost: {:.4f}".format(loss))


# %%
print("MLP")
train(
    mlp_model,
    trainloader,
    num_epochs,
    lossfun,
)
eval(
    mlp_model,
    evalloader,
    lossfun,
)

# %%
print("CNN")
train(
    conv_model,
    trainloader,
    num_epochs,
    lossfun,
)
eval(
    conv_model,
    evalloader,
    lossfun,
)

# %%
import scipy as sp
import numpy as np

x = np.arange(1, 5)
w = np.array([2, 1, 3])
y = np.array([2, 4])
b = 1
x, w

# %%
y_hat = np.correlate(x, w, mode="valid") + b

# %%
0.5 * (y - y_hat) @ (y - y_hat)

# %%
lw = np.correlate(y_hat - y, x, mode="valid")[::-1]
lw

# %%
lb = (y_hat - y).sum()
lb

# %%
w_new = w - 0.01 * lw
b_new = b - 0.01 * lb
w_new, b_new

# %%
y_hat_new = np.correlate(x, w_new, mode="valid") + b_new
y_hat_new

# %%
0.5 * (y - y_hat_new) @ (y - y_hat_new)
