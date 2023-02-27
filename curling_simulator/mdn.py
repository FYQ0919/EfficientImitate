import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

n_samples = 1000
#
# epsilon = torch.randn(n_samples)
# x_data = torch.linspace(-10, 10, n_samples)
# y_data = 7*np.sin(0.75*x_data) + 0.5*x_data + epsilon
#
# # y_data, x_data = y_data.view(-1, 1), x_data.view(-1, 1),
# y_data, x_data = x_data.view(-1, 1), y_data.view(-1, 1)

# plt.figure(figsize=(8, 8))
# plt.scatter(x_data, y_data, alpha=0.4)
# plt.show()
x_data = np.load("train_x.npy", allow_pickle=True)
y_data = np.load("train_y.npy", allow_pickle=True)

x_train = torch.tensor(x_data)
y_train = torch.tensor(y_data)

x_test = torch.tensor(x_data[-1:])
y_test = torch.tensor(y_data[-1:])

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class MixtureDensityNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, num_gaussians):
        super(MixtureDensityNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_gaussians = num_gaussians

        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3 * self.num_gaussians)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        alpha, mu, sigma = torch.split(x, self.num_gaussians, dim=1)
        alpha = torch.softmax(alpha, dim=1)
        sigma = torch.exp(sigma)
        return alpha, mu, sigma

    def mdn_loss(self, y_true, alpha, mu, sigma):
        normalizer = (2 * np.pi * sigma ** 2) ** 0.5
        exponent = -0.5 * ((y_true.unsqueeze(1) - mu) / sigma) ** 2
        gaussian = torch.exp(exponent) / normalizer
        density = torch.sum(gaussian * alpha, dim=1)
        loss = -torch.log(density + 1e-9)
        return loss.mean()


# 使用示例
# 定义模型
model = MixtureDensityNetwork(input_dim=64, output_dim=3, num_gaussians=3)
# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.01)
# 训练模型
for epoch in range(10000):
    optimizer.zero_grad()
    alpha, mu, sigma = model(x_train)
    loss = model.mdn_loss(y_train, alpha, mu, sigma)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')
# 进行推断

alpha, mu, sigma = model(x_test)

y_pred = torch.sum(alpha.unsqueeze(-1) * mu, dim=1)

