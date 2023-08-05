import time

import torch
from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3, 4),
            nn.LeakyReLU(),
            nn.Linear(4, 8),
            nn.LeakyReLU(),
            nn.Linear(8, 4),
            nn.LeakyReLU(),
            nn.Linear(4, 2),
        )

    def forward(self, x):
        output = self.network(x)
        return output


if __name__ == "__main__":

    model = NeuralNetwork()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)

    model.train()

    for i in range(0, 50):
        x = torch.rand(3)
        labels = torch.tensor([x[0] + x[2], x[0] * x[2] + x[1]])

        for j in range(0, 3):
            optimizer.zero_grad()

            output = model(x)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

    model.eval()

    test_data = torch.stack([torch.rand(3) for i in range(0, 10)])
    print(test_data)
    test_labels = torch.stack([torch.tensor([x[0] + x[2], x[0] * x[2] + x[1]]) for x in test_data])
    print(test_labels)

    with torch.no_grad():
        print(criterion(model(test_data), test_labels).item())
