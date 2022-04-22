from tqdm import tqdm
import sklearn.datasets as ds
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import plotly.express as px
from torch.nn import Module


class SyntheticData(Dataset):
    def __init__(self, n_samples=200, n_features=1, noise=20, random_state=42):
        x, y, coefs = ds.make_regression(n_samples=n_samples, n_features=n_features, noise=noise, coef=True,
                                         random_state=random_state)

        # x = np.c_[np.ones((n_samples, 1)), x]
        y = y.reshape(n_samples, 1)
        self.n_samples = n_samples

        self.X = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.coefs = coefs

    def __getitem__(self, item):
        return self.X[item], self.y[item]

    def __len__(self):
        return self.n_samples


class LinearModel:

    def __init__(self, n_dims):
        self.theta = torch.rand(n_dims, 1)

    def predict(self, batch):
        return batch @ self.theta

    def measure_cost(self, validation_loader):
        total_cost = 0
        n_samples = len(validation_loader)

        for x, y in validation_loader:
            prediction = self.predict(x)
            total_cost += torch.sum((prediction - y) ** 2)

        return float(total_cost / (2 * n_samples))

    def train(self, n_epochs, train_loader, validation_loader, step_size=0.01):
        costs = []
        for _ in tqdm(range(n_epochs)):

            for x, y in train_loader:
                gradient = x.T @ (self.predict(x) - y)
                gradient /= len(x)

                self.theta -= step_size * gradient

            cost = self.measure_cost(validation_loader)
            costs.append(cost)
        return costs


class PyTorchLinearModel(Module):

    def __init__(self, n_input, n_output=1):
        super().__init__()
        self.linear = torch.nn.Linear(n_input, n_output)

    def forward(self, x):
        return self.linear(x)


def train_pytorch_model(model, n_epochs, train_loader):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    pytorch_costs = []
    loss = 0
    for epoch in tqdm(range(n_epochs)):

        for x, y in train_loader:
            predictions = model(x)
            loss = criterion(predictions, y)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
        pytorch_costs.append(loss.item())

    return pytorch_costs


def evaluate_pytorch_model(model, validation_loader):
    with torch.no_grad():
        total_loss = 0

        for x, y in validation_loader:
            predictions = model(x)
            loss = criterion(predictions, y)
            total_loss += loss.item()

        total_loss /= len(validation_loader)

    return total_loss


def main():
    n_dims = 4
    batch_size = 16
    validation_split = 0.2

    n_epochs = 200

    dataset = SyntheticData(n_features=n_dims)
    dataset_size = len(dataset)

    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    validation_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

    model = LinearModel(n_dims)

    costs = model.train(n_epochs, train_loader, validation_loader)

    print(dataset.coefs)
    print(model.theta)
    print(costs[-1])

    pytorch_model = PyTorchLinearModel(n_dims)
    train_pytorch_model(pytorch_model)

    fig = px.line(pytorch_costs)
    fig.show()

    total_loss = evaluate_pytorch_model(pytorch_model)
    print(total_loss)

    for param in pytorch_model.parameters():
        print(param)


if __name__ == "__main__":
    main()
