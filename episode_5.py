from tqdm import tqdm
import sklearn.datasets as ds
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px

pio.templates.default = "simple_white"


class LinearModel:

    def __init__(self, theta_0=0.5, theta_1=1.25):
        self.theta_0 = theta_0
        self.theta_1 = theta_1

    def predict(self, x):
        return self.theta_0 + self.theta_1 * x

    def train(self, train_loader, n_epochs, step_size, validation_loader):
        costs = []
        for epoch in tqdm(range(n_epochs)):
            d_theta_0 = 0
            d_theta_1 = 0

            for x, y in train_loader:
                d_theta_0 += self.predict(x) - y
                d_theta_1 += (self.predict(x) - y) * x

            d_theta_0 /= len(train_loader)
            d_theta_1 /= len(train_loader)

            self.theta_0 -= step_size * d_theta_0
            self.theta_1 -= step_size * d_theta_1

            costs.append(self.measure_cost(validation_loader))

        return costs

    def measure_cost(self, validation_loader):
        total_cost = 0
        n_samples = len(validation_loader)
        for _, (x, y) in enumerate(validation_loader):
            total_cost += (self.predict(x) - y) ** 2
        return float(total_cost / (2 * n_samples))


class SyntheticData(Dataset):
    def __init__(self, n_samples=200, n_features=1, n_informative=1, noise=20, coefficients=True, random_state=42):
        X, y, coefs = ds.make_regression(n_samples=n_samples,
                                         n_features=n_features,
                                         n_informative=n_informative,
                                         noise=noise,
                                         coef=coefficients,
                                         random_state=random_state)
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.coefs = coefs

        self.n_samples = X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.y[i]

    def __len__(self):
        return self.n_samples


def main():
    synthetic_data = SyntheticData()
    batch_size = 1

    validation_split = 0.2
    dataset_size = len(synthetic_data)
    indices = list(range(dataset_size))

    split = int(np.floor(validation_split * dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(synthetic_data, batch_size=batch_size,
                                               sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(synthetic_data, batch_size=batch_size,
                                                    sampler=valid_sampler)

    # HYPERPARAMETERS
    step_size = 0.1
    n_epochs = 100

    linear_model = LinearModel()
    costs = linear_model.train(train_loader, n_epochs, step_size, validation_loader)

    fig = px.line(costs)
    fig.show()

    print(synthetic_data.coefs)
    print(f"{linear_model.theta_0} | {linear_model.theta_1}")


if __name__ == "__main__":
    main()
