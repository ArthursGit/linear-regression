import torch
import sklearn.datasets as ds
import matplotlib.pyplot as plt
import numpy as np


class LinearModel:

    def __init__(self, n_features):
        self.theta = torch.tensor(
            np.random.randn(n_features, 1)
        )

    def predict(self, data):
        if not torch.is_tensor(data): 
            data = torch.tensor(data)
        assert data.shape[1] == self.theta.shape[0], \
            f"Shape mismatch, expected {self.theta.shape[0]}, got {data.shape[1]}"
        return data @ self.theta

    def measure_cost(self, training_data, targets):
        n_samples = len(training_data)
        predictions = self.predict(training_data)
        return ((predictions - targets) ** 2).sum() / (2 * n_samples)

    def train(self, training_data, targets, learning_rate=0.01, n_episodes=500, batch_size=64):
        n_samples = len(training_data)
        for episode in range(n_episodes):
            i = 0
            while i < n_samples:
                if i + batch_size > n_samples:
                    batch_size = n_samples - i

                batch_x = training_data[i:i + batch_size]
                batch_y = targets[i:i+batch_size]

                prediction = self.predict(batch_x)
                self.theta -= learning_rate * ((batch_x.T @ (prediction - batch_y)) / n_samples)

                i += batch_size

            cost = self.measure_cost(training_data, targets)

            print(f"Episode {episode}, Cost = {cost}")


def get_synthetic_data(n_features=1, n_informative=1, noise=20, coefficients=True, random_state=42):
    X, y, coef = ds.make_regression(n_features=n_features,
                                    n_informative=n_informative,
                                    noise=noise,
                                    coef=coefficients,
                                    random_state=random_state)

    X = np.c_[np.ones((len(X), 1)), X]

    return X, y, coef


training_data, target, coefficients = get_synthetic_data()

training_data_tensor = torch.tensor(training_data)
target_tensor = torch.tensor(target).unsqueeze(1)

model = LinearModel(2)

model.train(training_data_tensor, target_tensor)

print(coefficients)
print(model.theta)


plt.scatter(training_data[:, 1], target)

x = np.linspace(-3, 3, 2)
x = np.c_[np.ones((len(x), 1)), x]
x = torch.tensor(x)

predictions = model.predict(x)

plt.plot(x[:, 1], predictions, color='red')

plt.show()
