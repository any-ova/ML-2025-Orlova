import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

import torchvision.transforms.functional as F


class Linear:
    def __init__(self, input_size, output_size):
        '''
        Creates weights and biases for linear layer.
        Dimention of inputs is *input_size*, of output: *output_size*.
        '''
        self.W = np.random.randn(input_size, output_size) * 0.01
        self.b = np.zeros(output_size)

    def forward(self, X):
        '''
        Passes objects through this layer.
        X is np.array of size (N, input_size).
        Returns output of size (N, output_size).
        Hint: You may need to store X for backward pass
        '''
        self.X = X
        return X.dot(self.W) + self.b

    def backward(self, dLdy):
        '''
        1. Compute dLdw and dLdx.
        2. Store dLdw for step() call
        3. Return dLdx
        '''
        self.dLdW = self.X.T.dot(dLdy)
        self.dLdb = dLdy.sum(0)
        self.dLdx = dLdy.dot(self.W.T)
        return self.dLdx

    def step(self, learning_rate):
        '''
        1. Apply gradient dLdw to network:
        w <- w - learning_rate*dLdw
        '''
        self.W = self.W - learning_rate * self.dLdW
        self.b = self.b - learning_rate * self.dLdb


class Sigmoid:
    def __init__(self):
        pass

    def forward(self, X):
        '''
        Passes objects through this layer.
        X is np.array of size (N, d)
        '''
        self.s = 1. / (1 + np.exp(-X))
        return self.s

    def backward(self, dLdy):
        '''
        1. Compute dLdx.
        2. Return dLdx
        '''
        return self.s * (1 - self.s) * dLdy

    def step(self, learning_rate):
        pass


class NLLLoss:
    def __init__(self):
        pass

    def forward(self, X, y):
        '''
        Passes objects through this layer.
        X is np.array of size (N, C), where C is the number of classes
        y is np.array of size (N), contains correct labels
        '''
        self.p = np.exp(X)
        self.p /= self.p.sum(1, keepdims=True)
        self.y = np.zeros((X.shape[0], X.shape[1]))
        self.y[np.arange(X.shape[0]), y] = 1
        return -(np.log(self.p) * self.y).sum(1).mean(0)

    def backward(self):
        '''
        Note that here dLdy = 1 since L = y
        1. Compute dLdx
        2. Return dLdx
        '''
        return (self.p - self.y) / self.y.shape[0]


class NeuralNetwork:
    def __init__(self, modules):
        '''
        Constructs network with *modules* as its layers
        '''
        self.modules = modules

    def forward(self, X):
        y = X
        for i in range(len(self.modules)):
            y = self.modules[i].forward(y)
        return y

    def backward(self, dLdy):
        '''
        dLdy here is a gradient from loss function
        '''
        for i in range(len(self.modules))[::-1]:
            dLdy = self.modules[i].backward(dLdy)

    def step(self, learning_rate):
        for i in range(len(self.modules)):
            self.modules[i].step(learning_rate)


class ReLU:
    def __init__(self):
        pass

    def forward(self, X):
        '''
        Passes objects through this layer.
        X is np.array of size (N, d)
        '''
        self.X = X
        return np.maximum(X, 0)

    def backward(self, dLdy):
        '''
        1. Compute dLdx.
        2. Return dLdx
        '''
        return (self.X > 0) * dLdy

    def step(self, learning_rate):
        pass


class ELU:
    '''
    ELU(x) = x, x > 0; a*(e^x - 1), x <= 0
    '''

    def __init__(self, a=1):
        self.a = a

    def forward(self, X):
        '''
        Passes objects through this layer.
        X is np.array of size (N, d)
        '''
        self.X = X
        return X * (X > 0) + self.a * (np.exp(X) - 1) * (X <= 0)

    def backward(self, dLdy):
        '''
        1. Compute dLdx.
        2. Return dLdx
        '''
        X = self.X
        dydX = (X > 0) + self.a * np.exp(X) * (X <= 0)
        return dLdy * dydX

    def step(self, learning_rate):
        pass


class Tanh:
    def __init__(self):
        pass

    def forward(self, X):
        '''
        Passes objects through this layer.
        X is np.array of size (N, d)
        '''
        exp_pos = np.exp(X)
        exp_neg = np.exp(-X)
        self.tanh_x = (exp_pos - exp_neg) / (exp_pos + exp_neg)
        return self.tanh_x

    def backward(self, dLdy):
        '''
        1. Compute dLdx.
        2. Return dLdx
        '''
        return (1 - (self.tanh_x) ** 2) * dLdy

    def step(self, learning_rate):
        pass


class Noise():
    def __init__(self, mean, stddev):
        self.mean = mean
        self.stddev = stddev

    def __call__(self, tensor):
        noise = torch.zeros_like(tensor).normal_(self.mean, self.stddev)
        return tensor.add_(noise)

    def __repr__(self):
        repr = f"{self.__class__.__name__}(mean={self.mean},stddev={self.stddev})"
        return repr


class Rotation():

    def __init__(self, angle_range=(-15, 15)):
        self.angle_range = angle_range

    def __call__(self, tensor):
        angle = np.random.uniform(self.angle_range[0], self.angle_range[1])
        return F.rotate(tensor, angle)

    def __repr__(self):
        repr = f"{self.__class__.__name__}(angle_range={self.angle_range})"
        return repr


class Shift():

    def __init__(self, shift_range=(-0.1, 0.1)):
        self.shift_range = shift_range

    def __call__(self, tensor):
        shift_x = np.random.uniform(self.shift_range[0], self.shift_range[1])
        shift_y = np.random.uniform(self.shift_range[0], self.shift_range[1])
        return F.affine(tensor, angle=0, translate=(shift_x, shift_y), scale=1.0, shear=0)

    def __repr__(self):
        repr = f"{self.__class__.__name__}(shift_range={self.shift_range})"
        return repr


def train(network, epochs, learning_rate, verbose=True, loss=None):
    loss = loss or NLLLoss()
    train_loss_epochs = []
    test_loss_epochs = []
    train_accuracy_epochs = []
    test_accuracy_epochs = []
    try:
        for epoch in range(epochs):
            losses = []
            accuracies = []
            for X, y in train_loader:
                X = X.view(X.shape[0], -1).numpy()
                y = y.numpy()
                prediction = network.forward(X)
                loss_batch = loss.forward(prediction, y)
                losses.append(loss_batch)
                dLdx = loss.backward()
                network.backward(dLdx)
                network.step(learning_rate)
                accuracies.append((np.argmax(prediction, 1) == y).mean())
            train_loss_epochs.append(np.mean(losses))
            train_accuracy_epochs.append(np.mean(accuracies))

            losses = []
            accuracies = []
            for X, y in test_loader:
                X = X.view(X.shape[0], -1).numpy()
                y = y.numpy()
                prediction = network.forward(X)
                loss_batch = loss.forward(prediction, y)
                losses.append(loss_batch)
                accuracies.append((np.argmax(prediction, 1) == y).mean())
            test_loss_epochs.append(np.mean(losses))
            test_accuracy_epochs.append(np.mean(accuracies))

            if verbose:
                print(f'Epoch {epoch}: Train Loss: {train_loss_epochs[-1]:.3f}, '
                      f'Test Loss: {test_loss_epochs[-1]:.3f}, '
                      f'Train Acc: {train_accuracy_epochs[-1]:.3f}, '
                      f'Test Acc: {test_accuracy_epochs[-1]:.3f}')

    except KeyboardInterrupt:
        print("\nTraining interrupted.")

    return train_loss_epochs, test_loss_epochs, train_accuracy_epochs, test_accuracy_epochs


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = MNIST('.', train=True, download=True, transform=transform)
    test_dataset = MNIST('.', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    network = NeuralNetwork([
        Linear(784, 100), Tanh(),
        Linear(100, 100), Tanh(),
        Linear(100, 10)
    ])
    loss = NLLLoss()

    tr_loss, ts_loss, tr_acc, ts_acc = train(network, 20, 0.01)

    print(f"\nFinal Results (Tanh, no augmentations):")
    print(f"Train Accuracy: {tr_acc[-1]:.3f}, Test Accuracy: {ts_acc[-1]:.3f}")
    print(f"Train Loss: {tr_loss[-1]:.3f}, Test Loss: {ts_loss[-1]:.3f}")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(tr_loss, label='Train Loss', linewidth=2)
    plt.plot(ts_loss, label='Test Loss', linewidth=2)
    plt.title("Loss over Epochs", fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(tr_acc, label='Train Accuracy', linewidth=2)
    plt.plot(ts_acc, label='Test Accuracy', linewidth=2)
    plt.title("Accuracy over Epochs", fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("training_plots.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Преобразования с аугментациями
    transform_rotate = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        Rotation()
    ])

    transform_shift = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        Shift()
    ])

    transform_noise = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        Noise(mean=0.0, stddev=0.1)
    ])

    transform_all = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        Rotation(),
        Shift(),
        Noise(mean=0.0, stddev=0.1)
    ])

    # Загруззики
    train_dataset_rotate = MNIST('.', train=True, download=True, transform=transform_rotate)
    train_dataset_shift = MNIST('.', train=True, download=True, transform=transform_shift)
    train_dataset_noise = MNIST('.', train=True, download=True, transform=transform_noise)
    train_dataset_all = MNIST('.', train=True, download=True, transform=transform_all)

    train_loader_rotate = DataLoader(train_dataset_rotate, batch_size=32, shuffle=True)
    train_loader_shift = DataLoader(train_dataset_shift, batch_size=32, shuffle=True)
    train_loader_noise = DataLoader(train_dataset_noise, batch_size=32, shuffle=True)
    train_loader_all = DataLoader(train_dataset_all, batch_size=32, shuffle=True)

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


    def create_network():
        return NeuralNetwork([
            Linear(784, 100), ReLU(),
            Linear(100, 100), ReLU(),
            Linear(100, 10)
        ])


    learning_rate = 0.01
    epochs = 20

    # Обучение с разными аугментациями
    print("Обучение без аугментаций")
    net_basic = create_network()
    _, ts_loss_basic, _, ts_acc_basic = train(net_basic, epochs, learning_rate, verbose=False)

    print("Обучение с вращениями")
    net_rotate = create_network()
    _, ts_loss_rotate, _, ts_acc_rotate = train(net_rotate, epochs, learning_rate, verbose=False)

    print("Обучение со сдвигами")
    net_shift = create_network()
    _, ts_loss_shift, _, ts_acc_shift = train(net_shift, epochs, learning_rate, verbose=False)

    print("Обучение с шумом")
    net_noise = create_network()
    _, ts_loss_noise, _, ts_acc_noise = train(net_noise, epochs, learning_rate, verbose=False)

    print("Обучение со всеми аугментациями")
    net_all = create_network()
    _, ts_loss_all, _, ts_acc_all = train(net_all, epochs, learning_rate, verbose=False)

    plt.figure(figsize=(14, 6))

    # График Test Loss
    plt.subplot(1, 2, 1)
    plt.title('Test Loss over Epochs', fontsize=16)
    plt.plot(ts_loss_basic, label='No Aug', linewidth=2, marker='o')
    plt.plot(ts_loss_rotate, label='Rotation', linewidth=2, marker='s')
    plt.plot(ts_loss_shift, label='Shift', linewidth=2, marker='^')
    plt.plot(ts_loss_noise, label='Noise', linewidth=2, marker='D')
    plt.plot(ts_loss_all, label='All', linewidth=2, marker='x', linestyle='--')
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # График Test Accuracy
    plt.subplot(1, 2, 2)
    plt.title('Test Accuracy over Epochs', fontsize=16)
    plt.plot(ts_acc_basic, label='No Aug', linewidth=2, marker='o')
    plt.plot(ts_acc_rotate, label='Rotation', linewidth=2, marker='s')
    plt.plot(ts_acc_shift, label='Shift', linewidth=2, marker='^')
    plt.plot(ts_acc_noise, label='Noise', linewidth=2, marker='D')
    plt.plot(ts_acc_all, label='All', linewidth=2, marker='x', linestyle='--')
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig("augmentation_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Без аугментаций:      {ts_acc_basic[-1]:.4f}")
    print(f"Только вращения:      {ts_acc_rotate[-1]:.4f}")
    print(f"Только сдвиги:        {ts_acc_shift[-1]:.4f}")
    print(f"Только шум:           {ts_acc_noise[-1]:.4f}")
    print(f"Все аугментации:      {ts_acc_all[-1]:.4f}")
