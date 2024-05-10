# Mirror descent on linear model with quadratic potential
# 
# Making code *less* modular, because at least this version seems to get the right gradients
# Will build from here to more modular code

import numpy as np
import torch as t
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

from utils import generate_linear_data, generate_train_test_val_loaders, compute_validation_loss

config = dict(
    # generate data
    d_feature = 5,
    n_samples = 100,
    w_cov = None,
    noise_var = 0.1,
    seed=None,           # manually choose random seed

    # train model (inner loop)
    model_lr = 0.01,
    n_model_epochs = 25,
    weight_initialization = None,
    return_iterates = False,

    # train potential (outer loop)
    potential_lr = 0.1,
    n_potential_epochs = 50,
)


def square_loss(w, x, y):
    return (t.inner(w, x) - y)**2

def batch_loss(w: t.Tensor, test: DataLoader) -> t.Tensor:
    loss = 0.0
    for x, y in test:
        loss += square_loss(w, x, y)
    return loss.mean()


def train_w(w0: t.Tensor, Q: t.Tensor, train: DataLoader, test: DataLoader = None, config: dict = config):
    d = config['d_feature']
    n_epochs = config['n_model_epochs']
    lr = config['model_lr']
    if config['return_iterates']: w_iterates, train_losses, test_losses = [w0.clone().detach()], [], []

    # dimension checks
    assert w0.shape == (1, d), f"{w0.shape}"
    assert Q.shape == (d, d), f"{Q.shape}"

    w = w0.clone().requires_grad_(True)
    for _ in range(n_epochs):
        for x, y in train:
            loss = square_loss(w, x, y)
            w_grad = t.autograd.grad(loss, w)[0]
            
            # mirror descent
            grad_phi_w_prev = (Q @ w.T).T - lr * w_grad
            mirror_update = (Q.inverse() @ grad_phi_w_prev.T).T
            w = mirror_update.requires_grad_(True)

        if config['return_iterates']:  # option to avoid computing these
            w_iterates.append(w.clone().detach())
            train_losses.append(batch_loss(w, train).item())
            if test: test_losses.append(batch_loss(w, test).item())
    
    if config['return_iterates']:
        return w_iterates, train_losses, test_losses
    return w, [], []


if __name__ == "__main__":
    Q = t.nn.Parameter(t.eye(config['d_feature']))
    X, y, w_star = generate_linear_data(config)
    train, test, val = generate_train_test_val_loaders(X, y)

    config['return_iterates'] = True
    
    w0 = 0.2 * t.ones((1,5))
    w_iterates, w_train_losses, w_test_losses = train_w(w0, Q, train, test, config)
    w_hat = w_iterates if isinstance(w_iterates, t.Tensor) else w_iterates[-1]

    print("Linear model trained.")
    print(f"w_star: {w_star}")
    print(f"w_hat: {w_hat}")
    print(f"Final distance: {t.linalg.norm(w_hat - w_star)}")
    if w_train_losses: print(f"Final train loss: {w_train_losses[-1]}")
    if w_test_losses: print(f"Final test loss: {w_test_losses[-1]}")

    w_distances = [np.linalg.norm(w.detach().numpy() - w_star.detach().numpy()) for w in w_iterates]

    # Create a figure and subplots
    fig, axs = plt.subplots(1, 2, figsize=(24, 12))
    axs[0].plot(w_distances)
    axs[0].set_title("Linear model: $d(w_i, w_*)$")
    axs[0].set_yscale('log')
    axs[1].plot(w_train_losses, label="train")
    axs[1].plot(w_test_losses, label="test")
    axs[1].legend()
    axs[1].set_title("Linear model: train and test losses")
    axs[1].set_yscale('log')
    plt.tight_layout()
    plt.show()