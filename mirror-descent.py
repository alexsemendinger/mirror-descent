# Mirror Descent and Implicit Regularization
#
# Alex Semendinger, 2024
#

import numpy as np
import matplotlib.pyplot as plt
import torch as t
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


# ------------- Potential class -----------------
class Potential():
    """
    Potential functions for mirror descent. Allows you to package together
        * a function φ
        * its gradient ∇ φ
        * the inverse of the gradient [∇ φ]^{-1}
        * the Bregman divergence D_φ
        * parameters of the potential function

    Usage: 
        p = Potential(phi, grad_phi, grad_phi_inverse, params)
        * call φ(x) via p(x)
        * call ∇ φ(x) via p.grad(x)
        * call [∇ φ]^{-1} (x) via p.grad_inverse(x)
        * call D_φ(x, y) via p.bregman_divergence(x, y)
        * update parameters via p.update_parameters(loss)
    """
    def __init__(self, phi, grad_phi, grad_phi_inverse, params=None):
        self._phi = phi
        self.grad = grad_phi
        self.grad_inverse = grad_phi_inverse
        self.params = params
    
    def __call__(self, x):
        return self._phi(x, self.params)
    
    def bregman_divergence(self, x, y):
        return self(y) - self(x) - t.inner(y-x, self.grad(x, self.params))
    
    def update_parameters(self, loss, lr=0.01):
        # TODO: check carefully, is this correct / concise?

        dummy_tensor = t.tensor(0.0, requires_grad=True)
        loss_tensor = dummy_tensor + loss

        if isinstance(self.params, dict):
            for name, param in self.params.items():
                if param.requires_grad:
                    grad_param = t.autograd.grad(loss_tensor, param, create_graph=True, allow_unused=True)[0]
                    if grad_param is not None:
                        with t.no_grad():
                            param -= lr * grad_param
        else:
            grad_params = t.autograd.grad(loss_tensor, self.params, create_graph=True, allow_unused=True)
            with t.no_grad():
                for param, grad in zip(self.params, grad_params):
                    if grad is not None:
                        param -= lr * grad

# --- potentials ---

def pd_potential(Q):
# TODO: enforce positive definite?

    Q = t.tensor(Q, requires_grad=True)

    def phi(x, params):
        Q = params['Q']
        return t.inner(x, Q @ x) / 2
    
    def grad_phi(x, params):
        Q = params['Q']
        return Q @ x.T
    
    def grad_phi_inverse(x, params):
        Q = params['Q']
        return Q.inverse() @ x
    
    params = {'Q': Q}
    
    return Potential(phi, grad_phi, grad_phi_inverse, params)


# TODO: add more potentials


# ------- mirror descent step ----------
def mirror_descent_step(w_prev, potential, step_size, grad_loss):
    """
    inputs:
        w_prev: weights of previous iteration, tensor
        potential: Potential object
        step_size: float
        grad_loss: gradient of the loss on w_prev
    outputs:
        tensor of same shape as w_prev, mirror descent update
    """
    assert isinstance(w_prev, t.Tensor), "w_prev must be a PyTorch tensor"
    assert isinstance(step_size, float), "step_size must be a float"

    params = potential.params
    mirror_update = potential.grad(w_prev, params) - step_size * grad_loss
    return potential.grad_inverse(mirror_update, params)


# ------------- LINEAR MODEL ----------------

default_config = dict(
    # generate data
    d_feature = 5,
    n_samples = 100,
    w_cov = None,
    noise_var = 1e-2,

    # train model (inner loop)
    model_step_size = 1e-3,
    n_model_epochs = 110,
    weight_initialization = None,

    # train potential (outer loop)
    potential_step_size = 1e-3,
    n_potential_epochs = 10
)

# ---- Dataset generation ------
def generate_linear_data(config):
    """
    Returns (X, y, w), where
    - X: (n_samples, d_feature) tensor of datapoints
    - w ~ N(0, w_cov), e_i ~ N(0, noise_var)
    - y: (n_samples,) tensor, y_i = <w, x_i> + e_i
    """
    n_samples, d_feature = config['n_samples'], config['d_feature']
    w_cov, noise_var = config.get('w_cov'), config.get('noise_var', 1e-2)

    if w_cov is None:
        w_cov = t.ones((d_feature,))

    X = t.rand(n_samples, d_feature)
    w = t.normal(mean=t.zeros((d_feature,)), std=w_cov)  # TODO: make multivariate normal
    e = t.normal(mean=t.zeros((n_samples,)), std=noise_var)

    y = X @ w + e
    return (X, y, w)


def generate_train_test_val_loaders(X, y):
    """
    Generates torch.util.data.DataLoaders for test, train, val
    from tensors X, y
    """
    n_samples, _ = X.shape
    assert n_samples == y.shape[0], f"Tensors are incompatible sizes to form dataset: {X.shape} {y.shape}"

    # Split the dataset into train, validation, and test sets
    train_ratio, val_ratio, test_ratio = 0.6, 0.2, 0.2
    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)

    X_train, X_val, X_test = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
    y_train, y_val, y_test = y[:train_size], y[train_size:train_size+val_size], y[train_size+val_size:]

    # Convert the data to PyTorch tensors and create datasets
    tensor_datasets = [
        TensorDataset(X, y.unsqueeze(1))
        for X, y in [(X_train, y_train), (X_val, y_val), (X_test, y_test)]
    ]

    # Create PyTorch data loaders
    train_loader = DataLoader(tensor_datasets[0], batch_size=1, shuffle=True)
    val_loader = DataLoader(tensor_datasets[1], batch_size=1)
    test_loader = DataLoader(tensor_datasets[2], batch_size=1)

    return train_loader, test_loader, val_loader


# --- Inner training loop ---
def square_loss(target, label):
        return (label - target) ** 2


class LinearModel(nn.Module):
    def __init__(self, config):
        d_feature = config["d_feature"]
        w0 = config["weight_initialization"]

        super(LinearModel, self).__init__()
        self.w = nn.Linear(d_feature, 1, bias=False)

        if w0 is not None:
            self.w.weight.data = w0
    
    def forward(self, x):
        return self.w(x)
    

def train_linear_model(model, potential, train_loader, test_loader, config):
    """
    Performs stochastic mirror descent with specified potential on the objective
       F(w) = |y_i - <w, x_i>|^2.
    """
    n_epochs = config['n_model_epochs']
    step_size = config['model_step_size']

    checkin = n_epochs // 50
    progress_bar = tqdm(range(n_epochs))
    for epoch in progress_bar:
        for input, target in train_loader:
            # forward pass
            output = model(input)
            loss = square_loss(output, target)

            model.zero_grad()
            loss.backward()
            
            # mirror descent update
            # TODO: check all of the logic here and elsewhere carefully
            with t.no_grad():
                param = model.w.weight
                grad = param.grad.view(-1)  # flatten the gradient tensor
                update = mirror_descent_step(param.view(-1), potential, step_size, grad)
                param.copy_(update.view_as(param))  # reshape the update to match the parameter shape
        
        if epoch % checkin == 0:
            train_loss, test_loss = 0, 0
            with t.no_grad():
                for input, target in test_loader:
                    output = model(input)
                    test_loss += square_loss(output, target)
                for input, target in train_loader:
                    output = model(input)
                    train_loss += square_loss(output, target)
            test_loss /= len(test_loader)
            train_loss /= len(train_loader)

            progress_bar.set_postfix(train_loss=train_loss.item(), test_loss=test_loss.item())
    
    return model


# --- Outer training loop ---
def compute_validation_loss(model, val_loader):
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    criterion = nn.MSELoss()  # Mean Squared Error loss

    with t.no_grad():  # Disable gradient computation
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    val_loss /= len(val_loader)  # Average the validation loss
    return val_loss


def train_potential(potential_init: Potential, ModelClass,
                    train_loader, test_loader, val_loader, config=default_config):

    n_epochs = config["n_potential_epochs"]
    potential = potential_init
    weights = []
    val_losses = []

    for epoch in range(n_epochs):
        model = ModelClass(config)
        model = train_linear_model(model, potential, train_loader, test_loader, config)
        loss = compute_validation_loss(model, test_loader)
        potential.update_parameters(loss)
        weights.append(model.w.weight.data)

        if epoch % (n_epochs // 10) == 0:
            val_loss = compute_validation_loss(model, val_loader)
            val_losses.append(val_loss)
            print(f"  Outer loop epoch {epoch}: val loss {val_loss}")  # TODO: make this more professional
    
    return potential, weights, val_losses


# ------------- MAIN METHOD -----------------

if __name__ == "__main__":
    config = default_config
    d_feature = config["d_feature"]

    X, y, w_star = generate_linear_data(config)
    train, test, val = generate_train_test_val_loaders(X, y)

    pd_sqrt = t.rand((d_feature, d_feature))
    potential = pd_potential(pd_sqrt.T @ pd_sqrt)

    trained_potential, trained_weights, val_losses = train_potential(potential, LinearModel, train, test, val, config)

    plt.plot(val_losses)
    plt.title("Validation loss vs epochs")
    plt.xlabel("Epoch (val loss computed every 10th epoch)")
    plt.ylabel("Val loss")
    plt.show()