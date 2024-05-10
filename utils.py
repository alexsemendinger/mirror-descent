# dumping some things in here so they don't clutter up other notebooks
# this is going to be a bit of a mess by design

import torch as t
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

default_config = dict(
    # generate data
    d_feature = 5,
    n_samples = 100,
    w_cov = None,
    noise_var = 1e-2,
    seed=None,         # manually choose random seed

    # train model (inner loop)
    model_step_size = 1e-3,
    n_model_epochs = 110,
    weight_initialization = None,
    return_loss = False,

    # train potential (outer loop)
    potential_step_size = 1e-3,
    n_potential_epochs = 50,
)

def generate_linear_data(config: dict = default_config) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    """
    Returns (X, y, w), where
    - X: (n_samples, d_feature) tensor of datapoints
    - w ~ N(0, w_cov), e_i ~ N(0, noise_var)
    - y: (n_samples,) tensor, y_i = <w, x_i> + e_i
    """
    n_samples, d_feature = config['n_samples'], config['d_feature']
    w_cov, noise_var = config.get('w_cov'), config.get('noise_var', 1e-2)
    seed = config.get('seed')
    if seed: t.manual_seed(seed)

    if w_cov is None:
        w_cov = t.eye(d_feature)
    
    X = t.rand(n_samples, d_feature)
    w = t.distributions.MultivariateNormal(t.zeros(d_feature), covariance_matrix=w_cov).sample()
    e = t.normal(mean=t.zeros((n_samples,)), std=noise_var)

    y = X @ w + e
    return X, y, w


def create_dataloader_splits(X: t.Tensor, y: t.Tensor, k: int) -> list[DataLoader]:
    """
    Generates k equally-sized DataLoaders for cross-validation
    from tensors X, y
    """
    n_samples, _ = X.shape
    assert n_samples == y.shape[0], f"Tensors are incompatible sizes to form dataset: {X.shape} {y.shape}"

    split_size = n_samples // k
    if n_samples % k != 0:
        print(f"\nWarning in utils.create_dataloader_splits: # of splits {k} does not divide n_samples {n_samples}.")
        print(f"(DataLoaders will still be created, but only {split_size * k} samples will be used.)\n")

    X_splits = [X[n * split_size : (n+1) * split_size] for n in range(k)]
    y_splits = [y[n * split_size : (n+1) * split_size] for n in range(k)]

    # tensor -> TensorDataset -> DataLoader
    dataloaders = [ DataLoader(TensorDataset(X, y.unsqueeze(1)), batch_size=1)
                    for X, y in zip(X_splits, y_splits) ]

    return dataloaders


def generate_train_test_val_loaders(X: t.Tensor, y: t.Tensor) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Generates torch.util.data.DataLoaders for test, train, val
    from tensors X, y
    """
    n_samples, _ = X.shape
    assert n_samples == y.shape[0], f"Tensors are incompatible sizes to form dataset: {X.shape} {y.shape}"

    # Split the dataset into train, validation, and test sets
    train_ratio, val_ratio, test_ratio = 0.6, 0.3, 0.1
    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)

    X_train, X_val, X_test = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
    y_train, y_val, y_test = y[:train_size], y[train_size:train_size+val_size], y[train_size+val_size:]

    # tensor -> TensorDataset -> DataLoader
    train_loader, test_loader, val_loader= [
        DataLoader(TensorDataset(X, y.unsqueeze(1)), batch_size=1)
        for X, y in [(X_train, y_train), (X_val, y_val), (X_test, y_test)]
    ]

    return train_loader, test_loader, val_loader


def compute_validation_loss(model: nn.Module, val_loader: DataLoader, grad_enabled=False) -> t.Tensor:
    #model.eval()  # Set the model to evaluation mode (only relevant for dropout, batchnorm, etc)
    val_loss = 0.0
    criterion = nn.MSELoss()  # Mean Squared Error loss

    with t.set_grad_enabled(grad_enabled):
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss

    return val_loss.mean()


def square_loss(target, label) :
        return (label - target) ** 2