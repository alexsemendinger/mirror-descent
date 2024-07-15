# Worked-out example of a single step of mirror descent.
# 
# L_ij_minimizer returns a matrix with the following property:
#  * fix a vector w_*. let y_i = <e_i, w_*> for i=1, ..., d.
#  * fix an initialization w0.
#  * if you train w0 on (e_i, y_i) and evaluate on (e_j, y_j)
#  * you'll have zero evaluation loss.
#
# This script demonstrates that if you train Q with naive gradient descent
#  on this data, you'll recover the L_ij_minimizer.


import numpy as np
import matplotlib.pyplot as plt

def L_ij_minimizer(w0, w_star, lr=0.5):
    """
    Returns matrix Q that makes the error L_[ij] = 0
    """
    d = np.shape(w0)[0]
    Q = np.zeros((d,d))
    for i in range(d):
        for j in range(d):
            Q[i,j] = ((w0[j] - w_star[j]) / (w0[i] - w_star[i])).item()
    return Q / (2 * lr)


def generate_std_basis_data(d_feature, n_samples, w=None):
    if w is None:
        w = np.random.randn(d_feature, 1)
    xs, ys = [], []
    for i in range(n_samples):
        ei = np.zeros((d_feature, 1))
        ei[i,0] = 1
        xs.append(ei)
        ys.append(w[i,0])
    return xs, ys, w


def mirror_descent_step(w, Q, lr, x, y):
    """
    Single step of mirror descent as described above.
    Returns updated weights.
    """
    return w - 2 * lr * (np.inner(w, x) - y) * (Q @ x)


# learning the Lij_minimizer for *off-diag* elements when the i != j version of potential update is used
#  which makes sense, since we're purposely not updating diagonals in that case.
# Let's try modifying it so we do update diagonal
def potential_update_with_diag(w, Q, outer_lr, inner_lr, xs, ys):
    """
    Derivative of the cross-validation loss (as implemented in `crossval` above)
    with respect to the matrix Q.

    Returns the updated matrix.
    """
    k = len(xs)
    def cv_derivative(w, Q, lr, xi, xj, yi, yj):
        err = (w.T @ xi - yi).item()
        return (w.T @ xj - 2 * lr * err * (xi.T @ Q @ xj) - yj).item() * err * np.outer(xi, xj)
    
    update = 0
    for i, (xi, yi) in enumerate(zip(xs, ys)):
        for j, (xj, yj) in enumerate(zip(xs, ys)):
            #print(f"\n\n(i,j) = {(i,j)}. \ndC/dQ=\n{cv_derivative(w, Q, inner_lr, xi, xj, yi, yj)}")
            update += cv_derivative(w, Q, inner_lr, xi, xj, yi, yj)
    update = - 2 * inner_lr * update / (k * (k - 1))

    return Q - outer_lr * update


def crossval(w, Q, lr, xs, ys):
    """
    For each (x_i, y_i) in zip(xs, ys):
    1. "Train" a model with a single step of mirror descent on (x_i, y_i)
    2. Evaluate it on the rest of the dataset
    Return the average loss over all i, j. (Here we're not restricting to i != j.)
    """
    k = len(xs)
    def L_ij(w, Q, xi, xj, yi, yj):
        return ( w.T @ xj - 2 * lr * (w.T @ xi - yi) * (xi.T @ Q @ xj) - yj )**2
    value = 0
    for i, (xi, yi) in enumerate(zip(xs, ys)):
        for j, (xj, yj) in enumerate(zip(xs, ys)):
            value += L_ij(w, Q, xi, xj, yi, yj).item()
    return value / (2 * k * (k-1))

if __name__ == '__main__':
    d_feature, n_samples = 3, 3
    xs, ys, w_star = generate_std_basis_data(d_feature, n_samples)
    w0 = np.ones((d_feature, 1))
    inner_lr, outer_lr = 1, 0.05
    Q_star = L_ij_minimizer(w0, w_star, inner_lr)
    Q0 = np.random.randn(d_feature, d_feature)

    n_potential_iterations = 2000
    Q = Q0
    crossvals = []
    Qs = []
    for iter in range(n_potential_iterations):
        crossvals.append(crossval(w0, Q, inner_lr, xs, ys))
        Qs.append(Q)
        Q = potential_update_with_diag(w0, Q, outer_lr, inner_lr, xs, ys)
    Q_dists = [np.linalg.norm(Q - Q_star) for Q in Qs]

    # Plotting
    plt.subplot(1, 2, 1)
    plt.plot(np.log(crossvals))
    plt.title("Log of crossval")

    plt.subplot(1, 2, 2)
    plt.plot(np.log(Q_dists))
    plt.title("$\log d(Q, Q_\star)$")

    plt.show()