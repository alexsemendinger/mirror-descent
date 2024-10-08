import matplotlib.pyplot as plt
import numpy as np

def plot_together(data1, data2, title=None, yscale=None):
    """
    Plot two datasets side by side for comparison.
    [ I'm not sure this is very good. ]
    
    Parameters:
        data1 (tuple or list): The first dataset to plot. Can be a tuple of (x, y) or a single list.
        data2 (tuple or list): The second dataset to plot. Can be a tuple of (x, y) or a single list.
        title (str, optional): Title for the figure.
        yscale (str, optional): Scale for the y-axis (e.g., 'log', 'linear', 'symlog').

    Usage:
        plot_together((x1, y1), (x2, y2), title="Comparison of Two Datasets")
        plot_together(data1, data2, title="Data Comparison", yscale='log')
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))  # Creates two subplots side by side

    ax1.plot(data1)  
    ax2.plot(data2)  

    # Setting titles and scales
    if title:
        fig.suptitle(title)
    if yscale:
        ax1.set_yscale(yscale)
        ax2.set_yscale(yscale)
    plt.show()


def plot_matrix_evolution(matrices, n_images=5, extra_matrix=None, extra_matrix_title=None, main_title=None, 
                          t_scale="log", figsize=None, cmap='RdBu'):
    """
    Visualize the evolution of matrices over time, optionally including an extra matrix for comparison.

    This function creates a figure with evenly sampled matrices from the input sequence,
    including the first and last matrices. It uses a diverging colormap (RdBu_r) to clearly
    show positive and negative values, with white representing zero. A colorbar is added
    to show the scale of values across all matrices.

    Example usage:
    With extra matrix (like w_cov):
      `plot_matrix_evolution(Qs, extra_matrix=w_cov, extra_matrix_title=f'w_cov, rank={rank}', main_title='Evolution of $Q$ during training')`

    Without extra matrix:
      `plot_matrix_evolution(Qs, main_title='Evolution of $Q$ during training')`

    Parameters:
    -----------
    matrices : numpy.ndarray
        A 3D array of matrices to visualize, where the first dimension represents time/iteration.
    n_images : int, optional (default=5)
        The number of matrices to display from the sequence (including first and last).
    extra_matrix : numpy.ndarray, optional (default=None)
        An additional matrix to display alongside the sequence (e.g., for comparison).
    extra_matrix_title : str, optional (default=None)
        Title for the extra matrix. If None and extra_matrix is provided, 'Extra Matrix' is used.
    main_title : str, optional (default='Evolution of Matrix')
        The main title for the entire figure.
    t_scale: str, "linear" (default) or "log"
        Determines whether to use np.linspace or np.logspace for choosing indices to sample
    figsize : tuple of int, optional (default=(20, 4))
        Figure size in inches (width, height).
    """
    assert t_scale in {'linear', 'log'}, f"Invalid t_scale option {t_scale}. Valid options are 'linear', 'log'."

    n_matrices = len(matrices)
    n_cols = n_images + 1 if extra_matrix is not None else n_images
    
    if figsize is None: figsize = (4 * (n_cols - 1), 4)
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)
    
    # Find global min and max for consistent color scale
    vmin = matrices.min()
    vmax = matrices.max()
    if extra_matrix is not None:
        vmin = min(vmin, extra_matrix.min())
        vmax = max(vmax, extra_matrix.max())
    
    # Make the colormap symmetrical around zero
    abs_max = max(abs(vmin), abs(vmax))
    vmin, vmax = -abs_max, abs_max
    
    images = []
    
    # Generate indices for even sampling, including first and last
    if n_images > 1:
        if t_scale == "linear":
            indices = np.linspace(0, n_matrices - 1, n_images, dtype=int)
        if t_scale == "log":
            indices = np.logspace(-1, np.log(n_matrices - 1), n_images, dtype=int, base=2)
            if indices[1] == 0: indices[1] = max(1, indices[2] // 2)  # cludge, don't repeat zero index
    else:
        indices = [0]

    if indices[0] != 0:
        print(f"Warning: not plotting matrix at index 0. \nIndices being plotted are: {indices}.")
    
    for j, i in enumerate(indices):
        ax = axes[j] if n_cols > 1 else axes
        im = ax.imshow(matrices[i], cmap=cmap, vmin=vmin, vmax=vmax)
        images.append(im)
        ax.axis('off')
        ax.set_title(f't={i}', fontsize='14')
    
    # Plot extra matrix if provided
    if extra_matrix is not None:
        ax = axes[-1]
        im = ax.imshow(extra_matrix, cmap=cmap, vmin=vmin, vmax=vmax)
        images.append(im)
        ax.axis('off')
        ax.set_title(extra_matrix_title or '', fontsize='14')
    
    fig.suptitle(main_title, fontsize='18')
    
    # Add a colorbar that refers to all images
    fig.subplots_adjust(bottom=0.2)
    cbar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.05])
    cbar = fig.colorbar(images[0], cax=cbar_ax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=10)
    
    # Set colorbar ticks to show negative, zero, and positive values
    tick_locator = plt.MaxNLocator(nbins=5)
    cbar.locator = tick_locator
    cbar.update_ticks()

    plt.show()


