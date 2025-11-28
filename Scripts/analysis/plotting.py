import numpy as np
import matplotlib.pyplot as plt
from analysis.fitting import FitResult


# ================================
# 1) PLOT ISTOGRAMMA + FIT 1D
# ================================
def plot_histogram_with_fit(x, y, fit: FitResult, title="Histogram + Fit"):
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.bar(x, y, width=(x[1] - x[0]), alpha=0.6, label="Data")

    if fit is not None:
        # Estrai parametri dal FitResult
        A, mu, sigma, offset = fit.params

        xfit = np.linspace(min(x), max(x), 500)
        yfit = A * np.exp(-(xfit - mu) ** 2 / (2 * sigma**2)) + offset

        ax.plot(xfit, yfit, "r-", linewidth=2, label="Gaussian Fit")

    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Counts")
    ax.legend()

    return fig, ax


# ================================
# 2) PLOT HEATMAP SEMPLICE
# ================================
def plot_heatmap(matrix, title="Heatmap", cmap="viridis"):
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(matrix, origin="lower", cmap=cmap)
    fig.colorbar(im, ax=ax)

    ax.set_title(title)
    ax.set_xlabel("X pixel")
    ax.set_ylabel("Y pixel")

    return fig, ax


# ================================
# 3) PLOT HEATMAP + FIT 2D
# ================================
def plot_heatmap_with_fit2D(matrix, fit2d: FitResult, title="Heatmap + 2D Fit", cmap="viridis"):
    fig, ax = plt.subplots(figsize=(5, 5))

    # Mostra la mappa
    im = ax.imshow(matrix, origin="lower", cmap=cmap)
    fig.colorbar(im, ax=ax)

    if fit2d is not None:
        # Estrai parametri del fit 2D
        A, x0, y0, sx, sy, offset = fit2d.params

        nx = matrix.shape[1]
        ny = matrix.shape[0]

        x = np.arange(nx)
        y = np.arange(ny)
        xgrid, ygrid = np.meshgrid(x, y)

        # Calcola il fit 2D
        zfit = (
            A * np.exp(-(((xgrid - x0) ** 2) / (2 * sx ** 2)
                          + ((ygrid - y0) ** 2) / (2 * sy ** 2)))
            + offset
        )

        # Overlay dei contorni del fit
        ax.contour(xgrid, ygrid, zfit, colors="red", linewidths=1.2)

    ax.set_title(title)
    ax.set_xlabel("X pixel")
    ax.set_ylabel("Y pixel")

    return fig, ax

