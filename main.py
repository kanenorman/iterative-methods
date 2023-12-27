import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science"])


def phi(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Represents the heat equation

    Φ(x,y) = sin(πy) * e^(πx)

    Parameters
    ----------
    x
    y

    Returns
    -------
    Solution to equation Φ(x,y)
    """
    return np.sin(np.pi * y) * np.exp(np.pi * x)


def main():
    x = np.linspace(0, 1, 1000)
    y = np.linspace(0, 1, 1000)

    X, Y = np.meshgrid(x, y)
    Z = phi(X, Y)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot the 3D surface on the first subplot

    axis = fig.add_subplot(121, projection="3d")
    axis.plot_surface(X, Y, Z, cmap="hot")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_title(r"$\phi(x,y)$")
    axis.view_init(azim=90, elev=30)

    # Plot the contour plot on the second subplot
    axis = axes[1]
    contours = axis.contour(X, Y, Z, cmap="hot")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_title("Contour Plot")

    cbar = plt.colorbar(contours, ax=axes[1])
    cbar.set_label(r"$\phi(x,y)$")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
