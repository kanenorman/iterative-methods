import numpy as np
import matplotlib.pyplot as plt

# Uncomment the next two lines if scienceplots is installed
# import scienceplots
# plt.style.use(["science"])


def phi(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Represents the heat equation Φ(x,y) = sin(πy) * e^(πx).
    """
    return np.sin(np.pi * y) * np.exp(np.pi * x)


def initialize_matrix(h):
    """
    Initializes the matrix with boundary conditions for the heat equation.
    """
    n = int(1 / h) - 1
    w = np.zeros((n + 2, n + 2))

    for i in range(1, n + 3):
        for j in range(1, n + 3):
            w[-1, j - 1] = np.exp(np.pi) * np.sin(np.pi * (j - 1) * h)
            w[0, j - 1] = np.sin(np.pi * (j - 1) * h)
            w[i - 1, -1] = 0
            w[i - 1, 0] = 0

    return w


def jacobi(h, n_iterations=10000, tol=1e-8):
    """
    Solves the heat equation using the Jacobi method.
    """
    w_old = initialize_matrix(h)
    f = np.zeros_like(w_old)
    ratio = 1
    k = 0

    while ratio > tol and k < n_iterations:
        k += 1
        w_new = np.copy(w_old)

        for i in range(1, len(w_old) - 1):
            for j in range(1, len(w_old) - 1):
                w_new[i, j] = (
                    w_old[i + 1, j]
                    + w_old[i - 1, j]
                    + w_old[i, j - 1]
                    + w_old[i, j + 1]
                    + f[i, j] * h**2
                ) / 4

        ratio = np.max(np.abs(w_new - w_old))
        w_old = w_new

    return w_old


def gauss_seidel(h, n_iterations=10000, tol=1e-8):
    """
    Solves the heat equation using the Gauss-Seidel method.
    """
    w_old = initialize_matrix(h)
    f = np.zeros_like(w_old)
    ratio = 1
    k = 0

    while ratio > tol and k < n_iterations:
        k += 1
        w_new = np.copy(w_old)

        for i in range(1, len(w_old) - 1):
            for j in range(1, len(w_old) - 1):
                w_new[i, j] = (
                    w_new[i + 1, j]
                    + w_new[i - 1, j]
                    + w_new[i, j - 1]
                    + w_new[i, j + 1]
                    + f[i, j] * h**2
                ) / 4

        ratio = np.max(np.abs(w_new - w_old))
        w_old = w_new

    return w_old


def SOR(h, omega=1.5, n_iterations=10000, tol=1e-8):
    """
    Solves the heat equation using the Successive Over-Relaxation (SOR) method.
    """
    w_old = initialize_matrix(h)
    f = np.zeros_like(w_old)
    ratio = 1
    k = 0

    while ratio > tol and k < n_iterations:
        k += 1
        w_new = np.copy(w_old)

        for i in range(1, len(w_old) - 1):
            for j in range(1, len(w_old) - 1):
                new_value = (
                    w_new[i + 1, j]
                    + w_new[i - 1, j]
                    + w_new[i, j - 1]
                    + w_new[i, j + 1]
                    + f[i, j] * h**2
                ) / 4
                w_new[i, j] = omega * new_value + (1 - omega) * w_old[i, j]

        ratio = np.max(np.abs(w_new - w_old))
        w_old = w_new

    return w_old


def true_solution(n_points=1000):
    """
    Plots the true solution of the heat equation.
    """
    x = np.linspace(0, 1, n_points)
    y = np.linspace(0, 1, n_points)
    X, Y = np.meshgrid(x, y)
    Z = phi(X, Y)

    plt.contourf(X, Y, Z, cmap="hot")
    plt.colorbar()
    plt.title("True Solution")


def plot_solution(method, h, n_points, index, num_rows, num_cols):
    """
    Plots the solution for a given method and grid spacing.
    """
    w = method(h)
    x = np.linspace(0, 1, n_points)
    y = np.linspace(0, 1, n_points)
    X, Y = np.meshgrid(x, y)

    plt.subplot(num_rows, num_cols, index)
    plt.contourf(X, Y, w, cmap="hot")
    plt.colorbar()
    plt.title(f"{method.__name__.capitalize()} Solution with h={h}")


def main():
    H_LIST = [1 / 2, 1 / 4, 1 / 8]
    methods = [jacobi, gauss_seidel, SOR]
    num_methods = len(methods)

    plt.figure(figsize=(20, 10))
    num_cols = len(H_LIST) + 1
    num_rows = num_methods

    # Plot solutions for each method
    for row, method in enumerate(methods, start=1):
        for col, h in enumerate(H_LIST):
            index = (row - 1) * num_cols + col + 1
            plt.subplot(num_rows, num_cols, index)
            plot_solution(method, h, int(1/h) + 1, index, num_rows, num_cols)
            if col == 0:
                plt.ylabel(f'{method.__name__.capitalize()}')

    # Plot the true solution in the last column
    for row in range(1, num_rows + 1):
        plt.subplot(num_rows, num_cols, row * num_cols)
        true_solution(n_points=int(1/H_LIST[-1]) + 1)
        if row == 1:
            plt.title('True Solution')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()


