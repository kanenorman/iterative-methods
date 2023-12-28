import sys

import matplotlib.pyplot as plt
import numpy as np
import scienceplots

plt.style.use(["science"])


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


def solve_heat_equation(h, update_strategy, n_iterations=10000, tol=1e-8):
    """
    Solves the heat equation using a general iterative method.
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
                new_value = update_strategy(w_old, w_new, i, j, h, f)
                w_new[i, j] = new_value

        ratio = np.max(np.abs(w_new - w_old))
        w_old = w_new

    return w_old


def jacobi_update(w_old, w_new, i, j, h, f):
    return (
        w_old[i + 1, j]
        + w_old[i - 1, j]
        + w_old[i, j - 1]
        + w_old[i, j + 1]
        + f[i, j] * h**2
    ) / 4


def gauss_seidel_update(w_old, w_new, i, j, h, f):
    return (
        w_new[i + 1, j]
        + w_new[i - 1, j]
        + w_new[i, j - 1]
        + w_new[i, j + 1]
        + f[i, j] * h**2
    ) / 4


def sor_update(w_old, w_new, i, j, h, f):
    omega = 2 / (1 + np.sqrt(1 - np.cos(np.pi * h) ** 2))
    new_value = (
        w_new[i + 1, j]
        + w_new[i - 1, j]
        + w_new[i, j - 1]
        + w_new[i, j + 1]
        + f[i, j] * h**2
    ) / 4
    return omega * new_value + (1 - omega) * w_old[i, j]


def plot_solution(method, h, first_row=False):
    """
    Plots the solution for a given method and grid spacing.
    """
    w = method(h)
    n_points = w.shape[0]
    x = np.linspace(0, 1, n_points)
    y = np.linspace(0, 1, n_points)
    X, Y = np.meshgrid(x, y)

    contour = plt.contourf(
        X, Y, w, cmap="hot", levels=np.linspace(np.min(w), np.max(w), num=20)
    )
    if first_row:
        plt.title(f"h={h}")

    return contour


def true_solution_plot(n_points, first_row=False):
    """
    Plots the true solution of the heat equation.
    """
    x = np.linspace(0, 1, n_points)
    y = np.linspace(0, 1, n_points)
    X, Y = np.meshgrid(x, y)
    Z = phi(X, Y)

    contour = plt.contourf(
        X, Y, Z, cmap="hot", levels=np.linspace(np.min(Z), np.max(Z), num=20)
    )

    if first_row:
        plt.title("True Solution")

    return contour


def main() -> int:
    h_list = [1 / 2, 1 / 4, 1 / 8]
    methods = {
        lambda h: solve_heat_equation(h, jacobi_update): "Jacobi",
        lambda h: solve_heat_equation(h, gauss_seidel_update): "Gauss-Seidel",
        lambda h: solve_heat_equation(
            h,
            sor_update,
        ): "Successive Over Relaxation",
    }
    num_methods = len(methods)

    plt.figure(figsize=(20, 10))
    num_cols = len(h_list) + 1  # Additional column for the True Solution
    num_rows = num_methods

    # Plot solutions for each method
    for row, method in enumerate(methods.keys(), start=1):
        for col, h in enumerate(h_list):
            first_row = row == 1
            plt.subplot(num_rows, num_cols, (row - 1) * num_cols + col + 1)
            plot_solution(method, h, first_row)
            if col == 0:
                plt.ylabel(methods[method])

    # Plot the true solution in the last column
    for row in range(1, num_rows + 1):
        first_row = row == 1
        plt.subplot(num_rows, num_cols, row * num_cols)
        n_points = int(1 / h_list[-1]) + 1
        true_solution_plot(n_points, first_row)

    plt.suptitle(
        r"Approximate Solutions to $\phi(x,y) = sin(\pi y)  e^{\pi x}$",
        fontsize="xx-large",
    )
    plt.tight_layout()
    plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
