import sys

import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from matplotlib.animation import FuncAnimation

plt.style.use(["science"])


def phi(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Represents the heat equation Φ(x,y) = sin(πx) * e^(πy).
    """
    return np.sin(np.pi * x) * np.exp(np.pi * y)


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


def generic_plot(X, Y, Z, first_row, title):
    """
    A generic plot function for contour plots.
    """
    contour = plt.contourf(
        X, Y, Z, cmap="hot", levels=np.linspace(np.min(Z), np.max(Z), num=20)
    )
    if first_row:
        plt.title(title)

    return contour


def approximate_solution_plot(method, h, first_row=False):
    """
    Plots the solution for a given method and grid spacing.
    """
    w = method(h)
    n_points = w.shape[0]
    x = np.linspace(0, 1, n_points)
    y = np.linspace(0, 1, n_points)
    X, Y = np.meshgrid(x, y)

    return generic_plot(X, Y, w, first_row, f"h={h}")


def true_solution_plot(n_points, first_row=False):
    """
    Plots the true solution of the heat equation.
    """
    x = np.linspace(0, 1, n_points)
    y = np.linspace(0, 1, n_points)
    X, Y = np.meshgrid(x, y)
    Z = phi(X, Y)

    return generic_plot(X, Y, Z, first_row, "True Solution")


def generic_plot_3d(ax, X, Y, Z, first_row, title):
    """
    A generic plot function for 3D surface plots.
    """
    surface = ax.plot_surface(X, Y, Z, cmap="hot", rstride=1, cstride=1, alpha=0.8)
    if first_row:
        ax.set_title(title)

    return surface


def approximate_solution_plot_3d(ax, method, h, first_row=False):
    """
    Plots the solution for a given method and grid spacing in 3D.
    """
    w = method(h)
    n_points = w.shape[0]
    x = np.linspace(0, 1, n_points)
    y = np.linspace(0, 1, n_points)
    X, Y = np.meshgrid(x, y)

    return generic_plot_3d(ax, X, Y, w, first_row, f"h={h}")


def true_solution_plot_3d(ax, n_points, first_row=False):
    """
    Plots the true solution of the heat equation in 3D.
    """
    x = np.linspace(0, 1, n_points)
    y = np.linspace(0, 1, n_points)
    X, Y = np.meshgrid(x, y)
    Z = phi(X, Y)

    return generic_plot_3d(ax, X, Y, Z, first_row, "True Solution")


def animate_solutions(h, file_name, max_iterations=1000, tol=1e-2):
    """
    Animates the solution process for all methods and saves it to a file.
    """
    w_jacobi = initialize_matrix(h)
    w_gauss_seidel = initialize_matrix(h)
    w_sor = initialize_matrix(h)
    n_points = w_jacobi.shape[0]
    x = np.linspace(0, 1, n_points)
    y = np.linspace(0, 1, n_points)
    X, Y = np.meshgrid(x, y)
    Z_true = phi(X, Y)

    # Iteration counters
    iter_jacobi = iter_gauss_seidel = iter_sor = 0
    # Previous errors to check for convergence
    prev_error_jacobi = prev_error_gauss_seidel = prev_error_sor = np.inf

    fig = plt.figure(figsize=(18, 6))

    ax_jacobi = fig.add_subplot(1, 3, 1, projection="3d")
    ax_jacobi.title.set_text("Iteration: 0")
    ax_gauss_seidel = fig.add_subplot(1, 3, 2, projection="3d")
    ax_gauss_seidel.title.set_text("Iteration: 0")
    ax_sor = fig.add_subplot(1, 3, 3, projection="3d")
    ax_sor.title.set_text("Iteration: 0")

    error_texts = [
        fig.text(0.175 + 0.3 * i, 0.02, "", transform=fig.transFigure) for i in range(3)
    ]

    def update(frame):
        nonlocal w_jacobi, w_gauss_seidel, w_sor
        nonlocal iter_jacobi, iter_gauss_seidel, iter_sor
        nonlocal prev_error_jacobi, prev_error_gauss_seidel, prev_error_sor

        # Update Jacobi method and check for convergence
        if iter_jacobi < max_iterations:
            w_jacobi, error_jacobi = solve_heat_equation_step(
                w_jacobi, jacobi_update, h, Z_true
            )
            if np.abs(error_jacobi - prev_error_jacobi) > tol:
                prev_error_jacobi = error_jacobi
                iter_jacobi += 1

        # Update Gauss-Seidel method and check for convergence
        if iter_gauss_seidel < max_iterations:
            w_gauss_seidel, error_gauss_seidel = solve_heat_equation_step(
                w_gauss_seidel, gauss_seidel_update, h, Z_true
            )
            if np.abs(error_gauss_seidel - prev_error_gauss_seidel) > tol:
                prev_error_gauss_seidel = error_gauss_seidel
                iter_gauss_seidel += 1

        # Update SOR method and check for convergence
        if iter_sor < max_iterations:
            w_sor, error_sor = solve_heat_equation_step(w_sor, sor_update, h, Z_true)
            if np.abs(error_sor - prev_error_sor) > tol:
                prev_error_sor = error_sor
                iter_sor += 1

        # Update plot titles and error texts
        for ax, w, method, iteration, error in zip(
            [ax_jacobi, ax_gauss_seidel, ax_sor],
            [w_jacobi, w_gauss_seidel, w_sor],
            ["Jacobi", "Gauss-Seidel", "SOR"],
            [iter_jacobi, iter_gauss_seidel, iter_sor],
            [prev_error_jacobi, prev_error_gauss_seidel, prev_error_sor],
        ):
            ax.clear()
            ax.plot_surface(X, Y, w, cmap="hot")
            ax.set_zlim(0, np.exp(np.pi))
            ax.title.set_text(f"Iteration: {iteration}")
            error_texts[["Jacobi", "Gauss-Seidel", "SOR"].index(method)].set_text(
                f"{method} Error: {error:.8e}"
            )

    anim = FuncAnimation(fig, update, frames=max_iterations, interval=50)
    anim.save(f"plots/{file_name}", writer="imagemagick")


def solve_heat_equation_step(w_old, update_strategy, h, Z_true):
    """
    Performs a single step of the heat equation solution using the specified update strategy.
    """
    w_new = np.copy(w_old)
    for i in range(1, len(w_old) - 1):
        for j in range(1, len(w_old) - 1):
            w_new[i, j] = update_strategy(w_old, w_new, i, j, h, np.zeros_like(w_old))
    error = np.max(np.abs(w_new - Z_true))
    return w_new, error


def main() -> int:
    h = 1 / 16  # Choose a fixed h for animation
    animate_solutions(h, "heat_equation_animation.gif")

    h_list = [1 / 2**x for x in range(1, 6)]  # 1/2, 1/4, ...1/64
    methods = {
        lambda h: solve_heat_equation(h, jacobi_update): "Jacobi",
        lambda h: solve_heat_equation(h, gauss_seidel_update): "Gauss-Seidel",
        lambda h: solve_heat_equation(h, sor_update): "Successive Over Relaxation",
    }
    num_methods = len(methods)
    num_cols = len(h_list) + 1  # Additional column for the True Solution

    # Create 2D plot figure
    plt.figure(figsize=(25, 10))
    for row, method in enumerate(methods.keys(), start=1):
        for col, h in enumerate(h_list):
            first_row = row == 1
            plt.subplot(num_methods, num_cols, (row - 1) * num_cols + col + 1)
            approximate_solution_plot(method, h, first_row)
            if col == 0:
                plt.ylabel(methods[method])

    # Plot True Solution in 2D
    for row in range(1, num_methods + 1):
        first_row = row == 1
        plt.subplot(num_methods, num_cols, row * num_cols)
        n_points = int(1 / h_list[-1]) + 1
        true_solution_plot(n_points, first_row)

    plt.suptitle(
        r"2D Contour Plots: Approximate Solutions to $\phi(x,y) = sin(\pi x)  e^{\pi y}$",
        fontsize="xx-large",
    )
    plt.tight_layout()
    plt.savefig("plots/2d_contour.png")

    # Create 3D plot figure
    plt.figure(figsize=(20, 10))
    for row, method in enumerate(methods.keys(), start=1):
        for col, h in enumerate(h_list):
            first_row = row == 1
            ax = plt.subplot(
                num_methods, num_cols, (row - 1) * num_cols + col + 1, projection="3d"
            )
            approximate_solution_plot_3d(ax, method, h, first_row)

            # Add vertical labels for each row
            if col == 0:
                ax.text2D(
                    -0.1,
                    0.5,
                    methods[method],
                    transform=ax.transAxes,
                    rotation=90,
                    verticalalignment="center",
                    horizontalalignment="left",
                    fontsize=10,
                )

    # Plot True Solution in 3D
    for row in range(1, num_methods + 1):
        first_row = row == 1
        ax = plt.subplot(num_methods, num_cols, row * num_cols, projection="3d")
        n_points = int(1 / h_list[-1]) + 1
        true_solution_plot_3d(ax, n_points, first_row)

    plt.suptitle(
        r"3D Surface Plots: Approximate Solutions to $\phi(x,y) = sin(\pi x)  e^{\pi y}$",
        fontsize="xx-large",
    )
    plt.tight_layout(pad=3.0)
    plt.savefig("plots/3d_surface.png")

    return 0


if __name__ == "__main__":
    sys.exit(main())
