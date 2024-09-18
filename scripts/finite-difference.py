"""Solve a model wave equation problem with Neumann boundary conditions
using the finite difference method.

If run as a script, the result is plotted. This file can also be
imported as a module and convergence tests run on the solver.
"""
from argparse import ArgumentParser
from alive_progress import alive_bar
import numpy as np

from wave_solvers import FiniteDifferenceWaveEquation
from wave_solvers.utils import bump_function, cosine

import matplotlib.pyplot as plt


if __name__ == "__main__":

    parser = ArgumentParser(
        description="""Solve a Poisson problem on the unit square.""")
    parser.add_argument("--analytic", action="store_true",
                        help="Plot the analytic solution instead of solving the finite element problem.")
    parser.add_argument("--error", action="store_true",
                        help="Plot the error instead of the solution.")
    parser.add_argument("max_step", type=int, nargs=1,
                        help="The maximum number of time steps.")
    parser.add_argument("time_step", type=float, nargs=1,
                        help="The time step.")
    parser.add_argument("wave_speed", type=float, nargs=1,
                        help="The wave speed.")
    parser.add_argument("resolution", type=int, nargs=1,
                        help="The number of grid points in each direction on the mesh.")
    args = parser.parse_args()
    max_step = args.max_step[0]
    time_step = args.time_step[0]
    wave_speed = args.wave_speed[0]
    resolution = args.resolution[0]
    analytic = args.analytic
    plot_error = args.error

    # Generate the initial conditions.
    if plot_error:
        u_0 = lambda x, y: cosine(x, y, wave_speed)
    else:
        u_0 = lambda x, y: bump_function(x, y, 0.5, 0.5, 0.1)
    u_1 = u_0

    # Create solver object.
    solver = FiniteDifferenceWaveEquation(resolution, wave_speed, time_step, u_0, u_1)
    print(f"CFL number is {solver.cfl}/{1 / np.sqrt(2)}.")

    if plot_error:
        # Iterate numerical solution.
        with alive_bar(max_step, title="Iterating...") as bar: 
            solver.advance(max_step, bar)
        (xx, yy), values = solver.evaluate()

        # Evaluate the exact solution.
        final_time = solver.time_step * time_step
        exact = cosine(xx, yy, wave_speed, final_time)

        # Compute the mean squared error.
        error = np.sqrt(np.mean((values - exact)**2))
        print(f"Root mean squared error is {error}.")

        # Plot the error.
        plt.figure()

        plt.pcolormesh(xx, yy, values - exact, shading="auto")

        plt.colorbar()

        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Error in the numerical solution")

        plt.show()

    else:
        # solver.animate(frames=max_step)
        solver.cplot(frames=max_step)
