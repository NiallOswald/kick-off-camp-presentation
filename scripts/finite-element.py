"""Solve a model wave equation problem with Neumann boundary conditions
using the finite element method.

If run as a script, the result is plotted. This file can also be
imported as a module and convergence tests run on the solver.
"""
from argparse import ArgumentParser
from alive_progress import alive_bar
import numpy as np

from wave_solvers import FiniteElementWaveEquation
from wave_solvers.utils import bump_function, cosine

import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation


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
                        help="The number of cells in each direction on the mesh.")
    parser.add_argument("degree", type=int, nargs=1,
                        help="The degree of the polynomial basis for the function space.")
    args = parser.parse_args()
    max_step = args.max_step[0]
    time_step = args.time_step[0]
    wave_speed = args.wave_speed[0]
    resolution = args.resolution[0]
    degree = args.degree[0]
    analytic = args.analytic
    plot_error = args.error

    # Generate the initial conditions.
    if plot_error:
        u_0 = lambda x: cosine(x[0], x[1], wave_speed)
    else:
        u_0 = lambda x: bump_function(x[0], x[1], 0.5, 0.5, 0.1)
    u_1 = u_0

    # Create solver object.
    solver = FiniteElementWaveEquation(resolution, degree, wave_speed, time_step, u_0, u_1)
    
    if plot_error:
        # Iterate numerical solution.
        with alive_bar(max_step, title="Iterating...") as bar: 
            solver.advance(max_step, bar)
        coords, values = solver.evaluate()

        coords, index = np.unique(coords, axis=0, return_index=True)
        values = values[index]

        # Evaluate the exact solution.
        final_time = solver.time_step * time_step
        exact = cosine(coords[:, 0], coords[:, 1], wave_speed, final_time)

        # Compute the mean squared error.
        error = np.sqrt(np.mean((values - exact)**2))
        print(f"Root mean squared error: {error}")

        # Plot the error.
        # Evaluate numerical solution.
        coords, values, triangles = solver.evaluate(return_triangles=True)

        # Evaluate the exact solution.
        exact = cosine(coords[:, :, 0], coords[:, :, 1], wave_speed, final_time)

        # Plot the error.
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        
        for c in range(coords.shape[0]):
            ax.plot_trisurf(Triangulation(coords[c, :, 0], coords[c, :, 1], triangles),
                                          values[c, :], color="blue", linewidth=0)
            
        for c in range(coords.shape[0]):
            ax.plot_trisurf(Triangulation(coords[c, :, 0], coords[c, :, 1], triangles),
                                          exact[c, :], color="red", linewidth=0)

        ax.set_title("Error")
        plt.show()
    
    else:
        solver.animate(frames=max_step)
        # solver.cplot(frames=max_step)
