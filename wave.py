"""Solve a model wave equation problem with Neumann boundary conditions
using the finite element method.

If run as a script, the result is plotted. This file can also be
imported as a module and convergence tests run on the solver.
"""
from __future__ import division
from fe_utils import *
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from argparse import ArgumentParser
from alive_progress import alive_bar


def bump_function(x, x0, radius=1.0):
    """A bump function centred at x0 with a given radius."""

    if np.linalg.norm(x - x0) < radius:
        return np.exp(1 + radius**2 / (np.linalg.norm(x - x0)**2 - radius**2))
    else:
        return 0.0


def assemble(fs, f, wave_speed, dt):
    """Assemble the finite element system for the Helmholtz problem given
    the function space in which to solve and the right hand side
    function."""

    fe = fs.element
    mesh = fs.mesh

    # Create an appropriate (complete) quadrature rule.
    Q = gauss_quadrature(fe.cell, 2 * fe.degree)

    # Tabulate the basis functions and their gradients at the quadrature points.
    phi = fe.tabulate(Q.points)
    grad_phi = fe.tabulate(Q.points, grad=True)

    # Create the left hand side matrix and right hand side vector.
    # This creates a sparse matrix because creating a dense one may
    # well run your machine out of memory!
    A = sp.lil_matrix((fs.node_count, fs.node_count))
    l = np.zeros(fs.node_count)  # noqa: E741

    # Now loop over all the cells and assemble A and l
    for c in range(mesh.entity_counts[-1]):
        J = mesh.jacobian(c)
        invJ = np.linalg.inv(J)
        detJ = abs(np.linalg.det(J))

        nodes = fs.cell_nodes[c, :]

        A[np.ix_(nodes, nodes)] += np.einsum(
            "ba,qib,ya,qjy,q->ij", invJ, grad_phi, invJ,
            grad_phi, Q.weights, optimize=True
        ) * detJ * wave_speed**2 * dt**2
        A[np.ix_(nodes, nodes)] += np.einsum(
            "qi,qj,q->ij", phi, phi, Q.weights, optimize=True
        ) * detJ

        l[nodes] += np.einsum(
            "qi,k,qk,q->i", phi, f.values[nodes], phi, Q.weights, optimize=True
        ) * detJ

    return A, l


def solve_wave(degree, resolution, wave_speed, time_step, max_step, analytic=False, return_error=False):
    """Solve a model wave equation problem on a unit square mesh with
    ``resolution`` elements in each direction, using equispaced
    Lagrange elements of degree ``degree``."""

    # Set up the mesh, finite element and function space required.
    mesh = UnitSquareMesh(resolution, resolution)
    fe = LagrangeElement(mesh.cell, degree)
    fs = FunctionSpace(mesh, fe)

    # Create a function to hold the analytic solution for comparison purposes.
    analytic_answer = Function(fs)
    analytic_answer.interpolate(lambda x: 0.)  # TODO: Add analytic solution

    # If the analytic answer has been requested then bail out now.
    if analytic:
        return analytic_answer, 0.0

    # Create the initial conditions for the wave equation.
    u0 = Function(fs)
    u0.interpolate(lambda x: bump_function(x, np.array([0.5, 0.5]), 0.1))
    u0.values[:] = np.nan_to_num(u0.values, 0.0)

    u1 = Function(fs)
    u1.values[:] = u0.values

    with alive_bar(max_step) as bar:
        for step in range(max_step):
            # Create the right hand side function and populate it with the
            # correct values.
            f = Function(fs)
            f.values[:] = 2*u1.values - u0.values

            # Assemble the finite element system.
            A, l = assemble(fs, f, wave_speed, time_step)

            # Create the function to hold the solution.
            u = Function(fs)

            # Cast the matrix to a sparse format and use a sparse solver for
            # the linear system. This is vastly faster than the dense
            # alternative.
            A = sp.csr_matrix(A)
            u.values[:] = splinalg.spsolve(A, l)

            # Update the time step.
            u0.values[:] = u1.values
            u1.values[:] = u.values

            # Update the progress bar.
            bar()

    # Compute the L^2 error in the solution for testing purposes.
    error = errornorm(analytic_answer, u)

    if return_error:
        u.values -= analytic_answer.values

    # Return the solution and the error in the solution.
    return u, error


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

    u, error = solve_wave(degree, resolution, wave_speed, time_step, max_step, analytic, plot_error)

    u.plot()
