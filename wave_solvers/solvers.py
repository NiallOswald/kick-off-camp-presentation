"""Solver classes for the wave equation."""

from abc import ABC, abstractmethod
import numpy as np
import scipy.sparse as sp
from fe_utils import LagrangeElement, FunctionSpace, UnitSquareMesh, Function, gauss_quadrature


class WaveEquation(ABC):
    """A base class for wave equation solvers."""

    def __init__(self, c, dt, u_0, u_1):
        """Initialise the wave equation solver.
        
        :param c: The wave speed.
        :param dt: The time step.
        :param u_0: The previous state of the wave equation.
        :param u_1: The current state of the wave equation.
        """
        self.c = c
        self.dt = dt
        self.u_0, self.u_1 = u_0, u_1

        self.time_step = 0

    @abstractmethod
    def step(self):
        """Advance the wave equation by one time step."""
        self.time_step += 1
    
    @abstractmethod
    def evaluate(self, x):
        """Evaluate the wave equation solution at collection of points."""
        raise NotImplementedError


class FiniteElementWaveEquation(WaveEquation):
    """A finite element solver for the wave equation."""

    def __init__(self, resolution, degree, c, dt, u_0, u_1):
        """Initialise the finite element wave equation solver.
        
        :param resolution: The number of cells in each direction.
        :param degree: The degree of the finite element.
        :param c: The wave speed.
        :param dt: The time step.
        :param u_0: The previous state of the wave equation.
        :param u_1: The current state of the wave equation.
        """
        # Set up the mesh, finite element and function space required.
        self.mesh = UnitSquareMesh(resolution, resolution)
        self.fe = LagrangeElement(self.mesh.cell, degree)
        self.fs = FunctionSpace(self.mesh, self.fe)

        # Replace initial conditions with functions from the function space
        fs_u_0 = Function(self.fs)
        fs_u_0.interpolate(u_0)
        fs_u_0.values[:] = np.nan_to_num(fs_u_0.values, 0.0)

        fs_u_1 = Function(self.fs)
        fs_u_1.interpolate(u_1)
        fs_u_1.values[:] = np.nan_to_num(fs_u_1.values, 0.0)

        super().__init__(c, dt, fs_u_0, fs_u_1)

    @staticmethod
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

    def step(self):
        """Advance the wave equation by one time step."""
        # Create the right hand side function and populate it with the correct
        # values.
        f = Function(self.fs)
        f.values[:] = 2 * self.u_1.values - self.u_0.values

        # Assemble the finite element system.
        A, l = self.assemble(self.fs, f, self.c, self.dt)

        # Create the function to hold the solution.
        u = Function(self.fs)

        # Cast the matrix to a sparse format and use a sparse solver for the
        # linear system. This is vastly faster than the dense alternative.
        A = sp.csr_matrix(A)
        u.values[:] = sp.spsolve(A, l)

        # Update the time step.
        self.u_0.values[:] = self.u_1.values
        self.u_1.values[:] = u.values

        super().step()

    def evaluate(self, x):
        """Evaluate the wave equation solution at collection of points."""
        return self.u_1.evaluate(x)


class FiniteDifferenceWaveEquation(WaveEquation):
    """A finite difference solver for the wave equation."""

    raise NotImplementedError


class SpectralWaveEquation(WaveEquation):
    """A spectral solver for the wave equation."""

    raise NotImplementedError
