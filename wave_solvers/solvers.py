"""Solver classes for the wave equation."""

from abc import ABC, abstractmethod
from alive_progress import alive_bar
import numpy as np
import scipy.sparse as sp
from fe_utils import LagrangeElement, FunctionSpace, UnitSquareMesh, Function, gauss_quadrature
from wave_solvers.utils import boundary_nodes

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors
from matplotlib.tri import Triangulation

plt.rcParams.update({'font.size': 12})


class WaveEquation(ABC):
    """A base class for wave equation solvers."""

    def __init__(self, c, dt, u_0, u_1, boundary_conditions="neumann"):
        """Initialise the wave equation solver.
        
        :param c: The wave speed.
        :param dt: The time step.
        :param u_0: The previous state of the wave equation.
        :param u_1: The current state of the wave equation.
        """
        self.c = c
        self.dt = dt
        self.u_0, self.u_1 = u_0, u_1
        self.boundary_conditions = boundary_conditions

        self.time_step = 0

    @abstractmethod
    def step(self):
        """Advance the wave equation by one time step."""
        self.time_step += 1
    
    @abstractmethod
    def evaluate(self, x):
        """Evaluate the wave equation solution at collection of points."""
        raise NotImplementedError
    
    @abstractmethod
    def animate(self, **kwargs):
        """Produce an animation of the wave equation solutions."""
        raise NotImplementedError
    
    @abstractmethod
    def _animate(self, k, bar=lambda: None):
        """Advance the wave equation up to time step k."""
        raise NotImplementedError
    
    @abstractmethod
    def _neumann(self):
        """Apply Neumann boundary conditions to the solution."""
        raise NotImplementedError
    
    @abstractmethod
    def _dirichlet(self):
        """Apply Dirichlet boundary conditions to the solution."""
        raise NotImplementedError
    
    def advance(self, k, bar=lambda: None):
        """Advance the wave equation up to time step k."""
        for _ in range(k - self.time_step):
            self.step()
            bar()


class FiniteElementWaveEquation(WaveEquation):
    """A finite element solver for the wave equation."""

    def __init__(self, resolution, degree, c, dt, u_0, u_1, boundary_conditions="neumann"):
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

        super().__init__(c, dt, fs_u_0, fs_u_1, boundary_conditions)

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

        # Apply the boundary conditions.
        if self.boundary_conditions == "dirichlet":
            self._dirichlet(A, l)

        # Create the function to hold the solution.
        u = Function(self.fs)

        # Cast the matrix to a sparse format and use a sparse solver for the
        # linear system. This is vastly faster than the dense alternative.
        A = sp.csr_matrix(A)
        u.values[:] = sp.linalg.spsolve(A, l)

        # Update the time step.
        self.u_0.values[:] = self.u_1.values
        self.u_1.values[:] = u.values

        super().step()

    def evaluate(self, *args, **kwargs):
        """Evaluate the wave equation solution at collection of points."""
        return self.u_1.evaluate(*args, **kwargs)
    
    def _neumann(self):
        """Apply Neumann boundary conditions to the solution."""
        pass
    
    def _dirichlet(self, A, l):
        """Apply Dirichlet boundary conditions to the solution."""
        boundary = boundary_nodes(self.fs)

        A[boundary, :] = 0
        A[boundary, boundary] = 1
        l[boundary] = 0

    def animate(self, subdivisions=None, **kwargs):
        """Produce an animation of the wave equation solutions."""
        coords, values, triangles = self.evaluate(return_triangles=True)

        fig = plt.figure()
        self.ax = fig.add_subplot(projection="3d")

        self.plots = []
        for c in range(coords.shape[0]):
            self.plots.append(self.ax.plot_trisurf(
                Triangulation(coords[c, :, 0], coords[c, :, 1], triangles),
                              values[c, :], linewidth=0),
            )

        self.ax.set_xlabel(r"$x$")
        self.ax.set_ylabel(r"$y$")
        self.ax.set_zlabel(r"$\eta$")

        self.ax.set_zlim(-1, 1)

        fig.tight_layout()

        if isinstance(kwargs["frames"], int):
            frame_count = kwargs["frames"]
        else:
            frame_count = len(kwargs["frames"])

        with alive_bar(frame_count, title="Generating frames...") as bar:
            ani = animation.FuncAnimation(
                fig, self._animate, init_func=lambda: None, fargs=(subdivisions, bar,),
                **kwargs
            )

            writer = animation.PillowWriter(fps=15)
            ani.save("../figures/fem-surf.gif", writer=writer)
    
    def _animate(self, k, subdivisions, bar=lambda: None):
        """Advance the wave equation up to time step k."""
        self.advance(k)
        bar()

        coords, values, triangles = self.evaluate(subdivisions, return_triangles=True)

        for i in range(len(self.plots)):
            self.plots[i].remove()
            self.plots[i] = self.ax.plot_trisurf(
                Triangulation(coords[i, :, 0], coords[i, :, 1], triangles),
                              values[i, :], linewidth=0,
            )

        return self.plots
    
    def cplot(self, **kwargs):
        fig, self.ax = plt.subplots()

        coords, values, triangles = self.evaluate(return_triangles=True)

        norm = colors.SymLogNorm(linthresh=1e-2, linscale=1.0,
                                 vmin=-1.0, vmax=1.0, base=10)
        self.plots = []
        for c in range(coords.shape[0]):
            self.plots.append(self.ax.tripcolor(
                coords[c, :, 0], coords[c, :, 1], triangles, values[c, :],
                norm=norm, cmap='RdBu_r', shading='gouraud'
            ))
        fig.colorbar(self.plots[0], ax=self.ax, extend='both', label=r"$u$")

        self.ax.set_xlabel(r"$x$")
        self.ax.set_ylabel(r"$y$")

        fig.tight_layout()

        if isinstance(kwargs["frames"], int):
            frame_count = kwargs["frames"]
        else:
            frame_count = len(kwargs["frames"])
        
        with alive_bar(frame_count, title="Generating frames...") as bar:
            ani = animation.FuncAnimation(
                fig, self._cplot, init_func=lambda: None, fargs=(norm, bar,),
                **kwargs
            )

            writer = animation.PillowWriter(fps=15)
            ani.save("../figures/fem-cplot.gif", writer=writer)

    def _cplot(self, k, norm, bar=lambda: None):
        """Advance the wave equation up to time step k."""
        self.advance(k)
        bar()

        coords, values, triangles = self.evaluate(return_triangles=True)

        for i in range(len(self.plots)):
            self.plots[i].remove()
            self.plots[i] = self.ax.tripcolor(
                coords[i, :, 0], coords[i, :, 1], triangles, values[i, :],
                norm=norm, cmap='RdBu_r', shading='gouraud'
            )
        
        return self.plots


class FiniteDifferenceWaveEquation(WaveEquation):
    """A finite difference solver for the wave equation."""

    def __init__(self, n, c, dt, u_0, u_1, boundary_conditions="neumann"):
        """Initialise the finite difference wave equation solver.
        
        :param n: The number of grid points in each direction.
        :param c: The wave speed.
        :param dt: The time step.
        :param u_0: The previous state of the wave equation.
        :param u_1: The current state of the wave equation.
        """
        self.n = n

        # Compute the spatial step size.
        self.dx = 1 / (n - 1)

        # Create the grid.
        x = np.linspace(0, 1, n)
        self.xx, self.yy = np.meshgrid(x, x)

        # Create the initial conditions.
        u_0 = u_0(self.xx, self.yy)
        u_1 = u_1(self.xx, self.yy)

        np.nan_to_num(u_0, copy=False)
        np.nan_to_num(u_1, copy=False)

        # Compute the CFL number.
        self.cfl = c * dt / self.dx

        super().__init__(c, dt, u_0, u_1, boundary_conditions)

    def step(self):
        """Advance the wave equation by one time step."""
        # Update the interior of the domain.
        u = np.zeros_like(self.u_1)
        u[1:-1, 1:-1] = (self.c * self.dt / self.dx)**2 * (
            self.u_1[1:-1, :-2] + self.u_1[1:-1, 2:] + self.u_1[:-2, 1:-1] + self.u_1[2:, 1:-1] - 4 * self.u_1[1:-1, 1:-1]
        ) + 2 * self.u_1[1:-1, 1:-1] - self.u_0[1:-1, 1:-1]

        # Update the boundary conditions.
        if self.boundary_conditions == "neumann":
            self._neumann(u)
        elif self.boundary_conditions == "dirichlet":
            self._dirichlet(u)

        self.u_0, self.u_1 = self.u_1, u

        super().step()
    
    def evaluate(self):
        """Evaluate the wave equation solution at collection of points."""
        return (self.xx, self.yy), self.u_1
    
    def _neumann(self, u):
        """Apply Neumann boundary conditions to the solution."""
        # Apply the edge conditions.
        u[0, 1:-1] = (4 * u[1, 1:-1] - u[2, 1:-1]) / 3
        u[-1, 1:-1] = (4 * u[-2, 1:-1] - u[-3, 1:-1]) / 3
        u[1:-1, 0] = (4 * u[1:-1, 1] - u[1:-1, 2]) / 3
        u[1:-1, -1] = (4 * u[1:-1, -2] - u[1:-1, -3]) / 3

        # Apply the corner conditions.
        u[0, 0] = (u[0, 1] + u[1, 0]) / 2
        u[0, -1] = (u[0, -2] + u[1, -1]) / 2
        u[-1, 0] = (u[-1, 1] + u[-2, 0]) / 2
        u[-1, -1] = (u[-1, -2] + u[-2, -1]) / 2
    
    def _dirichlet(self, u):
        """Apply Dirichlet boundary conditions to the solution."""
        u[0, :] = 0
        u[-1, :] = 0
        u[:, 0] = 0
        u[:, -1] = 0

    def animate(self, **kwargs):
        """Produce an animation of the wave equation solutions."""
        (xx, yy), values = self.evaluate()

        fig = plt.figure()
        self.ax = fig.add_subplot(projection="3d")

        self.plots = [self.ax.plot_surface(xx, yy, values, cmap="viridis")]

        self.ax.set_xlabel(r"$x$")
        self.ax.set_ylabel(r"$y$")
        self.ax.set_zlabel(r"$\eta$")

        self.ax.set_zlim(-1, 1)

        fig.tight_layout()

        if isinstance(kwargs["frames"], int):
            frame_count = kwargs["frames"]
        else:
            frame_count = len(kwargs["frames"])

        with alive_bar(frame_count, title="Generating frames...") as bar:
            ani = animation.FuncAnimation(
                fig, self._animate, init_func=lambda: None, fargs=(bar,),
                **kwargs
            )

            writer = animation.PillowWriter(fps=15)
            ani.save("../figures/fdm-surf.gif", writer=writer)
    
    def _animate(self, k, bar=lambda: None):
        """Advance the wave equation up to time step k."""
        self.advance(k)
        bar()

        (xx, yy), values = self.evaluate()

        self.plots[0].remove()
        self.plots[0] = self.ax.plot_surface(xx, yy, values, cmap="viridis")

        return self.plots
    
    def cplot(self, **kwargs):
        fig, self.ax = plt.subplots()

        norm = colors.SymLogNorm(linthresh=1e-2, linscale=1.0,
                                 vmin=-1.0, vmax=1.0, base=10)
        self.plots = [self.ax.pcolormesh(self.xx, self.yy, self.u_1, norm=norm,
                                         cmap='RdBu_r', shading='auto')]
        fig.colorbar(self.plots[0], ax=self.ax, extend='both', label=r"$u$")

        self.ax.set_xlabel(r"$x$")
        self.ax.set_ylabel(r"$y$")

        fig.tight_layout()

        if isinstance(kwargs["frames"], int):
            frame_count = kwargs["frames"]
        else:
            frame_count = len(kwargs["frames"])

        with alive_bar(frame_count, title="Generating frames...") as bar:
            ani = animation.FuncAnimation(
                fig, self._cplot, init_func=lambda: None, fargs=(norm, bar,),
                **kwargs
            )

            writer = animation.PillowWriter(fps=15)
            ani.save("../figures/fdm-cplot.gif", writer=writer)

    def _cplot(self, k, norm, bar=lambda: None):
        self.advance(k)
        bar()

        (xx, yy), values = self.evaluate()

        self.plots[0].remove()
        self.plots[0] = self.ax.pcolormesh(xx, yy, values, norm=norm,
                                           cmap='RdBu_r', shading='auto')
        
        return self.plots

