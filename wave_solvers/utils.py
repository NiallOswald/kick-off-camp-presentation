"""Additional functions for the wave solvers."""

import numpy as np
from fe_utils import Function


def bump_function(x, y, x0, y0, radius=1.0):
    """A bump function centred at x0 with a given radius."""
    val = (x - x0)**2 + (y - y0)**2 - radius**2
    return np.where(val < 0, np.exp(1 + radius**2 / val), 0.0)


def cosine(x, y, c=1.0, t=0.0):
    """A cosine function with a given period."""
    return np.cos(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * c * t * np.sqrt(2))


def boundary_nodes(fs):
    """Find the list of boundary nodes in fs. This is a
    unit-square-specific solution. A more elegant solution would employ
    the mesh topology and numbering.
    """
    eps = 1.e-10

    f = Function(fs)

    def on_boundary(x):
        """Return 1 if on the boundary, 0. otherwise."""
        if x[0] < eps or x[0] > 1 - eps or x[1] < eps or x[1] > 1 - eps:
            return 1.
        else:
            return 0.

    f.interpolate(on_boundary)

    return np.flatnonzero(f.values)


def cheb(n):
    """Return the Chebyshev differentiation matrix of order n."""
    if n == 0:
        return 0, 1

    x = np.array([np.cos(np.pi * np.arange(n + 1) / n)]).T
    c = np.array(
        [np.concatenate([[2], np.ones(n - 1), [2]]) * (-1) ** np.arange(n + 1)]
    ).T
    X = np.tile(x, (1, n + 1))
    dX = X - X.T
    D = (c @ (1 / c).T) / (dX + np.eye(n + 1))
    D = D - np.diag(np.sum(D.T, axis=0))

    return D, x.T[0]
