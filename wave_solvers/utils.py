"""Additional functions for the wave solvers."""

import numpy as np


def bump_function(x, y, x0, y0, radius=1.0):
    """A bump function centred at x0 with a given radius."""
    val = (x - x0)**2 + (y - y0)**2 - radius**2
    return np.where(val < 0, np.exp(1 + radius**2 / val), 0.0)
