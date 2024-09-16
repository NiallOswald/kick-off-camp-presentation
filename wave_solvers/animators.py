"""Module for animating wave equation solutions."""

from alive_progress import alive_bar

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.tri import Triangulation

plt.rcParams.update({'font.size': 12})


class WaveAnimation:
    """Produce an animation of the wave equation solutions."""

    def __init__(self, solver, subdivisions=None):
        """Initialise the wave equation animation.

        :param solver: The wave equation solver.
        """
        self.solver = solver
        self.subdivisions = subdivisions

    def __call__(self, **kwargs):
        """Produce an animation of the wave equation solutions."""
        coords, values, triangles = self.solver.evaluate()

        fig = plt.figure()
        self.ax = fig.add_subplot(projection="3d")

        self.plots = []
        for c in range(coords.shape[0]):
            self.plots.append(self.ax.plot_trisurf(
                Triangulation(coords[c, :, 0], coords[c, :, 1], triangles),
                              values[c, :], linewidth=0)
            )

        self.ax.set_xlabel(r"$x$")
        self.ax.set_ylabel(r"$y$")

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
            ani.save("../figures/wave_2d.gif", writer=writer)
    
    def _animate(self, k, bar=lambda: None):
        """Advance the wave equation up to time step k."""
        for _ in range(k - self.solver.time_step):
            self.solver.step()
        bar()

        coords, values, triangles = self.solver.evaluate(self.subdivisions)

        for i in range(len(self.plots)):
            self.plots[i].remove()
            self.plots[i] = self.ax.plot_trisurf(
                Triangulation(coords[i, :, 0], coords[i, :, 1], triangles),
                              values[i, :], linewidth=0
            )

        return self.plots
