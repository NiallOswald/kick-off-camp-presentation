"""Module for animating wave equation solutions."""

from alive_progress import alive_bar

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors

plt.rcParams.update({'font.size': 12})


class WaveAnimation:
    """Produce an animation of the wave equation solutions."""

    def __init__(self, solver, xx, yy):
        """Initialise the wave equation animation.

        :param solver: The wave equation solver.
        """
        self.solver = solver
        self.xx, self.yy = xx, yy

    def __call__(self, **kwargs):
        """Produce an animation of the wave equation solutions."""
        fig, ax = plt.subplots()

        norm = colors.SymLogNorm(linthresh=1e-2, linscale=1.0,
                                 vmin=-1.0, vmax=1.0, base=10)
        self.cplot = ax.pcolormesh(self.xx, self.yy,
                                   self.solver.evaluate(self.xx, self.yy),
                                   norm=norm, cmap='RdBu_r', shading='auto')
        fig.colorbar(self.cplot, ax=ax, extend='both', label=r"$u$")

        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")

        fig.tight_layout()

        if isinstance(kwargs["frames"], int):
            frame_count = kwargs["frames"]
        else:
            frame_count = len(kwargs["frames"])

        with alive_bar(frame_count, title="Generating plot...") as bar:
            ani = animation.FuncAnimation(
                fig, self._animate, init_func=lambda: None, fargs=(bar,),
                **kwargs
            )

            writer = animation.PillowWriter(fps=30)
            ani.save("figures/wave_2d.gif", writer=writer)  # , dpi=300)
    
    def _animate(self, k, bar=lambda: None):
        """Advance the wave equation up to time step k."""
        for _ in range(k - self.solver.time_step):
            self.solver.step()
        bar()

        self.cplot.set_array(self.solver.evaluate(self.xx, self.yy).flatten())

        return self.cplot
