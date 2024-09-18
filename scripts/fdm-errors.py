import numpy as np
import matplotlib.pyplot as plt

ERRORS = [0.0631933496201165, 0.02260175050457134, 0.010638979945212073, 0.0052985192602288496, 0.0026586684211154046, 0.001333307745832683, 0.0006678369458465305, 0.00033423714816136467]
RESOLUTIONS = [8, 16, 32, 64, 128, 256, 512, 1024]

# Plot the convergence of the finite difference method.
plt.loglog(RESOLUTIONS, ERRORS, "k-o")

# Plot the line of best fit and label the gradient.
m, c = np.polyfit(np.log(RESOLUTIONS[3:]), np.log(ERRORS[3:]), 1)
plt.loglog(RESOLUTIONS, np.exp(c) * RESOLUTIONS**m, "k--", label=r"$n^{"f"{m:.2f}""}$")

plt.xlabel(r"Resolution $n$")
plt.ylabel("Root Mean Squared Error (RMSE)")
plt.title("Convergence of the finite difference method")

plt.legend()

plt.savefig("../figures/fdm-convergence.png", dpi=300)
