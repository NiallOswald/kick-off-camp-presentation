import numpy as np
import matplotlib.pyplot as plt

ERRORS = [0.031448968022956304, 0.019750021529295652, 0.011308933844055702, 0.006114984004581622, 0.0031667037722979577]
RESOLUTIONS = [4, 8, 16, 32, 64]

# Plot the convergence of the finite element method.
plt.loglog(RESOLUTIONS, ERRORS, "k-o")

# Plot the line of best fit and label the gradient.
m, c = np.polyfit(np.log(RESOLUTIONS[3:]), np.log(ERRORS[3:]), 1)
plt.loglog(RESOLUTIONS, np.exp(c) * RESOLUTIONS**m, "k--", label=r"$n^{"f"{m:.2f}""}$")

plt.xlabel(r"Resolution $n$")
plt.ylabel("Root Mean Squared Error (RMSE)")
plt.title("Convergence of the finite element method")

plt.legend()

plt.savefig("../figures/fem-convergence.png", dpi=300)
