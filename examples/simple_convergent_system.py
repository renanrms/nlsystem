import nlsystem

import numpy as np
import matplotlib.pyplot as plt


def system(x, t):
	return (x[1], -x[0] - x[1])


arrows = [
	np.linspace(-3, 3, 17),
	np.linspace(-3, 3, 17)
]

nlsystem.plot_phase_plane(system, (), arrows)

plt.show()
