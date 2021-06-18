import numpy as np
import matplotlib.pyplot as plt
from itertools import product

import nls


class simple_convergent_system(nls.System):
    def model(self, x, t):
        return (x[1], -x[0] - x[1])


system = simple_convergent_system()

system.set_simulation_data(t=np.linspace(0, 10, 10000))

system.plot_phase_plan([-1,1], [-1,1], n=17, curves=[(-1,0), (1,0), (0,-1), (0,1)])

plt.show()

system.simulate((1,1))

plt.plot(system.t, system.x[0])

plt.show()