import numpy as np
import matplotlib.pyplot as plt
from itertools import product

import nlsystem as nls


class simple_divergent_system(nls.System):
    """ Sistema simples e instável, sem entradas.

    .. math::
        \dot{x_0} = x_0 sign(x_0 x_1)
        \dot{x_1} = x_1 sign(x_0 x_1)
    """

    dim = 2
    
    def model(self, x, t):
        return  (
            x[0]*np.arctan(10*x[0]*x[1]),
            x[1]*np.arctan(10*x[0]*x[1])
        )


system = simple_divergent_system()

system.set_simulation_data(t=np.linspace(0, 10, 10000))

system.plot_phase_plan([-1,1], [-1,1], n=17, balance_kwargs={'plot_trace':True, 'min_resolution':0.5, 'max_resolution':1e-2})

plt.show()

system.simulate((1,1))

plt.plot(system.t, system.x[0])

plt.show()
