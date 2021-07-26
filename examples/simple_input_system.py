import numpy as np
import matplotlib.pyplot as plt
from itertools import product

import nlsystem as nls


class simple_input_system(nls.System):
    """ Sistema simples e globalmente estável, com entrada.

    .. math::
        \dot{x_0} = x_1
        \dot{x_1} = -x_0 - x_1 + u_0(t)
    """

    dim = 2
    
    def model(self, x, t):
        self.save_signal('v', x[0] + x[1], t)
        return (x[1], -x[0] - x[1] + self.input('u0', t))


t = np.linspace(0, 50, 10000)
u0 = np.vectorize(lambda s: 0 if s < 25 else 1)(t)

system = simple_input_system()
system.set_simulation_data(t=t, inputs={'u0': np.zeros_like(u0)})
system.plot_phase_plan([-1,1], [-1,1], n=17, curves=[(0,1), (0,-1), (1,0), (-1,0)])

plt.show()

system.set_simulation_data(t=t, inputs={'u0': u0})
system.simulate((1,1))
plt.plot(system.t, system.x[0])
plt.plot(system.t, u0)

plt.show()