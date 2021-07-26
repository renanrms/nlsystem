import numpy as np
import matplotlib.pyplot as plt
from itertools import product

import nlsystem as nls


class simple_convergent_system(nls.System):
    """ Sistema simples e globalmente estável, sem entradas.

    .. math::
        \dot{x_0} = x_1
        \dot{x_1} = -x_0 - x_1
    """

    dim = 2
    
    def model(self, x, t):
        return (x[1], -x[0] - x[1])


system = simple_convergent_system()

system.set_simulation_data(t=np.linspace(0, 10, 10000))

system.plot_phase_plan([-1,1], [-1,1], n=17, curves=[(-1,0), (1,0), (0,-1), (0,1)], 
    balance_kwargs={'plot_trace':True, 'resolution_factor':0.1})

plt.show()

print('Pontos de equilíbrio do sistema:')
print(system.balance_points([-1,1], [-1,1]))
