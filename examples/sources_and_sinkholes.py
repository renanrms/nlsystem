import numpy as np
import matplotlib.pyplot as plt
from itertools import product

import nlsystem as nls

def module(x):
    return np.sqrt(sum(np.square(x)))

class sources_and_sinkholes(nls.System):
    """ Sistema que simula a forma do campo gerado por duas cargas.
    """

    dim = 2
    
    def model(self, x, t):
        x = np.array(x)

        vector_s1 = x - np.array((-0.5, 0))
        if module(vector_s1) == 0:
            field_s1 = (np.array((np.NaN,np.NaN)))
        else:
            field_s1 = vector_s1 / module(vector_s1)**3

        vector_d1 = x - np.array((0.5, 0))
        if module(vector_d1) == 0:
            field_d1 = np.array((np.NaN,np.NaN))
        else:
            field_d1 = -vector_d1 / module(vector_d1)**3

        return field_s1 + field_d1


system = sources_and_sinkholes()

system.set_simulation_data(t=np.linspace(0, 10, 10000))

system.plot_phase_plan([-1,1], [-1,1], n=17, plot_balance_points=True, balance_kwargs={'plot_trace':True, 'min_resolution':0.1})

plt.show()
