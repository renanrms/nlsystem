import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import itertools

def plot_phase_plane(model, args, arrows, t0=0, t_curves=None, X0_curves=None, cmap=None):
	initial_states = [(i, j) 
										for i in arrows[0]
										for j in arrows[1]]

	x = []
	y = []
	dx = []
	dy = []

	for w0 in initial_states:
		w = model(w0, t0, *args)
		x.append(w0[0])
		y.append(w0[1])
		dx.append(w[0])
		dy.append(w[1])

	dx = np.array(dx)
	dy = np.array(dy)
	abs_derivate = np.sqrt(np.square(dx) + np.square(dy))

	dx_norm = dx / abs_derivate
	dy_norm = dy / abs_derivate

	# Plotting Vector Field with QUIVER
	if cmap == None:
		cmap = matplotlib.cm.get_cmap('hot')

	plt.axes([0.025, 0.025, 0.95, 0.95])
	plt.quiver(x, y, dx_norm, dy_norm, abs_derivate, angles='xy', pivot='middle', alpha=.8, cmap=cmap)
	plt.quiver(x, y, dx_norm, dy_norm, edgecolor='k', angles='xy', facecolor='None', linewidth=.1, pivot='middle')

	# Plota algumas curvas apenas para ilustrar
	if not (t_curves is None or X0_curves is None):
		ys = np.linspace(-4, 4, 4)
		Dys = np.linspace(-10, 10, 6)

		for w0 in itertools.product(*X0_curves):
			w = odeint(model, w0, t_curves, args=args)
			plt.plot(w[:,0], w[:,1], color='blue', linewidth=1)

	# Configura as anotações e informações adicionais do gráfico
	sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(abs_derivate), vmax=max(abs_derivate)))
	# plt.colorbar(sm, shrink=0.6, label=r'$abs(\dot{x})$')

	# Setting x, y boundary limits
	x_min = min(arrows[0])
	x_max = max(arrows[0])
	x_step = (x_max - x_min) / (len(arrows[0]) - 1)

	y_min = min(arrows[1])
	y_max = max(arrows[1])
	y_step = (y_max - y_min) / (len(arrows[1]) - 1)

	plt.xlim(x_min - x_step, x_max + x_step)
	plt.ylim(y_min - y_step, y_max + y_step)