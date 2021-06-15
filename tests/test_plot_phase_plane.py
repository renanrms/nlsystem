import pytest

import numpy as np

import nlsystem


def test_plot_phase_plane():
	def system(x, t):
		return (x[1], -x[0])

	arrows = [
		np.linspace(-3, 3, 17),
		np.linspace(-3, 3, 17)
	]

	nlsystem.plot_phase_plane(system, (), arrows)
