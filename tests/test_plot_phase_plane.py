import pytest

import numpy as np

import nlsystem as nls


def test_plot_phase_plane():
	class test_system(nls.System):
		def model(self, x, t):
			return (x[1], -x[0] - x[1])

	system = test_system()
	system.plot_phase_plan([-1,1], [-1,1], n=17)
