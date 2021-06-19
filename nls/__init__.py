import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy import interpolate
from itertools import product


class System:

	def __init__(self):
		self.t = None
		self.x = None
		self.inputs = None
		self.signals = None
		self._interpolated_inputs = None
		self._raw_signals = None
		self._avoid_save_signals = False

	def model(self, x, t):
		""" Método abstrato. Deve retornar uma tupla com as derivadas do estado x.
		"""
		raise Exception("Modelo não implementado.")

	def set_simulation_data(self, t, inputs=None):
		""" Armazena os dados para a simulação do sistema.

		Parâmetros
		----------
		t : iterável
			Iterável contento os instantes de tempo para simulação.
		inputs : dict
			Dicionário contendo os sinais de entrada para simulação do modelo.
			Deve ter a forma { 'nome_sinal': sinal, ... } , onde o sinal deve ter o mesmo comprimento de t.
		"""

		if len(t) == 0:
			raise Exception("A série temporal não pode ter comprimento 0.")
		self.t = np.array(t)

		self.inputs = None
		self._interpolated_inputs = None

		if inputs is not None:
			self.inputs = {}
			self._interpolated_inputs = {}
			for key in inputs.keys():
				if len(inputs[key]) != len(self.t):
					raise Exception(
						f'O comprimento do sinal de input "{key}" é diferente do comprimento da série temporal.')
				self.inputs[key] = np.array(inputs[key])
				self._interpolated_inputs[key] = interpolate.interp1d(t, inputs[key], fill_value="extrapolate")
		
		self._raw_signals = None
		self.signals = None

	def _clean_regressive_time_serie_end_to_begin(self, time_serie):
		""" Elimina regiões de regressão da série temporal, que podem ter sido geradas por recálculos de um método de aproximação. O método parte do fim e retira cada amostra com instante de tempo porterior ao da amostra sucessora.

		Parâmetros
		----------
		time_serie : dict
			Um dicionário no formato {'time': [], 'data': []}
		"""
		i = len(time_serie['time']) - 1
		while i > 0:
			if time_serie['time'][i-1] > time_serie['time'][i]:
				time_serie['time'].pop(i-1)
				time_serie['data'].pop(i-1)
			i -= 1

	def simulate(self, initial_state):
		""" Simula o sistema.
		
		Também armazena as saídas e os estados do sistema na simulação, correspondendo aos instantes de tempo do vetor t.
		
		Parâmetros
		----------
		initial_state : tuple
			Uma tupla definindo o estado inicial do sistema para a simulação.
		"""

		self._raw_signals = {}
		self.signals = {}

		states = odeint(self.model, initial_state, self.t)

		self.x = [states[:, i] for i in range(states.shape[1])]

		for name in self._raw_signals.keys():
			self._clean_regressive_time_serie_end_to_begin(self._raw_signals[name])
			interp = interpolate.interp1d(self._raw_signals[name]['time'], self._raw_signals[name]['data'], fill_value="extrapolate")
			self.signals[name] = interp(self.t)

	def save_signal(self, name, sample, t):
		""" Salva amostra do sinal acesso a este após a simulação.
		
		Deve ser usada dentro do método `model` para salvar sinais ao longo da simulação.

		Parâmetros
		----------
		name : str
			Nome do sinal salvo.
		sample : float
			Amostra do sinal a ser armazenada.
		t : float
			Instante de tempo na simulação.
		"""

		if self._avoid_save_signals:
			return

		if not name in self._raw_signals.keys():
			self._raw_signals[name] = {'data':[], 'time':[]}

		self._raw_signals[name]['data'].append(sample)
		self._raw_signals[name]['time'].append(t)
	
	def input(self, name, t):
		""" Retorna a amostra do sinal de entrada `name` para o instante `t` através de uma interpolação do sinal.

		Parâmetros
		----------
		name : str
			Nome do sinal de entrada.
		t : float
			Instante de tempo na simulação.
		"""

		if self.inputs is None:
			raise Exception("Não foram forneciso sinais de entrada.")

		if not name in self._interpolated_inputs.keys():
			raise Exception("Sinal de entrada não fornecido.")
		
		return float(self._interpolated_inputs[name](t))

	def plot_phase_plan(self, x_limits, y_limits, n, curves=None, colorbar=True, cmap=None):
		""" Plota o plano de fase do sistema.

		Parâmetros
		----------
		x_limits, y_limits : list
			Lista com os limites inferior e superior de cada eixo.
		n : int ou list
			Quantidade de posições em cada eixo em que estarão as setas do plano de fase.
			Se n for uma lista serão definidas quantidades diferentes em cada eixo.
			se for um inteiro, o valor será repetido para ambos os eixos.
		curves : iterável, optional
			Um iterável contendo os pares de estado inicial para traçar as curvas sobre o plano de fase.
		colorbar : bool, optional
			Determina se será incluída uma colorbar ao labo do gráfico.
		cmap : Colormap, optional
			Define as cores usadas nas setas do plano de fase.
		"""

		self._avoid_save_signals = True

		t0 = 0
		if self.t is not None:
			t0 = self.t[0]

		x_min, x_max = x_limits
		y_min, y_max = y_limits

		if type(n) == int:
			x_n, y_n = (n, n)
		else:
			x_n, y_n = n

		x, y, dx, dy = [], [], [], []

		arrows = list(product(
			np.linspace(x_min, x_max, x_n),
			np.linspace(y_min, y_max, y_n)
		))

		for w0 in arrows:
			w = self.model(w0, t0)
			x.append(w0[0])
			y.append(w0[1])
			dx.append(w[0])
			dy.append(w[1])

		dx = np.array(dx)
		dy = np.array(dy)
		abs_derivate = np.sqrt(np.square(dx) + np.square(dy))

		dx_norm = np.zeros_like(abs_derivate)
		dy_norm = np.zeros_like(abs_derivate)

		for i in range(len(abs_derivate)):
			if abs_derivate[i] == 0:
				# Com o valor None não será plotada nenhuma seta.
				dx_norm[i] = None
				dy_norm[i] = None
			else:
				dx_norm[i] = dx[i] / abs_derivate[i]
				dy_norm[i] = dy[i] / abs_derivate[i]

		# Protando o campo vetorial com QUIVER
		if cmap == None:
			cmap = matplotlib.cm.get_cmap('hot')

		plt.axes([0.025, 0.025, 0.95, 0.95])
		plt.quiver(x, y, dx_norm, dy_norm, abs_derivate,
				   angles='xy', pivot='middle', alpha=.8, cmap=cmap)
		plt.quiver(x, y, dx_norm, dy_norm, edgecolor='k', angles='xy',
				   facecolor='None', linewidth=.1, pivot='middle')

		# Plota algumas curvas apenas para ilustrar
		if curves is not None:
			if self.t is None:
				raise Exception("Para plotar as curvas é preciso definir o array de instantes de tempo. Ver método set_simulation_data().")
			for w0 in curves:
				w = odeint(self.model, w0, self.t)
				plt.plot(w[:, 0], w[:, 1], color='blue', linewidth=1)

		# Configura as anotações e informações adicionais do gráfico
		sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(
			vmin=min(abs_derivate), vmax=max(abs_derivate)))
		if colorbar == True:
			plt.colorbar(sm, shrink=0.6)

		# Define os limites de exibição dos eixos
		x_step = (x_max - x_min) / (x_n - 1)
		y_step = (y_max - y_min) / (y_n - 1)
		plt.xlim(x_min - x_step, x_max + x_step)
		plt.ylim(y_min - y_step, y_max + y_step)

		self._avoid_save_signals = False
