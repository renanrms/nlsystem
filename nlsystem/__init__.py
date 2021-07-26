import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from matplotlib.cm import get_cmap
from scipy.integrate import odeint
from scipy.interpolate import interp1d


class System:

	def __init__(self):
		self.t = None
		self.x = None
		self.inputs = {}
		self.signals = {}
		self._interpolated_inputs = {}
		self._raw_signals = {}

	def model(self, x, t):
		""" Método abstrato. Deve retornar uma tupla com as derivadas do estado x.
		"""
		raise Exception("Modelo não implementado.")

	def set_simulation_data(self, t, inputs={}):
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

		self.inputs = {}
		self._interpolated_inputs = {}

		for key in inputs.keys():
			if len(inputs[key]) != len(self.t):
				raise Exception(
					f'O comprimento do sinal de input "{key}" é diferente do comprimento da série temporal.')
			self.inputs[key] = np.array(inputs[key])
			self._interpolated_inputs[key] = interp1d(t, inputs[key], fill_value="extrapolate")
		
		self._raw_signals = {}
		self.signals = {}

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
		
		Também armazena as saídas e os estados do sistema na simulação, correspondendo aos instantes de tempo do vetor `t`.
		
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
			interp = interp1d(self._raw_signals[name]['time'], self._raw_signals[name]['data'], fill_value="extrapolate")
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

		if not name in self._interpolated_inputs.keys():
			raise Exception("Sinal de entrada não fornecido.")
		
		return float(self._interpolated_inputs[name](t))

	def balance_points(self, x_limits, y_limits, resolution_factor=None, plot_trace=False):
		""" Obtém os pontos de equilíbrio do sistema pelo método de bisecção.
		O valor de retorno é dado por lista de dicionários com chaves `x` e `y`.

		Parâmetros
		----------
		x_limits, y_limits : list
			Lista com os limites inferior e superior de cada eixo.
		resolution_factor : float
			Determina a granularidade mínima com que se dividirá a região para descartar as subregiões que parecem não conter pontos de equilíbrio.

			O método irá procurar os pontos de equilíbrio dividindo sussecivamente cada subregião até que sua maior dimensão se torne menor que o tamanho da menor dimensão do região original multiplicada por `resolution_factor`.
		plot_trace : bool
			Se for `True`, desenha os retângulos que estão sendo verificados quanto à existência de pontos de equilíbrio.
		"""

		t0, x_min, x_max, y_min, y_max = self._parse_context_data(x_limits, y_limits)
		
		if resolution_factor is None:
			resolution_factor = 0.01

		delta = resolution_factor * min(x_max - x_min, y_max - y_min)

		return self._get_balance_points(t0, x_min, x_max, y_min, y_max, delta, plot_trace=plot_trace)

	def _get_balance_points(self, t0, x_min, x_max, y_min, y_max, delta, eps=2e-32, plot_trace=False):
		""" Obtém os pontos de equilíbrio do sistema pelo método de bisecção.
		O valor de retorno é dado por lista de dicionários com chaves `x` e `y`.
		
		O retângulo será definido como a seguir. ::

		  :           ^
		  :           |   f3      f0 = (fx0, fy0)
		  :    y_max -+  o--------o
		  :           |  |        |
		  :           |  |        |
		  :           |  |        |
		  :    y_min -+  o--------o
		  :           |   f2      f1
		  :           o--+--------+---->
		  :              x_min    x_max	

		Parâmetros
		----------
		t0 : float
			Instante de referência para os cálculos.
		x_min, x_max, y_min, y_max : float
			Limites inferior e superior de cada eixo.
		delta : float
			Largura maxima que uma subregião pode ter para descartar a região.
		eps : float
			precisão usada para buscar os pontos.
		plot_trace : bool
			Se for `True`, desenha os retângulos que estão sendo verificados quanto à existência de raízes.
		"""

		width = x_max - x_min
		height = y_max - y_min
		x_middle = (x_min + x_max)/2
		y_middle = (y_min + y_max)/2

		if plot_trace:
			plt.gca().add_patch(plt.Rectangle((x_min, y_min), width, height, linewidth=0.2, edgecolor='g', facecolor='none'))

		# Obtém cada coordenada da resposta do sistema nos pontos de teste.
		fx0, fy0 = self.model((x_max, y_max), t0)
		fx1, fy1 = self.model((x_max, y_min), t0)
		fx2, fy2 = self.model((x_min, y_min), t0)
		fx3, fy3 = self.model((x_min, y_max), t0)

		# Verifica a variação de sinal ou não da cada coordenada em cada um dos eixos.
		Gxx = min(fx2*fx1, fx3*fx0)
		Gxy = min(fx0*fx1, fx3*fx2)
		Gyx = min(fy2*fy1, fy3*fy0)
		Gyy = min(fy0*fy1, fy3*fy2)

		# Verifica a variação de sinal de cada coordenada em pelo menos um dos eixos.
		Gx = min(Gxx, Gxy)
		Gy = min(Gyx, Gyy)

		largest_dim = max(width, height)
		contain_zeros_inside = max(Gx, Gy) < 0 or (fx2 == 0 and fy2 == 0)

		if contain_zeros_inside and largest_dim <= eps:
			return [{'x':x_min, 'y':y_min}]
		elif contain_zeros_inside or largest_dim > delta:
			if width >= height:
				return self._get_balance_points(t0, x_min, x_middle, y_min, y_max, delta, plot_trace=plot_trace) + self._get_balance_points(t0, x_middle, x_max, y_min, y_max, delta, plot_trace=plot_trace)
			else:
				return self._get_balance_points(t0, x_min, x_max, y_min, y_middle, delta, plot_trace=plot_trace) + self._get_balance_points(t0, x_min, x_max, y_middle, y_max, delta, plot_trace=plot_trace)
		else:
			return []

	def _parse_context_data(self, x_limits, y_limits, n=None):
		""" Método interno para tratar alguns dados de entrada de funções.
		"""

		t0 = 0
		if self.t is not None:
			t0 = self.t[0]

		x_min, x_max = x_limits
		y_min, y_max = y_limits

		if n is None:
			return t0, x_min, x_max, y_min, y_max

		if type(n) == int:
			x_n, y_n = (n, n)
		else:
			x_n, y_n = n
		
		return t0, x_min, x_max, y_min, y_max, x_n, y_n


	def plot_phase_plan(self, x_limits, y_limits, n, curves=None, colorbar=True, cmap=None, balance_kwargs={}):
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
		balance_kwargs : dict
			Argumentos passados para o método `blance_points`.
		"""

		t0, x_min, x_max, y_min, y_max, x_n, y_n = self._parse_context_data(x_limits, y_limits, n)

		arrows = list(product(
			np.linspace(x_min, x_max, x_n),
			np.linspace(y_min, y_max, y_n)
		))

		x, y, dx, dy = [], [], [], []
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

		if cmap == None:
			cmap = get_cmap('hot')

		plt.axes([0.025, 0.025, 0.95, 0.95])
		plt.quiver(x, y, dx_norm, dy_norm, abs_derivate,
				   angles='xy', pivot='middle', alpha=.8, cmap=cmap)
		plt.quiver(x, y, dx_norm, dy_norm, edgecolor='k', angles='xy',
				   facecolor='None', linewidth=.1, pivot='middle')

		if curves is not None:
			if self.t is None:
				raise Exception("Para plotar as curvas é preciso definir o array de instantes de tempo. Ver método set_simulation_data().")
			for w0 in curves:
				w = odeint(self.model, w0, self.t)
				plt.plot(w[:, 0], w[:, 1], color='#1f77b4', linewidth=1)

		sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(
			vmin=min(abs_derivate), vmax=max(abs_derivate)))
		if colorbar == True:
			plt.colorbar(sm, shrink=0.6)

		x_step = (x_max - x_min) / (x_n - 1)
		y_step = (y_max - y_min) / (y_n - 1)
		plt.xlim(x_min - x_step, x_max + x_step)
		plt.ylim(y_min - y_step, y_max + y_step)

		balance_points = self.balance_points(x_limits, y_limits, **balance_kwargs)

		for point in balance_points:
			plt.plot(point['x'], point['y'], 'bo', label='Pontos de Equilíbrio')

		if balance_points != []:
			plt.legend(loc='upper right')
