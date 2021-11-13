def clean_regressive_time_series_end_to_begin(time_serie):
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
