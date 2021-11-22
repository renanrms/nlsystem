import numpy as np

def jacobian_matrix(func, x0, delta=1e-8):
    matrix = np.empty(shape=(len(x0), len(x0)))
    for n in range(len(x0)):
        xi = np.array(x0)
        xf = np.array(x0)

        xi[n] -= delta/2
        xf[n] += delta/2

        matrix[:,n] = (np.array(func(xf, 0)) - np.array(func(xi, 0))) / delta
    return matrix
        
