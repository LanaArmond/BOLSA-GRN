import numpy as np

class Solvers:
    def __init__(self):
        pass
    
    
    #declaraçoes de funções
    @staticmethod
    def norm_hardcoded(val, max):
        return val / max


    # Método de Euler
    @staticmethod
    def euler_step(fun, t, y, h):
        return y + h * np.array(fun(t, y))

    # Método de RK6
    @staticmethod
    def rk6_step(fun, t, y, h):
        k1 = h * np.array(fun(t, y))
        k2 = h * np.array(fun(t + h / 3, y + k1 / 3))
        k3 = h * np.array(fun(t + h / 3, y + k1 / 6 + k2 / 6))
        k4 = h * np.array(fun(t + h / 2, y + k1 / 8 + 3 * k3 / 8))
        k5 = h * np.array(fun(t + 2 * h / 3, y + k1 / 2 - 3 * k3 / 2 + 2 * k4))
        k6 = h * np.array(fun(t + h, y - 3 * k1 / 2 + 2 * k3 - 3 * k4 / 2 + k5))
        return y + (k1 + 4 * k4 + k6) / 6
    
    @staticmethod
    def solve_manual(fun, y0, method, t_eval, h):
        y = np.zeros((len(y0), len(t_eval)))
        y[:, 0] = y0
        for i in range(1, len(t_eval)):
            if method == 'Euler':
                y[:, i] = Solvers.euler_step(fun, t_eval[i - 1], y[:, i - 1], h)
            elif method == 'RK6':
                y[:, i] = Solvers.rk6_step(fun, t_eval[i - 1], y[:, i - 1], h)
        return y
