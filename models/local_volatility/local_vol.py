# models/local_volatility/local_vol.py

import numpy as np

class LocalVolatilityModel:
    """
    Local Volatility Model (e.g., Dupire's model).
    """

    def __init__(self, spot, local_vol_surface, dt=1/252):
        """
        :param spot: Initial stock price
        :param local_vol_surface: Callable function sigma(S, t)
        :param dt: Time step
        """
        self.spot = spot
        self.local_vol_surface = local_vol_surface
        self.dt = dt

    def simulate_paths(self, T, n_paths):
        """
        Simulate asset paths under local volatility dynamics.
        :param T: Time horizon
        :param n_paths: Number of paths
        :return: Simulated paths (numpy array)
        """
        N = int(T / self.dt)
        paths = np.zeros((N + 1, n_paths))
        paths[0] = self.spot

        for t in range(1, N + 1):
            time = t * self.dt
            z = np.random.standard_normal(n_paths)
            sigma = self.local_vol_surface(paths[t - 1], time)
            paths[t] = paths[t - 1] * np.exp(
                -0.5 * sigma ** 2 * self.dt + sigma * np.sqrt(self.dt) * z
            )
        return paths
