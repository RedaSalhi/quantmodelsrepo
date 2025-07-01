# models/black_scholes/sde.py

import numpy as np

class BlackScholesSDE:
    """
    Black-Scholes Stochastic Differential Equation (SDE) Model
    dS_t = mu * S_t * dt + sigma * S_t * dW_t
    """

    def __init__(self, spot, mu, sigma, dt=1/252):
        self.spot = spot      # Initial stock price
        self.mu = mu          # Drift
        self.sigma = sigma    # Volatility
        self.dt = dt          # Time step

    def simulate_paths(self, T, n_paths):
        """
        Simulate GBM paths using Euler-Maruyama scheme.
        :param T: Time horizon (in years)
        :param n_paths: Number of simulated paths
        :return: Simulated paths (numpy array)
        """
        N = int(T / self.dt)
        paths = np.zeros((N + 1, n_paths))
        paths[0] = self.spot

        for t in range(1, N + 1):
            z = np.random.standard_normal(n_paths)
            paths[t] = paths[t - 1] * np.exp(
                (self.mu - 0.5 * self.sigma ** 2) * self.dt +
                self.sigma * np.sqrt(self.dt) * z
            )
        return paths
