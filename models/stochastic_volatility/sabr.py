# models/stochastic_volatility/sabr.py

import numpy as np

class SABRModel:
    """
    SABR Model:
    dF = sigma * F^beta dW1
    dsigma = nu * sigma dW2
    Corr(dW1, dW2) = rho
    """

    def __init__(self, F0, alpha, beta, rho, nu, dt=1/252):
        self.F0 = F0          # Initial forward
        self.alpha = alpha    # Initial volatility
        self.beta = beta      # Elasticity parameter
        self.rho = rho        # Correlation
        self.nu = nu          # Vol of vol
        self.dt = dt

    def simulate_paths(self, T, n_paths):
        """
        Simulate forward and volatility paths under SABR dynamics.
        :param T: Time horizon
        :param n_paths: Number of paths
        :return: Tuple (F, sigma)
        """
        N = int(T / self.dt)
        F = np.zeros((N + 1, n_paths))
        sigma = np.zeros((N + 1, n_paths))
        F[0] = self.F0
        sigma[0] = self.alpha

        for t in range(1, N + 1):
            z1 = np.random.standard_normal(n_paths)
            z2 = np.random.standard_normal(n_paths)
            W1 = z1
            W2 = self.rho * z1 + np.sqrt(1 - self.rho ** 2) * z2

            F_prev = F[t - 1]
            sigma_prev = sigma[t - 1]

            F[t] = F_prev + sigma_prev * F_prev ** self.beta * np.sqrt(self.dt) * W1
            sigma[t] = sigma_prev + self.nu * sigma_prev * np.sqrt(self.dt) * W2

            # Ensure positive vol
            sigma[t] = np.maximum(sigma[t], 0)

        return F, sigma
