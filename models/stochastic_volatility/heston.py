# models/stochastic_volatility/heston.py

import numpy as np

class HestonModel:
    """
    Heston Stochastic Volatility Model:
    dS_t = mu * S_t dt + sqrt(v_t) * S_t dW1_t
    dv_t = kappa*(theta - v_t) dt + xi * sqrt(v_t) dW2_t
    Corr(dW1, dW2) = rho
    """

    def __init__(self, spot, mu, v0, kappa, theta, xi, rho, dt=1/252):
        self.spot = spot
        self.mu = mu
        self.v0 = v0          # Initial variance
        self.kappa = kappa    # Mean reversion speed
        self.theta = theta    # Long-term variance
        self.xi = xi          # Vol of vol
        self.rho = rho        # Correlation
        self.dt = dt

    def simulate_paths(self, T, n_paths):
        """
        Simulate asset paths under the Heston model.
        :param T: Time horizon
        :param n_paths: Number of paths
        :return: Tuple (S, v)
        """
        N = int(T / self.dt)
        S = np.zeros((N + 1, n_paths))
        v = np.zeros((N + 1, n_paths))
        S[0] = self.spot
        v[0] = self.v0

        for t in range(1, N + 1):
            z1 = np.random.standard_normal(n_paths)
            z2 = np.random.standard_normal(n_paths)
            W1 = z1
            W2 = self.rho * z1 + np.sqrt(1 - self.rho ** 2) * z2

            v_prev = np.maximum(v[t - 1], 0)
            v[t] = v_prev + self.kappa * (self.theta - v_prev) * self.dt + \
                   self.xi * np.sqrt(v_prev * self.dt) * W2
            v[t] = np.maximum(v[t], 0)  # Keep variance positive

            S[t] = S[t - 1] * np.exp(
                (self.mu - 0.5 * v_prev) * self.dt + np.sqrt(v_prev * self.dt) * W1
            )
        return S, v
