# models/black_scholes/pde.py

import numpy as np

class BlackScholesPDE:
    """
    Black-Scholes PDE solver using explicit finite difference scheme.
    """

    def __init__(self, S_max, K, T, r, sigma, M=100, N=1000):
        self.S_max = S_max  # Max asset price
        self.K = K          # Strike
        self.T = T          # Maturity
        self.r = r          # Risk-free rate
        self.sigma = sigma  # Volatility
        self.M = M          # Number of asset price steps
        self.N = N          # Number of time steps

    def solve_call(self):
        """
        Solves the Black-Scholes PDE for a European Call option.
        Returns: asset price grid, option price grid
        """
        dt = self.T / self.N
        dS = self.S_max / self.M
        S = np.linspace(0, self.S_max, self.M + 1)
        V = np.maximum(S - self.K, 0)  # Payoff at maturity

        for n in reversed(range(self.N)):
            V_prev = V.copy()
            for i in range(1, self.M):
                delta = (V_prev[i + 1] - V_prev[i - 1]) / (2 * dS)
                gamma = (V_prev[i + 1] - 2 * V_prev[i] + V_prev[i - 1]) / (dS ** 2)
                V[i] = V_prev[i] + dt * (
                    0.5 * self.sigma ** 2 * S[i] ** 2 * gamma +
                    self.r * S[i] * delta -
                    self.r * V_prev[i]
                )
            V[0] = 0  # S = 0
            V[-1] = self.S_max - self.K * np.exp(-self.r * (self.T - n * dt))
        return S, V
