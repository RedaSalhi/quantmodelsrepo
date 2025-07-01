# models/jump_models/jump_diffusion.py

import numpy as np

class JumpDiffusionModel:
    """
    Merton's Jump Diffusion Model:
    dS_t/S_t = (mu - lambda * k) dt + sigma dW_t + J dq_t
    where:
        J ~ log-normal jump size
        q_t ~ Poisson process with intensity lambda
    """

    def __init__(self, spot, mu, sigma, lamb, mu_j, sigma_j, dt=1/252):
        self.spot = spot      # Initial stock price
        self.mu = mu          # Drift
        self.sigma = sigma    # Diffusion volatility
        self.lamb = lamb      # Jump intensity (lambda)
        self.mu_j = mu_j      # Mean jump size (in log-space)
        self.sigma_j = sigma_j  # Jump size volatility
        self.dt = dt

    def simulate_paths(self, T, n_paths):
        """
        Simulate paths under Merton's jump diffusion.
        :param T: Time horizon
        :param n_paths: Number of paths
        :return: Simulated paths
        """
        N = int(T / self.dt)
        paths = np.zeros((N + 1, n_paths))
        paths[0] = self.spot

        for t in range(1, N + 1):
            z = np.random.standard_normal(n_paths)
            # Poisson jumps
            jumps = np.random.poisson(self.lamb * self.dt, n_paths)
            jump_sizes = np.random.normal(self.mu_j, self.sigma_j, n_paths)
            J = np.exp(jump_sizes) - 1

            paths[t] = paths[t - 1] * np.exp(
                (self.mu - self.lamb * self.mu_j - 0.5 * self.sigma ** 2) * self.dt +
                self.sigma * np.sqrt(self.dt) * z
            ) * (1 + J * jumps)

        return paths
