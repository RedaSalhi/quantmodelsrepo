# models/hull_white/hull_white.py

import numpy as np

class HullWhiteModel:
    """
    Hull-White One-Factor Short Rate Model:
    dr_t = [theta(t) - a * r_t] dt + sigma dW_t

    theta(t) can be constant or calibrated to fit the term structure.
    """

    def __init__(self, r0, a, sigma, theta=0.0, dt=1/252):
        """
        :param r0: Initial short rate
        :param a: Mean reversion speed
        :param sigma: Volatility
        :param theta: Can be constant or function theta(t)
        :param dt: Time step
        """
        self.r0 = r0
        self.a = a
        self.sigma = sigma
        self.theta = theta
        self.dt = dt

    def theta_t(self, t):
        """
        Return theta(t). Supports scalar or callable.
        """
        if callable(self.theta):
            return self.theta(t)
        return self.theta

    def simulate_short_rate(self, T, n_paths):
        """
        Simulate short rate paths.
        :param T: Time horizon
        :param n_paths: Number of paths
        :return: Simulated short rate paths
        """
        N = int(T / self.dt)
        r = np.zeros((N + 1, n_paths))
        r[0] = self.r0

        for t in range(1, N + 1):
            time = t * self.dt
            z = np.random.standard_normal(n_paths)
            r[t] = r[t - 1] + (self.theta_t(time) - self.a * r[t - 1]) * self.dt + \
                   self.sigma * np.sqrt(self.dt) * z

        return r

    def zero_coupon_bond_price(self, r_t, T, t=0):
        """
        Closed-form ZCB price under Hull-White model.
        :param r_t: Current short rate at time t
        :param T: Bond maturity
        :param t: Current time
        :return: ZCB price
        """
        B = (1 - np.exp(-self.a * (T - t))) / self.a
        A = np.exp(
            (B - (T - t)) * (self.theta_t(t) - (self.sigma ** 2) / (2 * self.a ** 2))
            - (self.sigma ** 2) * B ** 2 / (4 * self.a)
        )
        return A * np.exp(-B * r_t)
