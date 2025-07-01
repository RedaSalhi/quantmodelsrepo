# models/hull_white/extensions.py

import numpy as np

class HullWhiteExtensions:
    """
    Example: Multi-factor Hull-White or pricing exotics.
    Extend the base Hull-White model for swaptions, caps/floors.
    """

    def __init__(self, base_model):
        """
        :param base_model: An instance of HullWhiteModel
        """
        self.model = base_model

    def price_caplet(self, r_t, T, K):
        """
        Illustrative: approximate Black formula style caplet pricing.
        This is a placeholder — you’d typically integrate the model’s
        implied forward rate distribution here.
        :param r_t: Current short rate
        :param T: Maturity
        :param K: Strike rate
        :return: Caplet price (approx.)
        """
        P = self.model.zero_coupon_bond_price(r_t, T)
        forward_rate = -np.log(P) / T
        d1 = (forward_rate - K) / (self.model.sigma * np.sqrt(T))
        caplet_price = P * max(forward_rate - K, 0)
        return caplet_price

    def multi_factor_extension(self):
        """
        Stub for multi-factor Hull-White models.
        For example, use a second factor with different a2, sigma2.
        """
        raise NotImplementedError("Multi-factor extension not implemented yet.")
