import numpy as np

def Qt_ratio(VaR,CVaR):
    ratio = np.mean(CVaR / VaR)
    return ratio