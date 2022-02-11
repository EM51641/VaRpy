import numpy as np 
from arch import arch_model
import scipy 
from varpy.tools.innovation_finder import Extract_Excess_Innovations
from varpy.tools.arch_mdl import Arch_data
 
def Normal_VaR(return_matrix, theta,Horizon): #500 datas needed 

        """
        Compute the Value-at-Risk and Conditional Value-at-Risk
        Parameters
        ----------
        risk_returns : np.ndarray
        theta : np.float64
        Horizon : np.int16
        Returns
        ----------
        np.ndarray,np.ndarray   VaR , CVaR
        """
        
        mean_forecast,var_forecast,conditional_volatility = Arch_data(return_matrix , Horizon )
        excess_innovations = Extract_Excess_Innovations(return_matrix , mean_forecast , conditional_volatility )
        mu,scale  = Dist_parameters(excess_innovations)
        VaR,CVaR = Var_CVaR_extractor(mean_forecast,var_forecast,scale,mu,theta)
        return VaR,CVaR


def Dist_parameters(excess_innovations):
        mu , scale = scipy.stats.norm.fit(excess_innovations)
        return mu , scale

def Var_CVaR_extractor(mean_forecast,var_forecast,scale,mu,theta,):
        unconditional_VaR = scipy.stats.norm.ppf(1-theta) * scale - mu
        VaR    = - ( mean_forecast + np.sqrt(var_forecast) * unconditional_VaR ) 
        CVaR   = -( np.sqrt(var_forecast) * (scipy.stats.norm.pdf(scipy.stats.norm.ppf(1-theta))/(theta)) - mean_forecast )
        return VaR , CVaR

