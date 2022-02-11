import numpy as np 
import scipy
from varpy.tools.innovation_finder import Extract_Excess_Innovations
from varpy.tools.arch_mdl import Arch_data

 
def EVT_VaR ( return_matrix, theta,Horizon):

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
        Excess_innovations ,Excess_returns,last_innovation_term = Extract_Excess_Innovations(return_matrix , mean_forecast , conditional_volatility , True )
        ret_params,innovation_params = Dist_parameters(Excess_returns, Excess_innovations )
        VaR,CVaR = Var_CVaR_extractor(mean_forecast,var_forecast,theta,Excess_innovations,innovation_params,return_matrix.size,Excess_innovations.size,last_innovation_term)
        return VaR , CVaR

def Dist_parameters(excess_returns, excess_innovations ):
    ret_params = scipy.stats.genpareto.fit(excess_returns , floc=0)
    innovation_params = scipy.stats.genpareto.fit(excess_innovations , floc=0) 
    return ret_params , innovation_params

def Var_CVaR_extractor(mean_forecast,var_forecast,theta,excess_innovations,innovation_params,total_length,sample_length,last_innovation_term):
    uncond_Var = last_innovation_term + (innovation_params[2]/innovation_params[0]) * ( ( total_length * theta / sample_length) ** (- innovation_params[0]) - 1 )
    VaR = -( mean_forecast + np.sqrt(var_forecast) * uncond_Var )
    CVaR = -( mean_forecast + uncond_Var * np.sqrt(var_forecast) * (  1 / (1 - innovation_params[0]) + \
            (innovation_params[2] - innovation_params[0] * last_innovation_term ) / ((1-innovation_params[0]) * uncond_Var) ) )
    return VaR , CVaR