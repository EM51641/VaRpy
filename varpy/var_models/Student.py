import numpy as np 
from arch import arch_model
import scipy
from varpy.tools.innovation_finder import Extract_Excess_Innovations
from varpy.tools.arch_mdl import Arch_data
from scipy.stats import t
 
def Student_VaR(return_matrix, theta,Horizon): #500 datas needed 

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
        
        mean_forecast,var_forecast,conditional_volatility = Arch_data(return_matrix , Horizon, 'studentst' )
        innovations = Extract_Excess_Innovations(return_matrix , mean_forecast , conditional_volatility)
        alpha  = Dist_parameters(innovations)
        VaR,CVaR = Var_CVaR_extractor(mean_forecast,var_forecast,alpha,theta)   
        return VaR,CVaR

def Dist_parameters(innovations):
        params = scipy.stats.t.fit(innovations)[0] #alpha
        return params

def Var_CVaR_extractor(mean_forecast,var_forecast,alpha,theta):
        unconditional_VaR = np.sqrt((alpha-2)/alpha)*scipy.stats.t.ppf(1-theta,alpha)
        VaR    = - (mean_forecast + np.sqrt(var_forecast) * unconditional_VaR ) 
        CVaR   = - (scipy.stats.t.pdf(scipy.stats.t.ppf(1-theta,alpha),alpha)/theta) * ( alpha+scipy.stats.t.ppf(1-theta,alpha)**2)/(alpha-1) \
                * np.sqrt(var_forecast) - mean_forecast
        return VaR , CVaR

