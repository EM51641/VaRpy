import numpy as np 
from varpy import EVT_VaR , Normal_VaR , Student_VaR



def Backtest(return_mtx :np.ndarray ,maximum_data_point : np.int64 ,step : np.int64, theta : np.float64 ,model = 'gaussian') : 
    """
    Backtest the VaR model and test its time significance

    Parameters
    -------
    
    return_mtx :np.ndarray
    maximum_data_point : np.int64
    step : np.int64
    theta : np.float64
    model = np.str

    Returns 
    -------

    VaR , CVaR
    """ 

    paths = int((return_mtx.size - maximum_data_point))#/step)
    VaR , CVaR = np.ones(paths),np.ones(paths)

    if model == 'Gaussian':
        VaR , CVaR = Gaussian_simulation(return_mtx , maximum_data_point , step , VaR , CVaR , paths , theta)
    
    elif model == 'EVT':
        VaR , CVaR = EVT_simulation(return_mtx , maximum_data_point , step , VaR , CVaR , paths , theta )
    
    elif model == 'Studentst' : 
        VaR , CVaR = Student_simulation(return_mtx , maximum_data_point , step , VaR , CVaR , paths, theta )

    return VaR , CVaR 


def Gaussian_simulation(return_mtx :np.ndarray ,maximum_data_point : np.int64 ,step : np.int64,VaR : np.ndarray \
                        , CVaR : np.ndarray , paths : np.int64 , theta :np.float64):
    
    for i,j in zip(range(0,return_mtx.size - maximum_data_point,step),range(0,paths)):
            returns = return_mtx[i:maximum_data_point+i]
            VaR[i:i+step] , CVaR[i:i+step] =  Normal_VaR(returns,theta,step)

    return VaR , CVaR 


def EVT_simulation(return_mtx :np.ndarray ,maximum_data_point : np.int64 ,step : np.int64,VaR : np.ndarray \
                    , CVaR : np.ndarray , paths : np.int64 , theta :np.float64):
    
    for i,j in zip(range(0,return_mtx.size - maximum_data_point,step),range(0,paths)):
            returns = return_mtx[i:maximum_data_point+i]
            VaR[i:i+step] , CVaR[i:i+step] =  EVT_VaR(returns,theta,step)

    return VaR , CVaR 

def Student_simulation(return_mtx :np.ndarray ,maximum_data_point : np.int64 ,step : np.int64,VaR : np.ndarray \
                        , CVaR : np.ndarray , paths : np.int64 , theta :np.float64):
    
    for i,j in zip(range(0,return_mtx.size - maximum_data_point,step),range(0,paths)):
            returns = return_mtx[i:maximum_data_point+i]
            VaR[i:i+step] , CVaR[i:i+step] =  Student_VaR(returns,theta,step)

    return VaR , CVaR