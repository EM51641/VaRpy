import numpy as np

def Sorting(return_matrix,mean,volatility,isevt) : 
        z  = (return_matrix - mean)/volatility 
        z_sorted = - np.sort(z[~np.isnan(z)]) #Remove NaNs
        if isevt :
                N = int(np.round(return_matrix.size * 0.10)) #Retained terms
                return N , z_sorted
        else : 
                return z_sorted


def Extract_Excess_Innovations(return_matrix,mean,volatility,isevt = False):
        if isevt :
                N,z_sorted = Sorting(return_matrix,mean,volatility,isevt) #Retained terms
                selected_z_terms = z_sorted[:N]
                selected_ret_terms = - np.sort(return_matrix)[:N]
                Excess_innovations = selected_z_terms - selected_z_terms[-1]
                Excess_returns = selected_ret_terms 
                return Excess_innovations,Excess_returns,selected_z_terms[-1]
        else : 
                z_sorted = Sorting(return_matrix,mean,volatility,isevt) #Retained terms
                Excess_innovations = z_sorted
                return Excess_innovations
