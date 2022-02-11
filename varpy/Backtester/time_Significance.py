import numpy as np
import tabulate
from varpy.Backtester.chtest import Christophersen_computation , POF , TUFF , Hass
from varpy.Backtester.Binomial_test import Binomial_test
from varpy.Backtester.Quantile_ratio import Qt_ratio

class Testing :  


    def __init__(self,return_matrix,VaR,CVaR,maximum_data_point,theta):

        self.VaR_violations  = self.count_Violation(return_matrix,maximum_data_point,VaR) 
        self.CVaR_violations = self.count_Violation(return_matrix,maximum_data_point,CVaR) 
        self.VaR_violation_mtx = self.Violation_mtx(return_matrix,maximum_data_point,VaR) 
        self.CVaR_violation_mtx = self.Violation_mtx(return_matrix,maximum_data_point,CVaR) 
        self.log_ccr_VaR = Christophersen_computation(self.VaR_violation_mtx, theta,self.VaR_violations)
        self.bin_test_VaR = Binomial_test(self.VaR_violations, self.VaR_violation_mtx.size,theta)
        self.Kupiec_test = POF(self.VaR_violations,theta,self.VaR_violation_mtx.size)
        self.Tuff_test = TUFF(self.VaR_violation_mtx,self.VaR_violations,theta)
        self.Hass_test = Hass(self.VaR_violation_mtx,self.VaR_violations,theta,self.VaR_violation_mtx.size)
        self.Q_ratio = Qt_ratio(VaR,CVaR)
        self.summary = self.summary(self.VaR_violations,self.CVaR_violations,self.log_ccr_VaR ,\
                                    self.bin_test_VaR,self.Q_ratio,self.Kupiec_test,self.Tuff_test,\
                                    self.Hass_test)

    def Violation_mtx(self,return_matrix,maximum_data_point,VaR):
        violations = np.where(return_matrix[maximum_data_point:] - VaR > 0, 1 , 0)
        return violations

    def count_Violation(self,return_matrix,maximum_data_point,VaR):
        count_violations = (return_matrix[maximum_data_point:] - VaR < 0).sum()
        return count_violations
    
    def summary(self,VaR_violations_nb,CVaR_violations_nb,log_ccr_VaR ,bin_test_VaR,Q_ratio,Kupiec_test,TUFF_test,Hass_test) :

        top_right = [('Number of VaR Violations:',
                      "%#8.3f" % VaR_violations_nb),
                      ('Number of CVaR Violations:',
                      "%#8.3f" % CVaR_violations_nb),
                     ('Christophersen Test:',
                      "%#8.3f" % log_ccr_VaR),
                     ('Prob (Z-statistic):', "%#8.4g" % bin_test_VaR),
                     ('Quantile ratio:', Q_ratio),
                     ('Kupiec Test:',
                      "%#8.3f" % Kupiec_test),
                     ('TUFF Test:',
                      "%#8.3f" % TUFF_test),
                     ('Hass Test:',
                      "%#8.3f" % Hass_test)
                     ]

        synoptic = tabulate.tabulate(top_right ,headers = ['Outputs','Statistics'])

        return synoptic




        





