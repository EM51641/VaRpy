import scipy


def Binomial_test(Violations_nb,Total_data_points,theta):
    Z = scipy.stats.binomtest(Violations_nb, Total_data_points, theta).pvalue
    return Z