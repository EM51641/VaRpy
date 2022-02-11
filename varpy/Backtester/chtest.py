from joblib import Parallel
import numpy as np
import numba as nb


@nb.njit()
def chris_test(Matrix:np.ndarray):
    m_00 , m_01 , m_10 , m_11 = 0 , 0 , 0 , 0
    for i in range(0,Matrix.size - 1):
        if   Matrix[i] == 0 and Matrix[i+1] == 0 :
            m_00=+1
        elif Matrix[i] == 0 and Matrix[i+1] != 0 :
            m_01=+1
        elif Matrix[i] != 0 and Matrix[i+1] == 0 :
            m_10=+1
        elif Matrix[i] != 0 and Matrix[i+1] != 0 :
            m_11=+1
    return m_00 , m_01 , m_10 , m_11

def Christophersen_computation(Matrix,prob,violations_number):

    m_00 , m_01 , m_10 , m_11 = chris_test(Matrix)
    term_1 = CCI(m_00 , m_01 , m_10 , m_11)
    term_2 = POF(violations_number,prob,Matrix.size)
    return term_1 + term_2

def CCI(m_00 , m_01 , m_10 , m_11):

    pi  = ( m_01 + m_11 ) / ( m_00 + m_01 + m_10 + m_11 )
    pi1 = m_11 / ( m_10 + m_11 )
    pi0 = m_01 / ( m_00 + m_01 )
    A   =  2 * ( m_00 + m_01 ) * np.log( 1 - pi ) - 2 * ( m_10 + m_11 ) * np.log( pi )
    B   =  2 * m_00 * np.log( 1 - pi0 ) + 2 * m_01 * np.log( pi0 ) + 2 * m_10 * np.log( 1 - pi1 ) + 2 * m_11 * np.log( pi1 )
    return - A + B


def POF(total_violations,prob,total_points):
    A = 2 * ( total_points - total_violations ) * np.log( 1 - prob ) + 2 * total_violations * np.log(prob)
    B = 2 * ( total_points - total_violations ) * np.log( 1 - (total_violations/total_points)) + 2 * total_violations * np.log( total_violations / total_points )
    return - A + B


def TUFF(Matrix,total_violations,prob) :
    inter_time_counter = first_failure(Matrix)
    A = 2 * np.log( prob ) + ( inter_time_counter - 1 ) *  np.log( 1 - prob)
    B = 2 * ( 1/inter_time_counter ) + ( inter_time_counter - 1 ) *  np.log( 1 - 1 / inter_time_counter )
    return - A + B

def Hass(Matrix,total_violations,prob,total_points) : 
    term_1 = POF(total_violations,prob,total_points)
    term_2 = TBFI_test(Matrix,prob)
    return term_1 + term_2

@nb.njit()
def TBFI_test(Matrix:np.ndarray,prob : np.float64):
    A , B  , inter_time_counter = 0 , 0 , 0 
    for i in range(0,Matrix.size - 1):
        if   Matrix[i] == 0 :
            A =+ 2 * np.log( prob ) + (i - inter_time_counter - 1 ) *  np.log( 1 - prob)
            B =+ 2 * ( i - inter_time_counter - 1 ) + ( i - inter_time_counter - 1 ) *  np.log( 1 - 1 /( i - inter_time_counter))
            inter_time_counter = i
    return - A + B 



def first_failure(Matrix):
    First_fail = np.where(Matrix == 0)[0][0]
    return First_fail