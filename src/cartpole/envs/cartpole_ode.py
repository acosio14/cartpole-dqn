import numpy as np

def ordinay_differantial_equation(state, force, constants):
    
    mp, M, L, g = constants

    x, x_dot, theta, theta_dot = state
    
    temp = (force + mp*L*theta_dot**2*np.sin(theta)) / (M+mp)
    
    theta_ddot = (g*np.sin(theta) - np.cos(theta)*temp) / (L*(4/3 - mp*np.cos(theta)**2/(M+mp)))
    
    x_ddot = temp - (mp*L*theta_ddot*np.cos(theta))/(M+mp)

    return np.array([x_dot, x_ddot, theta_dot, theta_ddot])