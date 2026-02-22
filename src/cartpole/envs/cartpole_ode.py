import numpy as np

def ordinay_differantial_equation(state, force, constants):
    
    mp = constants[0]
    M = constants[1]
    L = constants[2]
    g = constants[3]
    
    # Get Derivatives
    x_dot= state[1] # cart velocity
    x_ddot = (
        -mp * L * np.sin(state[2]) * np.square(state[3])
        + mp * g * np.cos(state[2]) * np.sin(state[3])
        + force / (M + mp * np.square(np.sin(state[2]))) 
    ) # cart acceleration
    theta_dot = state[3] # angular velocity
    theta_ddot = (
        -(M + mp) * g * np.sin(state[2]) 
        - mp * L * np.sin(state[2]) * np.cos(state[2]) * np.square(state[3]) 
        - force * np.cos(state[2]) / (L * (M + mp * np.square(np.sin(state[2]))) ) 
    )# angular acceleration

    return np.array([x_dot, x_ddot, theta_dot, theta_ddot])