

def euler(df, x, x_dot, timestep):
    
    return  x + timestep * x_dot



def runge_kutta_fourth_order(df, state, force, timestep):
    k1 = df(state, force)
    k2 = df(state + k1/2, force)
    k3 = df(state + k2/2, force)
    k4 = df(state + k3, force)

    return state + (1/6)*timestep*(k1 + 2*k2 + 2*k3 + k4) # state_1 (n+1)