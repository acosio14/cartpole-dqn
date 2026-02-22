
from envs.cartpole_ode import ordinay_differantial_equation as df

def euler(x, x_dot, timestep):
    
    return  x + timestep * x_dot


# Can't import df as ode because its a method of cartpoleenv
def runge_kutta_fourth_order(state, force, timestep, constants):
    k1 = df(state, force, constants)
    k2 = df(state + k1/2, force, constants)
    k3 = df(state + k2/2, force, constants)
    k4 = df(state + k3, force, constants)

    return state + (1/6)*timestep*(k1 + 2*k2 + 2*k3 + k4) # state_1 (n+1)