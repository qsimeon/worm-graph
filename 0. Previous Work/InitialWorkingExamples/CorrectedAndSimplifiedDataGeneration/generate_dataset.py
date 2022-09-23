"""
Where possible variable names match those in the original paper for readability:
https://www.frontiersin.org/articles/10.3389/fncom.2019.00008/full
"""

import numpy as np
from scipy import integrate, linalg
from tqdm import tqdm

n_neurons = 279
Gc = 0.1  # Cell membrane conductance (pS)
C = 0.015  # Cell Membrane Capacitance

Gg = np.load('Gg.npy')  # shape = (279,279)
# Gg[i,j] is the total conductivity of gap junctions between neurons i and j

gsyn = 1.0
Gs = np.load('Gs.npy')  # shape = (279,279)
# Gs[i,j] is the total conductivity of synapses to i from j # TODO "TO i FROM J"!? Make sure directionality isn't a problem

E_cell = -35.0  # Leakage potential (mV)

E = np.load('emask.npy')
E = (-48.0 * E).reshape(-1)  # shape = (279,)
# "directionality" of each neuron (0 if excitatory or −48 mV if inhibitory)

a_r = 1.0 / 1.5  # Synaptic activity rise time
a_d = 5.0 / 1.5  # Synaptic activity decay time
beta = 0.125  # Width of the sigmoid (mv^-1)

atol = 1e-3  # Absolute tolerance for ODE solver
dt = 0.01  # Time step size used for ODE solver


def compute_Vth():
    """
    Solves equation (11) for x (really V) in the original paper
    https://www.frontiersin.org/articles/10.3389/fncom.2019.00008/full

    Variable names follow those in the paper for readability

    TODO: I think the paper is wrong in its description of how this is calculated, and that this code is right
    """
    s_eq = a_r / (a_r + 2 * a_d)

    M1 = Gc * np.eye(n_neurons)
    M2 = np.diag(Gg.sum(
        axis=1)) - Gg  # M_2 matrix, but corrected for what I think is a mistake in the paper
    M3 = s_eq * np.diag(Gs.sum(axis=1))
    A = M1 + M2 + M3

    b1 = Gc * E_cell * np.ones(n_neurons)
    b3 = s_eq * (Gs @ E)
    b = b1 + b3
    # I_ext is zero since these simulations do not include external stimulation

    Vth = linalg.solve(A, b)

    return Vth


Vth = compute_Vth()  # shape = (279,)
# Threshold potential for each neuron, only depends on Gg and Gs, doesn't change during simulation


def run_simulation(total_steps: int) -> np.array:
    # data = np.zeros((total_steps, 2 * n_neurons))
    initial_voltages = np.random.uniform(-70, 30, size=n_neurons)
    initial_synaptic_activities = 10 ** (-2) * np.random.normal(size=n_neurons)
    initial_state = np.concatenate((initial_voltages, initial_synaptic_activities))
    # 2 * n_neurons because storing voltages and synaptic activities for each neuron
    # The scipy ODE interface requires them to be stored in a 1D array

    # # Initialize ODE solver
    # solver = integrate.ode(membrane_voltage_ode_rhs)
    # solver.set_integrator(
    #     name='vode',
    #     atol=atol,
    #     min_step=dt * 1e-6,
    #     method='bdf'
    # )
    # solver.set_initial_value(initial_state)
    #
    # for s in tqdm(range(total_steps)):
    #     solver.integrate(solver.t + dt)
    #     if not solver.successful():
    #         raise ValueError('Integration was not successful')
    #     data[s, :] = solver.y

    data = integrate.odeint(func=membrane_voltage_ode_rhs,
                            y0=initial_state,
                            t=np.linspace(0, 1, total_steps),
                            tfirst=True,
                            printmessg=True)

    # integrate.solve_ivp(fun=membrane_voltage_ode_rhs,
    #                     t_span=(0, total_steps),
    #                     y0=initial_state)

    np.save('test', data)




def membrane_voltage_ode_rhs(t: float, y: np.array):
    """
    Variable names match equations (2) through (6) in the original paper
    https://www.frontiersin.org/articles/10.3389/fncom.2019.00008/full
    """
    # First N values in y are voltages, second N are synaptic activities
    V, s = np.split(y, 2)

    I_gap = V * Gg.sum(axis=1) - Gg @ V  # Equation 3
    I_syn = V * (Gs @ s) - Gs @ (s * E)  # Equation 4
    # I_ext is zero since these simulations do not include external stimulation
    dV = (-Gc * (V - E_cell) - I_gap - I_syn) / C  # Equation 2

    phi = 1 / (1 + np.exp(-beta * (V - Vth)))  # Equation 6
    ds = a_r * phi * (1 - s) - a_d * s  # Equation 5

    return np.concatenate((dV, ds))


if __name__ == '__main__':
    run_simulation(20)
