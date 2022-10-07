#@title Quilee Simeon made EDITS to M. Skuhersky's `generate_dataset.py` script below.
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

ggap = 1.0 # gap junction weight
gsyn = 1.0 # synaptic connection weight

def compute_Vth():
    """
    Solves equation (11) for x (really V) in the original paper
    https://www.frontiersin.org/articles/10.3389/fncom.2019.00008/full

    Variable names follow those in the paper for readability

    TODO: I think the paper is wrong in its description of how this is calculated, and that this code is right
    """
    s_eq = a_r / (a_r + 2 * a_d)

    M1 = Gc * np.eye(n_neurons)
    M2 = np.diag(Gg.sum(axis=1))
    # M2 = np.diag(Gg.sum(
    #     axis=1)) - Gg  # M_2 matrix, but corrected for what I think is a mistake in the paper
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

    data = integrate.odeint(func=membrane_voltage_ode_rhs,
                            y0=initial_state,
                            t=np.linspace(0, 1, total_steps),
                            Dfun=compute_jacobian,
                            tfirst=True,
                            printmessg=True)

    np.save('data', data)
    return data


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


def compute_jacobian(t, y):

    N = Gg.shape[0]
    EMat = np.tile(np.reshape(E, N), (N, 1))

    Vvec, SVec = np.split(y, 2)
    Vrep = np.tile(Vvec, (N, 1))

    J1_M1 = -np.multiply(Gc, np.eye(N))
    Ggap = np.multiply(ggap, Gg)
    Ggapsumdiag = -np.diag(Ggap.sum(axis = 1))
    J1_M2 = np.add(Ggap, Ggapsumdiag) 
    Gsyn = np.multiply(gsyn, Gs)
    J1_M3 = np.diag(np.dot(-Gsyn, SVec))

    J1 = (J1_M1 + J1_M2 + J1_M3) / C

    J2_M4_2 = np.subtract(EMat, np.transpose(Vrep))
    J2 = np.multiply(Gsyn, J2_M4_2) / C

    sigmoid_V = np.reciprocal(1.0 + np.exp(-B*(np.subtract(Vvec, Vth))))
    J3_1 = np.multiply(ar, 1 - SVec)
    J3_2 = np.multiply(B, sigmoid_V)
    J3_3 = 1 - sigmoid_V
    J3 = np.diag(np.multiply(np.multiply(J3_1, J3_2), J3_3))

    J4 = np.diag(np.subtract(np.multiply(-ar, sigmoid_V), ad))

    J_row1 = np.hstack((J1, J2))
    J_row2 = np.hstack((J3, J4))
    J = np.vstack((J_row1, J_row2))

    return J

    if __name__ == "__main__":
        #@title Simulated C. elegans neuron voltage traces with `generate_dataset.py`. 
        #@markdown The output is a file called data.npy.
        data = run_simulation(100)