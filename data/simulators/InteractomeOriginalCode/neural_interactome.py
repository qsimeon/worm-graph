# coding: utf-8

import time
import os
from os.path import join as oj
import numpy as np
from scipy import integrate, signal, sparse, linalg


def run_NI_sim(input_Array, Gg_Static, Gs_Static, E,
               Gc = 0.1, C = 0.015, ggap = 1.0, gsyn = 1.0, Ec = -35.0,
               ar = 1.0/1.5, ad = 5.0/1.5, B = 0.125, Iext = 100000, rate = 0.025, offset = 0.15,
               max_time = 100, init_buffer = 50, t_Delta = 0.01, atol = 0.001, 
               out_dir = "saved_dynamics", out_file = "saved_dynamics"):
    
    """ driver function to simulate data from Neural Interactome
    
    Parameters
    ----------
    - input_Array : N-length vector of initial voltages; see data/neuron_names.txt for order of neurons
    - Gg_Static : N x N connection matrix for gap (electrical) junctions
    - Gs_Static : N x N connection matrix for synaptic (chemical) junctions
    - E : N x 1 directionality vector
    - Gc : cell membrane conductance (pS)
    - C : cell membrane capacitance
    - ggap : gap junction weight
    - gsyn : synaptic connection weight
    - Ec : leakage potential (mV)
    - ar : synaptic activity's rise time
    - ad : synaptic activity's decay time
    - B : width of the sigmoid (mv^-1)
    - Iext : parameter for voltage threshold computation
    - rate : rate of transition parameter    
    - offset : offset of transition parameter
    - max_time : number of time points to run simulation
    - init_buffer : number of time points for initialization buffer
    - t_Delta : step size for ODE
    - atol : tolerance for ODE solver
    - out_dir : name of directory to write output
    - out_file : name of file to write output
    
    Returns
    -------
    numpy matrix of size max_time x #Neurons with simulated dynamics
    and writes dynamics out to file "out_dir/out_file.npy"
    """
    
    # Error checking
    assert Gg_Static.shape == Gs_Static.shape
    assert Gg_Static.shape[0] == Gg_Static.shape[1]

    global transit_Mat, Gg_Dynamic, Gs_Dynamic, oldMask, t_Switch, t_Tracker, transit_End
    
    # Initialize
    N = Gg_Static.shape[0]
    t_Tracker = 0
    t_Switch = 0
    transit_End = 0.3
    EMat = np.tile(np.reshape(E, N), (N, 1))
    EffVth(Gg_Static, Gs_Static, E, Gc, Ec, ggap, gsyn, ar, ad)
    dt = t_Delta
    InitCond = 10**(-4)*np.random.normal(0, 0.94, 2*N)
    data_Mat = np.zeros((max_time + init_buffer, N))
    data_Mat[0, :] = InitCond[:N]
    
    transit_Mat = np.zeros((2, N))
    Gg_Dynamic = Gg_Static.copy()
    Gs_Dynamic = Gs_Static.copy()

    transit_Mask(input_Array, Iext)

    # Configuring the ODE Solver
    r = integrate.ode(membrane_voltageRHS).set_integrator('vode', atol = atol, 
                                                          min_step = dt*1e-6, method = 'bdf',
                                                          with_jacobian = True)
    r.set_initial_value(InitCond, 0)
    r.set_f_params(EMat, Gc, Ec, offset, Iext, ar, ad, B, C, rate)
    
    # Solve ODE over time
    k = 1
    while r.successful() and k < max_time + init_buffer:
        #print k
        r.integrate(r.t + dt)
        data = np.subtract(r.y[:N], Vth)
        data_Mat[k, :] = voltage_filter(data, 500, 1)
        t_Tracker = r.t
        k += 1
    
    out = data_Mat[init_buffer:, :]
    
    # Save Results
    if out_file != "saved_dynamics":
        np.save(oj(out_dir, out_file + '.npy'), out)
    
    return out


""" Mask transition """
def transit_Mask(input_Array, Iext):

    global t_Switch, oldMask, newMask, transit_End, Vth_Static

    transit_Mat[0,:] = transit_Mat[1,:]

    t_Switch = t_Tracker

    transit_Mat[1,:] = input_Array

    oldMask = transit_Mat[0,:]
    newMask = transit_Mat[1,:]

    Vth_Static = EffVth_rhs(Iext, newMask)
    transit_End = t_Switch + 0.3
    

def update_Mask(old, new, t, tSwitch, rate):

    return np.multiply(old, 0.5-0.5*np.tanh((t-tSwitch)/rate)) +\
            np.multiply(new, 0.5+0.5*np.tanh((t-tSwitch)/rate))


""" Ablation """
def modify_Connectome(ablation_Array, Gg_Static, Gs_Static,
                      E, Gc, Ec, ggap, gsyn, ar, ad):

    N = Gg_Static.shape[0]
    
    global Vth_Static, Gg_Dynamic, Gs_Dynamic

    apply_Col = np.tile(ablation_Array, (N, 1))
    apply_Row = np.transpose(apply_Col)

    apply_Mat = np.multiply(apply_Col, apply_Row)

    Gg_Dynamic = np.multiply(Gg_Static, apply_Mat)
    Gs_Dynamic = np.multiply(Gs_Static, apply_Mat)

    try:
        newMask

    except NameError:
        EffVth(Gg_Dynamic, Gs_Dynamic, E, Gc, Ec, ggap, gsyn, ar, ad)
        if np.sum(ablation_Array) != N:
            print('Neurons %s are ablated' %np.where(ablation_Array == False)[0])
        else:
            print("All Neurons healthy")
        print("EffVth Recalculated")
        
    else:
        EffVth(Gg_Dynamic, Gs_Dynamic)
        Vth_Static = EffVth_rhs(Iext, newMask)
        if np.sum(ablation_Array) != N:
            print('Neurons %s are ablated' %np.where(ablation_Array == False)[0])
        else:
            print("All Neurons healthy")
        print("EffVth Recalculated")
        print("Vth Recalculated")
        
        

""" Efficient V-threshold computation """
def EffVth(Gg, Gs, E, Gc, Ec, ggap, gsyn, ar, ad):

    N = Gg.shape[0]
    
    Gcmat = np.multiply(Gc, np.eye(N))
    EcVec = np.multiply(Ec, np.ones((N, 1)))

    M1 = -Gcmat
    b1 = np.multiply(Gc, EcVec)

    Ggap = np.multiply(ggap, Gg)
    Ggapdiag = np.subtract(Ggap, np.diag(np.diag(Ggap)))
    Ggapsum = Ggapdiag.sum(axis = 1)
    Ggapsummat = sparse.spdiags(Ggapsum, 0, N, N).toarray()
    M2 = -np.subtract(Ggapsummat, Ggapdiag)

    Gs_ij = np.multiply(gsyn, Gs)
    s_eq = round((ar/(ar + 2 * ad)), 4)
    sjmat = np.multiply(s_eq, np.ones((N, N)))
    S_eq = np.multiply(s_eq, np.ones((N, 1)))
    Gsyn = np.multiply(sjmat, Gs_ij)
    Gsyndiag = np.subtract(Gsyn, np.diag(np.diag(Gsyn)))
    Gsynsum = Gsyndiag.sum(axis = 1)
    M3 = -sparse.spdiags(Gsynsum, 0, N, N).toarray()

    b3 = np.dot(Gs_ij, np.multiply(s_eq, E))

    M = M1 + M2 + M3

    global LL, UU, bb

    (P, LL, UU) = linalg.lu(M)
    bbb = -b1 - b3
    bb = np.reshape(bbb, N)


def EffVth_rhs(Iext, InMask):

    InputMask = np.multiply(Iext, InMask)
    b = np.subtract(bb, InputMask)

    Vth = linalg.solve_triangular(UU, linalg.solve_triangular(LL, b, lower = True, check_finite=False), check_finite=False)

    return Vth


def voltage_filter(v_vec, vmax, scaler):
    
    filtered = vmax * np.tanh(scaler * np.divide(v_vec, vmax))
    
    return filtered


""" Right hand side """
def membrane_voltageRHS(t, y, EMat, Gc, Ec, offset, Iext, ar, ad, B, C, rate):
    
    N = Gg_Dynamic.shape[0]
    
    """ Split the incoming values """
    Vvec, SVec = np.split(y, 2)

    """ Gc(Vi - Ec) """
    VsubEc = np.multiply(Gc, (Vvec - Ec))

    """ Gg(Vi - Vj) Computation """
    Vrep = np.tile(Vvec, (N, 1))
    GapCon = np.multiply(Gg_Dynamic, np.subtract(np.transpose(Vrep), Vrep)).sum(axis = 1)

    """ Gs*S*(Vi - Ej) Computation """
    VsubEj = np.subtract(np.transpose(Vrep), EMat)
    SynapCon = np.multiply(np.multiply(Gs_Dynamic, np.tile(SVec, (N, 1))), VsubEj).sum(axis = 1)

    global InMask, Vth

    if t >= t_Switch and t <= transit_End:

        InMask = update_Mask(oldMask, newMask, t, t_Switch + offset, rate)
        Vth = EffVth_rhs(Iext, InMask)

    else:

        InMask = newMask
        Vth = Vth_Static

    """ ar*(1-Si)*Sigmoid Computation """
    SynRise = np.multiply(np.multiply(ar, (np.subtract(1.0, SVec))),
                          np.reciprocal(1.0 + np.exp(-B*(np.subtract(Vvec, Vth)))))

    SynDrop = np.multiply(ad, SVec)

    """ Input Mask """
    Input = np.multiply(Iext, InMask)

    """ dV and dS and merge them back to dydt """
    dV = (-(VsubEc + GapCon + SynapCon) + Input)/C
    dS = np.subtract(SynRise, SynDrop)

    return np.concatenate((dV, dS))