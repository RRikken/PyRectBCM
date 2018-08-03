import copy
from math import ceil
import numpy as np
import numexpr as ne
from sympy import EulerGamma

g = 9.81
eg = np.float64(EulerGamma.evalf(16))

def r_iteration(Input, l2, phij):
    """ Iterates the friction coefficient and yields flow velocities.
    Computes the flow solution in the inlets and basin, and iterates the velocity scale
    until it matches that found in the hydrodynamic solution (given a certain tolerance).

    Args:
        Input (Class): Class containing all data for the various geograpical elements.
            Consists of the Basin, OpenInlets, Ocean, and Pars classes.
        l2 (np.ndarray): 3d-array containing the L2-norm for the eigenfunctinos \phi(m, n, j).
        phij (np.ndarray): 3d-array containing the integrated eigenfunctions over all j inlets.

    Returns:
        returnObject (dict):
            uj (np.ndarray): flow velocity in the inlets.
            ub (float): flow velocity in the basin.
            mui2 (np.ndarray): frictional correction factor for the inlets.
            mub2 (complex): frictional correction factor.
    """
    Basin = Input.Basin
    Inlets = Input.OpenInlets
    Ocean = Input.Ocean
    Pars = Input.Pars

    A1 = np.zeros((Inlets.numinlets, Inlets.numinlets), dtype=complex)
    A2 = np.zeros((Inlets.numinlets, Inlets.numinlets), dtype=complex)
    A3 = np.zeros((Inlets.numinlets, Inlets.numinlets), dtype=complex)
    A = np.zeros((Inlets.numinlets, Inlets.numinlets), dtype=complex)

    A1c = 1j*Ocean.tidefreq*Inlets.lengths/g * np.eye(Inlets.numinlets)

    a_width = np.squeeze(Inlets.widths)
    beta = Inlets.widths/a_width[:, np.newaxis]
    betap = (beta+1)/2 * (1 - np.eye(np.sum(Inlets.numinlets)))
    betam = (beta-1)/2
    alpha = abs(np.transpose(Inlets.locations) - Inlets.locations)/a_width[:, np.newaxis]
    alpha[alpha == 0] = 1e-7
    A2 = ((Inlets.depths*Ocean.tidefreq*a_width[:, np.newaxis])
                                        / (2*g*Ocean.depth)
                                        * (beta + 2j/np.pi*((3/2 - eg)*beta
                                        + betam**2*np.log(Ocean.ko*a_width[:, np.newaxis]/2*np.sqrt(alpha**2-betam**2))
                                        - betap**2*np.log(Ocean.ko*a_width[:, np.newaxis]/2*np.sqrt(alpha**2-betap**2))
                                        + alpha*(betam*np.log((alpha+betam)/(alpha-betam)) - betap*np.log((alpha+betap)/(alpha-betap)))
                                        + alpha**2*np.log(np.sqrt((alpha**2-betam**2)/(alpha**2-betap**2)))
                                        )))
    A2self = np.squeeze((Inlets.depths*Ocean.tidefreq*Inlets.widths)/(2*g*Ocean.depth)
            *(1 + 2j/np.pi*(3/2 - eg - np.log(Ocean.ko*Inlets.widths/2))))
    A2[np.eye(Inlets.numinlets) == 1] = A2self

    A3c = (Ocean.tidefreq/(g*1j)
            * np.reshape(
                    (Inlets.widths[np.newaxis, np.newaxis, :, :]*Inlets.depths[np.newaxis, np.newaxis, :, :]
                    * (phij[:, :, :, np.newaxis]*phij[:, :, np.newaxis, :])
                    / (l2[:, :, :, np.newaxis]*Basin.depth)),
                    (-1, np.sum(Inlets.numinlets), np.sum(Inlets.numinlets))
                    )
           )
    cmnqc = (Inlets.widths[np.newaxis, :, :]*Inlets.depths[np.newaxis, :, :] * (phij[:, :, :])
            / (Basin.depth))
    ubc = cmnqc/np.sqrt(l2)
    kmn2 = np.reshape(Pars.kmn2[:, :, np.newaxis, np.newaxis], (-1, 1, 1))
    kb2 = Basin.kb**2

    uj = Inlets.uj
    ub = Basin.ub
    res = 42
    r = 1
    while res > 1e-10 and r < 50:
        rb = 8/(3*np.pi)*Basin.cd*ub
        rj = 8/(3*np.pi)*Inlets.cd*uj
        mub2 = 1 - 1j*rb/(Ocean.tidefreq*Basin.depth)
        mui2 = 1 - 1j*rj/(Ocean.tidefreq*Inlets.depths)

        A1 = A1c*mui2
        A3 = mub2*ne.evaluate('sum(A3c/(kmn2 - mub2*kb2), axis=0)')
        # A3 = mub2*np.sum(A3c/(kmn2 - mub2*kb2), axis = 0)

        A = A1 + A2 - A3
        B = np.squeeze(-Ocean.tideamp*np.exp(1j*Ocean.wavenumber*Inlets.locations))
        sol = np.linalg.solve(A, B)
        ubn = (np.sqrt(1/(Basin.width*Basin.length) * np.sum(
                Pars.kmn2 * np.abs(
                    np.sum(ubc*sol[np.newaxis, np.newaxis, :], axis = 2)
                               /(Pars.kmn2 - mub2*Basin.kb**2)
                                  )**2)))
        ires = np.abs(np.squeeze(uj) - abs(sol))
        bres = np.abs(ub - ubn)
        res = np.max(np.r_[ires, bres])
        uj = (abs(abs(sol[np.newaxis, :])) + uj)/2
        ub = (ub + abs(ubn))/2
        r = r + 1

    return uj, ub, mui2, mub2

def rec_model(Input, silent = None):
    """ Runs the barrier coast model.
    Based on input geometry and parameters,
    computes the barrier coast evolution
    over a given time-period.

    Args:
        Input (Input class): Class containing the input data as generated
            by the input_generator.
        silent (boolean): Flag for output per timestep

    Returns:
        Output (Input class): Class containing the updated inputed classes,
            containing the computed evolution of the barrier coast.
    """

    Output = copy.deepcopy(Input)
    Basin = Output.Basin
    Inlets = Output.Inlets
    Ocean = Output.Ocean
    Pars = Output.Pars
    Output.OpenInlets = copy.copy(Inlets)
    t = Pars.tstart
    ndt = ceil((Pars.tend-Pars.tstart)/Pars.dt)
    Inlets.wit = np.repeat(Inlets.wit, ndt, axis = 0)
    l2 = np.full((Pars.mtrunc+1, Pars.ntrunc+1, Basin.numinlets), Basin.length*Basin.width)
    l2[1:, :, :] = l2[1:, :, :]/2
    l2[:, 1:, :] = l2[:, 1:, :]/2
    signs = np.array([1, -1])
    signs = np.tile(signs[:, np.newaxis, np.newaxis], (ceil((Pars.mtrunc+1)/2), Pars.ntrunc+1, Basin.numinlets))
    uj = Inlets.uj

    while t < Pars.tend and max(abs(1-uj[uj!=0])) > 1e-15 and np.sum(Inlets.widths) > 0:
        inlets_zip = np.squeeze(Inlets.widths > 0)
        phij = np.zeros(l2.shape)
        phij[:, 0, inlets_zip] = 1
        phij[:, 1:, inlets_zip] = (
            Basin.width/(Pars.nrange[:, 1:, np.newaxis]*np.pi*Inlets.widths[np.newaxis, :, inlets_zip])
            * ( np.sin(Pars.nrange[:, 1:, np.newaxis]*np.pi/Basin.width*(
                Inlets.locations[np.newaxis, :, inlets_zip]
                + Inlets.widths[np.newaxis, :, inlets_zip]/2))
            -np.sin(Pars.nrange[:, 1:, np.newaxis]*np.pi/Basin.width*(
                Inlets.locations[np.newaxis, :, inlets_zip]
                - Inlets.widths[np.newaxis, :, inlets_zip]/2)) ))
        phij = phij * signs[0:Pars.mtrunc+1, 0:Pars.ntrunc+1, 0:(Basin.numinlets)]

        Output.OpenInlets.widths = Inlets.widths[:, inlets_zip]
        Output.OpenInlets.depths = Inlets.depths[:, inlets_zip]
        Output.OpenInlets.lengths = Inlets.lengths[inlets_zip]
        Output.OpenInlets.locations = Inlets.locations[:, inlets_zip]
        Output.OpenInlets.uj = Inlets.uj[:, inlets_zip]
        Output.OpenInlets.cd - Inlets.cd
        Output.OpenInlets.numinlets = np.sum(inlets_zip)

        uj, ub, mui2, mub2 = r_iteration(Output, l2[:, :, inlets_zip], phij[:, :, inlets_zip])

        Inlets.uj[:, inlets_zip] = uj
        Basin.ub = ub

        da = Inlets.sedimport/Inlets.lengths * ((Inlets.uj/Inlets.ueq)**3 - 1) * Pars.dt
        newi = Inlets.widths**2 + da/Inlets.shape
        nix = np.squeeze(newi > 0)
        Inlets.widths[:, nix] = (np.sqrt(newi[:, nix]))
        Inlets.widths[:, ~nix] = 0

        Inlets.depths = Inlets.shape * Inlets.widths
        Inlets.uj[Inlets.widths == 0] = 0

        if np.isnan(Inlets.widths).any():
            print(Inlets.wit)
            print(Inlets.depths)
            print(Inlets.uj, Basin.ub, mui2, mub2, da)
            raise NameError ("aiaiai")

        if silent is None:
            print(t, Inlets.uj, da)
        t = t + Pars.dt
        tix = round((t-Pars.tstart)/Pars.dt)
        Inlets.wit[tix-1, :] = Inlets.widths

    Inlets.mui = mui2
    Basin.mub = mub2
    return Output
