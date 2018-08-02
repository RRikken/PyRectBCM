import copy
from math import ceil
import numpy as np
import numexpr as ne
from sympy import EulerGamma

g = 9.81
eg = np.float64(EulerGamma.evalf(16))

def r_iteration(Input, l2, phij, inlets_zip):
    global g
    global eg
    Basin = Input.Basin
    Inlets = Input.Inlets
    Ocean = Input.Ocean
    Pars = Input.Pars

    A1 = np.zeros((Basin.numinlets, Basin.numinlets), dtype=complex)
    A2 = np.zeros((Basin.numinlets, Basin.numinlets), dtype=complex)
    A3 = np.zeros((Basin.numinlets, Basin.numinlets), dtype=complex)
    A = np.zeros((Basin.numinlets, Basin.numinlets), dtype=complex)

    a_width = np.squeeze(Inlets.widths)
    beta = Inlets.widths[:, inlets_zip]/a_width[inlets_zip, np.newaxis]
    betap = (beta+1)/2 * (1 - np.eye(np.sum(inlets_zip)))
    betam = (beta-1)/2
    alpha = abs(np.transpose(Inlets.locations[:, inlets_zip]) - Inlets.locations[:, inlets_zip])/a_width[inlets_zip, np.newaxis]
    alpha[alpha == 0] = 1e-7
    A2[np.ix_(inlets_zip, inlets_zip)] = ((Inlets.depths[:, inlets_zip]*Ocean.tidefreq*a_width[inlets_zip, np.newaxis])
                                        / (2*g*Ocean.depth)
                                        * (beta + 2j/np.pi*((3/2 - eg)*beta
                                        + betam**2*np.log(Ocean.ko*a_width[inlets_zip, np.newaxis]/2*np.sqrt(alpha**2-betam**2))
                                        - betap**2*np.log(Ocean.ko*a_width[inlets_zip, np.newaxis]/2*np.sqrt(alpha**2-betap**2))
                                        + alpha*(betam*np.log((alpha+betam)/(alpha-betam)) - betap*np.log((alpha+betap)/(alpha-betap)))
                                        + alpha**2*np.log(np.sqrt((alpha**2-betam**2)/(alpha**2-betap**2)))
                                        )))
    A2self = np.squeeze((Inlets.depths*Ocean.tidefreq*Inlets.widths)/(2*g*Ocean.depth)
            *(1 + 2j/np.pi*(3/2 - eg - np.log(Ocean.ko*Inlets.widths/2))))
    A2self[~inlets_zip] = 0
    A2[np.eye(Basin.numinlets) == 1] = A2self
    A3c = Ocean.tidefreq/(g*1j)
    uj = Inlets.uj
    ub = Basin.ub
    res = 42
    r = 1

    cmnqc = (Inlets.widths[np.newaxis, :, :]*Inlets.depths[np.newaxis, :, :] * (phij[:, :, :])
            / (Basin.depth))
    ubc = cmnqc/np.sqrt(l2)
    kmn2 = np.reshape(Pars.kmn2[:, :, np.newaxis, np.newaxis], (-1, 1, 1))
    kb2 = Basin.kb**2

    A1c = 1j*Ocean.tidefreq*Inlets.lengths/g * np.eye(Basin.numinlets)
    A3cn = np.reshape((Inlets.widths[np.newaxis, np.newaxis, :, inlets_zip]*Inlets.depths[np.newaxis, np.newaxis, :, inlets_zip]
           * (phij[:, :, inlets_zip, np.newaxis]*phij[:, :, np.newaxis, inlets_zip])
           / (l2[:, :, inlets_zip, np.newaxis]*Basin.depth)), (-1, np.sum(inlets_zip), np.sum(inlets_zip)))
    while res > 1e-10 and r < 50:
        rb = 8/(3*np.pi)*Basin.cd*ub
        rj = 8/(3*np.pi)*Inlets.cd*uj
        mub2 = 1 - 1j*rb/(Ocean.tidefreq*Basin.depth)
        mui2 = 1 - 1j*rj/(Ocean.tidefreq*Inlets.depths)

        A1 = A1c*mui2
        A3[np.ix_(inlets_zip, inlets_zip)] = A3c*mub2*ne.evaluate('sum(A3cn/(kmn2 - mub2*kb2), axis=0)')
        # A3[np.ix_(inlets_zip, inlets_zip)] = A3c*mub2*np.sum(A3cn/(kmn2 - mub2*kb2), axis = 0)

        A = A1 + A2 - A3
        B = np.squeeze(-Ocean.tideamp*np.exp(1j*Ocean.wavenumber*Inlets.locations))
        sol = np.zeros(Basin.numinlets, dtype=complex)
        sol[inlets_zip] = np.linalg.solve(A[np.ix_(inlets_zip, inlets_zip)], B[inlets_zip])
        ubn = (np.sqrt(1/(Basin.width*Basin.length) * np.sum(Pars.kmn2
               * np.abs(np.sum(ubc*sol[np.newaxis, np.newaxis, :], axis = 2)
                               /(Pars.kmn2 - mub2*Basin.kb**2))**2)))
        ires = np.abs(np.squeeze(uj) - abs(sol))
        bres = np.abs(ub - ubn)
        res = np.max(np.r_[ires, bres])
        uj = (abs(abs(sol[np.newaxis, :])) + uj)/2
        ub = (ub + abs(ubn))/2
        r = r + 1

    return uj, ub, mui2, mub2

def rec_model(Input):
    Output = copy.deepcopy(Input)
    Basin = Output.Basin
    Inlets = Output.Inlets
    Ocean = Output.Ocean
    Pars = Output.Pars

    t = Pars.tstart
    ndt = ceil((Pars.tend-Pars.tstart)/Pars.dt)
    Inlets.wit = np.repeat(Inlets.wit, ndt, axis = 0)
    l2 = np.full((Pars.mtrunc+1, Pars.ntrunc+1, Basin.numinlets), Basin.length*Basin.width)
    l2[1:, :, :] = l2[1:, :, :]/2
    l2[:, 1:, :] = l2[:, 1:, :]/2
    signs = np.array([1, -1])
    signs = np.tile(signs[:, np.newaxis, np.newaxis], (ceil((Pars.mtrunc+1)/2), Pars.ntrunc+1, Basin.numinlets))
    # signs = signs[Pars.mtrunc, Pars.ntrunc, Basin.numinlets-1]
    phij = np.copy(l2)*0
    uj = Inlets.uj

    while t < Pars.tend and max(abs(1-uj[uj!=0])) > 1e-15 and np.sum(Inlets.widths) > 0:
        inlets_zip = np.squeeze(Inlets.widths > 0)
        phij = phij*0
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
        uj, ub, mui2, mub2 = r_iteration(Output, l2, phij, inlets_zip)
        # uj, ub, mui2, mub2 = r_iteration_wraper(Output, l2, phij, inlets_zip)
        da = Inlets.sedimport/Inlets.lengths * ((uj/Inlets.ueq)**3 - 1) * Pars.dt
        newi = Inlets.widths**2 + da/Inlets.shape
        nix = np.squeeze(newi > 0)
        Inlets.widths[:, nix] = (np.sqrt(newi[:, nix]))
        Inlets.widths[:, ~nix] = 0

        Inlets.depths = Inlets.shape * Inlets.widths
        uj[Inlets.widths == 0] = 0

        if np.isnan(Inlets.widths).any():
            print(Inlets.wit)
            print(Inlets.depths)
            print(uj,ub,mui2,mub2, da)
            raise NameError ("aiaiai")

        print(t, uj, da)
        t = t + Pars.dt
        tix = round((t-Pars.tstart)/Pars.dt)

        Inlets.wit[tix-1, :] = Inlets.widths
        Inlets.uj = uj
        Basin.ub = ub

    Inlets.mui = mui2
    Basin.mub = mub2
    return Output
