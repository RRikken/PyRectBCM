import copy
from math import ceil
import numpy as np
import numexpr as ne
from sympy import EulerGamma
from pyrectbcm.data_classes import ModelData, Basin, Inlets, Ocean, Parameters
from pyrectbcm.constants import GRAVITY

eg = np.float64(EulerGamma.evalf(16))


def hydrodynamic_model(
        basin: Basin,
        inlets: Inlets,
        ocean: Ocean,
        parameters: Parameters,
        l2: np.ndarray,
        phij: np.ndarray
) -> dict:
    """ Iterates the friction coefficient and yields flow velocities.
    Computes the flow solution in the inlets and basin, and iterates the velocity scale
    until it matches that found in the hydrodynamic solution (given a certain tolerance).
    Solves the system :math:`\mathbf{A} \mathbf{\hat{u}} = \mathbf{f}`
    for the flow velocity amplitude :math:`\mathbf{\hat{u}}`. The forcing therm :math:`\mathbf{f}`
    consists of a tidal forcing, and matrix :math:`\mathbf{A} = \mathbf{A}_1 + \mathbf{A}_2 - \mathbf{A}_3`.
    The first term is associated with friction in the channel.
    The second with the sea impedance.
    The third term is associated with basin impedance.

    Args:
        basin (Basin)
        inlets (Inlets)
        ocean (Ocean)
        parameters (Pars)
        l2 (np.ndarray): 3d-array containing the L2-norm for the eigenfunctions :math:`\phi_{m, n}(x, y)`.
        phij (np.ndarray): 3d-array containing the integrated eigenfunctions over all j inlets.

    Returns:
        returnObject (dict):
            uj (np.ndarray): flow velocity in the inlets.
            ub (float): flow velocity in the basin.
            mui2 (np.ndarray): frictional correction factor for the inlets.
            mub2 (complex): frictional correction factor.
    """

    # Initialize main matrix A and three constructor matrices A1, A2, A3 such that A = A1 + A2 - A3
    A1 = np.zeros((Inlets.numinlets, Inlets.numinlets), dtype=complex)
    A2 = np.zeros((Inlets.numinlets, Inlets.numinlets), dtype=complex)
    A3 = np.zeros((Inlets.numinlets, Inlets.numinlets), dtype=complex)
    A = np.zeros((Inlets.numinlets, Inlets.numinlets), dtype=complex)

    # Fill the diagonal of A1
    A1c = 1j * Ocean.tidefreq * Inlets.lengths / g * np.eye(Inlets.numinlets)

    # Compute A2
    a_width = np.squeeze(Inlets.widths)
    beta = Inlets.widths / a_width[:, np.newaxis]
    betap = (beta + 1) / 2 * (1 - np.eye(np.sum(Inlets.numinlets)))
    betam = (beta - 1) / 2
    alpha = (
        abs(np.transpose(inlets.locations) - inlets.locations) / a_width[:, np.newaxis]
    )
    alpha[alpha == 0] = 1e-7  # Ensure that alpha**2 != 0
    A2 = (
        (inlets.depths * ocean.tidefreq * a_width[:, np.newaxis])
        / (2 * GRAVITY * ocean.depth)
        * (
            beta
            + (2j / np.pi)
            * (
                (3 / 2 - eg) * beta
                + betam ** 2
                * np.log(
                    ocean.ko
                    * a_width[:, np.newaxis]
                    / 2
                    * np.sqrt(alpha ** 2 - betam ** 2)
                )
                - betap ** 2
                * np.log(
                    ocean.ko
                    * a_width[:, np.newaxis]
                    / 2
                    * np.sqrt(alpha ** 2 - betap ** 2)
                )
                + alpha
                * (
                    betam * np.log((alpha + betam) / (alpha - betam))
                    - betap * np.log((alpha + betap) / (alpha - betap))
                )
                + alpha ** 2
                * np.log(np.sqrt((alpha ** 2 - betam ** 2) / (alpha ** 2 - betap ** 2)))
            )
        )
    )
    A2self = np.squeeze(
        (inlets.depths * ocean.tidefreq * inlets.widths)
        / (2 * GRAVITY * ocean.depth)
        * (1 + 2j / np.pi * (3 / 2 - eg - np.log(ocean.ko * inlets.widths / 2)))
    )
    A2[np.eye(inlets.numinlets) == 1] = A2self

    # Compute the friction independent part of A3
    A3c = (
        ocean.tidefreq
        / (GRAVITY * 1j)
        * np.reshape(
            (
                inlets.widths[np.newaxis, np.newaxis, :, :]
                * inlets.depths[np.newaxis, np.newaxis, :, :]
                * (phij[:, :, :, np.newaxis] * phij[:, :, np.newaxis, :])
                / (l2[:, :, :, np.newaxis] * basin.depth)
            ),
            (-1, np.sum(inlets.numinlets), np.sum(inlets.numinlets)),
        )
    )

    # Prepare for friction iteration
    cmnqc = (
        inlets.widths[np.newaxis, :, :]
        * inlets.depths[np.newaxis, :, :]
        * (phij[:, :, :])
        / (basin.depth)
    )
    ubc = cmnqc / np.sqrt(l2)
    kmn2 = np.reshape(parameters.kmn2[:, :, np.newaxis, np.newaxis], (-1, 1, 1))
    kb2 = basin.kb ** 2

    uj = inlets.uj
    ub = basin.ub
    res = 42
    r = 1

    # Friction iteration loop
    while res > 1e-10 and r < 50:
        rb = 8 / (3 * np.pi) * basin.cd * ub
        rj = 8 / (3 * np.pi) * inlets.cd * uj
        mub2 = 1 - 1j * rb / (ocean.tidefreq * basin.depth)
        mui2 = 1 - 1j * rj / (ocean.tidefreq * inlets.depths)

        A1 = A1c * mui2
        A3 = mub2 * ne.evaluate("sum(A3c/(kmn2 - mub2*kb2), axis=0)")
        # A3 = mub2*np.sum(A3c/(kmn2 - mub2*kb2), axis = 0)

        # Construct matrix A and solve linear system
        A = A1 + A2 - A3
        B = np.squeeze(
            -ocean.tideamp * np.exp(1j * ocean.wavenumber * inlets.locations)
        )
        sol = np.linalg.solve(A, B)

        # Compute velocity scale in the basin
        ubn = np.sqrt(
            1
            / (basin.width * basin.length)
            * np.sum(
                parameters.kmn2
                * np.abs(
                    np.sum(ubc * sol[np.newaxis, np.newaxis, :], axis=2)
                    / (parameters.kmn2 - mub2 * basin.kb ** 2)
                )
                ** 2
            )
        )

        # Determine residuals in inlets and basin
        ires = np.abs(np.squeeze(uj) - abs(sol))
        bres = np.abs(ub - ubn)
        res = np.max(np.r_[ires, bres])
        uj = (abs(abs(sol[np.newaxis, :])) + uj) / 2
        ub = (ub + abs(ubn)) / 2
        r = r + 1

    return uj, ub, mui2, mub2


def rec_model(model_input: ModelData, silent: bool = False) -> ModelData:
    """ Runs the barrier coast model.
    Based on input geometry and parameters,
    computes the barrier coast evolution
    over a given time-period.

    Args:
        model_input (ModelData class): Class containing the input data as generated
            by the input_generator.
        silent (boolean): Flag for output per timestep

    Returns:
        output (ModelData class): Class containing the updated ModelData classes,
            containing the computed evolution of the barrier coast.
    """

    # Create handles for convenience
    uj = Inlets.uj
    output = copy.deepcopy(model_input)
    basin = output.get_basin()
    inlets = output.get_inlets()
    ocean = output.get_ocean()
    parameters = output.get_parameters()

    # initialize parameters and arrays
    output.OpenInlets = copy.copy(inlets)
    open_inlets = output.OpenInlets

    t = parameters.tstart
    ndt = ceil((parameters.tend - parameters.tstart) / parameters.dt)
    inlets.wit = np.repeat(inlets.wit, ndt, axis=0)
    l2 = np.full(
        (parameters.mtrunc + 1, parameters.ntrunc + 1, basin.get_number_of_inlets()), basin.get_length() * basin.get_width()
    )
    l2[1:, :, :] = l2[1:, :, :] / 2
    l2[:, 1:, :] = l2[:, 1:, :] / 2
    signs = np.array([1, -1])
    signs = np.tile(
        signs[:, np.newaxis, np.newaxis],
        (ceil((parameters.mtrunc + 1) / 2), parameters.ntrunc + 1, basin.get_number_of_inlets()),
    )
    uj = inlets.uj

    # Main morphodynamic loop
    while (
        t < parameters.tend
        and max(abs(1 - uj[uj != 0])) > 1e-15
        and np.sum(inlets.get_widths()) > 0
    ):
        # Determine open inlets
        inlets_zip = np.squeeze(inlets.get_widths() > 0)

        # Integrate Eigenfunctions over inlets
        phij = np.zeros(l2.shape)
        phij[:, 0, inlets_zip] = 1
        phij[:, 1:, inlets_zip] = (
            basin.get_width()
            / (
                parameters.nrange[:, 1:, np.newaxis]
                * np.pi
                * inlets.get_widths()[np.newaxis, :, inlets_zip]
            )
            * (
                np.sin(
                    parameters.nrange[:, 1:, np.newaxis]
                    * np.pi
                    / basin.get_width()
                    * (
                        inlets.get_locations()[np.newaxis, :, inlets_zip]
                        + inlets.get_widths()[np.newaxis, :, inlets_zip] / 2
                    )
                )
                - np.sin(
                    parameters.nrange[:, 1:, np.newaxis]
                    * np.pi
                    / basin.get_width()
                    * (
                        inlets.get_locations()[np.newaxis, :, inlets_zip]
                        - inlets.get_widths()[np.newaxis, :, inlets_zip] / 2
                    )
                )
            )
        )
        phij = (
            phij
            * signs[0 : parameters.mtrunc + 1, 0 : parameters.ntrunc + 1, 0 : (basin.get_number_of_inlets())]
        )

        # Create new handle for hydroloop
        open_inlets.widths = inlets.get_widths()[:, inlets_zip]
        open_inlets.depths = inlets.depths[:, inlets_zip]
        open_inlets.lengths = inlets.get_lengths()[inlets_zip]
        open_inlets.locations = inlets.get_locations()[:, inlets_zip]
        open_inlets.uj = inlets.uj[:, inlets_zip]
        open_inlets.cd = inlets.cd
        open_inlets.numinlets = np.sum(inlets_zip)

        # Run hydrodynamic model
        uj, ub, mui2, mub2 = hydrodynamic_model(
            output, l2[:, :, inlets_zip], phij[:, :, inlets_zip]
        )

        # Update inlet mophology
        inlets.uj[:, inlets_zip] = uj
        basin.ub = ub

        da = (
            inlets.get_sediment_import()
            / inlets.get_lengths()
            * ((inlets.uj / inlets.get_equilibrium_velocity()) ** 3 - 1)
            * parameters.dt
        )
        newi = inlets.get_widths() ** 2 + da / inlets.get_shape_factor()
        nix = np.squeeze(newi > 0)
        inlets.get_widths()[:, nix] = np.sqrt(newi[:, nix])
        inlets.get_widths()[:, ~nix] = 0

        inlets.depths = inlets.get_shape_factor() * inlets.get_widths()
        inlets.uj[inlets.get_widths() == 0] = 0

        # Throw error if invalid inlet widhts are present
        if np.isnan(inlets.get_widths()).any():
            print(inlets.wit)
            print(inlets.depths)
            print(inlets.uj, basin.ub, mui2, mub2, da)
            raise NameError("aiaiai")

        # Finish timestep and prepare for next
        if silent is None:
            print(t, inlets.uj, da)
        else:
            print(t)
        t = t + parameters.dt
        tix = round((t - parameters.tstart) / parameters.dt)
        inlets.wit[tix - 1, :] = inlets.get_widths()

    inlets.mui = mui2
    basin.mub = mub2
    return output
