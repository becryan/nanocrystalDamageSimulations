"""
Utilities for constructing 2D and 3D nanocrystal models from an asymmetric
unit (e.g. a single protein molecule).

These functions replicate the atomic coordinates of the asymmetric unit
across a periodic lattice to generate a full crystal suitable for coherent
scattering or diffraction simulations. The replication is performed using
fully vectorised NumPy broadcasting, avoiding Python loops and enabling
efficient construction of large crystals.

Two variants are provided:

    - build_newcrystal():    2D lattice replication in the (x, y) plane.
    - build_newcrystal_3d(): full 3D replication using lattice constants
                             (a, b, c) along x, y, and z.

Both functions return a flat array of atomic coordinates and atomic numbers
in the format expected by downstream scattering code:

        [x, y, z, Z]

where Z is the atomic number. The functions assume a simple orthorhombic
lattice; more complex space groups or arbitrary lattice vectors can be
added if needed.

"""

import numpy as np


def build_newcrystal(x, y, charge, ncell, a=39.0, b=35.0):
    """
    Vectorised version of the MATLAB code:
    for i = -4:4
        for j = -4:4
            newcrystal = [x + a*i, y + b*j, charge]
    """

    # lattice indices: [-ncell, ..., +ncell]
    cells = np.arange(-ncell, ncell + 1)  # e.g. [-4..4] → 9 cells

    # broadcast shifts
    shift_x = cells[:, None] * a  # shape (L, 1)
    shift_y = cells[None, :] * b  # shape (1, L)

    # full grid of shifts
    grid_x = shift_x[:, :, None]  # (L, L, 1)
    grid_y = shift_y[:, :, None]  # (L, L, 1)

    # broadcast original coordinates
    x0 = x[None, None, :]  # (1, 1, N)
    y0 = y[None, None, :]  # (1, 1, N)
    Z0 = charge[None, None, :]  # (1, 1, N)

    # apply shifts to all atoms at once
    X = x0 + grid_x  # (L, L, N)
    Y = y0 + grid_y  # (L, L, N)
    Z = np.broadcast_to(Z0, X.shape)  # (L, L, N)

    # reshape to (L*L*N, 3)
    newcrystal = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    return newcrystal


def build_newcrystal_3d(x, y, z, charge, ncell, a=39.0, b=35.0, c=50.0):
    """
    Build a 3D crystal by replicating the asymmetric unit across
    a cubic lattice from -ncell to +ncell in each dimension.

    Returns:
        newcrystal: array of shape (N_total, 4)
                    columns = [x, y, z, Z]
    """

    # lattice indices: [-ncell, ..., +ncell]
    cells = np.arange(-ncell, ncell + 1)  # e.g. [-4..4] → 9 cells

    # 3D shift grids
    shift_x = cells[:, None, None] * a  # (L, 1, 1)
    shift_y = cells[None, :, None] * b  # (1, L, 1)
    shift_z = cells[None, None, :] * c  # (1, 1, L)

    # broadcast to full 3D grid
    grid_x = shift_x[:, :, :, None]  # (L, L, L, 1)
    grid_y = shift_y[:, :, :, None]  # (L, L, L, 1)
    grid_z = shift_z[:, :, :, None]  # (L, L, L, 1)

    # broadcast original coordinates
    x0 = x[None, None, None, :]  # (1, 1, 1, N)
    y0 = y[None, None, None, :]
    z0 = z[None, None, None, :]
    Z0 = charge[None, None, None, :]

    # apply shifts to all atoms at once
    X = x0 + grid_x  # (L, L, L, N)
    Y = y0 + grid_y
    Z = z0 + grid_z
    Znum = np.broadcast_to(Z0, X.shape)

    # reshape to (L*L*L*N, 4)
    newcrystal = np.column_stack([X.ravel(), Y.ravel(), Z.ravel(), Znum.ravel()])

    return newcrystal
