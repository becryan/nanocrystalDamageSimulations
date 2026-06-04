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


def build_newcrystal(x, y, charge, ncell, a=39.0, b=35.0, centered=True):
    """
    Build a 2D crystal by replicating the asymmetric unit across a regular
    lattice in x and y.

    Args:
        x, y: coordinates of the asymmetric unit.
        charge: atomic numbers for each atom.
        ncell: number of unit cells along each axis.
        a, b: lattice constants in x and y.
        centered: whether to center the replicated lattice at the origin.

    Example:
        ncell=2 produces a 2x2 crystal grid.
    """

    if centered:
        x = x - 0.5 * (x.min() + x.max())
        y = y - 0.5 * (y.min() + y.max())

    # lattice indices: [0, ..., ncell-1]
    cells = np.arange(ncell, dtype=float)
    if centered:
        cells -= (ncell - 1) / 2.0

    # lattice cell indices and shifts
    coords = np.column_stack([x, y])  # (N, 2)

    if centered:
        coords[:, 0] -= 0.5 * (coords[:, 0].min() + coords[:, 0].max())
        coords[:, 1] -= 0.5 * (coords[:, 1].min() + coords[:, 1].max())

    cells = np.arange(ncell, dtype=float)
    if centered:
        cells -= (ncell - 1) / 2.0

    sx, sy = np.meshgrid(cells * a, cells * b, indexing="ij")
    shifts = np.column_stack([sx.ravel(), sy.ravel()])

    tiled_coords = np.tile(coords, (shifts.shape[0], 1))
    rep_shifts = np.repeat(shifts, coords.shape[0], axis=0)

    newcoords = tiled_coords + rep_shifts
    newZ = np.tile(charge, shifts.shape[0])

    newcrystal = np.column_stack([newcoords[:, 0], newcoords[:, 1], newZ])
    return newcrystal


def build_newcrystal_3d(x, y, z, charge, ncell, a=39.0, b=35.0, c=50.0, centered=True):
    """Tile-based 3D replication: returns array of [x,y,z,Z].

    This function is a drop-in corrected implementation using explicit
    tiling and repeating of the asymmetric unit coordinates. It produces
    exactly ncell**3 translations and centers the lattice when requested.
    """

    coords = np.column_stack([x, y, z])  # (N, 3)

    if centered:
        coords[:, 0] -= 0.5 * (coords[:, 0].min() + coords[:, 0].max())
        coords[:, 1] -= 0.5 * (coords[:, 1].min() + coords[:, 1].max())
        coords[:, 2] -= 0.5 * (coords[:, 2].min() + coords[:, 2].max())

    cells = np.arange(ncell, dtype=float)
    if centered:
        cells -= (ncell - 1) / 2.0

    sx, sy, sz = np.meshgrid(cells * a, cells * b, cells * c, indexing="ij")
    shifts = np.column_stack([sx.ravel(), sy.ravel(), sz.ravel()])  # (M,3)

    tiled_coords = np.tile(coords, (shifts.shape[0], 1))  # (M*N, 3)
    rep_shifts = np.repeat(shifts, coords.shape[0], axis=0)  # (M*N, 3)

    newcoords = tiled_coords + rep_shifts
    newZ = np.tile(charge, shifts.shape[0])

    newcrystal = np.column_stack(
        [newcoords[:, 0], newcoords[:, 1], newcoords[:, 2], newZ]
    )
    return newcrystal
