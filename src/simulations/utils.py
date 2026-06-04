"""

General utility functions such as plotting, reading files ,called by other scripts
"""

import numpy as np
from Bio.PDB import PDBParser
import matplotlib.pyplot as plt


def pdbreadatom(filename):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pdb", filename)

    xs, ys, zs, elems = [], [], [], []

    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    x, y, z = atom.get_coord()
                    xs.append(x)
                    ys.append(y)
                    zs.append(z)
                    elems.append(atom.element)

    return {
        "x": np.array(xs),
        "y": np.array(ys),
        "z": np.array(zs),
        "element": np.array(elems),
    }


def plot_crystal_3d(newcrystal, s=1):
    x = newcrystal[:, 0]
    y = newcrystal[:, 1]
    z = newcrystal[:, 2]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(x, y, z, s=s, alpha=0.5)

    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.set_zlabel("z (Å)")
    ax.set_title("3D Crystal Structure")

    plt.show()


def plot_crystal_2d(crystal, s=5, alpha=0.7, cmap="viridis", color_by_element=False):
    """Plot a 2D projection of crystal atom positions.

    Args:
        crystal: NumPy array with shape (N, 3) or (N, 4).
            Columns should be [x, y, z] or [x, y, z, Z].
        s: marker size.
        alpha: marker alpha.
        cmap: matplotlib colormap name for element coloring.
        color_by_element: if True, color atoms by atomic number.
    """
    crystal = np.asarray(crystal)
    if crystal.ndim != 2 or crystal.shape[1] not in (3, 4):
        raise ValueError("crystal must be a 2D array with 3 or 4 columns")

    x = crystal[:, 0]
    y = crystal[:, 1]

    plt.figure(figsize=(8, 8))

    if color_by_element and crystal.shape[1] == 4:
        z = crystal[:, 3]
        scatter = plt.scatter(x, y, c=z, cmap=cmap, s=s, alpha=alpha)
        cbar = plt.colorbar(scatter)
        cbar.set_label("Atomic number Z")
    else:
        plt.scatter(x, y, s=s, alpha=alpha, color="tab:blue")

    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("x (Å)")
    plt.ylabel("y (Å)")
    plt.title("2D Crystal Projection")
    plt.tight_layout()
    plt.show()


def sample_atoms(crystal, n_samples, seed=None):
    """Randomly sample atoms from a crystal array without replacement.

    Args:
        crystal: NumPy array with shape (N, 3) or (N, 4).
            Columns should be [x, y, z] or [x, y, z, Z].
        n_samples: Number of atoms to sample.
        seed: Optional random seed for reproducibility.

    Returns:
        A NumPy array of sampled rows with the same number of columns.
    """
    crystal = np.asarray(crystal)
    if crystal.ndim != 2 or crystal.shape[1] not in (3, 4):
        raise ValueError("crystal must be a 2D array with 3 or 4 columns")

    if n_samples < 0:
        raise ValueError("n_samples must be non-negative")

    if n_samples > crystal.shape[0]:
        raise ValueError("n_samples cannot exceed number of atoms")

    rng = np.random.default_rng(seed)
    indices = rng.choice(crystal.shape[0], size=n_samples, replace=False)
    return crystal[indices]
