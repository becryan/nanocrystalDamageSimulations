"""
Main -- pipeline file to :

- read in the xyz, build nanocrystal

- simulate the diffraction pattern


"""

from build_nanocrystal import crystal_from_pdb_2d, crystal_from_pdb_3d
from utils import plot_crystal_2d, plot_crystal_3d, plot_diffraction
from fqdam import simulate_diffraction
import matplotlib.plt as plt
from pathlib import Path

import argparse
import numpy as np
from input_output import compute_hist2d


def main():
    p = argparse.ArgumentParser(
        description="Generate and output nanocrystal diffraction patterns"
    )

    p.add_argument("--pdbname", type=Path, default="1JA6", help="Path to pdb file")

    p.add_argument(
        "--plot", type=bool, default=False, help="True/false plot during simulation"
    )

    p.add_argument(
        "--output",
        type=Path,
        default="output",
        help="Output directory for diffraction patterns",
    )

    p.add_argument(
        "--maxq",
        type=float,
        default=10.0,
        help="Maximum q value for diffraction simulation",
    )

    p.add_argument(
        "--wavelength",
        type=float,
        default=1.0,
        help="Wavelength of incident radiation, Angstrom",
    )

    p.add_argument(
        "--gridsize",
        type=int,
        default=256,
        help="Grid size (number of pixels) for diffraction simulation",
    )

    p.add_argument(
        "--dimensions",
        type=int,
        default=2,
        help="Dimensions of the simulation (2 or 3)",
    )

    args = p.parse_args()

    protein_file = args.pdbname
    inline_plot = args.plot
    # output_dir = args.output
    ngrid = args.gridsize
    dims = args.dimensions

    crystal = crystal_from_pdb_2d(protein_file, ncell=4, a=39.0, b=39.0)
    if inline_plot:
        plot_crystal_2d(crystal, s=2, alpha=0.8, color_by_element=True)

    if dims == 2:
        intensity, q_grid, radial_q = simulate_diffraction(
            crystal, dims=dims, NGRID=ngrid
        )

        plot_diffraction(intensity, dims=2)
        histo = compute_hist2d(intensity, intensity, bins=(20, 20))
        if inline_plot:
            plt.imshow(np.log1p(histo[0]), cmap="inferno", origin="lower")
            plt.show()

    elif dims == 3:
        crystal3d = crystal_from_pdb_3d(protein_file, ncell=2, a=39.0, b=39.0, c=50.0)
        plot_crystal_3d(crystal3d, s=1)
        intensity3d, q_grid3d, radial_q3d = simulate_diffraction(
            crystal3d, dims=dims, NGRID=(128, 128, 128)
        )
        plot_diffraction(intensity3d, dims=3)
        histo = compute_hist2d(intensity3d, intensity3d, bins=(20, 20))
        plt.imshow(np.log1p(histo[0]), cmap="inferno", origin="lower")
        plt.show()


if __name__ == "__main__":
    main()
