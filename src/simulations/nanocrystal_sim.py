import numpy as np
import time
from Bio.PDB import PDBParser
from fqdam import fqdam
import matplotlib.pyplot as plt
from build_nanocrystal import build_newcrystal


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


# ---------------------------------------------------------
# basic nanocrystal simulation (vectorised)
# ---------------------------------------------------------

t0 = time.time()

# --- Load PDB (replace with your own loader) ---
protein = pdbreadatom("1JA6.pdb")  # expects dict-like: x, y, z, element
x = np.asarray(protein["x"])  # shape (N,)
y = np.asarray(protein["y"])
z = np.asarray(protein["z"])
atom = np.asarray(protein["element"])  # e.g. ['C', 'N', 'O', 'S', ...]

N = x.size

# --- Parameters ---
lambda_x = 1.24
NGRID = 512
qmax = 1 / 5
dq = 2 * qmax / (NGRID - 1)

qgrid = np.linspace(-qmax, qmax, NGRID)
qx, qy = np.meshgrid(qgrid, qgrid)  # shape (G, G)

radial_q = np.sqrt(qx**2 + qy**2)  # shape (G, G)
qz = 1.0 / lambda_x


# ---------------------------------------------------------
# Assign atomic charges (Z)  <-- MUST happen before crystal build
# ---------------------------------------------------------
charge = np.zeros(N, dtype=int)
charge[atom == "C"] = 6
charge[atom == "N"] = 7
charge[atom == "O"] = 8
charge[atom == "S"] = 16

# ---------------------------------------------------------
# Build crystal using atomic numbers, NOT element symbols
# ---------------------------------------------------------
newcrystal = build_newcrystal(x, y, charge, a=39.0, b=35.0, ncell=4)

# ---------------------------------------------------------
# Precompute form factors (same grid for all atoms)
# ---------------------------------------------------------
fc1 = fqdam(radial_q, 2, 4, 0, 6)
fn1 = fqdam(radial_q, 2, 5, 0, 7)
fo1 = fqdam(radial_q, 2, 6, 0, 8)
fs1 = fqdam(radial_q, 2, 8, 6, 16)

# ---------------------------------------------------------
# Vectorised scattering from isolated protein
# ---------------------------------------------------------
# phases: shape (N, G, G)
phase = np.exp(
    -2j
    * np.pi
    * (qx[None, :, :] * x[:, None, None] + qy[None, :, :] * y[:, None, None])
)

# masks per element
mask_C = charge == 6
mask_N = charge == 7
mask_O = charge == 8
mask_S = charge == 16

fn = np.zeros_like(radial_q, dtype=complex)

if np.any(mask_C):
    fn += np.sum(phase[mask_C] * fc1, axis=0)
if np.any(mask_N):
    fn += np.sum(phase[mask_N] * fn1, axis=0)
if np.any(mask_O):
    fn += np.sum(phase[mask_O] * fo1, axis=0)
if np.any(mask_S):
    fn += np.sum(phase[mask_S] * fs1, axis=0)

# ---------------------------------------------------------
# Vectorised scattering from crystal coordinates
# ---------------------------------------------------------
print("getting coordinates of atoms in crystal")

# newcrystal: shape (M, 3) -> [x, y, Z]
newcrystal = np.asarray(newcrystal)
xc = newcrystal[:, 0]
yc = newcrystal[:, 1]
Zc = newcrystal[:, 2].astype(int)
plot_crystal_3d(newcrystal, s=1)
phase_c = np.exp(
    -2j
    * np.pi
    * (qx[None, :, :] * xc[:, None, None] + qy[None, :, :] * yc[:, None, None])
)

mask_Cc = Zc == 6
mask_Nc = Zc == 7
mask_Oc = Zc == 8
mask_Sc = Zc == 16

fnshift = np.zeros_like(radial_q, dtype=complex)

if np.any(mask_Cc):
    fnshift += np.sum(phase_c[mask_Cc] * fc1, axis=0)
if np.any(mask_Nc):
    fnshift += np.sum(phase_c[mask_Nc] * fn1, axis=0)
if np.any(mask_Oc):
    fnshift += np.sum(phase_c[mask_Oc] * fo1, axis=0)
if np.any(mask_Sc):
    fnshift += np.sum(phase_c[mask_Sc] * fs1, axis=0)

# ---------------------------------------------------------
# Damage model would go here
# ---------------------------------------------------------

print("Elapsed time (vectorised):", time.time() - t0)
