import numpy as np
from numpy.fft import fft2, fftshift
from fqdam import fq_dam as fq
from build_nanocrystal import build_newcrystal
from utils import pdbreadatom, plot_crystal_3d
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Load PDB and build crystal
# ---------------------------------------------------------
protein = pdbreadatom("1JA6.pdb")
x = protein["x"]
y = protein["y"]
atom = protein["element"]
N = len(x)

# atomic numbers
Z = np.zeros(N, dtype=int)
Z[atom == "C"] = 6
Z[atom == "N"] = 7
Z[atom == "O"] = 8
Z[atom == "S"] = 16

# build 2D crystal
newcrystal = build_newcrystal(x, y, Z, ncell=4, a=39.0, b=39.0)
xc = newcrystal[:, 0]
yc = newcrystal[:, 1]
Zc = newcrystal[:, 2]

plot_crystal_3d(newcrystal, s=1)

# ---------------------------------------------------------
# Real-space grid
# ---------------------------------------------------------
NGRID = 256
# define bounding box
margin = 0
xmin, xmax = xc.min() - margin, xc.max() + margin
ymin, ymax = yc.min() - margin, yc.max() + margin

dx = (xmax - xmin) / NGRID
dy = (ymax - ymin) / NGRID

dx = (xmax - xmin) / NGRID
dy = (ymax - ymin) / NGRID

# one density map per Z
rho_C = np.zeros((NGRID, NGRID), dtype=float)
rho_N = np.zeros((NGRID, NGRID), dtype=float)
rho_O = np.zeros((NGRID, NGRID), dtype=float)
rho_S = np.zeros((NGRID, NGRID), dtype=float)

ix = np.clip(((xc - xmin) / dx).astype(int), 0, NGRID - 1)
iy = np.clip(((yc - ymin) / dy).astype(int), 0, NGRID - 1)

mask_C = Zc == 6
mask_N = Zc == 7
mask_O = Zc == 8
mask_S = Zc == 16

np.add.at(rho_C, (iy[mask_C], ix[mask_C]), 1.0)
np.add.at(rho_N, (iy[mask_N], ix[mask_N]), 1.0)
np.add.at(rho_O, (iy[mask_O], ix[mask_O]), 1.0)
np.add.at(rho_S, (iy[mask_S], ix[mask_S]), 1.0)

# ---------------------------------------------------------
# FFT → structure factor per element
# ---------------------------------------------------------
F_C = fftshift(fft2(rho_C))
F_N = fftshift(fft2(rho_N))
F_O = fftshift(fft2(rho_O))
F_S = fftshift(fft2(rho_S))

# build q-grid consistent with FFT spacing
qx = np.fft.fftfreq(NGRID, d=dx)
qy = np.fft.fftfreq(NGRID, d=dy)
qx, qy = np.meshgrid(qx, qy)
radial_q = np.sqrt(qx**2 + qy**2)

# ---------------------------------------------------------
# Optional: apply atomic form factor in q-space
# (here using carbon-like as an example; you can mix by Z if you want)
# ---------------------------------------------------------
fC = fq(radial_q, 2, 4, 0, 6)  # C: 1s2 2s2 2p2
fN = fq(radial_q, 2, 5, 0, 7)  # N: 1s2 2s2 2p3
fO = fq(radial_q, 2, 6, 0, 8)  # O: 1s2 2s2 2p4
fS = fq(radial_q, 2, 8, 6, 16)  # S: 1s2 2s2 2p6 3s2 3p4

F_total = F_C * fC + F_N * fN + F_O * fO + F_S * fS
# ---------------------------------------------------------
# Intensity
# ---------------------------------------------------------
Intensity = np.abs(F_total) ** 2


plt.imshow(np.sqrt(Intensity))
plt.show()
