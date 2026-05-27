import numpy as np
import matplotlib.pyplot as plt

# Load diffraction pattern data
I_ideal = np.load("ideal.npy")
I_obs   = np.load("observed.npy")

# Flatten and log-scale
x = np.log(I_ideal.flatten() + 1e-12)
y = np.log(I_obs.flatten() + 1e-12)

# Compute 2D histogram
H, xedges, yedges = np.histogram2d(x, y, bins=200)

# Plot
plt.figure(figsize=(6,5))
plt.imshow(H.T, origin='lower', 
           extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
           cmap='viridis', aspect='auto')
plt.xlabel("log(I_ideal)")
plt.ylabel("log(I_observed)")
plt.title("2D Joint Histogram")
plt.colorbar(label="Counts")
plt.show()
