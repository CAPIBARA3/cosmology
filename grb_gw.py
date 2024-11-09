import matplotlib.pyplot as plt
import numpy as np

# Data: Masses (in solar masses) and estimated frequency ranges (Hz)
# X-axis: masses of different astrophysical objects
masses = [1.4, 5, 10, 20, 50]  # Solar masses: typical values for neutron stars, stellar-mass BHs, etc.
labels = ["Neutron Star", "Light Black Hole", "Stellar-Mass Black Hole", "Heavy Black Hole", "Hypernova/Collapsar"]

# Y-axis: corresponding gravitational wave frequency ranges (Hz)
frequencies = [500, 200, 100, 30, 1]  # Approximate frequency peaks in Hz

# Plot
plt.figure(figsize=(10, 6))
plt.plot(masses, frequencies, marker='o', linestyle='-', color='b')

# Adding labels for each type of object
for i, label in enumerate(labels):
    plt.text(masses[i], frequencies[i] + 20, label, ha='center', fontsize=10)

# Labels and titles
plt.xlabel("Mass of Object (in Solar Masses)")
plt.ylabel("Peak Gravitational Wave Frequency (Hz)")
plt.title("Approximate Gravitational Wave Frequencies from Different Astrophysical Objects")
plt.yscale('log')  # Log scale for frequency as the range is large

# Customizing ticks and grid for clarity
plt.xticks(masses, labels)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

# Show plot
plt.show()