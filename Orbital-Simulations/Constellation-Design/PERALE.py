import numpy as np
import matplotlib.pyplot as plt
#from astropy.constants import R_earth
from astropy import units as u
from poliastro.bodies import Earth
from poliastro.twobody import Orbit

# Function to convert orbital elements to position in 3D space
def position_from_orbital_elements(orbit, num_points=100):
    """Extract the position data in 3D space."""
    positions = orbit.sample(num_points)
    return np.array([positions.x.value,
                     positions.y.value,
                     positions.z.value])

# Function to set equal aspect ratio for 3D plots
def set_equal_aspect_3d(ax):
    """Set equal aspect ratio for a 3D plot."""
    extents = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])
    centers = np.mean(extents, axis=1)
    max_size = max(extents[:, 1] - extents[:, 0])
    ax.set_xlim(centers[0] - max_size / 2, centers[0] + max_size / 2)
    ax.set_ylim(centers[1] - max_size / 2, centers[1] + max_size / 2)
    ax.set_zlim(centers[2] - max_size / 2, centers[2] + max_size / 2)

# Define parameters
num_planes = 5
num_satellites = 5
altitude = 500 * u.km
inclination = 53 * u.deg
raan_spacing = 360 / num_planes  # Right Ascension of the Ascending Node (RAAN) spacing
theta_spacing = 360 / num_satellites  # True anomaly spacing

# Generate orbits and satellites
orbits = []
positions_satellites = []

earth_radius = 6371 * u.km

for i in range(num_planes):
    raan = i * raan_spacing * u.deg
    for j in range(num_satellites):
        true_anomaly = j * theta_spacing * u.deg
        orbit = Orbit.from_classical(Earth, Earth.R + altitude, 0 * u.one, inclination, raan, 0 * u.deg, true_anomaly)
        orbits.append(orbit)
        positions_satellites.append(orbit.r.to_value(u.km))

# Plotting the Earth
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Earth Sphere
u, v = np.linspace(0, 2 * np.pi, 100), np.linspace(0, np.pi, 100)
x = earth_radius * np.outer(np.cos(u), np.sin(v))
y = earth_radius * np.outer(np.sin(u), np.sin(v))
z = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))

ax.plot_surface(x, y, z, color='blue', alpha=0.5)

# Plot orbits and satellites
for orbit in orbits:
    x, y, z = position_from_orbital_elements(orbit)
    ax.plot(x, y, z, label=f'Orbit Plane', alpha=0.7)

for position in positions_satellites:
    ax.scatter(*position, color='red', s=10)

# Setting plot limits and labels
ax.set_xlim([-8000, 8000])
ax.set_ylim([-8000, 8000])
ax.set_zlim([-8000, 8000])

# Ensure equal aspect ratio
set_equal_aspect_3d(ax)

ax.set_xlabel("X (km)")
ax.set_ylabel("Y (km)")
ax.set_zlabel("Z (km)")
ax.set_title("5 Orbital Planes with 25 Satellites in LEO")

plt.show()
