import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from astropy import units as u
from astropy.coordinates import CartesianRepresentation

# Constants
earth_radius = 6371.0  # Earth radius in km
G = 6.67430e-20  # Gravitational constant in km^3/kg/s^2
M_earth = 5.972e24  # Mass of Earth in kg

# Read your input data
file_path_alt = "C:/Users/aless/PycharmProjects/SMDsimulations/master_results.txt"
columns_alt = ["Altitude", "Expl-Fragm", "Coll_Fragm", "Launch/Mis", "NaK-Drops", "SRM-Slag", "SRM-Dust",
               "Paint Flks", "Ejecta", "MLI", "Cloud 1", "Cloud 2", "Cloud 3", "Cloud 4", "Cloud 5",
               "Human-made", "Meteoroids", "Streams", "Total"]

data_alt = pd.read_csv(file_path_alt, delim_whitespace=True, skiprows=2, names=columns_alt)
file_path_incl = "C:/Users/aless/PycharmProjects/SMDsimulations/MASTER2_declination_distribution.txt"
columns_incl = ["Declination", "Expl-Fragm", "Coll_Fragm", "Launch/Mis", "NaK-Drops", "SRM-Slag", "SRM-Dust",
                "Paint Flks", "Ejecta", "MLI", "Cloud 1", "Cloud 2", "Cloud 3", "Cloud 4", "Cloud 5",
                "Human-made", "Meteoroids", "Streams", "Total"]

data_incl = pd.read_csv(file_path_incl, delim_whitespace=True, skiprows=2, names=columns_incl)
file_path_diam = "C:/Users/aless/PycharmProjects/SMDsimulations/MASTER2_diameter_distribution.txt"
columns_diam = ["Diameter", "Expl-Fragm", "Coll_Fragm", "Launch/Mis", "NaK-Drops", "SRM-Slag", "SRM-Dust",
                "Paint Flks", "Ejecta", "MLI", "Cloud 1", "Cloud 2", "Cloud 3", "Cloud 4", "Cloud 5",
                "Human-made", "Meteoroids", "Streams", "Total"]

data_diam = pd.read_csv(file_path_diam, delim_whitespace=True, skiprows=2, names=columns_diam)

# Total number of particles to simulate
total_particles = 100

# Compute probability distributions
data_alt['Probability'] = data_alt['Total'] / data_alt['Total'].sum()
data_incl['Probability'] = data_incl['Expl-Fragm'] / data_incl['Expl-Fragm'].sum()
data_diam['Probability'] = data_diam['Total'] / data_diam['Total'].sum()

# Select random altitudes based on computed probabilities
altitudes = np.random.choice(data_alt['Altitude'], size=total_particles, p=data_alt['Probability'])

# Select random orbit inclinations based on computed probabilities
latitudes = np.random.choice(data_incl['Declination'], size=total_particles, p=data_incl['Probability'])

# Select random diameters based on computed probabilities
diameters = np.random.choice(data_diam['Diameter'], size=total_particles, p=data_diam['Probability'])


# Function to generate random positions
def positionsfun(altitudes, latitudes, total_particles):
    positions = np.empty((total_particles, 3))
    i = 0
    for altitude in altitudes:
        latitude = np.random.choice(latitudes)
        longitude = np.random.uniform(0, 2 * np.pi)
        x = (altitude + earth_radius) * np.cos(np.radians(latitude)) * np.cos(longitude)
        y = (altitude + earth_radius) * np.cos(np.radians(latitude)) * np.sin(longitude)
        z = (altitude + earth_radius) * np.sin(np.radians(latitude))
        positions[i, :] = [x, y, z]
        i += 1
    return positions


positions = positionsfun(altitudes, latitudes, total_particles)


# Function to set equal axis scale for 3D plots
def set_axes_equal(ax):
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    midpoints = limits.mean(axis=1)
    max_extent = max(limits.ptp(axis=1))
    ax.set_xlim3d([midpoints[0] - max_extent / 2, midpoints[0] + max_extent / 2])
    ax.set_ylim3d([midpoints[1] - max_extent / 2, midpoints[1] + max_extent / 2])
    ax.set_zlim3d([midpoints[2] - max_extent / 2, midpoints[2] + max_extent / 2])


# Create orbits
orbits = []
for pos, alt in zip(positions, altitudes):
    radius = alt + earth_radius
    semi_major_axis = radius * u.km  # semi-major axis
    orbital_velocity_val = np.sqrt(G * M_earth / radius)  # Circular velocity at this altitude

    position = CartesianRepresentation(pos[0], pos[1], pos[2]) * u.km

    # Create orbit with zero eccentricity
    orbit = Orbit.from_classical(Earth, semi_major_axis, 0 * u.deg, 0 * u.deg, 0 * u.deg, 0 * u.deg, 0, position)

    orbits.append(orbit)

debris_orbits = orbits

# Generate velocities
velocities = np.array([orbit.v for orbit in debris_orbits])

# Initialize the plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("Generated debris field")
ax.set_xlabel("X (km)")
ax.set_ylabel("Y (km)")
ax.set_zlabel("Z (km)")

# Plot Earth
phi1 = np.linspace(0, np.pi, 50)
theta1 = np.linspace(0, 2 * np.pi, 50)
phi1, theta1 = np.meshgrid(phi1, theta1)
x = earth_radius * np.sin(phi1) * np.cos(theta1)
y = earth_radius * np.sin(phi1) * np.sin(theta1)
z = earth_radius * np.cos(phi1)
ax.plot_surface(x, y, z, rstride=1, cstride=1, color='b', alpha=0.6, edgecolor='k')

# Colors for debris
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
debris_points = [ax.plot([], [], [], 'o', color=colors[i % len(colors)])[0] for i in range(len(positions))]
trajectories_lines = [ax.plot([], [], [], '-', color=colors[i % len(colors)])[0] for i in range(len(positions))]

# Initialize state tracking
current_positions = [pos for pos in positions]
trajectories = [[] for _ in range(total_particles)]


def update(frame):
    global current_positions, trajectories

    for i, orbit in enumerate(debris_orbits):
        state = orbit.propagate(frame * u.day)

        position = state.represent_as(CartesianRepresentation).xyz.to_value(u.km)
        trajectories[i].append(position)
        debris_points[i].set_data([position[0]], [position[1]])
        debris_points[i].set_3d_properties([position[2]])

        trajectory = np.array(trajectories[i])
        trajectories_lines[i].set_data(trajectory[:, 0], trajectory[:, 1])
        trajectories_lines[i].set_3d_properties(trajectory[:, 2])

    return debris_points + trajectories_lines


set_axes_equal(ax)
ani = FuncAnimation(fig, update, frames=100, interval=50, blit=False)
ax.set_box_aspect([1, 1, 1])
plt.show()





