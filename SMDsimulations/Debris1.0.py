import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from astropy import units as u
from astropy.coordinates import CartesianRepresentation
from astropy.time import Time, TimeDelta

# Read your input data
file_path_alt = "C:/Users/aless/PycharmProjects/SMDsimulations/master_results.txt"

columns_alt = [
    "Altitude", "Expl-Fragm", "Coll_Fragm", "Launch/Mis", "NaK-Drops", "SRM-Slag", "SRM-Dust",
    "Paint Flks", "Ejecta", "MLI", "Cloud 1", "Cloud 2", "Cloud 3", "Cloud 4", "Cloud 5", "Human-made",
    "Meteoroids", "Streams", "Total"
]

data_alt = pd.read_csv(file_path_alt, delim_whitespace=True, skiprows=2, names=columns_alt)

# Convert the data values
#conversion_factor = 1e-9
#for column in data_alt.columns[1:]:
#    data_alt[column] *= conversion_factor

file_path_incl = "C:/Users/aless/PycharmProjects/SMDsimulations/MASTER2_declination_distribution.txt"

columns_incl = [
    "Declination", "Expl-Fragm", "Coll_Fragm", "Launch/Mis", "NaK-Drops", "SRM-Slag", "SRM-Dust",
    "Paint Flks", "Ejecta", "MLI", "Cloud 1", "Cloud 2", "Cloud 3", "Cloud 4", "Cloud 5", "Human-made",
    "Meteoroids", "Streams", "Total"
]

data_incl = pd.read_csv(file_path_incl, delim_whitespace=True, skiprows=2, names=columns_incl)


total_density = data_incl['Expl-Fragm']
total_sum = total_density.sum()

file_path_diam = "C:/Users/aless/PycharmProjects/SMDsimulations/MASTER2_diameter_distribution.txt"

columns_diam = [
    "Diameter", "Expl-Fragm", "Coll_Fragm", "Launch/Mis", "NaK-Drops", "SRM-Slag", "SRM-Dust",
    "Paint Flks", "Ejecta", "MLI", "Cloud 1", "Cloud 2", "Cloud 3", "Cloud 4", "Cloud 5", "Human-made",
    "Meteoroids", "Streams", "Total"
]

data_diam = pd.read_csv(file_path_diam, delim_whitespace=True, skiprows=2, names= columns_diam)



# Total number of particles (to simulate)
total_particles = 1000

# Compute probability distributions
data_alt['Probability'] = data_alt['Total'] / data_alt['Total'].sum()
data_incl['Probability'] = total_density/total_sum
data_diam['Probability'] = data_diam['Total'] / data_diam['Total'].sum()

#plot probability distributions
plt.plot(data_incl['Declination'], data_incl['Probability'])
plt.xlabel('Declination [deg]')
plt.ylabel('Probability')
plt.title('Probability Distribution of Space Debris by Declination')
plt.show()

plt.plot(data_alt['Altitude'], data_alt['Probability'])
plt.xlabel('Altitude [km]')
plt.ylabel('Probability')
plt.title('Probability distribution of space debris by Altitude')
plt.show()

plt.plot(data_diam['Diameter'], data_diam['Probability'])
plt.xlabel('Diameter []')
plt.ylabel('Probability')
plt.title('Probability distribution of space debris by Diameter')
plt.show()



# Select random altitudes based on the computed probabilities
altitudes = np.random.choice(data_alt['Altitude'], size=total_particles, p=data_alt['Probability'])



# Earth radius in kilometers
earth_radius = 6371.0  # in km


#Select random orbit inclinations based on the computed probabilities
latitudes = np.random.choice(data_incl['Declination'], size=total_particles, p=data_incl['Probability'])
latitudes += 90
print("Latitudes:",  latitudes)
print("Altitudes:", altitudes)
print(data_incl['Probability'].sum())
print(data_alt['Probability'].sum())
print(data_diam['Probability'].sum())


#Select random diameters based on the computed probabilities
diameters = np.random.choice(data_diam['Diameter'], size=total_particles, p=data_diam['Probability'])

print('Diameters [m]: ', diameters)



eccentricities = np.random.uniform(-0.05, 0.05, total_particles)*u.one
r_ascension = np.random.uniform(0,360, total_particles)*u.deg
arg_periapsis = np.random.uniform(0, 360, total_particles)*u.deg
true_anomaly = np.random.uniform(0, 360, total_particles)*u.deg
radii = (altitudes+earth_radius)*u.km


debris_orbits = [
    Orbit.from_classical(Earth, rad, ecc, inc*u.deg, raan, argp, nu, Time.now()) for rad, inc, ecc, raan, argp, nu in zip(radii, latitudes, eccentricities, r_ascension, arg_periapsis, true_anomaly)
]


positions = np.zeros((total_particles, 3))


for i in range(0, total_particles):
    state = debris_orbits[i].propagate(0*u.s)
    positions[i, :] = state.represent_as(CartesianRepresentation).xyz.to_value(u.km)






fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')



ax.set_title("Generated debris field")
ax.set_xlabel("X (km)")
ax.set_ylabel("Y (km)")
ax.set_zlabel("Z (km)")

# Colors for debris
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
debris_points = [ax.plot([], [], [], 'o', color=colors[i % len(colors)])[0] for i in range(len(positions))]
trajectories_lines = [ax.plot([], [], [], '-', color=colors[i % len(colors)])[0] for i in range(len(positions))]

#plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='r', marker='o', label='Points')
phi1 = np.linspace(0, np.pi, 50)  # Latitude angle
theta1 = np.linspace(0, 2 * np.pi, 50)  # Longitude angle
phi1, theta1 = np.meshgrid(phi1, theta1)
x = earth_radius * np.sin(phi1) * np.cos(theta1)
y = earth_radius * np.sin(phi1) * np.sin(theta1)
z = earth_radius * np.cos(phi1)
ax.plot_surface(x, y, z, rstride=1, cstride=1, color='b', alpha=0.6, edgecolor='k')


ax.set_box_aspect([1, 1, 1])
plt.show()



M_earth = 5.972e24  # Earth mass in kg, only needed if you are manually calculating T
G = 6.67430e-20  # Gravitational constant in km^3 / (kg * s^2)
start_time = np.empty((total_particles, 1), dtype=object)
time_of_flight = 1 * u.day
time_step = 10 * u.s
n_steps = int((time_of_flight / time_step).decompose())  # Total steps
time_points = np.empty((n_steps, total_particles), dtype=object)
for i,orbit in enumerate(debris_orbits):
    r = orbit.a * 1e3

    # Calculate the orbital period T using the semi-major axis
    T = 2 * np.pi * np.sqrt(r ** 3 / (G * M_earth))*u.s  # Orbital period in seconds

    # Add a random phase to the orbital period
    phase = np.random.uniform(0, T.value) * u.s  # Random phase in seconds

    start_time[i] = phase
    time_points[:, i] = [phase + (k * time_step) for k in range(n_steps)]




# Prepare for animation
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')



ax.set_title("Simulated debris field")
ax.set_xlabel("X (km)")
ax.set_ylabel("Y (km)")
ax.set_zlabel("Z (km)")

ax.plot_surface(x, y, z, rstride=1, cstride=1, color='b', alpha=0.2, edgecolor='k')


# Colors for debris
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
debris_points = [ax.plot([], [], [], 'o', color=colors[i % len(colors)])[0] for i in range(len(positions))]
trajectories_lines = [ax.plot([], [], [], '-', color=colors[i % len(colors)])[0] for i in range(len(positions))]

# Initialize state tracking
current_positions = [pos for pos in positions]
trajectories = [[] for _ in range(total_particles)]  # Store the evolving trajectories

from astropy.time import TimeDelta
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt

def update(frame):

    global current_positions, trajectories, start_time

    # Update debris positions
    for i, orbit in enumerate(debris_orbits):

        time = time_points[frame, i]
        start_time_i = start_time[i,0]*u.s
        # Propagate the orbit by the phase time (in seconds)
        time_delta = (time - start_time_i)
        time_delta = time_delta
        state = orbit.propagate(time_delta)

        # Get the position in Cartesian coordinates (x, y, z in km)
        position = state.represent_as(CartesianRepresentation).xyz.to_value(u.km)
        
        # Ensure position is a valid sequence (numpy array or list)
        if len(position) == 3:  # Check position has x, y, z
            # Update positions and trajectories
            trajectories[i].append(position)
            debris_points[i].set_data([position[0]], [position[1]])  # x and y
            debris_points[i].set_3d_properties([position[2]])  # z

            # Update trajectory lines
            trajectory = np.array(trajectories[i])
            trajectories_lines[i].set_data(trajectory[:, 0], trajectory[:, 1])  # x and y
            trajectories_lines[i].set_3d_properties(trajectory[:, 2])  # z
            trajectories_lines[i].set_linestyle('--')
            trajectories_lines[i].set_linewidth(0.5)
        else:
            print(f"Invalid position for debris {i} at frame {frame}: {position}")


    return debris_points + trajectories_lines

# Create animation
ani = FuncAnimation(fig, update, frames=int(n_steps), interval=1, blit=False)
ax.set_box_aspect([1, 1, 1])
plt.show()




