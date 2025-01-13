import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from astropy import units as u
from astropy.coordinates import CartesianRepresentation
from astropy.time import Time, TimeDelta
import plotly.graph_objects as go
import random

M_earth = 5.972e24  # Earth mass in kg, only needed if you are manually calculating T
G = 6.67430e-20  # Gravitational constant in km^3 / (kg * s^2)

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

# Select random altitudes based on the computed probabilities
altitudes = np.random.choice(data_alt['Altitude'], size=total_particles, p=data_alt['Probability'])



# Earth radius in kilometers
earth_radius = 6371.0  # in km


#Select random orbit inclinations based on the computed probabilities
latitudes = np.random.choice(data_incl['Declination'], size=total_particles, p=data_incl['Probability'])
latitudes += 90

#Select random diameters based on the computed probabilities
diameters = np.random.choice(data_diam['Diameter'], size=total_particles, p=data_diam['Probability'])

eccentricities = np.random.uniform(-0.05, 0.05, total_particles)*u.one
r_ascension = np.random.uniform(0,360, total_particles)*u.deg
arg_periapsis = np.random.uniform(0, 360, total_particles)*u.deg
true_anomaly = np.random.uniform(0, 360, total_particles)*u.deg
radii = (altitudes+earth_radius)*u.km


debris_orbits = [
    Orbit.from_classical(Earth, rad, ecc, inc*u.deg, raan, argp, nu, Time.now()) for rad, inc, ecc, raan, argp, nu in zip(radii, latitudes, eccentricities, r_ascension, arg_periapsis, true_anomaly)
]

def position_from_orbital_elements(orbit, num_points=100):
    """Extract the position data in 3D space."""
    positions = orbit.sample(num_points)
    return np.array([positions.x.value,
                     positions.y.value,
                     positions.z.value])



# Define parameters
num_planes = 13
num_satellites = 40
altitude = 450 * u.km
inclination = 89 * u.deg
raan_spacing = 360 / num_planes  # Right Ascension of the Ascending Node (RAAN) spacing
theta_spacing = 360 / num_satellites  # True anomaly spacing

# Generate orbits and satellite
sat_orbits = []
positions_satellites = []

earth_radius = 6371 * u.km

for i in range(num_planes):
    raan = i * raan_spacing * u.deg
    for j in range(num_satellites):
        true_anomaly = np.random.uniform(0, 360)*u.deg + j * theta_spacing * u.deg
        orbit = Orbit.from_classical(Earth, Earth.R + altitude + (75*i*u.km), 0 * u.one, inclination, raan, 0 * u.deg, true_anomaly)
        sat_orbits.append(orbit)
        positions_satellites.append(orbit.r.to_value(u.km))


positions = np.zeros((total_particles, 3))


for i in range(0, total_particles):
    state = debris_orbits[i].propagate(0*u.s)
    positions[i, :] = state.represent_as(CartesianRepresentation).xyz.to_value(u.km)




# Define initial constellation parameters
num_planes = 13
num_satellites = 4
altitude = 450 * u.km
inclination = 89 * u.deg
raan_spacing = 360 / num_planes  # Right Ascension of the Ascending Node (RAAN) spacing
theta_spacing = 360 / num_satellites  # True anomaly spacing
total_sats = num_planes*num_satellites

# Generate orbits and satellites
sat_orbits = []
positions_satellites = []

earth_radius = 6371 * u.km

for i in range(num_planes):
    raan = i * raan_spacing * u.deg
    for j in range(num_satellites):
        true_anomaly = np.random.uniform(0, 360)*u.deg + j * theta_spacing * u.deg
        orbit = Orbit.from_classical(Earth, Earth.R + altitude + (75*i*u.km), 0 * u.one, inclination, raan, 0 * u.deg, true_anomaly)
        sat_orbits.append(orbit)
        positions_satellites.append(orbit.r.to_value(u.km))


positions = np.zeros((total_particles, 3))


for i in range(0, total_particles):
    state = debris_orbits[i].propagate(0*u.s)
    positions[i, :] = state.represent_as(CartesianRepresentation).xyz.to_value(u.km)


time_of_flight = 2 * u.hour
time_step = 0.1 * u.s
n_steps = int((time_of_flight / time_step).decompose())  # Total steps
time_points = [(k * time_step) for k in range(n_steps)]

max_range = 100
FOV = 120


def det_probability(probability_function, velocity, distance, size):
    detection_probability = probability_function(velocity, distance, size)
    if not (0 <= detection_probability <= 1):
        raise ValueError("Probability function returned a value outside [0,1].")

    return 1 if random.random () < detection_probability else 0

# example probability distribution
# (probability decreases with distance, increases with size and has small velocity effect)


def ex_pf(vel, distance, dim):

    probability = max(0,min(0.5 - 0.1 * distance + 0.1 * dim - 0.05 * abs(vel).value))
    print('Probability: ', probability)

    return probability


position_deb = np.zeros((total_particles, 3))
position_sat = np.zeros((total_sats, 3))
v_debs = np.zeros((total_particles, 3))
soi = []
doi = []
det_deb = []

for t in time_points:

    fov_check = np.zeros((total_sats, total_particles))
    print('t = ', t/(3600*u.s), 'h')

    for i, orbit_deb in enumerate(debris_orbits):
        state_deb = orbit_deb.propagate(t)
        position_deb[i] = state_deb.represent_as(CartesianRepresentation).xyz.to_value(u.km)

    for i, orbit_sat in enumerate(sat_orbits):
        state_sat = orbit_sat.propagate(t)
        position_sat[i] = state_sat.represent_as(CartesianRepresentation).xyz.to_value(u.km)

    for sat in range(total_sats):
        radar_direction = position_sat[sat]/np.linalg.norm(position_sat[sat])
        for deb in range(total_particles):

            rel_position = position_sat[sat] - position_deb[deb]
            dist = np.linalg.norm(rel_position)

            if dist < max_range:
                ran = 1
            else:
                ran = 0

            cos_angle = np.dot(rel_position/dist, radar_direction)
            angle = np.arccos(cos_angle) * (180 / np.pi)

            if angle < FOV:
                fov = 1
            else:
                fov = 0
            if fov*ran == 1:
                v_deb = debris_orbits[deb].propagate(t).v
                v_sat = sat_orbits[sat].propagate(t).v
                v_rel = v_deb - v_sat
                print('debris number: ', deb, ' enters fov')

                size = diameters[deb]

                detection = det_probability(ex_pf, v_rel, dist, size)
                if detection == 1:
                    det_deb = [det_deb, deb]
                    print('detected debris number: ', deb)
                    print('at time: ', t/3600, 'h')
                    det_deb = np.unique(det_deb)

    for deb in doi:
        v_debs[deb] = debris_orbits[deb].propagate(t).v

    v_sats = np.zeros((total_sats, 1))
    for sat in soi:
        v_sats[sat] = sat_orbits[sat].propagate(t).v

fig = go.Figure()

# Plot the Earth
u_ang = np.linspace(0, 2 * np.pi, 100)
v_ang = np.linspace(0, np.pi, 100)
x = 6371 * np.outer(np.cos(u_ang), np.sin(v_ang))
y = 6371 * np.outer(np.sin(u_ang), np.sin(v_ang))
z = 6371 * np.outer(np.ones(np.size(u_ang)), np.cos(v_ang))
earth_surface = go.Surface(x=x, y=y, z=z, colorscale='Blues', opacity=0.6, name='Earth')
fig.add_trace(earth_surface)
max_distance = earth_radius.to_value(u.km)
# Plot positions of the generated debris field
for i in det_deb:
    fig.add_trace(
        go.Scatter3d(x=[position_deb[i, 0]], y=[position_deb[i, 1]], z=[position_deb[i, 2]], mode='markers',
                     marker=dict(color='red', size=2), name=f'Debris {i+1}')
    )
    max_distance = max(max_distance, np.max(np.abs(position_deb)))



max_distance += 1000

fig.update_layout(scene=dict(
    xaxis=dict(range=[-max_distance, max_distance], title='X (km)'),
    yaxis=dict(range=[-max_distance, max_distance], title='Y (km)'),
    zaxis=dict(range=[-max_distance, max_distance], title='Z (km)'),
    aspectmode='manual',
    aspectratio=dict(x=1, y=1, z=1)
),
    title='Detected debris',
    margin=dict(l=0, r=0, b=0, t=50))

fig.show()










