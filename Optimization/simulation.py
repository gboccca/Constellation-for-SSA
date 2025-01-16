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
import csv
from generic_tools import format_time

################################## INPUTS ##################################

#### Constants
M_earth = 5.972e24  # Earth mass in kg, only needed if you are manually calculating T
earth_radius = 6371.0 *u.km # in km
G = 6.67430e-20  # Gravitational constant in km^3 / (kg * s^2)


#### Simulation parameters
time_of_flight = 0.1 * u.hour
time_step = 10 * u.s
use_new_dataset = False  # Set to True to use the test dataset, False to use the MASTER-2009 model
total_particles = 100  # Number of debris particles to simulate. if not using the test dataset, can be arbitrarily chosen. 
                        # if using the test dataset, must be one of the following: 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000
start_time = 0*u.s      # Start time of the simulation



#### Radar data
max_range = 100 #km
FOV = 120       #degrees


#### Constellation data
sat_planes_number = 13             # Number of orbital planes for the satellites
sat_number = 40         # Number of satellites per plane
sat_min_altitude = 450 * u.km       # Altitude of the lowest satellite orbit
sat_inclination = 89 * u.deg
sat_raan_spacing = (360 / sat_number)*u.deg  # Right Ascension of the Ascending Node (RAAN) spacing
sat_theta_spacing = (360 / sat_number)*u.deg  # True anomaly spacing
total_sats = sat_planes_number*sat_number
initialize_random_anomalies = False     # Set to True to initialize satellites with random true anomalies, 
                                        # False to initialize with a standard true anomaly that starts at 0 for the lowest orbit and increases by 360/num_satellites for each subsequent satellite plane


################################## DEBRIS INTIALIZATION ##################################

# depending on use_new_dataset we use the MASTER-2009 model instead of a pre-generated dataset (generated still using the MASTER-2009 model, but already saved in a file)


def generate_debris(use_new_dataset, total_particles):

    """
    Generate a debris field based on the MASTER-2009 model or a pre-generated dataset.
    
    Args:
        use_new_dataset (bool): Set to True to use the test dataset, False to use the MASTER-2009 model.
        total_particles (int): Number of debris particles to simulate. If not using the test dataset, can be arbitrarily chosen.
    
    Returns:
        list: List of Orbit objects representing the debris orbits.
        list: List of diameters of the debris particles.
        
    """	


    if not use_new_dataset:

        ######## to initialize the debris from test datasets ########


        if total_particles not in np.append(np.arange(100,1001,100), np.arange(2000,10001,1000)):
            raise ValueError("The total number of particles must be one of the following: 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000")


        filename = f'Optimization/test_datasets/{total_particles}debris.csv'
        # Read CSV into a dictionary
        with open(filename, "r") as file:
            reader = csv.DictReader(file)
            dataset = {key: [] for key in reader.fieldnames}  # Initialize an empty dictionary with column headers
            for row in reader:
                for key in row:
                    dataset[key].append(row[key])  # Append each value to the corresponding list

        print(f"Dictionary reconstructed from '{filename}':")
        


        latitudes = np.array(dataset['latitudes'], dtype=np.float64)*u.deg
        eccentricities = np.array(dataset['eccentricities'], dtype=np.float64)*u.one
        r_ascension = np.array(dataset['r_ascension'], dtype=np.float64)*u.deg
        arg_periapsis = np.array(dataset['arg_periapsis'], dtype=np.float64)*u.deg
        true_anomaly = np.array(dataset['true_anomaly'], dtype=np.float64)*u.deg
        diameters = np.array(dataset['diameters'], dtype=np.float64) # unitless as of now
        altitudes = np.array(dataset['altitudes'], dtype=np.float64)*u.km
        radii = (altitudes+earth_radius)



        # Generate debris orbits
        debris_orbits = [
            Orbit.from_classical(Earth, rad, ecc, inc, raan, argp, nu, Time.now()) for rad, inc, ecc, raan, argp, nu in zip(radii, latitudes, eccentricities, r_ascension, arg_periapsis, true_anomaly)
        ]


    else:
        ######## to initizalize a random debris field based on the MASTER-2009 model ########


        file_path_alt = "SMDsimulations/master_results.txt"
        file_path_incl = "SMDsimulations/MASTER2_declination_distribution.txt"
        file_path_diam = "SMDsimulations/MASTER2_diameter_distribution.txt"

        columns_alt = [
            "Altitude", "Expl-Fragm", "Coll_Fragm", "Launch/Mis", "NaK-Drops", "SRM-Slag", "SRM-Dust",
            "Paint Flks", "Ejecta", "MLI", "Cloud 1", "Cloud 2", "Cloud 3", "Cloud 4", "Cloud 5", "Human-made",
            "Meteoroids", "Streams", "Total"
        ]

        columns_incl = [
            "Declination", "Expl-Fragm", "Coll_Fragm", "Launch/Mis", "NaK-Drops", "SRM-Slag", "SRM-Dust",
            "Paint Flks", "Ejecta", "MLI", "Cloud 1", "Cloud 2", "Cloud 3", "Cloud 4", "Cloud 5", "Human-made",
            "Meteoroids", "Streams", "Total"
        ]

        columns_diam = [
            "Diameter", "Expl-Fragm", "Coll_Fragm", "Launch/Mis", "NaK-Drops", "SRM-Slag", "SRM-Dust",
            "Paint Flks", "Ejecta", "MLI", "Cloud 1", "Cloud 2", "Cloud 3", "Cloud 4", "Cloud 5", "Human-made",
            "Meteoroids", "Streams", "Total"
        ]

        data_alt = pd.read_csv(file_path_alt, delim_whitespace=True, skiprows=2, names=columns_alt)
        data_incl = pd.read_csv(file_path_incl, delim_whitespace=True, skiprows=2, names=columns_incl)
        data_diam = pd.read_csv(file_path_diam, delim_whitespace=True, skiprows=2, names= columns_diam)

        total_density = data_incl['Expl-Fragm']
        total_sum = total_density.sum()


        # Convert the data values
        #conversion_factor = 1e-9
        #for column in data_alt.columns[1:]:
        #    data_alt[column] *= conversion_factor


        # Compute probability distributions of debris based on input master model
        data_alt['Probability'] = data_alt['Total'] / data_alt['Total'].sum()
        data_incl['Probability'] = total_density/total_sum
        data_diam['Probability'] = data_diam['Total'] / data_diam['Total'].sum()

        # Select random orbit altitudes, inclinations and diameters based on the computed probabilities
        altitudes = np.random.choice(data_alt['Altitude'], size=total_particles, p=data_alt['Probability'])*u.km
        latitudes = np.random.choice(data_incl['Declination'], size=total_particles, p=data_incl['Probability'])*u.deg
        latitudes += 90
        diameters = np.random.choice(data_diam['Diameter'], size=total_particles, p=data_diam['Probability'])

        # Choose remaining orbital elements randomly
        eccentricities = np.random.uniform(-0.05, 0.05, total_particles)*u.one
        r_ascension = np.random.uniform(0,360, total_particles)*u.deg
        arg_periapsis = np.random.uniform(0, 360, total_particles)*u.deg
        true_anomaly = np.random.uniform(0, 360, total_particles)*u.deg
        radii = (altitudes+earth_radius)



        # Generate debris orbits
        debris_orbits = [
            Orbit.from_classical(Earth, rad, ecc, inc, raan, argp, nu, Time.now()) for rad, inc, ecc, raan, argp, nu in zip(radii, latitudes, eccentricities, r_ascension, arg_periapsis, true_anomaly)
        ]

    return debris_orbits, diameters


def intialize_debris_positions(debris_orbits):

    """
    Extract the position data in 3D space.
    
    Args:
        debris_orbits (list): List of Orbit objects representing the debris orbits.
        
    Returns:
        np.array: 3D positions of the debris in Cartesian coordinates.
        
    """
    positions = np.zeros((total_particles, 3))
    for i in range(0, len(debris_orbits)):
        state = debris_orbits[i].propagate(0*u.s)
        positions[i, :] = state.represent_as(CartesianRepresentation).xyz.to_value(u.km)
    return positions


################################## CONSTELLATION INITIALIZATION ##################################


def generate_satellites(num_planes:int, num_satellites:int, altitude, inclination, raan_spacing, theta_spacing, initialize_random_anomalies):
    """
    Generate a constellation of satellites in a given number of planes with a given number of satellites per plane.

    Args:
        num_planes (int): Number of orbital planes for the satellites.
        num_satellites (int): Number of satellites per plane.
        altitude (km): Altitude of the lowest satellite orbit.
        inclination (deg): Inclination of the satellite orbits.
        raan_spacing (deg): Right Ascension of the Ascending Node (RAAN) spacing.
        theta_spacing (deg): True anomaly spacing.
        initialize_random_anomalies (bool): Set to True to initialize satellites with random true anomalies, False to initialize with a standard true anomaly that starts at 0 for the lowest orbit and increases by 360/num_satellites for each subsequent satellite plane.
    Returns:
        list: List of Orbit objects representing the satellite orbits.
        list: List of 3D positions of the satellites in Cartesian coordinates.
        
    """	

    sat_orbits = []
    positions_satellites = []

    for i in range(num_planes):
        raan = i * raan_spacing
        for j in range(num_satellites):
            if initialize_random_anomalies:
                true_anomaly = np.random.uniform(0, 360)*u.deg + j * theta_spacing
            else:
                true_anomaly = (j * theta_spacing + i*360*u.deg/num_planes)
            orbit = Orbit.from_classical(Earth, Earth.R + altitude + (75*i*u.km), 0 * u.one, inclination, raan, 0 * u.deg, true_anomaly)
            sat_orbits.append(orbit)
            positions_satellites.append(orbit.r.to_value(u.km))
    return sat_orbits, positions_satellites




################################## DEFINITION OF PROBABILITY OF DETECTION ##################################


# example probability distribution
# (probability decreases with distance, increases with size and has small velocity effect)
def ex_pf(vel, distance, dim):

    probability = max(0,min(0.5 - 0.1 * distance + 0.1 * dim - 0.05 * abs(vel).value))
    print('Probability: ', probability)

    return probability

# implementation of the probability function
def detection(probability_function, velocity, distance, size):
    detection_probability = probability_function(velocity, distance, size)
    if not (0 <= detection_probability <= 1):
        raise ValueError("Probability function returned a value outside [0,1].")

    return 1 # overruling the actual return value of the probability function to detect every debris that enters the FOV
    return 1 if random.random () < detection_probability else 0

################################## ORBIT PROPAGATION ##################################

def propagate_all_orbits(orbits:list, positions:np.array, time):
    """
    Propagate all orbits to a given time.
    
    Args:
        orbits (list): List of Orbit objects to propagate.
        positions (np.array): 3D positions of the orbits in Cartesian coordinates up until t-1.
        time (Time): Time to propagate the orbits to.
        
    Returns:
        np.array: 3D positions of the orbits in Cartesian coordinates.
        
    """
    for i, orbit in enumerate(orbits):
        state = orbit.propagate(time)
        positions[i, :] = state.represent_as(CartesianRepresentation).xyz.to_value(u.km)
    return positions

################################## SIMULATION ##################################

position_deb = np.zeros((total_particles, 3)) # debris positions in cartesian coordinates
position_sat = np.zeros((total_sats, 3))      # satellite positions in cartesian coordinates
v_debs = np.zeros((total_particles, 3))
soi = []
doi = []
det_deb = []
t   = start_time

debris_orbits, diameters = generate_debris(use_new_dataset, total_particles)
sat_orbits, positions_satellites = generate_satellites(sat_planes_number, sat_number, sat_min_altitude, sat_inclination, sat_raan_spacing, sat_theta_spacing, initialize_random_anomalies)


while t < time_of_flight:

    t += time_step

    fov_check = np.zeros((total_sats, total_particles))
    print(format_time(t.to_value(u.s)))

    # Calculate the positions of the debris and satellites at the current time
    for i, orbit_deb in enumerate(debris_orbits):
        state_deb = orbit_deb.propagate(t)
        position_deb[i] = state_deb.represent_as(CartesianRepresentation).xyz.to_value(u.km)


    for i, orbit_sat in enumerate(sat_orbits):
        state_sat = orbit_sat.propagate(t)
        position_sat[i] = state_sat.represent_as(CartesianRepresentation).xyz.to_value(u.km)

    # Check for debris in the field of view of the satellites
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
                #print('debris number: ', deb, ' enters fov')

                size = diameters[deb]

                
                if detection(ex_pf, v_rel, dist, size) == 1:
                    det_deb = np.append(det_deb,deb)
                    print(f'detected debris number {deb} at time {t/3600} h')
                    #det_deb = np.unique(deb)

    for deb in doi:
        v_debs[deb] = debris_orbits[deb].propagate(t).v

    v_sats = np.zeros((total_sats, 1))
    for sat in soi:
        v_sats[sat] = sat_orbits[sat].propagate(t).v

det_deb = np.unique(det_deb)
det_deb = det_deb.astype(int)
print(f"Detected Debris: {det_deb}")

################################## PLOTTING RESULTS ##################################

fig = go.Figure()

u_ang = np.linspace(0, 2 * np.pi, 100)
v_ang = np.linspace(0, np.pi, 100)
x = 6371 * np.outer(np.cos(u_ang), np.sin(v_ang))
y = 6371 * np.outer(np.sin(u_ang), np.sin(v_ang))
z = 6371 * np.outer(np.ones(np.size(u_ang)), np.cos(v_ang))
earth_surface = go.Surface(x=x, y=y, z=z, colorscale='Blues', opacity=0.6, name='Earth')
fig.add_trace(earth_surface)
max_distance = earth_radius.to_value(u.km) # remove unit
# Plot positions of the generated debris field
for i in det_deb:
    fig.add_trace(
        go.Scatter3d(x=[position_deb[i, 0]], y=[position_deb[i, 1]], z=[position_deb[i, 2]], mode='markers',
                     marker=dict(color='red', size=2), name=f'Debris {i+1}') # why i+1?
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