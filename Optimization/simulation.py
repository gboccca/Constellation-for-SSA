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
import time
from generic_tools import format_time

################################## INPUTS ##################################

#### Constants
M_earth = 5.972e24  # Earth mass in kg, only needed if you are manually calculating T
earth_radius = 6371.0 *u.km # in km
G = 6.67430e-20  # Gravitational constant in km^3 / (kg * s^2)


#### Simulation parameters
time_of_flight = 0.1 * u.hour

use_new_dataset = False  # Set to True to use the test dataset, False to use the MASTER-2009 model
total_debris = 100  # Number of debris particles to simulate. if not using the test dataset, can be arbitrarily chosen. 
                        # if using the test dataset, must be one of the following: 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000
start_time = 0*u.s      # Start time of the simulation
min_timestep_default = 0.5 * u.s
max_timestep_default = 10 * u.s
time_step = max_timestep_default
safe_range_default = 200         # used for dynamic timestep


#### Radar data
max_range = 100 #km
FOV = 60        #field of view half-width, degrees




################################## DEBRIS INTIALIZATION ##################################

# depending on use_new_dataset we use the MASTER-2009 model instead of a pre-generated dataset (generated still using the MASTER-2009 model, but already saved in a file)


def generate_debris(use_new_dataset, total_debris):

    """
    Generate a debris field based on the MASTER-2009 model or a pre-generated dataset.
    
    Args:
        use_new_dataset (bool): Set to True to use the test dataset, False to use the MASTER-2009 model.
        total_debris (int): Number of debris particles to simulate. If not using the test dataset, can be arbitrarily chosen.
    
    Returns:
        list: List of Orbit objects representing the debris orbits.
        list: List of diameters of the debris particles.
        
    """	


    if not use_new_dataset:

        ######## to initialize the debris from test datasets ########


        if total_debris not in np.append(np.arange(100,1001,100), np.arange(2000,10001,1000)):
            raise ValueError("The total number of particles must be one of the following: 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000")


        filename = f'Optimization/test_datasets/{total_debris}debris.csv'
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
        altitudes = np.random.choice(data_alt['Altitude'], size=total_debris, p=data_alt['Probability'])*u.km
        latitudes = np.random.choice(data_incl['Declination'], size=total_debris, p=data_incl['Probability'])*u.deg
        latitudes += 90
        diameters = np.random.choice(data_diam['Diameter'], size=total_debris, p=data_diam['Probability'])

        # Choose remaining orbital elements randomly
        eccentricities = np.random.uniform(-0.05, 0.05, total_debris)*u.one
        r_ascension = np.random.uniform(0,360, total_debris)*u.deg
        arg_periapsis = np.random.uniform(0, 360, total_debris)*u.deg
        true_anomaly = np.random.uniform(0, 360, total_debris)*u.deg
        radii = (altitudes+earth_radius)



        # Generate debris orbits
        debris_orbits = [
            Orbit.from_classical(Earth, rad, ecc, inc, raan, argp, nu, Time.now()) for rad, inc, ecc, raan, argp, nu in zip(radii, latitudes, eccentricities, r_ascension, arg_periapsis, true_anomaly)
        ]

    return debris_orbits, diameters


################################## CONSTELLATION INITIALIZATION ##################################


def generate_satellites_old(num_planes:int, num_satellites:int, altitude, inclination, raan_spacing, theta_spacing, initialize_random_anomalies):
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
            print(true_anomaly)
            
            sat_orbits.append(orbit)
            positions_satellites.append(orbit.r.to_value(u.km))
    return sat_orbits, positions_satellites


class Constellation:

    def __init__(self, altitudes:list, sat_distribution:list, inclination, raan_spacing, argument_periapsis, eccentricity = 0):
        self.altitudes = altitudes
        self.sat_distribution = sat_distribution
        self.inclination = inclination
        self.raan_spacing = raan_spacing
        self.argument_periapsis = argument_periapsis
        self.eccentricity = eccentricity
        self.total_sats = sum(sat_distribution)


    def generate_satellites(self):
        """
        Generate a constellation of satellites in a given number of planes with a given number of satellites per plane.

        Args:
            altitudes (list): List of altitudes for the satellite planes.
            sat_distribution (list): List of number of satellites per plane.
            inclination (deg): Inclination of the satellite orbits.
            raan_spacing (deg): Right Ascension of the Ascending Node (RAAN) spacing.
            argument_periapsis (deg): Argument of Periapsis.
            eccentricity (float): Eccentricity of the satellite orbits. Default is 0.

        Returns:
            list: List of Orbit objects representing the satellite orbits.
        """

        self.sat_orbits = []
        num_planes = len(self.altitudes)     # Number of orbital planes

        for i in range(num_planes):
            altitude = self.altitudes[i]                 # Altitude of the current plane
            num_sats = sat_distribution[i]          # Number of satellites in the current plane
            theta_spacing = 360/num_sats * u.deg    # Spacing between satellites in the plane (equally spaced)
            theta_offset = 360*i*u.deg/num_planes   # Offset per orbit to distribute satellites evenly in the constellation
            raan = i * self.raan_spacing                 # RAAN for the current plane


            for j in range(num_sats):
                true_anomaly = j * theta_spacing + theta_offset  # Spacing satellites evenly in the plane
                orbit = Orbit.from_classical(
                    Earth,
                    Earth.R + altitude,  # Semi-major axis (altitude above Earth's radius)
                    self.eccentricity * u.one,  # Eccentricity
                    self.inclination,  # Inclination (defaulting to polar orbit, can be adjusted if needed)
                    raan,  # Right Ascension of Ascending Node
                    self.argument_periapsis,  # Argument of Periapsis
                    true_anomaly,  # True Anomaly
                )
                self.sat_orbits.append(orbit)



################################## DYNAMIC TIMESTEP UPDATES ##################################

def update_timestep(distances, timestep, max_timestep=max_timestep_default, min_timestep=min_timestep_default, safe_range=safe_range_default):

    if np.any(np.argwhere(distances < safe_range)):                                     # update timestep if debris is too close to a satellite
        timestep = max(min_timestep, timestep - 0.1*u.s)
        #print('Debris too close to satellite, reducing timestep to ', timestep)
        return timestep
    else: 
        timestep = max_timestep
        #print('No debris too close to satellite, increasing timestep to ', timestep)
        return max_timestep   


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

################################## DETECTION ##################################


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

def detect_debris_old(total_sats, total_debris, position_sat, position_deb, debris_orbits, sat_orbits, debris_diameters, t, max_range, FOV, ex_pf, det_deb):
# note: switched this for detect_debris2, which is a more efficient implementation of the detection algorithm using numpy arrays instead of for loops
# for some reason, the new function gives more detected debris than the old one - fixed: i swithed the sign of rel_position in the angle calculation, it was was wrong here. the correct is deb-sat, not sat-deb

    for sat in range(total_sats):
        radar_direction = position_sat[sat]/np.linalg.norm(position_sat[sat])
        for deb in range(total_debris):

            # Check for debris in the field of view of the satellites
            rel_position =  position_deb[deb] - position_sat[sat]
            dist = np.linalg.norm(rel_position)
            cos_angle = np.dot(rel_position/dist, radar_direction)
            angle = np.arccos(cos_angle) * (180 / np.pi)
            if dist < max_range and angle<FOV:

                #print('debris number: ', deb, ' enters fov')
                # Get debris velocity and size to calculate detection probability
                v_deb = debris_orbits[deb].propagate(t).v
                v_sat = sat_orbits[sat].propagate(t).v
                v_rel = v_deb - v_sat
                size = diameters[deb]
                
                # Check if debris is detected
                if detection(ex_pf, v_rel, dist, size) == 1:
                    det_deb = np.append(det_deb,deb)
                    print(f'Detected debris number {deb} at time {t/3600} h')
    
    return det_deb


def detect_debris(position_sat, position_deb, debris_orbits, sat_orbits, debris_diameters, t, max_range, FOV, ex_pf, det_deb, timestep):

    """
    Detect debris in the field of view of the satellites.

    Args:
        position_sat (np.array): 3D positions of the satellites in Cartesian coordinates.
        position_deb (np.array): 3D positions of the debris in Cartesian coordinates.
        debris_orbits (list): List of Orbit objects representing the debris orbits.
        sat_orbits (list): List of Orbit objects representing the satellite orbits.
        debris_diameters (list): List of diameters of the debris particles.
        t (Time): Current time.
        max_range (float): Maximum detection range of the satellites.
        FOV (float): Field of view half-width of the satellites.
        ex_pf (function): Probability function.
        det_deb (list): List of detected debris.
        timestep (Time): Current timestep.

    Returns:
        list: updated list of detected debris.
        Time: Updated timestep.

    """

    for sat in range(len(position_sat)):
        rel_positions = position_deb - position_sat[sat]                # correct formula
        distances = np.linalg.norm(rel_positions, axis=1)
        cosines = np.dot(rel_positions, position_sat[sat])/(np.linalg.norm(position_sat[sat])*distances)
        angles = np.arccos(cosines)* (180 / np.pi)
        # note: removed radar direction vector in angle calculation because it is the same as the satellite position vector

        timestep = update_timestep(distances,timestep)
        debris_in_FOV = np.where((distances < max_range) & (angles < FOV))[0] # row indices of debris in FOV

        det_deb = np.append(det_deb, debris_in_FOV)

    """ Temporarily removed this section because the probabiliyu function is not implemented

        for deb in debris_in_FOV:
            v_deb = debris_orbits[deb].propagate(t).v
            v_sat = sat_orbits[sat].propagate(t).v
            v_rel = v_deb - v_sat
            size = debris_diameters[deb]
            if detection(ex_pf, v_rel, distances[deb], size) == 1:
                det_deb = np.append(det_deb, deb)
                print(f'Detected debris number {deb} at time {t/3600} h')

    """

    return det_deb,timestep


def detect_debris2(position_sat, position_deb, max_range, FOV, det_deb, timestep):

    """
    Slightly more efficient implementation of the detection algorithm using numpy arrays instead of for loops. Verifired that it the same outputs as detect_debris.

    Args:
        position_sat (np.array): 3D positions of the satellites in Cartesian coordinates.
        position_deb (np.array): 3D positions of the debris in Cartesian coordinates.
        max_range (float): Maximum detection range of the satellites.
        FOV (float): Field of view half-width of the satellites.
        det_deb (list): List of detected debris.
        timestep (Time): Current timestep.

    Returns:
        list: updated list of detected debris.
        Time: Updated timestep.

    """

    rel_positions = position_deb[:, np.newaxis, :] - position_sat[np.newaxis, :, :]     # relative positions of debris to satellites. shape: (total_debris, total_sats, 3)
    distances = np.linalg.norm(rel_positions, axis=2)                                   # distances between debris and satellites. shape: (total_debris, total_sats)
    dotproducts = np.sum(rel_positions*position_sat, axis=2)                            # dot product between relative positions and satellite positions. shape: (total_debris, total_sats)
    normproducts = np.linalg.norm(position_sat, axis=1)*distances                       # product of norms of relative and satellite positions. shape: (total_debris, total_sats)
    angles = np.arccos(dotproducts/normproducts) * (180 / np.pi)                        # angles between debris and satellites. shape: (total_debris, total_sats)

    timestep = update_timestep(distances, timestep)  
         
    debris_in_FOV = np.argwhere((distances < max_range) & (angles < FOV))               # row indices of debris in FOV
    debris_in_FOV = np.unique(debris_in_FOV[:, 0])
    det_deb = np.append(det_deb, debris_in_FOV)

    return det_deb, timestep


################################## SIMULATION ##################################




def simulation_loop(time_of_flight, time_step, start_time, debris_orbits, sat_orbits, total_debris, total_sats, max_range, FOV, diameters, ex_pf=ex_pf,):
    

    t = start_time
    position_deb = np.zeros((total_debris, 3))      # debris positions in cartesian coordinates
    position_sat = np.zeros((total_sats, 3))        # satellite positions in cartesian coordinates
    v_debs = np.zeros((total_debris, 3))
    det_deb = []


    while t < time_of_flight:

        t += time_step
        #print(format_time(t.to_value(u.s)))

        # Calculate the positions of the debris and satellites at the current time
        position_deb = propagate_all_orbits(debris_orbits, position_deb, t)
        position_sat = propagate_all_orbits(sat_orbits, position_sat, t)

        # Apply detection algorithm
        #det_deb = detect_debris_old(total_sats, total_debris, position_sat, position_deb, debris_orbits, sat_orbits, diameters, t, max_range, FOV, ex_pf, det_deb)
        #det_deb, time_step = detect_debris(position_sat, position_deb, debris_orbits, sat_orbits, diameters, t, max_range, FOV, ex_pf, det_deb, time_step)   # slightly slower than detect_debris2
        det_deb, time_step = detect_debris2(position_sat, position_deb, max_range, FOV, det_deb, time_step)


    det_deb = np.unique(det_deb)
    det_deb = det_deb.astype(int)

    return det_deb, position_deb, position_sat

################################## PLOTTING RESULTS ##################################

def plot_simulation_results(det_deb, position_deb):

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

if __name__ == "__main__":


    #### Constellation data
    sat_planes_number = 13             # Number of orbital planes for the satellites
    sat_number = 40         # Number of satellites per plane
    sat_min_altitude = 450 * u.km       # Altitude of the lowest satellite orbit
    sat_inclination = 90 * u.deg
    sat_raan_spacing = (360 / sat_planes_number)*u.deg  # Right Ascension of the Ascending Node (RAAN) spacing
    sat_theta_spacing = (360 / sat_number)*u.deg  # True anomaly spacing
    sat_argument_periapsis = 0*u.deg
    total_sats = sat_planes_number*sat_number
    initialize_random_anomalies = False     # Set to True to initialize satellites with random true anomalies, 
                                            # False to initialize with a standard true anomaly that starts at 0 for the lowest orbit and increases by 360/num_satellites for each subsequent satellite plane

    sat_altitudes = [sat_min_altitude + 75*i*u.km for i in range(sat_planes_number)]
    sat_distribution = [sat_number for i in range(sat_planes_number)]

    debris_orbits, diameters = generate_debris(use_new_dataset, total_debris)
    #sat_orbits, positions_satellites = generate_satellites_old(sat_planes_number, sat_number, sat_min_altitude, sat_inclination, sat_raan_spacing, sat_theta_spacing, initialize_random_anomalies)

    test_constellation = Constellation(sat_altitudes, sat_distribution, sat_inclination, sat_raan_spacing, sat_argument_periapsis)
    test_constellation.generate_satellites()
    #generate_satellites(sat_altitudes, sat_distribution, sat_inclination, sat_raan_spacing, 0*u.deg)
    start_stopwatch = time.time()
    det_deb, position_deb, position_sat = simulation_loop(time_of_flight, time_step, start_time, debris_orbits, test_constellation.sat_orbits, total_debris, test_constellation.total_sats, max_range, FOV, diameters)
    end_stopwatch = time.time()
    elapsed_time = end_stopwatch - start_stopwatch
    print(f"Elapsed time: {elapsed_time:.2f} s")
    plot_simulation_results(det_deb, position_deb)
    #constellation_efficiency = len(det_deb)/total_debris



