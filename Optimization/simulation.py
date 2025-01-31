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
import warnings
warnings.simplefilter("ignore", category=UserWarning)

################################## INPUTS ##################################

#### Constants
M_earth = 5.972e24  # Earth mass in kg, only needed if you are manually calculating T
earth_radius = 6371.0 *u.km # in km
G = 6.67430e-20  # Gravitational constant in km^3 / (kg * s^2)


################################## DEBRIS INTIALIZATION ##################################

# depending on use_new_dataset we use the MASTER-2009 model instead of a pre-generated dataset (generated still using the MASTER-2009 model, but already saved in a file)


def generate_debris(total_debris, use_new_dataset):

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

        print(f"Dictionary reconstructed from '{filename}'")
        


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


class Constellation:
    
    keys = ["altitudes", "sat_distribution", "inclination", "raan_spacing", "argument_periapsis", "eccentricity"] # class variable

    def __init__(self, **kwargs):

        self.altitudes = kwargs.get("altitudes")
        self.sat_distribution = kwargs.get("sat_distribution")
        self.inclination = kwargs.get("inclination", 90)
        self.raan_spacing = kwargs.get("raan_spacing", 360/len(self.altitudes))
        self.argument_periapsis = kwargs.get("argument_periapsis", 0)
        self.eccentricity = kwargs.get("eccentricity", 0)
        self.total_sats = sum(self.sat_distribution)
        if type(self.altitudes) == None or type(self.sat_distribution) == None:
            raise ValueError("Constellation: Altitudes and satellite distribution must be provided.")
        self.asdict = {"altitudes": self.altitudes, "sat_distribution": self.sat_distribution, "inclination": self.inclination, "raan_spacing": self.raan_spacing, "argument_periapsis": self.argument_periapsis, "eccentricity": self.eccentricity, "total_sats": self.total_sats}
        self.asarray = [self.altitudes, self.sat_distribution, self.inclination, self.raan_spacing, self.argument_periapsis, self.eccentricity]

        self.altitudes *= u.km
        self.inclination *= u.deg
        self.raan_spacing *= u.deg
        self.argument_periapsis *= u.deg



    def __repr__(self):
        return f"Constellation:\n (altitudes={self.altitudes}\n sat_distribution={self.sat_distribution}\n inclination={self.inclination}\n raan_spacing={self.raan_spacing}\n argument_periapsis={self.argument_periapsis}\n eccentricity={self.eccentricity})"
    
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
            num_sats = int(self.sat_distribution[i])          # Number of satellites in the current plane
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

################################## DETECTION (unused) ##################################


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



################################## RADAR ##################################
class Radar:

    def __init__(self, max_range=100, FOV=60):
        self.max_range = max_range      #km
        self.FOV = FOV                  #field of view half-width, degrees


################################## SIMULATION ##############################

class Simulation:

    def __init__(self, simtime, starttime, max_timestep=10.0*u.s, min_timestep=0.5*u.s, safe_range=200, collision_range=1):
        self.simtime = simtime
        self.starttime = starttime
        self.max_timestep = max_timestep
        self.min_timestep = min_timestep
        self.safe_range = safe_range
        self.collision_range = collision_range
        self.timestep = max_timestep
        self.det_deb = None
        self.col_deb = None
        self.position_deb = None
        self.position_sat = None
        self.det_pos = None
        self.det_time = None
        
    def __repr__(self):
        return f"Simulation(simtime={self.simtime}, starttime={self.starttime}, max_timestep={self.max_timestep}, min_timestep={self.min_timestep}, safe_range={self.safe_range})"
################################## DYNAMIC TIMESTEP UPDATES ##################################
# removed because it slowed down too much and increasing debris number still renders the idea
    def update_timestep(self, distances):
        return None
        if np.any(np.argwhere(distances < self.safe_range)):                                     # update timestep if debris is too close to a satellite
            self.timestep = max(self.min_timestep, self.timestep - 0.5*u.s)
            #print(f'Debris too close to satellite, reducing timestep to {self.timestep:.2f}')

        else: 
            self.timestep = self.max_timestep
            #print('No debris too close to satellite, increasing timestep to ', timestep)


################################## DETECTION ALGORITHM ##################################

    def detect_debris2(self, position_sat, position_deb, radar:Radar):

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

        self.update_timestep(distances)  # not actually doing anything, timstep is constant 
        
        debris_in_collision = np.argwhere(distances < self.collision_range)                       # row indices of debris in collision
        debris_in_collision = np.unique(debris_in_collision[:, 0])
        debris_in_FOV = np.argwhere((distances < radar.max_range) & (angles < radar.FOV))               # row indices of debris in FOV
        debris_in_FOV = np.unique(debris_in_FOV[:, 0])
        return debris_in_FOV, debris_in_collision

    def simulation_loop(self, debris_orbits, total_debris, constellation:Constellation, radar:Radar, diameters, ex_pf=ex_pf,):
        

        t = 0
        self.timestep = self.max_timestep
        self.position_deb = np.zeros((total_debris, 3))                     # debris positions in cartesian coordinates
        self.position_sat = np.zeros((constellation.total_sats, 3))         # satellite positions in cartesian coordinates
        #v_debs = np.zeros((total_debris, 3))
        self.det_deb = []
        self.col_deb = []
        self.det_pos = np.zeros((total_debris, 3))
        self.det_time = np.zeros(total_debris)


        while t < self.simtime:

            t += self.timestep
            #print(format_time(t.to_value(u.s)))

            # Calculate the positions of the debris and satellites at the current time
            self.position_deb = propagate_all_orbits(debris_orbits, self.position_deb, t)
            self.position_sat = propagate_all_orbits(constellation.sat_orbits, self.position_sat, t)

            # Apply detection algorithm
            debris_in_FOV, debris_in_collision=self.detect_debris2(self.position_sat, self.position_deb, radar)
            self.det_deb = np.append(self.det_deb, debris_in_FOV)
            self.det_pos[debris_in_FOV] = self.position_deb[debris_in_FOV]
            self.det_time[debris_in_FOV] = t
            self.col_deb = np.append(self.col_deb, debris_in_collision)


        self.det_deb = np.unique(self.det_deb)
        self.det_deb = self.det_deb.astype(int)
        self.col_deb = np.unique(self.col_deb)
        self.col_deb = self.col_deb.astype(int)
    
################################## PLOTTING RESULTS ##################################

def plot_simulation_results(det_deb, position_deb, det_time):

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
                        marker=dict(color='red', size=2), name=f'D{i+1}@t={det_time[i]:.2f}') # why i+1?
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



def main(sim:Simulation, const:Constellation, deb_orbits, deb_diameters, rad:Radar, plot:bool=True, simid='test'):
    """
    This function runs the simulation of the input constellation and returns the efficiency of the constellation in detecting debris.

    Args:
        sim (Simulation): Simulation object.
        const (Constellation): Constellation object.
        deb_orbits (list): List of Orbit objects representing the debris orbits.
        deb_diameters (list): List of diameters of the debris particles.
        rad (Radar): Radar object.

    Returns:
        float: Efficiency of the constellation in detecting debris.
    """
    total_debris = len(deb_orbits)

    # Obtain Orbit objects for the debris and the satellites
    const.generate_satellites()
    
    # Run the simulation
    start_stopwatch = time.time()
    sim.simulation_loop(deb_orbits, total_debris, const, rad, deb_diameters)
    end_stopwatch = time.time()
    elapsed_time = end_stopwatch - start_stopwatch

    if plot:
        plot_simulation_results(sim.det_deb, sim.det_pos, sim.det_time)

    # Calculate the efficiency of the constellation
    collision_penalty = len(sim.col_deb)/len(deb_orbits)
    constellation_efficiency = len(sim.det_deb)/len(deb_orbits) * (1 - collision_penalty)


    # Plot the results
    print(f"S{simid} completed. Elapsed time: {elapsed_time:.2f} s")
    print(f'    - Detected debris: {sim.det_deb}. Efficiency: {constellation_efficiency}. Collisions: {sim.col_deb}. Collision penalty: {collision_penalty}/1')
    return constellation_efficiency






if __name__ == "__main__":

    #### Simulation 
    time_of_flight = 0.1 * u.hour
    start_time = 0*u.s      # Start time of the simulation
    test_sim = Simulation (time_of_flight, start_time, min_timestep=1.0*u.s)

    #### Constellation 
    sat_planes_number = 13             # Number of orbital planes for the satellites
    sat_number = 60         # Number of satellites per plane
    sat_min_altitude = 450       # Altitude of the lowest satellite orbit
    sat_raan_spacing = (360 / sat_planes_number)  # Right Ascension of the Ascending Node (RAAN) spacing
    sat_altitudes = [sat_min_altitude + 75*i for i in range(sat_planes_number)]
    sat_distribution = [sat_number for i in range(sat_planes_number)]
    test_constellation = Constellation(altitudes=sat_altitudes, sat_distribution=sat_distribution, raan_spacing=sat_raan_spacing)

    #### Debris 
    use_new_dataset = False  # Set to False to use the test dataset, True to use the MASTER-2009 model
    total_debris = 1000  # Number of debris particles to simulate. if not using the test dataset, can be arbitrarily chosen. 
                            # if using the test dataset, must be one of the following: 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000

    #### Radar
    radar = Radar()

    # Code execution
    debris_orbits, debris_diameters = generate_debris(total_debris, use_new_dataset)
    constellation_efficiency = main(test_sim, test_constellation, debris_orbits, debris_diameters, radar)
    print(f"Constellation efficiency: {constellation_efficiency:.2f}")