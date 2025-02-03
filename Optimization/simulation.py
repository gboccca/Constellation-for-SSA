import numpy as np
import cupy as cp
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.twobody.propagation.cowell import CowellPropagator
from astropy import units as u
from astropy.coordinates import CartesianRepresentation
from astropy.time import Time, TimeDelta
import plotly.graph_objects as go
import plotly.io as pio
import random
import csv
import time
from generic_tools import format_time, safe_norm
import warnings
from gaussian import doublegaussian_fit, satellite_dist
warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=FutureWarning)

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


        filename = fr'Optimization\test_datasets\{total_debris}debris.csv'
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
        debris_speeds = [debris_orbits[i].v for i in range(total_debris)]


    else:
        ######## to initizalize a random debris field based on the MASTER-2009 model ########


        file_path_alt = "Optimization/master_results.txt"
        file_path_incl = "Optimization/MASTER2_declination_distribution.txt"
        file_path_diam = "Optimization/MASTER2_diameter_distribution.txt"

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
        latitudes += (90*u.deg)
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

        # Basic Constellation Parameters. Altitudes and satellite distribution are required.
        self.altitudes = kwargs.get("altitudes")
        self.sat_distribution = kwargs.get("sat_distribution")
        self.i_spacing = kwargs.get("i_spacing", 0)
        self.i_0 = kwargs.get("i_0", 90)
        self.raan_spacing = kwargs.get("raan_spacing", 0)
        self.raan_0 = kwargs.get("raan_0", 0)
        self.argument_periapsis = kwargs.get("argument_periapsis", 0)
        self.eccentricity = kwargs.get("eccentricity", 0)
        self.total_sats = sum(self.sat_distribution)
        if type(self.altitudes) == None or type(self.sat_distribution) == None:
            raise ValueError("Constellation: Altitudes and satellite distribution must be provided.")
        
        # Store the Gaussian parameters used to create the constellation
        self.mu1 = kwargs.get("mu1", None)
        self.s1 = kwargs.get("s1", None)
        self.w1 = kwargs.get("w1", None)
        self.mu2 = kwargs.get("mu2", None)
        self.s2 = kwargs.get("s2", None)
        self.w2 = kwargs.get("w2", None)

        # Store the parameters as a dictionary and array, mainly for plotting in PSO runs. array is currently unused.
        self.asdict = { "altitudes": self.altitudes,
                        "sat_distribution": self.sat_distribution, 
                        "i_spacing": self.i_spacing,
                        "i_0": self.i_0,
                        "raan_spacing": self.raan_spacing,
                        "raam_0": self.raan_0,
                        "argument_periapsis": self.argument_periapsis, 
                        "eccentricity": self.eccentricity, 
                        "total_sats": self.total_sats, 
                        "mu1": self.mu1,
                        "s1": self.s1,
                        "w1": self.w1,
                        "mu2": self.mu2,
                        "s2": self.s2,
                        "w2": self.w2}
        
        #self.asarray = [self.altitudes, self.sat_distribution, self.inclination, self.raan_spacing, self.argument_periapsis, self.eccentricity, self.total_sats, self.mu1, self.s1, self.w1, self.mu2, self.s2, self.w2]

        # Convert to astropy units
        self.altitudes *= u.km
        self.i_spacing *= u.deg
        self.i_0 *= u.deg
        self.raan_spacing *= u.deg
        self.raan_0 *= u.deg
        self.argument_periapsis *= u.deg



    def __repr__(self):
        return str(self.asdict)
    def generate_satellites(self):
        """
        Generate a constellation of satellite Orbit objects in a given number of planes with a given number of satellites per plane.

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
        self.sat_speeds = []
        num_planes = len(self.altitudes)     # Number of orbital planes

        for i in range(num_planes):
            altitude = self.altitudes[i]                 # Altitude of the current plane
            num_sats = int(self.sat_distribution[i])          # Number of satellites in the current plane
            theta_spacing = 360/num_sats * u.deg    # Spacing between satellites in the plane (equally spaced)
            theta_offset = 360*i*u.deg/num_planes   # Offset per orbit to distribute satellites evenly in the constellation
            raan = self.raan_0 + i * self.raan_spacing                 # RAAN for the current plane
            inc = self.i_0 + i * self.i_spacing

            for j in range(num_sats):
                true_anomaly = j * theta_spacing + theta_offset  # Spacing satellites evenly in the plane
                orbit = Orbit.from_classical(
                    Earth,
                    (Earth.R + altitude)/(1-self.eccentricity),  # Semi-major axis (altitude above Earth's radius)
                    self.eccentricity * u.one,  # Eccentricity
                    inc,  # Inclination (defaulting to polar orbit, can be adjusted if needed)
                    raan,  # Right Ascension of Ascending Node
                    self.argument_periapsis,  # Argument of Periapsis
                    true_anomaly,  # True Anomaly
                )
                self.sat_orbits.append(orbit)



################################## ORBIT PROPAGATION ##################################

def propagate_all_orbits(orbits:list, positions:np.array, time):
    """
    Propagate all orbits to a given time, sequentially with a for loop (heavy on CPU).
    
    Args:
        orbits (list): List of Orbit objects to propagate.
        positions (np.array): 3D positions of the orbits in Cartesian coordinates up until t-1.
        time (Time): Time to propagate the orbits to.
        
    Returns:
        np.array: 3D positions of the orbits in Cartesian coordinates.
        
    """
    print(f'Propagating orbits to {format_time(time.to_value(u.s))}')
    for i, orbit in enumerate(orbits):
        state = orbit.propagate(time)
        positions[i, :] = state.represent_as(CartesianRepresentation).xyz.to_value(u.km)
    
    print(f'Orbits propagated to {format_time(time.to_value(u.s))}')
    return positions




def propagate_all_orbits_gpu(positions:cp.array, end_times:np.array, orbits:list):
    """
    Propagate all orbits to a given time using Cowell method and GPU acceleration.
    Cowell is the only method that can propagate to an array of times, avoiding a for loop.

    Args:
        positions: (cp.array): empty array to store the 3D positions of the orbits in Cartesian coordinates.
        end_times (cp.array): Array of times to propagate the orbits to.
        orbits (list): List of Orbit objects to propagate.
        
    Returns:
        cp.array: 3D positions of the orbits in Cartesian coordinates.
        
    """

    k = Earth.k.to(u.km**3/u.s**2)  # Gravitational parameter of Earth in km^3/s^2
    propagator = CowellPropagator()

    # Propagate all orbits at all times using mikkola()
    for i, orbit in enumerate(orbits):
        state = orbit.propagate(0*u.s)
        start_pos = state.represent_as(CartesianRepresentation).xyz.to_value(u.km)
        start_v = state.v.to_value(u.km/u.s)
        
        final_pos, _ = propagator.propagate_many(state._state, end_times*u.s)
        positions[:, i, :] = cp.asarray(final_pos/u.s)

    return positions

################################## DETECTION (unused) ##################################



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

        # Empty Result Arrays
        self.det_deb = None
        self.det_pos = None
        self.det_time = None
        self.det_sat = None
        
        self.col_deb = None
        self.col_pos = None
        self.col_time = None
        self.col_sat = None

        self.position_deb = None
        self.position_sat = None

    def __repr__(self):
        return f"Simulation(simtime={self.simtime}, starttime={self.starttime}, max_timestep={self.max_timestep}, min_timestep={self.min_timestep}, safe_range={self.safe_range})"
    
    ################################## DYNAMIC TIMESTEP UPDATES ##################################




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

        #self.update_timestep(distances)  # not actually doing anything, timstep is constant 
        
        debris_in_collision = np.argwhere(distances < self.collision_range)                       # row indices of debris in collision
        debris_in_collision = np.unique(debris_in_collision[:, 0])
        debris_in_FOV = np.argwhere((distances < radar.max_range) & (angles < radar.FOV))               # row indices of debris in FOV
        debris_in_FOV = np.unique(debris_in_FOV[:, 0])
        return debris_in_FOV, debris_in_collision
    
    def simulation_loop(self, debris_orbits, total_debris, constellation:Constellation, radar:Radar, diameters):
        
        t = 0
        self.timestep = self.max_timestep
        self.position_deb = np.zeros((total_debris, 3))                     # debris positions in cartesian coordinates
        self.position_sat = np.zeros((constellation.total_sats, 3))         # satellite positions in cartesian coordinates
        self.det_deb = []
        self.col_deb = []
        self.det_pos = np.zeros((total_debris, 3))
        self.det_time = np.zeros(total_debris)


        while t < self.simtime:

            t += self.timestep

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


    def detect_debris_array_GPU(self, position_sat, position_deb, radar):
        """
        Fully vectorized GPU implementation of the detection algorithm.
        Processes all timesteps at once.
        
        Args:
            position_sat (cp.array): Shape (timesteps, total_sats, 3) - Satellite positions over time.
            position_deb (cp.array): Shape (timesteps, total_debris, 3) - Debris positions over time.
            radar (Radar): Radar object containing max_range and FOV attributes.
            

        Returns:
            cp.array: detections array. Each row is a collision, column 1 is the time index, column 2 is the debris #, coumn 3 is te satellite #.
            cp.array: collisions array, same indexing as above.
        """
        
        # Careful: this is 7GB of RAM for 1000 debris, 440 satellites and 1440 timesteps
        rel_positions = (position_deb[:, :, cp.newaxis, :] - position_sat[:, cp.newaxis, :, :]) # Shape: (T, D, S, 3)

        distances = cp.linalg.norm(rel_positions, axis=3)                                       # Shape: (T, D, S)

        dotproducts = cp.sum(rel_positions * position_sat[:, cp.newaxis, :, :], axis=3)         # Shape: (T, D, S)
        normproducts = cp.linalg.norm(position_sat, axis=2)[:, cp.newaxis, :] * distances       # Shape: (T, D, S)
        angles = cp.arccos(cp.clip(dotproducts / normproducts, -1.0, 1.0)) * (180 / cp.pi)      # Shape: (T, D, S)
       
        # Find debris in collision range: (T, num_collisions)
        collisions = cp.unique(cp.argwhere(distances < self.collision_range), axis=0)

        # Find debris in radar FOV: (T, num_detected)
        detections = cp.argwhere((distances < radar.max_range) & (angles < radar.FOV))
        
        return detections, collisions


    def simulation_loop_array_GPU(self, debris_orbits, total_debris, const:Constellation, radar:Radar, ):
        """
        Run the simulation "loop" using GPU parallelization and vectorized operations.
        Runs batch_size timesteps at a time. Stores the results in the Simulation object. 
        
        Args:
            debris_orbits (list): List of Orbit objects representing the debris orbits.
            total_debris (int): Number of debris particles to simulate.
            const (Constellation): Constellation object.
            radar (Radar): Radar object.
            batch_size (int): Number of timesteps to process at once. Default is 500.

        Returns:
            None
        """


        # Check GPU memory
        device = cp.cuda.Device(0)  # Get the first GPU
        # print("GPU id:", device.id)
        # print("Total Memory (GB):", device.mem_info[1] / 1e9)

        detections = cp.zeros((1,3))
        collisions = cp.zeros((1,3))

        times = np.arange(self.starttime.to_value(u.s), self.simtime.to_value(u.s), self.timestep.to_value(u.s))

        # Calculate batch size based on available memory
        free_memory, _ = cp.cuda.runtime.memGetInfo()
        # print("Free Memory (GB):", free_memory / 1e9)

        batch_size = int(free_memory* 0.9/ (2*(3*total_debris+3*const.total_sats+7*const.total_sats*total_debris)) ) # 4 bytes per float32, 0.8 factor for overhead
        batch_size = 500 # because i cant get the batch calculated properly and this kinda works

        # Process timesteps in batches
        for i in range(0, (len(times)),batch_size):
        
            batch_times = times[i:i+batch_size]

            # Initialize position arrays. Each row is a timestep, each row is a satellite/debris. Each element is a 3D vector storing the position
            # Using float32 to keep acccuracy and reduce memory usage. float16 cannot be used because of overflow in cp.linalg.norm
            self.position_deb = cp.zeros((len(batch_times), total_debris, 3), dtype=cp.float32)           # Shape: (T, S, 3)
            self.position_sat = cp.zeros((len(batch_times), const.total_sats, 3) , dtype=cp.float32)      # Shape: (T, D, 3)

            # Propagate all orbits at all times using cowell(). cowell is the only function in poliastro that can propagate to an array of times
            self.position_deb = propagate_all_orbits_gpu(self.position_deb, batch_times, debris_orbits)
            self.position_sat = propagate_all_orbits_gpu(self.position_sat, batch_times, const.sat_orbits)


            
            # Run detection for all timesteps in batch at once
            batch_detections, batch_collisions = self.detect_debris_array_GPU(self.position_sat, self.position_deb, radar)
            detections = cp.concatenate((detections, batch_detections), axis=0)
            collisions = cp.concatenate((collisions, batch_collisions), axis=0)

        # Remove duplicates
        _, detection_indices = cp.unique(detections[:,1], return_index=True)
        _, collision_indices = cp.unique(collisions[:,1], return_index=True)
        detections = detections[detection_indices]
        collisions = collisions[collision_indices]

        free_memory, _ = cp.cuda.runtime.memGetInfo()
        # print("Free Memory (GB):", free_memory / 1e9)

        # Store results in the Simulation object
        detections = detections[1:]                                     # Remove first row (0,0,0)
        self.det_deb = detections[:,1].astype(int)                      # column 1 of detections array is the debris index
        self.det_time = detections[:,0].astype(int)                     # column 0 of detections array is the detection time index (not the actual time)
        self.det_sat = detections[:,2].astype(int)                      # column 2 of detections array is the index of the satellite responsible for the detection
        self.det_pos= self.position_deb[self.det_time, self.det_deb, :]     # 3D positions of detected debris, obtained by extracting the positions of the detected debris at the detection time

        collisions = collisions[1:]                                     # Same as above for collisions     
        self.col_deb = collisions[:,1].astype(int)  
        self.col_time =  collisions[:,0].astype(int)
        self.col_sat = collisions[:,2].astype(int)
        self.col_pos= self.position_deb[self.col_time, self.col_deb, :]
        times = cp.asarray(times)

        
        # Convert results back to CPU for plotting
        self.det_deb = self.det_deb.get()
        self.det_pos = self.det_pos.get()
        print(self.det_pos)
        self.det_time = self.det_time.get()

        self.position_sat = self.position_sat.get()
        self.position_deb = self.position_deb.get()
    
        self.col_deb = self.col_deb.get()
        self.col_pos = self.col_pos.get()
        self.col_time = self.col_time.get()




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

def plot_simulation_results_gpu(sim:Simulation):
    """
    Plot the results of the simulation. This function is used when the simulation is run on the GPU.
    
    Args:
        sim (Simulation): Simulation object containing the results of the simulation.
        
    Returns:
        None
    """

    fig = go.Figure()

    u_ang = np.linspace(0, 2 * np.pi, 100)
    v_ang = np.linspace(0, np.pi, 100)
    x = 6371 * np.outer(np.cos(u_ang), np.sin(v_ang))
    y = 6371 * np.outer(np.sin(u_ang), np.sin(v_ang))
    z = 6371 * np.outer(np.ones(np.size(u_ang)), np.cos(v_ang))
    earth_surface = go.Surface(x=x, y=y, z=z, colorscale='Blues', opacity=0.6, name='Earth', showscale=False)
    fig.add_trace(earth_surface)
    max_distance = earth_radius.to_value(u.km) # remove unit
    # Plot positions of the generated debris field
    for i in range(len(sim.det_deb)):
        fig.add_trace(
            go.Scatter3d(x=[sim.det_pos[i, 0]], y=[sim.det_pos[i, 1]], z=[sim.det_pos[i, 2]], mode='markers',
                        marker=dict(color='red', size=2), name=f'D{sim.det_deb[i]+1}@t={sim.det_time[i]*sim.timestep:.0f}') # why i+1?
        )
        max_distance = max(max_distance, np.max(np.abs(sim.position_deb)))


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
    pio.write_image(fig, r"Optimization\sim_results\lastsim.png")
    fig.show()


def main(sim:Simulation, const:Constellation, deb_orbits, deb_diameters, rad:Radar, plot:bool=False, gpu:bool=True, simid='test'):
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
    

    #sim.simulation_loop(deb_orbits, total_debris, const, rad, deb_diameters)
    #sim.simulation_loop_gpu(deb_orbits, total_debris, const, rad, deb_diameters)
    #sim.simulation_loop_array(deb_orbits, total_debris, const, rad)
    if gpu:
        sim.simulation_loop_array_GPU(deb_orbits, total_debris, const, rad)
    else:
        sim.simulation_loop(deb_orbits, total_debris, const, rad, deb_diameters)
    
    end_stopwatch = time.time()
    elapsed_time = end_stopwatch - start_stopwatch

    if plot:
        if not gpu:
            plot_simulation_results(sim.det_deb, sim.det_pos, sim.det_time)
        else:
            plot_simulation_results_gpu(sim)

    # Calculate the efficiency of the constellation
    collision_penalty = len(sim.col_deb)/len(deb_orbits)
    constellation_efficiency = len(sim.det_deb)/len(deb_orbits) * (1 - collision_penalty)


    # Plot the results
    print(f"S{simid} completed. Elapsed time: {elapsed_time:.2f} s")
    print(f'    - Detected debris: {sim.det_deb}. Efficiency: {constellation_efficiency}. Collisions: {sim.col_deb}. Collision penalty: {collision_penalty}/1')
    return constellation_efficiency


if __name__ == "__main__":

    #### Simulation 
    time_of_flight = 2 * u.hour
    start_time = 0*u.s      # Start time of the simulation
    max_timestep = 5.0*u.s  # timestep of the simulation
    test_sim = Simulation (time_of_flight, start_time, max_timestep)

    #### Constellation 
    sat_planes_number = 12                                  # Number of orbital planes for the satellites
    sat_number = 4                                          # Number of satellites per plane
    total_sats = sat_number * sat_planes_number
    sat_min_altitude = 450                                  # Altitude of the lowest satellite orbit
    #sat_raan_spacing = (360 / sat_planes_number)           # Right Ascension of the Ascending Node (RAAN) spacing
    sat_raan_spacing = 0
    sat_raan_0 = 0
    sat_inc_spacing = 0
    sat_inc_0 = 90
    #sat_altitudes = [sat_min_altitude + 75*i for i in range(sat_planes_number)]
    #sat_distribution = [sat_number for i in range(sat_planes_number)]
    #w1, mu1, s1, wu2, mu2, s2 = doublegaussian_fit()
    w1, mu1, s1, wu2, mu2, s2 = 9.95456643, 789.88707892, 80.099526, 12.15301612, 1217.49579881, 458.32509372
    sat_distribution, sat_altitudes = satellite_dist(w1 = w1, mu1=mu1, s1 = s1, w2 = wu2, mu2 = mu2, s2 = s2, num_obrits=sat_planes_number, num_sats=sat_number)
    test_constellation = Constellation(altitudes=sat_altitudes, sat_distribution=sat_distribution, raan_spacing=sat_raan_spacing,  raan_0 = sat_raan_0, i_spacing = sat_inc_spacing, i_00=sat_inc_0)

    #### Debris 
    use_new_dataset = False                                 # Set to False to use the test dataset, True to use the MASTER-2009 model
    total_debris = 2000                                     # Number of debris particles to simulate. if not using the test dataset, can be arbitrarily chosen.
                                                            # if using the test dataset, must be one of the following: 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000

    #### Radar
    radar = Radar()

    # Code execution
    debris_orbits, debris_diameters = generate_debris(total_debris, use_new_dataset)
    constellation_efficiency = main(test_sim, test_constellation, debris_orbits, debris_diameters, radar, plot=True)
    print(f"Constellation efficiency: {constellation_efficiency:.2f}")