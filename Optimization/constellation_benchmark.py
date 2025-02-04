from simulation import Radar, Constellation, Simulation, main, generate_debris
from gaussian import satellite_dist
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u




#### Simulation 
time_of_flight = 2 * u.hour                                         # Time of flight of the simulation [h]
start_time = 0*u.s                                                  # Start time of the simulation [s]
max_timestep = 5.0*u.s                                              # timestep of the simulation [s]
sim = Simulation (time_of_flight, start_time, max_timestep)         # Create a simulation object

#### Radar
radar = Radar()                                                     # Create a radar object

#### Constellation
num_orbits = 13                                                     # Number of orbits in the constellation              
sats_per_orbit = 4                                                  # Number of satellites per orbit                  
num_sats = num_orbits * sats_per_orbit                              # Number of satellites in the constellation
hmin = 450                                                          # Minimum altitude of the constellation [km]
raan_spacing = 0                                                    # Spacing between the orbits in the RAAN
raan_0 = 0                                                          # RAAN of the first orbit
i_spacing = 0                                                       # Spacing between the orbits in the inclination
i_0 =  0                                                            # Inclination of the first orbit
e = 0.00                                                            # Eccentricity of the orbits

# Parameters of the Gaussian Mixture Model
w1, mu1, s1, w2, mu2, s2 = 9.95456643, 789.88707892, 80.099526, 12.15301612, 1217.49579881, 458.32509372                                                                

#### Debris
debris_num = 2000                                                   # Number of debris objects
use_new_dataset = False                                             # Use a new dataset for the debris
deb_orbits, deb_d = generate_debris(debris_num, use_new_dataset)    # Generate the debris objects

# Generate the satellite distribution
dist, altitudes = satellite_dist(num_orbits = num_orbits, num_sats = num_sats, w1=w1, mu1=mu1, s1=s1, w2=w2, mu2=mu2, s2=s2, hmin=hmin)                                 
# Create a constellation object
const = Constellation(altitudes = altitudes, sat_distribution = dist, i_spacing = i_spacing, i_0 = i_0, raan_spacing = raan_spacing, raan_0 = raan_0, eccentricity = e) 
# Run the simulation, plot the results and return the efficiency
consteff = main(sim, const, deb_orbits, deb_d, radar, plot = True, gpu = True, simid = input("Enter simulation ID: "), saveplot = True) 
