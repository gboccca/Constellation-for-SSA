from simulation import Simulation, Constellation, Radar, main, generate_debris
import numpy as np
from astropy import units as u

# PSO Hyperparameters
num_particles = 5
num_iterations = 100

# Simulation parameters
time_of_flight = 0.1 * u.hour
start_time = 0*u.s      # Start time of the simulation
sim = Simulation (time_of_flight, start_time)
radar = Radar()
deb_number = 100
use_new_dataset = False
deb_orbits, deb_diameters = generate_debris(deb_number, use_new_dataset)

# Constant constellation parameters
num_planes = 12
raan_spacing = 360/num_planes * u.deg
min_altitude = 450
max_altitude = 1350

# default constellation parameters
default_altitudes = np.array([min_altitude + i*(max_altitude-min_altitude)/num_planes for i in range(num_planes)])    # Altitude range (km)
default_distribution = np.array([40 for _ in range(num_planes)])                                            # satellites per plane

# bounds for the constellation parameters
altitude_bounds = np.array([np.array([-99,99]) + default_altitudes[i] for i in range(num_planes)])
distribution_bounds = np.array([np.array([-20,20]) + default_distribution[i] for i in range(num_planes)])
altitude_lbounds = altitude_bounds[:,0]
altitude_ubounds = altitude_bounds[:,1]
distribution_lbounds = distribution_bounds[:,0]
distribution_ubounds = distribution_bounds[:,1]

dim = 2*num_planes

# intialize particles
positions_alt = np.random.uniform(low=altitude_lbounds, high=altitude_ubounds, size=(int(num_particles), int(num_planes)))
positions_dist =  np.random.randint(low=distribution_lbounds, high=distribution_ubounds, size=(int(num_particles), int(num_planes)))
velocities_alt = np.random.uniform(low=-5, high=5, size=(int(num_particles), int(num_planes)))
velocities_dist  = np.random.randint(low=-2, high=2, size=(int(num_particles), int(num_planes)))           
positions = np.concatenate((positions_alt, positions_dist), axis=1)
velocities = np.concatenate((velocities_alt, velocities_dist), axis=1)
constellations = [Constellation(altitudes*u.km, distribution, raan_spacing) for altitudes, distribution in zip(positions_alt, positions_dist)]
print(constellations)
personal_best_positions = np.copy(positions)
personal_best_scores = np.array([])
for constellation in constellations:

    constellation_efficiency = main(sim, constellation, deb_orbits, deb_diameters, radar)
    personal_best_scores = np.append(personal_best_scores,constellation_efficiency)

global_best_score = np.max(personal_best_scores)
global_best_position = np.copy(personal_best_positions[np.argmax(personal_best_scores)])


print(personal_best_positions)
print(personal_best_scores)
print(global_best_score)
print(global_best_position)
