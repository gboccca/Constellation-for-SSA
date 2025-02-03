import numpy as np
import cupy as cp
from simulation import Radar, Constellation, Simulation, propagate_all_orbits


class Simulation:

    # removed because it slowed down too much and increasing debris number still renders the idea
    def update_timestep(self, distances):
        return None
        if np.any(np.argwhere(distances < self.safe_range)):                                     # update timestep if debris is too close to a satellite
            self.timestep = max(self.min_timestep, self.timestep - 0.5*u.s)
            #print(f'Debris too close to satellite, reducing timestep to {self.timestep:.2f}')

        else: 
            self.timestep = self.max_timestep
            #print('No debris too close to satellite, increasing timestep to ', timestep)


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


def detect_debris_gpu(self, position_sat, position_deb, radar:Radar):
    """
    GPU implementation of detect_debris2.
    Made to process one timestep at a time.

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

    # Move data to GPU
    position_sat = cp.asarray(position_sat)
    position_deb = cp.asarray(position_deb)


    rel_positions = position_deb[:, cp.newaxis, :] - position_sat[cp.newaxis, :, :]     # relative positions of debris to satellites. shape: (total_debris, total_sats, 3)
    distances = cp.linalg.norm(rel_positions, axis=2)                                   # distances between debris and satellites. shape: (total_debris, total_sats)
    dotproducts = cp.sum(rel_positions*position_sat, axis=2)                            # dot product between relative positions and satellite positions. shape: (total_debris, total_sats)
    normproducts = cp.linalg.norm(position_sat, axis=1)*distances                       # product of norms of relative and satellite positions. shape: (total_debris, total_sats)
    angles = cp.arccos(dotproducts/normproducts) * (180 / cp.pi)                        # angles between debris and satellites. shape: (total_debris, total_sats)

    self.update_timestep(distances)  # not actually doing anything, timstep is constant 
    
    debris_in_collision = cp.argwhere(distances < self.collision_range)                       # row indices of debris in collision
    debris_in_collision = cp.unique(debris_in_collision[:, 0])
    debris_in_FOV = cp.argwhere((distances < radar.max_range) & (angles < radar.FOV))               # row indices of debris in FOV
    debris_in_FOV = cp.unique(debris_in_FOV[:, 0])


    return debris_in_FOV.get(), debris_in_collision.get()

def simulation_loop_gpu(self, debris_orbits, total_debris, constellation:Constellation, radar:Radar, diameters, ex_pf=ex_pf,):
    """
    Run the simulation "loop" using GPU parallelization.
    Processes one timestep at a time. Stores the results in the Simulation object.
    
    Args:
        debris_orbits (list): List of Orbit objects representing the debris orbits.
        total_debris (int): Number of debris particles to simulate.
        constellation (Constellation): Constellation object.
        radar (Radar): Radar object.
        diameters (list): List of diameters of the debris particles. - unused
        ex_pf (function): Probability function to detect debris. - unused
    
    Returns:
        None
    """
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
        debris_in_FOV, debris_in_collision=self.detect_debris_gpu(self.position_sat, self.position_deb, radar)
        self.det_deb = np.append(self.det_deb, debris_in_FOV)
        self.det_pos[debris_in_FOV] = self.position_deb[debris_in_FOV]
        self.det_time[debris_in_FOV] = t
        self.col_deb = np.append(self.col_deb, debris_in_collision)


    self.det_deb = np.unique(self.det_deb)
    self.det_deb = self.det_deb.astype(int)
    self.col_deb = np.unique(self.col_deb)
    self.col_deb = self.col_deb.astype(int)

def detect_debris_array(self, position_sat, position_deb, radar):
    """
    Fully vectorized CPU implementation of the detection algorithm.
    Processes all timesteps at once.
    
    Args:
        position_sat (cp.array): Shape (timesteps, total_sats, 3) - Satellite positions over time.
        position_deb (cp.array): Shape (timesteps, total_debris, 3) - Debris positions over time.
        radar (Radar): Radar object containing max_range and FOV attributes.

    Returns:
        cp.array: Indices of detected debris per timestep.
        cp.array: Indices of collision debris per timestep.
    """

    # Compute relative positions: (T, D, S, 3)
    rel_positions = position_deb[:, :, np.newaxis, :] - position_sat[:, np.newaxis, :, :]

    # Compute distances: (T, D, S)
    distances = np.linalg.norm(rel_positions, axis=3)

    # Compute dot products: (T, D, S)
    dotproducts = np.sum(rel_positions * position_sat[:, np.newaxis, :, :], axis=3)

    # Compute norm products: (T, D, S)
    normproducts = np.linalg.norm(position_sat, axis=2)[:, np.newaxis, :] * distances

    # Compute angles: (T, D, S)
    angles = np.arccos(np.clip(dotproducts / normproducts, -1.0, 1.0)) * (180 / np.pi)

    # Find debris in collision range: (T, num_collisions)
    debris_in_collision = np.unique(np.argwhere(distances < self.collision_range)[:, [0, 1]], axis=0)

    # Find debris in radar FOV: (T, num_detected)
    debris_in_FOV = np.unique(np.argwhere((distances < radar.max_range) & (angles < radar.FOV))[:, [0, 1]], axis=0)

    return debris_in_FOV, debris_in_collision

def simulation_loop_array(self, debris_orbits, total_debris, constellation:Constellation, radar:Radar):
    # not implemented, quite useless
    # times = np.arange(self.starttime.to_value(u.s), self.simtime.to_value(u.s), self.timestep.to_value(u.s))

    # positions_deb = np.zeros((len(times), total_debris, 3))
    # print(positions_deb.shape)
    # positions_sat = np.zeros((len(times), constellation.total_sats, 3))
    # for i,t in enumerate(times):
    #     positions_deb[i] = propagate_all_orbits(debris_orbits, positions_deb[i], t*u.s)
    #     positions_sat[i] = propagate_all_orbits(constellation.sat_orbits, positions_sat[i], t*u.s)

    # positions_deb = propagate_all_orbits_gpu(positions_deb_0, deb_v, times, debris_orbits)

    # # Run detection for all timesteps at once
    # debris_in_FOV, debris_in_collision = self.detect_debris_array(positions_sat, positions_deb, radar)

    # # Flatten results across timesteps
    # self.det_deb = np.unique(debris_in_FOV[:, 1]).astype(int)  # Unique debris indices detected
    # self.col_deb = np.unique(debris_in_collision[:, 1]).astype(int)  # Unique debris indices in collision

    # # Store positions and detection times
    # self.det_pos = np.zeros((total_debris, 3))
    # self.det_time = np.zeros(total_debris)
    # for i, debris_idx in enumerate(self.det_deb):
    #     first_detection = debris_in_FOV[debris_in_FOV[:, 1] == debris_idx, 0].min()  # First timestep it was detected
    #     self.det_pos[debris_idx] = positions_deb[first_detection, debris_idx]
    #     self.det_time[debris_idx] = times[first_detection]
    pass
