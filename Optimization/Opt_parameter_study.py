from simulation import Simulation, Constellation, Radar, main, generate_debris
import numpy as np
from astropy import units as u
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import gc



radar = Radar()

altitudes = np.linspace(450 * u.km, 1500 * u.km, 13)
sat_distrib = np.ones(13) * 4
raan_spacing = (360/4) * u.deg
constellation = Constellation(altitudes, sat_distrib, raan_spacing)
deb_orbits, diam = generate_debris(500, use_new_dataset=False)

timestep_list = [20, 10, 5, 2.5, 1.225, 0.6125, 0.3] * u.s
flight_time = 1 * u.h


def study_timestep(radar, const, flighttime, timestep_list, deb_orbits):

    con_eff = np.zeros((len(timestep_list), 1))
    for i, timestep in enumerate(timestep_list):
        sim = Simulation(flighttime, 0*u.s, max_timestep=timestep)
        eff = main(sim, const, deb_orbits, diam, rad=radar)
        con_eff[i] = eff

    return con_eff


timestep_study = study_timestep(radar, constellation, flight_time, timestep_list, deb_orbits)

plt.scatter(timestep_list, timestep_study)
plt.xlabel("Simulation timestep")
plt.ylabel("Computed constellation efficiency")
plt.grid(True)
plt.show()


