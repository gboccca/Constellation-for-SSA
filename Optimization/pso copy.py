from simulation import Simulation, Constellation, Radar, main, generate_debris
import numpy as np
from astropy import units as u
import gc, warnings
from scipy.stats import norm

# PSO Hyperparameters
inertia = 0.5
cognitive = 0.5
social = 0.5

# Simulation parameters (constant)
time_of_flight = 0.1 * u.hour
start_time = 0*u.s      # Start time of the simulation
sim = Simulation (time_of_flight, start_time)
radar = Radar()
deb_number = 100
use_new_dataset = False
deb_orbits, deb_diameters = generate_debris(deb_number, use_new_dataset)

# generate constellation parameters


def satellite_dist(**kwargs):
    """
    Generate the satellite distribution from gasussian mixture model
    kwargs (all required):

        w1, mu1, sigma1, w2, mu2, sigma2, num_obrits, num_sats
    
    returns:
        discrete_dist, altitudes
    """

    required_kwargs = ['w1', 'mu1', 's1', 'w2', 'mu2', 's2', 'num_obrits', 'num_sats']


    # unpack parameters

    for key in required_kwargs:
        if key not in kwargs:
            raise ValueError(f'Missing required parameter {key}')

    superfluous_kwargs = []
    for key in kwargs:
        if key not in required_kwargs:
            superfluous_kwargs.append(key)
    
    if superfluous_kwargs:
        warnings.warn(f'Superfluous parameters: {superfluous_kwargs}')
        # remove superfluous keys from kwargs
        for key in superfluous_kwargs:
            kwargs.pop(key)
            
      
    num_obrits = kwargs['num_obrits']
    num_sats = kwargs['num_sats']
    w1 = kwargs['w1']
    mu1 = kwargs['mu1']
    sigma1 = kwargs['s1']
    w2 = kwargs['w2']
    mu2 = kwargs['mu2']
    sigma2 = kwargs['s2']


    # generate discrete positions along distribution
    altitudes = np.linspace(450, 1500, num_obrits)
    # generate distribution based on mean and std_deviation
    gmm_dist = (w1 * norm.pdf(altitudes, mu1, sigma1) +
                w2 * norm.pdf(altitudes, mu2, sigma2))  # Sum of two Gaussians

    # generate discrete distribution
    normalized_dist = gmm_dist / np.sum(gmm_dist)
    scaled_dist = normalized_dist * num_sats
    discrete_dist = np.round(scaled_dist).astype(int)

    # distribute remaining satellites in orbits closest to the distribution peak
    res = num_sats - np.sum(discrete_dist)
    peak_index = np.argmax(scaled_dist)
    sorted_indices = sorted(range(num_obrits), key=lambda x: abs(x - peak_index))
    for i in range(res):
        discrete_dist[sorted_indices[i % num_obrits]] += 1

    return discrete_dist, altitudes


def call_function_with_kwargs(func, a:dict, b:dict, **kwargs):
    """ call function 'func' with all its kwargs. kwargs are taken from a when they are in a, from b when they are in b. cannot be in both lists at the same time
    
    Args:
        func (function): function to call
        a (list): list of kwargs
        b (list): list of kwargs
        kwargs (dict): additional kwargs to pass to func
    returns: 
        outputs of func
    """
    
    kwargs = {key:kwargs[key] for key in kwargs}

    for key in list(a.keys())+list(b.keys()):
        if key in a:
            kwargs[key] = a[key]
        elif key in b:
            kwargs[key] = b[key]
        else:
            raise ValueError('Key not in a or b, check your lists')
    return func(**kwargs)






# intialize particles
default_num_planes = 12
default_min_alt = 450       #km
default_alt_spacing = 75    #km
default_altitudes = np.array([default_min_alt+i*default_alt_spacing for i in range(default_num_planes)])
default_inclination = 90    #deg
default_eccentricity = 0    #unitless
default_raan_spacing = 360/default_num_planes    #deg
default_sats_per_orbit = 40
default_distribution = np.array([default_sats_per_orbit for _ in range(default_num_planes)])
default_total_sats = np.sum(default_distribution)


w1=8.001299144911616
mu1=811.894743371767
s1=94.25661870684115
w2=13.976739547943197
mu2=1199.0493974102565
s2=463.32287670993446




default_values = {
    'inclination': default_inclination,
    'eccentricity': default_eccentricity,
    'raan_spacing': default_raan_spacing,
    'w1': w1,
    'mu1': mu1,
    's1': s1,
    'w2': w2,
    'mu2': mu2,
    's2': s2
}

default_bounds = {
                'raan_spacing': [0,180],
                'inclination': [0,90],
                'eccentricity': [0,0.9],
                'w1': [w1-5, w1+5],
                'mu1': [mu1-100, mu1+100],
                's1': [s1-50, s1+50],
                'w2': [w2-5, w2+5],
                'mu2': [mu2-100, mu2+100],
                's2': [s2-10, s2+10]
    }


def pso(n_particles, n_iterations, n_orbits, n_sats, opt_pars, **kwargs):
    """
    Particle Swarm Optimization on the selected constellation parameters. 
    Pass all the starting/default parameters as kwargs, and specify which you want to optimize with "opt_par=[]"
    
    opt_par = SET of strings containing elements from [gaussian, raan_spacing, inclination, eccentricity]

    """
    global default_values, default_bounds

    # add default values if not specified
    par_names = ['raan_spacing', 'inclination', 'eccentricity', 'w1', 'mu1', 's1', 'w2', 'mu2', 's2']
    for par in par_names:
        if par not in kwargs:
            kwargs[par] = default_values[par]
    
    
    parameters = {key:kwargs[key] for key in par_names}             # parameters that are not optimized
    for key in opt_pars:
        parameters.pop(key,None)

    bounds = {key:default_bounds[key] for key in opt_pars}          # parameters that are optimized
    dim = len(opt_pars)
    #print(f"optimized parameters: {opt_pars}")
    #print(f"non opt parameters: {parameters}")

    position = np.array([np.random.uniform(low=bounds[key][0], high=bounds[key][1], size=(n_particles)) for key in opt_pars])    # array of the parameters being optimized, one row per particle

    #print(f'starting positions: {position}')
    velocities = np.zeros((len(opt_pars), n_particles))
    pbest = position.copy()
    pbest_eff = np.zeros((n_particles,dim))

    # remember: do a first iteration to get the initial gbest
    gbest = np.zeros((dim,1))
    gbest_eff = 0

    # pso loop

    for i in range(n_iterations):
        print('Iteration:', i)

        # update velocity and position of all particles

        print(f'positions: {position}')
        print(f'pbest: {pbest}')
        print(f'gbest: {gbest}')
        cognitive_component = cognitive*np.random.rand()*(pbest-position)
        social_component = social*np.random.rand()*(position - gbest)

        print(f'velocities: {velocities}')
        print(f'cognitive: {cognitive_component}')
        print(f'social: {social_component}')
        velocities = velocities*inertia + cognitive_component + social_component
        position += velocities

        for p in range(n_particles):
            position_dict = {opt_pars[n]:position[n,p] for n in range(dim)}
            #print(position_dict)
            satdist, altitudes = call_function_with_kwargs(satellite_dist, parameters, position_dict, num_obrits=n_orbits, num_sats=n_sats)
            const = call_function_with_kwargs(Constellation, parameters, position_dict, sat_distribution=satdist, altitudes=altitudes)
            print(const)
            consteff = main(sim, const, deb_orbits, deb_diameters, radar)
            
            # update personal best
            if consteff > pbest_eff[p]:
                pbest[p] = position[p]
                pbest_eff[p] = consteff
            
            # update global best
            if consteff > gbest_eff:
                gbest = position[p]
                gbest_eff = consteff

        print(f'Iteration {i} completed. Best value: {gbest_eff}')


    print(parameters)
    print(bounds)
    

pso(5, 5, 14, 14*40, ['raan_spacing', 'inclination', 'eccentricity'])