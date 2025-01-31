from simulation import Simulation, Constellation, Radar, main, generate_debris
import numpy as np
from astropy import units as u
import gc, warnings
from scipy.stats import norm
from gaussian import doublegaussian_fit
import csv
import time
from matplotlib import pyplot as plt
import matplotlib.cm as cm



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
    
    # this will be triggered every time if the function is called correctly
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


def pso_default_parameters():

    default_num_planes = 12
    default_inclination = 90    #deg
    default_eccentricity = 0    #unitless
    default_raan_spacing = 360/default_num_planes    #deg

    w1, mu1, s1, w2, mu2, s2 = doublegaussian_fit()


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
    
    return default_values, default_bounds


def pso(n_particles, n_iterations, n_orbits, n_sats, opt_pars, inertia = 0.5, cognitive = 0.5, social = 0.5, **kwargs):
    """
    Particle Swarm Optimization on the selected constellation parameters. 
    Pass all the starting/default parameters as kwargs, and specify which you want to optimize with "opt_par=[]"
    
    Args:
        n_particles: number of particles in the swarm
        n_iterations: number of iterations
        n_orbits: number of orbits in the constellation
        n_sats: number of satellites per orbit
        opt_pars = SET of strings containing elements from [raan_spacing, inclination, eccentricity, w1, mu1, s1, w2, mu2, s2]
        inertia: inertia parameter
        cognitive: cognitive parameter
        social: social parameter
        kwargs:  values for raan_spacing, inclination, eccentricity, w1, mu1, s1, w2, mu2, s2. if not specified, default values are used from pso_default_parameters()

    Returns:
        best_const: constellation object with optimized parameters
        best_eff: efficiency of the best constellation
        pso_history: nparray of dimension n_particles x 2 containing every constellation simulated and its efficiency, at each iteration.


    """
    default_values, default_bounds = pso_default_parameters()

    # when a value is not specified in the kwargs, the default value is used
    par_names = ['raan_spacing', 'inclination', 'eccentricity', 'w1', 'mu1', 's1', 'w2', 'mu2', 's2']
    for par in par_names:
        if par not in kwargs:
            kwargs[par] = default_values[par]
    
    
    parameters = {key:kwargs[key] for key in par_names}             # parameters that are not optimized
    for key in opt_pars:
        parameters.pop(key,None)

    bounds = {key:default_bounds[key] for key in opt_pars}          # bounds parameters that are optimized
    lower_bound = np.array([bounds[key][0] for key in opt_pars])
    upper_bound = np.array([bounds[key][1] for key in opt_pars])
    # print('Lower bound:', lower_bound)
    # print('Upper bound:', upper_bound)
    dim = len(opt_pars)

    positions = np.array([np.random.uniform(low=bounds[key][0], high=bounds[key][1], size=(n_particles)) for key in opt_pars])    # array of the parameters being optimized, one row per particle
    # initialize random velocities between 0 and 1
    velocities = np.random.rand(dim, n_particles)
    
    # initialize particles best positions and values
    pbest = np.zeros((dim,n_particles))
    pbest_eff = np.zeros((1, n_particles))
    gbest = np.zeros((dim,1))
    gbest_eff = 0

    # initialize history
    pso_history = np.zeros((n_particles, n_iterations, 2), dtype=object) 
    gbest_history = np.zeros((n_iterations))


    # first evaluation of the particles
    print('\nStarting first evaluation (Iteration 0)')
    for p in range(n_particles):
        positions_dict = {opt_pars[n]:positions[n,p] for n in range(dim)}
        satdist, altitudes = call_function_with_kwargs(satellite_dist, parameters, positions_dict, num_obrits=n_orbits, num_sats=n_sats)
        const = call_function_with_kwargs(Constellation, parameters, positions_dict, sat_distribution=satdist, altitudes=altitudes)
        consteff = main(sim, const, deb_orbits, deb_diameters, radar, plot=False, simid=f'0.{p}')

        pbest[:,p] = positions[:,p]
        pbest_eff[0,p] = consteff

        pso_history[p,0,:] = [consteff, const]


    # initialize global best position and value
    gbest_eff = np.max(pbest_eff)
    gbest[:,0] = pbest[:,np.argmax(pbest_eff)]

    # initialize history
    # debugging prints
    # print('Initial positions:'); print(positions)
    # print('Initial velocities:'); print(velocities)
    # print('Initial pbest:'); print(pbest)
    # print('Initial pbest_eff:'); print(pbest_eff)
    # print('Initial gbest:'); print(gbest)
    # print('Initial gbest_eff:'); print(gbest_eff)


    # pso loop
    print('\nStarting PSO loop\n')
    for i in range(n_iterations):
        print('\nIteration:', i+1)
        i_start_time = time.time()

        # update velocity and position of all particles - there should be no problem with particles going out of range
        cognitive_component = cognitive*np.random.rand()*(pbest-positions)
        social_component = social*np.random.rand()*(gbest-positions)
        velocities = velocities*inertia + cognitive_component + social_component
        # print(f'Velocities in iteration {i}, before scaling:'); print(velocities)
        # scale velocities in each dimension so that the maximum velocity change is upper_bound - lower_bound in that dimension
        # this will solve the scaling/excessive velocity problem but will make it such that the fastest particle always oscillates between the bounds
        # update: this dont solve shit, need to figure out the fucking velocities holy fuck
        #update: i added that abs() and now it seems to not run into any problems
        # update: only apply scaling if one velocity exceeds the maximum. done to 

        for d in range(dim):
            if np.max(np.abs(velocities[d])) > upper_bound[d] - lower_bound[d]:
                velocities[d] *= (upper_bound[d] - lower_bound[d]) / np.max(np.abs(velocities[d]))

        # print('Velocities before:'); print(velocities)
        # print('Positions before:'); print(positions)
        
        # print(f'Velocities in iteration {i}, after scaling:'); print(velocities)
        positions += velocities
        # print(f'Positions in iteration {i}:'); print(positions)
        
        # implement reflection on the boundaries and reflect velocity: problem: if a velocity is too high, it is reflected out of the other bound
        for d in range(dim):
            for j in range(n_particles):
                if positions[d, j] < lower_bound[d]:
                    positions[d, j] = 2*lower_bound[d] - positions[d, j]
                    velocities[d, j] = -velocities[d, j]
                elif positions[d, j] > upper_bound[d]:
                    positions[d, j] = 2*upper_bound[d] - positions[d, j]
                    velocities[d, j] = -velocities[d, j]
        # print(f'Positions in iteration {i}, boundary condition applied:'); print(positions)

        # same but with np broadcasting:
        #positions = np.where(positions < lower_bound, 2*lower_bound - positions, positions)
        #positions = np.where(positions > upper_bound, 2*upper_bound - positions, positions)
        #velocities = np.where(positions < lower_bound, -velocities, velocities)
        #velocities = np.where(positions > upper_bound, -velocities, velocities)

        # print('Velocities after:'); print(velocities)
        # print('Positions after:'); print(positions)

        
        for p in range(n_particles):

            positions_dict = {opt_pars[n]:positions[n,p] for n in range(dim)}
            # print('Positions dict:'); print(positions_dict)
            satdist, altitudes = call_function_with_kwargs(satellite_dist, parameters, positions_dict, num_obrits=n_orbits, num_sats=n_sats)
            const = call_function_with_kwargs(Constellation, parameters, positions_dict, sat_distribution=satdist, altitudes=altitudes)
            # print(const)
            consteff = main(sim, const, deb_orbits, deb_diameters, radar, plot=False, simid=f'{i+1}.{p}')
            
            # update personal best - there is a broadcasting bug here
            if consteff > pbest_eff[0,p]:
                pbest[:,p] = positions[:,p]
                pbest_eff[0,p] = consteff

            # update history
            pso_history[p,i,:] = [consteff, const]

        # update global best position and value
        if np.max(pbest_eff) > gbest_eff:
            gbest_eff = np.max(pbest_eff)
            gbest[:,0] = pbest[:,np.argmax(pbest_eff)]

            

        
        gbest_history[i] = gbest_eff
        i_end_time = time.time()

        print(f'Iteration {i+1} completed in {(i_end_time - i_start_time):.2f} seconds . Best value: {gbest_eff}')

    return gbest, gbest_eff, pso_history, gbest_history

def save_pso_results(gbest, gbest_eff, pso_history, gbest_history, n_particles, n_iterations, opt_pars, run_name):
    """
    Save the results of the PSO optimization to a csv file
    """


    filename = f'Optimization/pso_results/{run_name}.csv'

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Best parameters:'])
        writer.writerow(gbest)
        writer.writerow(['Best efficiency:'])
        writer.writerow([gbest_eff])
        writer.writerow(['Number of particles:'])
        writer.writerow([n_particles])
        writer.writerow(['Number of iterations:'])
        writer.writerow([n_iterations])
        writer.writerow(['Optimized parameters:'])
        writer.writerow(opt_pars)
        writer.writerow(['History:'])
        writer.writerows(pso_history)

    print(f"Results saved successfully to '{filename}'")

    # plot efficiency as a function of iteration
    plt.scatter(range(n_iterations), gbest_history)
    plt.xlabel('Iteration')
    plt.ylabel('Efficiency')
    plt.title('Efficiency as a function of iteration')
    plt.savefig(f'Optimization/pso_plots/{run_name}/eta_vs_i.png')
    plt.close()

    # plot efficiency as a function of each opt_par
    # print(pso_history)
    # print(pso_history[1,:,1])
    for par in opt_pars:
        x = np.array([pso_history[j,i,1].asdict[par] for j in range(n_particles) for i in range(n_iterations)])
        y = np.array([pso_history[j,i,0] for j in range(n_particles) for i in range(n_iterations)])
        plt.scatter(x, y)
        plt.xlabel(par)
        plt.ylabel('Efficiency')
        plt.title(f'Efficiency as a function of {par}')
        plt.savefig(f'Optimization/pso_plots/{run_name}/eta_vs_{par}.png')
        plt.close()


    # generate 2d plots for each pair of parameters. efficiency is shown as the color of the points

    if len(opt_pars) == 2:
        x = np.array([pso_history[j, i, 1].asdict[opt_pars[0]] for j in range(n_particles) for i in range(n_iterations)])
        y = np.array([pso_history[j, i, 1].asdict[opt_pars[1]] for j in range(n_particles) for i in range(n_iterations)])
        z = np.array([pso_history[j, i, 0] for j in range(n_particles) for i in range(n_iterations)]) 

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Create scatter plot and store the mappable object
        scatter = ax.scatter(x, y, c=z, cmap='viridis')

        # Add colorbar and associate it with the scatter plot
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, aspect=20, pad=0.1)
        cbar.set_label('Efficiency')

        ax.set_xlabel(opt_pars[0])
        ax.set_ylabel(opt_pars[1])
        ax.set_title(f'Efficiency vs {opt_pars[0]} and {opt_pars[1]}')

        plt.savefig(f'Optimization/pso_plots/{run_name}/eta_vs_{opt_pars[0]}_{opt_pars[1]}.png')
        plt.close()


    # same as before, but now its a 3d plot and the color is still the efficiency
    if len(opt_pars) == 3:
        x = np.array([pso_history[j,i,1].asdict[opt_pars[0]] for j in range(n_particles) for i in range(n_iterations)])
        y = np.array([pso_history[j,i,1].asdict[opt_pars[1]] for j in range(n_particles) for i in range(n_iterations)])
        z = np.array([pso_history[j,i,1].asdict[opt_pars[2]] for j in range(n_particles) for i in range(n_iterations)])
        c = np.array([pso_history[j,i,0] for j in range(n_particles) for i in range(n_iterations)])
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(x, y, z, c=c, cmap='viridis')

        ax.set_xlabel(opt_pars[0])
        ax.set_ylabel(opt_pars[1])
        ax.set_zlabel(opt_pars[2])
        ax.set_title(f'Efficiency vs {opt_pars[0]}, {opt_pars[1]} and {opt_pars[2]}')

        cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, aspect=20, pad=0.1)
        cbar.set_label('Efficiency')

        plt.savefig(f'Optimization/pso_plots/{run_name}/eta_vs_{opt_pars[0]}_{opt_pars[1]}_{opt_pars[2]}.png')
        plt.close()
    
    print(f'Plots saved successfully to "Optimization/pso_plots/"')
        



if __name__ == '__main__':

    # PSO Hyperparameters
    inertia = 0.5   
    cognitive = 0.5
    social = 0.5
    n_particles = 2
    n_iterations = 5
    run_name = input('Enter the name of the run: ')

    # Simulation parameters (constant)
    time_of_flight = 0.01 * u.hour
    start_time = 0*u.s      # Start time of the simulation
    sim = Simulation (time_of_flight, start_time)
    radar = Radar()
    deb_number = 500
    use_new_dataset = False
    deb_orbits, deb_diameters = generate_debris(deb_number, use_new_dataset)

    # Constant constellation parameters
    n_orbits = 12
    n_sats = 40*12

    # PSO parameters
    opt_pars = ['raan_spacing', 'inclination', 'eccentricity']

    # Run PSO and save results
    pso_start_time = time.time()
    gbest, gbest_eff, pso_history, gbest_history = pso(n_particles, n_iterations, n_orbits, n_sats, opt_pars, inertia, cognitive, social)
    pso_end_time = time.time()
    print(f'PSO took {pso_end_time - pso_start_time} seconds to run {n_iterations} iterations with {n_particles} particles')
    save_pso_results(gbest, gbest_eff, pso_history, gbest_history, n_particles, n_iterations, opt_pars, run_name)