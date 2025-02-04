from simulation import Simulation, Constellation, Radar, main, generate_debris
import numpy as np
from astropy import units as u
import gc
from gaussian import doublegaussian_fit, satellite_dist
import csv
import time
from matplotlib import pyplot as plt



# generate constellation parameters




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

    default_num_planes = 13
    default_i0 = 0    #deg
    default_ispacing = 0    #deg
    default_eccentricity = 0    #unitless
    default_raan_spacing = 0    #deg
    default_raan_0 = 0    #deg

    w1, mu1, s1, w2, mu2, s2 = doublegaussian_fit()


    default_values = {
        'i_spacing' : default_ispacing,
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
                    'raan_spacing': [0,180/default_num_planes],
                    'i_spacing': [0,180/default_num_planes],
                    'eccentricity': [0,0.3],
                    'w1': [w1-5, w1+5],
                    'mu1': [mu1-100, mu1+100],
                    's1': [s1-50, s1+50],
                    'w2': [w2-5, w2+5],
                    'mu2': [mu2-100, mu2+100],
                    's2': [s2-10, s2+10]
        }
    
    return default_values, default_bounds


def pso(n_particles, n_iterations, deb_number, n_orbits, n_sats, hmin, opt_pars, inertia = 0.5, cognitive = 0.5, social = 0.5, gpu = True, use_new_dataset=False, **kwargs):
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
        gpu: whether to use the GPU for the simulation
        use_new_dataset: whether to generate a new debris dataset
        kwargs:  values for raan_spacing, inclination, eccentricity, w1, mu1, s1, w2, mu2, s2. if not specified, default values are used from pso_default_parameters()

    Returns:
        best_const: constellation object with optimized parameters
        best_eff: efficiency of the best constellation
        pso_history: nparray of dimension n_particles x 2 containing every constellation simulated and its efficiency, at each iteration.


    """
    # Debris
    if not use_new_dataset:
        deb_orbits, deb_diameters = generate_debris(deb_number, use_new_dataset)

    # default values and bounds
    default_values, default_bounds = pso_default_parameters()

    # when a value is not specified in the kwargs, the default value is used
    par_names = ['raan_spacing', 'i_spacing', 'eccentricity', 'w1', 'mu1', 's1', 'w2', 'mu2', 's2']
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
        if use_new_dataset:
            deb_orbits, deb_diameters = generate_debris(deb_number, use_new_dataset)

        positions_dict = {opt_pars[n]:positions[n,p] for n in range(dim)}
        satdist, altitudes = call_function_with_kwargs(satellite_dist, parameters, positions_dict, num_orbits=n_orbits, num_sats=n_sats, hmin = hmin)
        const = call_function_with_kwargs(Constellation, parameters, positions_dict, sat_distribution=satdist, altitudes=altitudes)
        consteff = main(sim, const, deb_orbits, deb_diameters, radar, plot=False, gpu=gpu, simid=f'0.{p}')

        pbest[:,p] = positions[:,p]
        pbest_eff[0,p] = consteff

        pso_history[p,0,:] = [consteff, const]
        gc.collect()


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
            satdist, altitudes = call_function_with_kwargs(satellite_dist, parameters, positions_dict, num_orbits=n_orbits, num_sats=n_sats, hmin=hmin)
            const = call_function_with_kwargs(Constellation, parameters, positions_dict, sat_distribution=satdist, altitudes=altitudes)
            # print(const)
            consteff = main(sim, const, deb_orbits, deb_diameters, radar, plot=False, simid=f'{i+1}.{p}')
            
            # update personal best - there is a broadcasting bug here
            if consteff > pbest_eff[0,p]:
                pbest[:,p] = positions[:,p]
                pbest_eff[0,p] = consteff

            # update history
            pso_history[p,i,:] = [consteff, const]
            gc.collect()

        # update global best position and value
        if np.max(pbest_eff) > gbest_eff:
            gbest_eff = np.max(pbest_eff)
            gbest[:,0] = pbest[:,np.argmax(pbest_eff)]

        # update history
        gbest_history[i] = gbest_eff

        # check convergence
        if i > 10:
            if np.std(gbest_history[i-10:i]) < 1e-3:
                print(f'Convergence reached after {i} iterations')
                break

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

    if len(opt_pars) == 6:
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

        x = np.array([pso_history[j,i,1].asdict[opt_pars[3]] for j in range(n_particles) for i in range(n_iterations)])
        y = np.array([pso_history[j,i,1].asdict[opt_pars[4]] for j in range(n_particles) for i in range(n_iterations)])
        z = np.array([pso_history[j,i,1].asdict[opt_pars[5]] for j in range(n_particles) for i in range(n_iterations)])
        c = np.array([pso_history[j,i,0] for j in range(n_particles) for i in range(n_iterations)])
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(x, y, z, c=c, cmap='viridis')

        ax.set_xlabel(opt_pars[3])
        ax.set_ylabel(opt_pars[4])
        ax.set_zlabel(opt_pars[5])
        ax.set_title(f'Efficiency vs {opt_pars[3]}, {opt_pars[4]} and {opt_pars[5]}')

        cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, aspect=20, pad=0.1)
        cbar.set_label('Efficiency')

        plt.savefig(f'Optimization/pso_plots/{run_name}/eta_vs_{opt_pars[3]}_{opt_pars[4]}_{opt_pars[5]}.png')
        plt.close()

    if len(opt_pars) == 8:

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


        x = np.array([pso_history[j,i,1].asdict[opt_pars[2]] for j in range(n_particles) for i in range(n_iterations)])
        y = np.array([pso_history[j,i,1].asdict[opt_pars[3]] for j in range(n_particles) for i in range(n_iterations)])
        z = np.array([pso_history[j,i,1].asdict[opt_pars[4]] for j in range(n_particles) for i in range(n_iterations)])
        c = np.array([pso_history[j,i,0] for j in range(n_particles) for i in range(n_iterations)])
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(x, y, z, c=c, cmap='viridis')

        ax.set_xlabel(opt_pars[2])
        ax.set_ylabel(opt_pars[3])
        ax.set_zlabel(opt_pars[4])
        ax.set_title(f'Efficiency vs {opt_pars[2]}, {opt_pars[3]} and {opt_pars[4]}')

        cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, aspect=20, pad=0.1)
        cbar.set_label('Efficiency')

        plt.savefig(f'Optimization/pso_plots/{run_name}/eta_vs_{opt_pars[2]}_{opt_pars[3]}_{opt_pars[4]}.png')
        plt.close()

        x = np.array([pso_history[j,i,1].asdict[opt_pars[5]] for j in range(n_particles) for i in range(n_iterations)])
        y = np.array([pso_history[j,i,1].asdict[opt_pars[6]] for j in range(n_particles) for i in range(n_iterations)])
        z = np.array([pso_history[j,i,1].asdict[opt_pars[7]] for j in range(n_particles) for i in range(n_iterations)])
        c = np.array([pso_history[j,i,0] for j in range(n_particles) for i in range(n_iterations)])
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(x, y, z, c=c, cmap='viridis')

        ax.set_xlabel(opt_pars[5])
        ax.set_ylabel(opt_pars[6])
        ax.set_zlabel(opt_pars[7])
        ax.set_title(f'Efficiency vs {opt_pars[5]}, {opt_pars[6]} and {opt_pars[7]}')

        cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, aspect=20, pad=0.1)
        cbar.set_label('Efficiency')

        plt.savefig(f'Optimization/pso_plots/{run_name}/eta_vs_{opt_pars[5]}_{opt_pars[6]}_{opt_pars[7]}.png')
        plt.close()



    if len(opt_pars) == 9:

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

        x = np.array([pso_history[j,i,1].asdict[opt_pars[3]] for j in range(n_particles) for i in range(n_iterations)])
        y = np.array([pso_history[j,i,1].asdict[opt_pars[4]] for j in range(n_particles) for i in range(n_iterations)])
        z = np.array([pso_history[j,i,1].asdict[opt_pars[5]] for j in range(n_particles) for i in range(n_iterations)])
        c = np.array([pso_history[j,i,0] for j in range(n_particles) for i in range(n_iterations)])
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(x, y, z, c=c, cmap='viridis')

        ax.set_xlabel(opt_pars[3])
        ax.set_ylabel(opt_pars[4])
        ax.set_zlabel(opt_pars[5])
        ax.set_title(f'Efficiency vs {opt_pars[3]}, {opt_pars[4]} and {opt_pars[5]}')

        cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, aspect=20, pad=0.1)
        cbar.set_label('Efficiency')

        plt.savefig(f'Optimization/pso_plots/{run_name}/eta_vs_{opt_pars[3]}_{opt_pars[4]}_{opt_pars[5]}.png')
        plt.close()

        x = np.array([pso_history[j,i,1].asdict[opt_pars[6]] for j in range(n_particles) for i in range(n_iterations)])
        y = np.array([pso_history[j,i,1].asdict[opt_pars[7]] for j in range(n_particles) for i in range(n_iterations)])
        z = np.array([pso_history[j,i,1].asdict[opt_pars[8]] for j in range(n_particles) for i in range(n_iterations)])
        c = np.array([pso_history[j,i,0] for j in range(n_particles) for i in range(n_iterations)])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(x, y, z, c=c, cmap='viridis')

        ax.set_xlabel(opt_pars[6])
        ax.set_ylabel(opt_pars[7])
        ax.set_zlabel(opt_pars[8])
        ax.set_title(f'Efficiency vs {opt_pars[6]}, {opt_pars[7]} and {opt_pars[8]}')

        cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, aspect=20, pad=0.1)
        cbar.set_label('Efficiency')

        plt.savefig(f'Optimization/pso_plots/{run_name}/eta_vs_{opt_pars[6]}_{opt_pars[7]}_{opt_pars[8]}.png')
        plt.close()


    print(f'Plots saved successfully to "Optimization/pso_plots/"')
        

if __name__ == '__main__':

    # PSO Hyperparameters
    inertia = 0.5   
    cognitive = 0.5
    social = 0.5
    n_particles = 15
    n_iterations = 100
    run_name = input('Enter the name of the run: ')

    # Simulation parameters (constant)
    time_of_flight = 1 * u.hour
    start_time = 0*u.s      # Start time of the simulation
    sim = Simulation (time_of_flight, start_time, max_timestep=5*u.s)
    radar = Radar()
    deb_number = 1000
    use_new_dataset = True
    gpu = True

    # Constant constellation parameters
    n_orbits = 13
    n_sats = 4*12
    hmin = 450

    # PSO parameters
    #opt_pars = ['raan_spacing', 'inclination', 'eccentricity']
    opt_pars = ['raan_spacing', 'i_spacing', 'w1', 'mu1', 's1', 'w2', 'mu2', 's2']

    # Run PSO and save results
    pso_start_time = time.time()
    gbest, gbest_eff, pso_history, gbest_history = pso(n_particles, n_iterations, deb_number, n_orbits, n_sats, hmin, opt_pars, inertia, cognitive, social, gpu=True, use_new_dataset=use_new_dataset)
    pso_end_time = time.time()
    print(f'PSO took {pso_end_time - pso_start_time} seconds to run {n_iterations} iterations with {n_particles} particles')
    save_pso_results(gbest, gbest_eff, pso_history, gbest_history, n_particles, n_iterations, opt_pars, run_name)