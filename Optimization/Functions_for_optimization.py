import numpy as np
import matplotlib.pyplot as plt

## FUNCTIONS 






def particle_swarm_optimization(
    objective:function, bounds, num_particles, num_iterations, debris_orbits, total_particles, fixed_total_sats
):
    """
    PSO for optimizing orbital parameters with variable altitudes and satellite distribution.
    """
    num_planes = 12  # Start with 12 orbital planes
    initial_altitudes = np.linspace(bounds[0, 0], bounds[0, 1], num_planes)  # Evenly spaced altitudes
    initial_satellites = np.full(num_planes, fixed_total_sats // num_planes)  # Equal satellites per plane

    # Initialize particles
    particles = []
    for _ in range(num_particles):
        altitudes = initial_altitudes + np.random.uniform(-10, 10, num_planes)  # Small initial variation
        satellite_distribution = np.random.dirichlet(np.ones(num_planes)) * fixed_total_sats
        satellite_distribution = np.round(satellite_distribution).astype(int)
        satellite_distribution[-1] += fixed_total_sats - np.sum(satellite_distribution)  # Fix rounding errors
        particles.append((altitudes, satellite_distribution))

    velocities = [np.zeros_like(particle[0]) for particle in particles]  # Zero velocities for altitudes
    personal_best_positions = particles.copy()
    personal_best_scores = np.full(num_particles, -np.inf)
    global_best_position = None
    global_best_score = -np.inf

    # Real-time plotting setup
    plt.ion()
    fig, ax = plt.subplots()

    for iteration in range(num_iterations):
        scores = []

        for i, (altitudes, satellite_distribution) in enumerate(particles):
            # Evaluate objective function
            params = (altitudes, satellite_distribution)
            score = objective(params, debris_orbits, total_particles)
            scores.append(score)

            # Update personal best
            if score > personal_best_scores[i]:
                personal_best_scores[i] = score
                personal_best_positions[i] = (altitudes.copy(), satellite_distribution.copy())

            # Update global best
            if score > global_best_score:
                global_best_score = score
                global_best_position = (altitudes.copy(), satellite_distribution.copy())

        # Update velocities and positions
        inertia = 0.5
        cognitive = 2
        social = 2

        for j, (altitudes, satellite_distribution) in enumerate(particles):
            r1, r2 = np.random.rand(2)

            # Update velocities for altitudes
            velocities[j,0] = (
                inertia * velocities[j,0]
                + cognitive * r1 * (personal_best_positions[j][0] - altitudes)
                + social * r2 * (global_best_position[0] - altitudes)
            )
            altitudes += velocities[j,0]

            velocities[j,1] = (
                inertia * velocities[j,1]
                + cognitive * r1 * (personal_best_positions[j][1] - satellite_distribution)
                + social * r2 * (global_best_position[1] - satellite_distribution)
            )

            

            # Adjust satellite distribution (stochastic perturbation)
            satellite_distribution += velocities[j,1]
            # Ensure altitudes stay within bounds
            altitudes = np.clip(altitudes, bounds[0, 0], bounds[0, 1])
            satellite_distribution = np.clip(satellite_distribution, 0, fixed_total_sats)

            satellite_distribution[-1] += fixed_total_sats - np.sum(satellite_distribution)  # Fix total count

            particles[j] = (altitudes, satellite_distribution)

        # Real-time plot
        ax.clear()
        ax.set_xlim(bounds[0, 0], bounds[0, 1])  # Altitude range
        ax.set_ylim(0, fixed_total_sats)  # Satellites per altitude range
        ax.set_title(f"Iteration {iteration + 1}")
        ax.set_xlabel("Altitude (km)")
        ax.set_ylabel("Number of Satellites")
        print(f"global best score: {global_best_score}")

        # Plot current particles
        for altitudes, satellite_distribution in particles:
            ax.scatter(altitudes, satellite_distribution, color="blue", alpha=0.5)

        plt.pause(0.1)

    plt.ioff()
    plt.show()

    return global_best_position, global_best_score




## CALLING THE FUNCTION
bounds = np.array([
    [450, 1350],    # Altitude range (km)
    [5, 15],       # Number of orbital planes
])
global_best_position = np.zeros(len(bounds))

num_particles = 3
num_iterations = 25

best_params, best_score = particle_swarm_optimization(constellation_efficiency, bounds, num_particles, num_iterations, debris_orbits, total_particles, fixed_total_sats=48)

print("Best Parameters:", best_params)
print("Best Detection Efficiency:", best_score)
