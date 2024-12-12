import numpy as np
import plotly.graph_objects as go


# Function to calculate orbital positions
def orbital_positions(semi_major_axis, eccentricity, inclination, raan, arg_perigee, num_points=500):
    theta = np.linspace(0, 2 * np.pi, num_points)
    r = (semi_major_axis * (1 - eccentricity ** 2)) / (1 + eccentricity * np.cos(theta))

    # Orbital position in perifocal coordinate system
    x_perifocal = r * np.cos(theta)
    y_perifocal = r * np.sin(theta)
    z_perifocal = np.zeros_like(theta)

    # Rotation matrices
    cos_o, sin_o = np.cos(raan), np.sin(raan)
    cos_w, sin_w = np.cos(arg_perigee), np.sin(arg_perigee)
    cos_i, sin_i = np.cos(inclination), np.sin(inclination)

    R_z_raan = np.array([[cos_o, -sin_o, 0], [sin_o, cos_o, 0], [0, 0, 1]])
    R_x_incl = np.array([[1, 0, 0], [0, cos_i, -sin_i], [0, sin_i, cos_i]])
    R_z_arg_perigee = np.array([[cos_w, -sin_w, 0], [sin_w, cos_w, 0], [0, 0, 1]])

    rotation_matrix = R_z_raan @ R_x_incl @ R_z_arg_perigee

    positions = np.dot(rotation_matrix, np.array([x_perifocal, y_perifocal, z_perifocal]))

    return positions


# Define random orbital parameters for three orbits
orbits = []
satellites = []
for i in range(3):
    semi_major_axis = 7000 #+ i * 2500  # Different altitudes
    eccentricity = 0.01 #+ i * 0.2  # Slightly different eccentricities
    inclination = np.radians(45)# + i * 10)  # Different inclinations
    raan = np.radians(30)# + i * 20)  # Different RAANs
    arg_perigee = np.radians(60 + i * 30)  # Different arguments of perigee

    # Get orbital positions
    orbit = orbital_positions(semi_major_axis, eccentricity, inclination, raan, arg_perigee)
    orbits.append(orbit)

    # Random initial position for the satellite
    satellite_position_index = np.random.randint(orbit.shape[1])
    satellite_position = orbit[:, satellite_position_index]
    satellites.append(satellite_position)

# Create interactive 3D plot using Plotly
fig = go.Figure()

# Plot the Earth
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = 6371 * np.outer(np.cos(u), np.sin(v))
y = 6371 * np.outer(np.sin(u), np.sin(v))
z = 6371 * np.outer(np.ones(np.size(u)), np.cos(v))
fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale='Blues', opacity=0.6, name='Earth'))

# Plot each orbit and its satellite
colors = ['red', 'green', 'orange']
max_distance = 6371
for i, (orbit, satellite_position) in enumerate(zip(orbits, satellites)):
    fig.add_trace(go.Scatter3d(x=orbit[0], y=orbit[1], z=orbit[2], mode='lines', line=dict(color=colors[i]),
                               name=f'Orbit {i + 1}'))
    fig.add_trace(
        go.Scatter3d(x=[satellite_position[0]], y=[satellite_position[1]], z=[satellite_position[2]], mode='markers',
                     marker=dict(color=colors[i], size=5), name=f'Satellite {i + 1}'))
    max_distance = max(max_distance, np.max(np.abs(orbit)))

# Set fixed axis limits with true proportions for Earth and orbits
max_distance += 1000  # Add some padding
fig.update_layout(scene=dict(
    xaxis=dict(range=[-max_distance, max_distance], title='X (km)'),
    yaxis=dict(range=[-max_distance, max_distance], title='Y (km)'),
    zaxis=dict(range=[-max_distance, max_distance], title='Z (km)'),
    aspectmode='manual',
    aspectratio=dict(x=1, y=1, z=1)
),
    title='Earth and Satellite Orbits',
    margin=dict(l=0, r=0, b=0, t=50))

fig.show()
