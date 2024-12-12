import numpy as np
import plotly.graph_objects as go

# Function to calculate orbital positions
def orbital_positions(semi_major_axis, eccentricity, inclination, raan, arg_perigee, true_anomaly):
    r = (semi_major_axis * (1 - eccentricity**2)) / (1 + eccentricity * np.cos(true_anomaly))
    
    # Orbital position in perifocal coordinate system
    x_perifocal = r * np.cos(true_anomaly)
    y_perifocal = r * np.sin(true_anomaly)
    z_perifocal = np.zeros_like(true_anomaly)

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

# Define orbital parameters for one orbit
semi_major_axis = 7000  # Altitude
eccentricity = 0.01  # Slight eccentricity
inclination = np.radians(45)  # Inclination
raan = np.radians(30)  # RAAN
arg_perigee = np.radians(60)  # Argument of perigee

# Generate time steps for animation
num_frames = 100
true_anomalies = np.linspace(0, 2 * np.pi, num_frames)

# Calculate orbit positions (fixed)
orbit = orbital_positions(semi_major_axis, eccentricity, inclination, raan, arg_perigee, true_anomalies)

# Create interactive 3D plot using Plotly
fig = go.Figure()

# Plot the Earth
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = 6371 * np.outer(np.cos(u), np.sin(v))
y = 6371 * np.outer(np.sin(u), np.sin(v))
z = 6371 * np.outer(np.ones(np.size(u)), np.cos(v))
earth_surface = go.Surface(x=x, y=y, z=z, colorscale='Blues', opacity=0.6, name='Earth')
fig.add_trace(earth_surface)

# Plot the orbit (fixed)
orbit_trace = go.Scatter3d(x=orbit[0], y=orbit[1], z=orbit[2], mode='lines', line=dict(color='red'), name='Orbit')
fig.add_trace(orbit_trace)

# Add satellite animation
frames = []
for i, true_anomaly in enumerate(true_anomalies):
    position = orbital_positions(semi_major_axis, eccentricity, inclination, raan, arg_perigee, np.array([true_anomaly]))
    frames.append(go.Frame(data=[
        go.Scatter3d(x=[position[0][0]], y=[position[1][0]], z=[position[2][0]], mode='markers',
                     marker=dict(color='blue', size=5), name='Satellite')
    ]))

fig.frames = tuple(frames)

# Set fixed axis limits with true proportions for Earth and orbits
max_distance = semi_major_axis + 1000  # Add some padding
fig.update_layout(
    scene=dict(
        xaxis=dict(range=[-max_distance, max_distance], title='X (km)'),
        yaxis=dict(range=[-max_distance, max_distance], title='Y (km)'),
        zaxis=dict(range=[-max_distance, max_distance], title='Z (km)'),
        aspectmode='manual',
        aspectratio=dict(x=1, y=1, z=1)
    ),
    updatemenus=[{
        'buttons': [
            {
                'args': [None, {'frame': {'duration': 50, 'redraw': True}, 'fromcurrent': True}],
                'label': 'Play',
                'method': 'animate'
            },
            {
                'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 0}}],
                'label': 'Pause',
                'method': 'animate'
            }
        ],
        'direction': 'left',
        'pad': {'r': 10, 't': 87},
        'showactive': False,
        'type': 'buttons',
        'x': 0.1,
        'xanchor': 'right',
        'y': 0,
        'yanchor': 'top'
    }],
    title='Earth and Animated Satellite Orbit',
    margin=dict(l=0, r=0, b=0, t=50)
)

fig.show()
