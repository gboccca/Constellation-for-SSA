import numpy as np
import matplotlib.pyplot as plt

def generate_hexagonal_array(side_length):
    points = []
    radius = side_length
    for q in range(-radius, radius + 1):
        r1 = max(-radius, -q - radius)
        r2 = min(radius, -q + radius)
        for r in range(r1, r2 + 1):
            x = side_length * (3/2 * q)
            y = side_length * (np.sqrt(3) * (r + q/2))
            points.append((x, y))
    return points

def plot_hexagonal_array(points):
    x_coords, y_coords = zip(*points)
    plt.figure(figsize=(8, 8))
    plt.scatter(x_coords, y_coords)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Hexagonal Array')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.show()









# Example usage
side_length = 10  # Side length of the hexagon
hex_array = generate_hexagonal_array(side_length)
plot_hexagonal_array(hex_array)
print(f"Total number of elements in the hexagonal array: {len(hex_array)}")

