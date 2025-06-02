
import cv2 as cv
import numpy as np
import random
import matplotlib.pyplot as plt



def plot_distortion_correct_rings(K, dist, ax, title, width=5496, height=3672, step=25, num_rings=5):
    # Create grid of points
    x, y = np.meshgrid(np.arange(0, width, step), np.arange(0, height, step))
    grid_points = np.stack([x, y], axis=-1).reshape(-1, 1, 2).astype(np.float32)

    # Undistort the points
    distorted_points = cv.undistortPoints(grid_points, K, dist, P=K)

    # Flatten for plotting
    original_points = grid_points.reshape(-1, 2)
    distorted_points = distorted_points.reshape(-1, 2)

    # Compute displacement vectors and magnitudes
    vectors = distorted_points - original_points
    magnitudes = np.linalg.norm(vectors, axis=1)

    # Normalize magnitudes for coloring
    norm = plt.Normalize(vmin=0, vmax=np.max(magnitudes))
    colors = plt.cm.jet(norm(magnitudes))

    # Plot distortion vectors with color-coded magnitude
    ax.quiver(original_points[:, 0], original_points[:, 1],
              vectors[:, 0], vectors[:, 1],
              magnitudes, cmap='jet', scale=1, angles='xy', scale_units='xy', width=(5e-7 * width))

    # Principal point and image center
    cx, cy = K[0, 2], K[1, 2]
    image_center = (width / 2, height / 2)

    # Plot cross at principal point (red) and star at image center (green)
    ax.plot(cx, cy, marker='+', markersize=12, color='red', label='Principal Point')
    ax.plot(image_center[0], image_center[1], marker='*', markersize=12, color='green', label='Image Center')

    # Generate rings based on fixed distortion magnitudes
    max_magnitude = np.max(magnitudes)
    ring_magnitudes = np.linspace(0, max_magnitude, num_rings + 1)[1:]  # Exclude 0

    # Create a grid for interpolation
    grid_x, grid_y = np.meshgrid(np.arange(0, width, step), np.arange(0, height, step))
    magnitude_grid = np.zeros_like(grid_x, dtype=np.float32)

    # Interpolate magnitudes onto the grid
    for i, (x, y) in enumerate(original_points):
        grid_idx_x = min(int(x // step), magnitude_grid.shape[1] - 1)
        grid_idx_y = min(int(y // step), magnitude_grid.shape[0] - 1)
        magnitude_grid[grid_idx_y, grid_idx_x] = max(magnitude_grid[grid_idx_y, grid_idx_x], magnitudes[i])

    # Generate contours for each ring magnitude
    for mag in ring_magnitudes:
        contours = ax.contour(grid_x, grid_y, magnitude_grid, levels=[mag], colors='black', linestyles='--', linewidths=1)
        for contour in contours.collections:
            for path in contour.get_paths():
                # Annotate the magnitude next to the contour line
                vertices = path.vertices
                if len(vertices) > 0:  # Ensure there are vertices to process
                    # Find a point on the contour to place the label
                    label_x, label_y = vertices[len(vertices) // 2]  # Use the midpoint of the contour
                    ax.text(label_x, label_y, f'{mag:.2f}', fontsize=8, color='black', ha='center', va='center',
                            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    # Formatting
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    ax.grid(True)




if __name__ == "__main__":

    try:
        K1 = np.load('mtx1.npy')
        K2 = np.load('mtx2.npy')
        dist1 = np.load('dist1.npy')
        dist2 = np.load('dist2.npy')
    except: 
        print("Failed to load documented camera parameters. Using computed values instead.")
        exit()

    fig, ax = plt.subplots()
    camera_matrix = np.array([
    [6.5746697944293521e+002,   0.0, 325.1],
    [  0.0, 6.5746697944293521e+002, 249.7],
    [  0.0,   0.0,   1.0]
    ])
    dist_coeffs = np.array([ -4.1802327176423804e-001, 5.0715244063187526e-001, 0.0, 0.0, -5.7843597214487474e-001]) 
    plot_distortion_correct_rings(camera_matrix, dist_coeffs, ax,width=680,height=480,step=5, title="Camera 1 Distortion")
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    plot_distortion_correct_rings(K1, dist1, axes[0], "Left Lens Distortion")
    plot_distortion_correct_rings(K2, dist2, axes[1], "Right Lens Distortion")
    plt.tight_layout()
    plt.show()