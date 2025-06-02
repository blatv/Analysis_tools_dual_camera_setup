import cv2 as cv
import numpy as np
import random
from scipy.stats import norm
import matplotlib.pyplot as plt
# Camera Properties
pixel_size = 0.0024
width = 5496
height = 3672


### PARAMETERS ###
# Used to scale the 3D points to a more reasonable size.
SCALING_FACTOR = 1014.58 #1.008076139 || Used for scaling the T vector when generating using SIFT
FOCAL_LENGTH = 100 # mm (used for K matrix if not available)

CALIBRATION_FILE_PATH = 'CalibrationData/100mm/' # Location of the .npy files
CALIBRATION_FILE_VERSION = "" # An ID to allow previous saved versions to be reused.
IMAGE_FILE_PATH = 'Datasets/100mm/Kenya_data/' #Overlap/Left/73.jpg || Test_beam_Data/Left/13.jpg
IMAGE_NAME = '208.jpg'
print(f"loading image '{IMAGE_NAME}'")
SHOW_STATISTICS = True

USE_DOCUMENTED_STEREO_PARAMETERS = True # R, T & E
USE_DOCUMENTED_CAMERA_PARAMETERS = False # K1, K2, dist1 & dist2





# Display Feature
show_epi_lines = True
show_distortion_graph = True
# Get the documented camera parameters or compute them if not available.
parameter_gather_camera_failed = False
if USE_DOCUMENTED_CAMERA_PARAMETERS:
    try:
        K1 = np.load(CALIBRATION_FILE_PATH+'mtx1.npy')
        K2 = np.load(CALIBRATION_FILE_PATH+'mtx2.npy')
        dist1 = np.load(CALIBRATION_FILE_PATH+'dist1.npy')
        dist2 = np.load(CALIBRATION_FILE_PATH+'dist2.npy')
    except: 
        print("Failed to load documented camera parameters. Using computed values instead.")
        parameter_gather_camera_failed = True

if not USE_DOCUMENTED_CAMERA_PARAMETERS or parameter_gather_camera_failed:
    # Temperary K matrix
    fx = fy = FOCAL_LENGTH / pixel_size
    K1 = np.array([[fx, 0, width/2],
            [0, fy, height/2],
            [0, 0, 1]])
    K2 = np.array([[fx, 0, width/2],
            [0, fy, height/2],
            [0, 0, 1]])
    dist1 = np.zeros((4,1))
    dist2 = np.zeros((4,1))


# Load the images and undistort them
img1 = cv.imread(IMAGE_FILE_PATH+'/Left/'+IMAGE_NAME) # 
img2 = cv.imread(IMAGE_FILE_PATH+'/Right/'+IMAGE_NAME) # 
# Undestorting them caused issues with using the F Matrix.
img1 = cv.undistort(img1, K1, dist1)
img2 = cv.undistort(img2, K2, dist2)





# Get stereo parameters from the documented values or compute them if not available.
parameter_gather_failed = False
if USE_DOCUMENTED_STEREO_PARAMETERS:
    try:
        R = np.load(CALIBRATION_FILE_PATH+'R'+CALIBRATION_FILE_VERSION+'.npy')
        T = np.load(CALIBRATION_FILE_PATH+'T'+CALIBRATION_FILE_VERSION+'.npy')
        E = np.load(CALIBRATION_FILE_PATH+'E'+CALIBRATION_FILE_VERSION+'.npy')
        F = np.load(CALIBRATION_FILE_PATH+'F'+CALIBRATION_FILE_VERSION+'.npy')
    except:
        print("Failed to load documented stereo parameters. Using computed values instead.")
        parameter_gather_failed = True
        
        
## COMPUTER R, T & F Matrix
if (not USE_DOCUMENTED_STEREO_PARAMETERS) or parameter_gather_failed:
    # Initialize SIFT detector
    sift = cv.SIFT_create(nfeatures=10000)

    # Detect features and compute descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    img1_keyp = cv.drawKeypoints(img1, keypoints1, None)
    img2_keyp = cv.drawKeypoints(img2, keypoints2, None)

    combined_display = np.hstack((img1_keyp, img2_keyp))

    combined_display = cv.resize(combined_display, (int(combined_display.shape[1]*0.2), int(combined_display.shape[0] * 0.2)))
    cv.imshow('Key-Points', combined_display)
    cv.waitKey(0)
    cv.destroyAllWindows()

    #### Use a matcher:
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    #-- Filter matches using the Lowe's ratio test
    ratio_thresh = 0.5  # Lower value means more strict matching
    good_matches = []
    for m,n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    #-- Draw matches
    draw_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
    image_matches = cv.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, draw_matches, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    image_matches = cv.resize(image_matches, (int(image_matches.shape[1]*0.2), int(image_matches.shape[0] * 0.2)))

    cv.imshow('Matches', image_matches)
    cv.waitKey(0)
    cv.destroyAllWindows()

    ## Generate Essential Matrix

    # Extract location of matched keypoints in both images
    points1 = np.zeros((len(good_matches), 2), dtype=np.float32)
    points2 = np.zeros((len(good_matches), 2), dtype=np.float32)
    for i, match in enumerate(good_matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
    # TODO: Only uses one of the K, but should use both K1 and K2

    # # Normalize the points using the intrinsic matrix and distortion coefficients before computing the essential matrix
    # # This is now handeld by undisorting the image itself after importing.
    # points1 = cv.undistortPoints(points1, K1, dist1) 
    # points2 = cv.undistortPoints(points2, K2, dist2)

    E, mask = cv.findEssentialMat(points1, points2, method=cv.RANSAC)

    print(f"Essential Matrix:\n{E}")


    # Get Rotation and Translation from Essential Matrix from Normilized image points
    _, R, T, mask = cv.recoverPose(E, points1, points2)
    
    # Add the scaling factor
    T = T* SCALING_FACTOR
    
    print(f"Translation Vector:{T}, Rotation Matrix:{R}")
    np.save(CALIBRATION_FILE_PATH+'R_'+CALIBRATION_FILE_VERSION+'.npy', R)
    np.save(CALIBRATION_FILE_PATH+'T_'+CALIBRATION_FILE_VERSION+'.npy', T)
    np.save(CALIBRATION_FILE_PATH+'E_'+CALIBRATION_FILE_VERSION+'.npy', E)


    ## Compute the Fundamental Matrix
    # Remove the principle points from the K matrix for calculating the F matrix.
    # The shift has already been applied by undistorting the image.
    K1_without_principle = K1.copy()
    K1_without_principle[0, 2] = 0
    K1_without_principle[1, 2] = 0
    K2_without_principle = K2.copy()
    K2_without_principle[0, 2] = 0
    K2_without_principle[1, 2] = 0


    K1_inv = np.linalg.inv(K1_without_principle)
    K2_inv = np.linalg.inv(K2_without_principle) 
    K2_inv_T = K2_inv.T

    F = K2_inv_T @ E @ K1_inv
    print(f"F-Matrix calculated: {F}")
    np.save(CALIBRATION_FILE_PATH+'F_'+CALIBRATION_FILE_VERSION+'.npy', F)
    
if show_epi_lines:
    ## Boolean to select points yourself, or to have them randomly assigned.
    use_selected_points = True
    
    
    if use_selected_points:
        

        clicked_points = []

        def on_click(event):
            if event.button == 2 and event.inaxes:  # Check if the middle click is inside the axes
                x_full = event.xdata * 1 
                y_full = event.ydata * 1
                clicked_points.append([int(x_full), int(y_full)])
                print(f"Currently {len(clicked_points)} points selected for epipolar lines")
                # Plot the clicked point
                ax.plot(event.xdata, event.ydata, 'ro', markersize=5)
                fig.canvas.draw()

        # Resize the image for display
        resized_img1 = img1.copy()

        # Convert BGR to RGB for Matplotlib
        resized_img1_rgb = cv.cvtColor(resized_img1, cv.COLOR_BGR2RGB)

        # Display the image using Matplotlib
        fig, ax = plt.subplots()
        ax.imshow(resized_img1_rgb)
        ax.set_title("Click Points")
        fig.canvas.mpl_connect('button_press_event', on_click)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
        plt.show()

        random_points = np.array(clicked_points)
        print(random_points)
    else:
        # Random point Generation for visualization
        h, w = img1.shape[:2]
        X = 20  # Number of random points you want
        # Generate random (x, y) points within image bounds 
        random_points = np.column_stack((
            np.random.randint(0, w, size=X),
            np.random.randint(0, h, size=X)
        ))
        print(random_points)
    if len(random_points) ==0:
        print("No points selected. Skipping")
    else:
        lines = cv.computeCorrespondEpilines(random_points.reshape(-1, 1, 2), 1, F)
        lines = lines.reshape(-1, 3)

        # Convert images to RGB for Matplotlib
        img1_rgb = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
        img2_rgb = cv.cvtColor(img2, cv.COLOR_BGR2RGB)

        # Create a Matplotlib figure
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(img1_rgb)
        ax2.imshow(img2_rgb)
        ax1.set_title('Image 1 with Points')
        ax2.set_title('Image 2 with Epipolar Lines')

        ax1.axis('off')
        ax2.axis('off')

        for idx, r in enumerate(lines):
            color = np.random.random(size=3)  # Random color for each line
            # Plot the marker on the first image
            ax1.plot(random_points[idx][0], random_points[idx][1], marker='x', color=color, markersize=10, markeredgewidth=2)

            # Calculate the epipolar line endpoints for the second image
            x0, y0 = 0, -r[2] / r[1]
            x1, y1 = img2.shape[1], -(r[2] + r[0] * img2.shape[1]) / r[1]

            # Plot the epipolar line on the second image
            ax2.plot([x0, x1], [y0, y1], color=color, linewidth=1)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
        
        plt.show()



###########
####  This section below will handel clicks on the image, and calculating 3D points.
###########

# Create Projection Matrices for the cameras:

P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))  # Projection matrix for cam1
P2 = K2 @ np.hstack((R, T.reshape(3, 1)))  # Projection matrix for cam2


click_count = 0
points_3d_last = None
image_scaling = 1
point_plot = None
line_plot = None
distance_last = None
def click_event(event):
    global click_count, point_in_frame1, point_in_frame2, points_3d_last, point_plot, line_plot, distance_last
    if event.button == 2:  # Middle mouse button
        x_full = event.xdata * (1 / image_scaling)
        y_full = event.ydata * (1 / image_scaling)
        if x_full < img1.shape[1]:  # Click is in the left image
            # Reset if clicking in the left image again
            click_count = 0
            point_in_frame1 = None
            point_in_frame2 = None

            # Remove previous plots
            if point_plot:
                point_plot.remove()
            if line_plot:
                line_plot.remove()

            # First click: point in frame 1
            point_in_frame1 = np.array([x_full, y_full], dtype=np.float32).reshape(1, 1, 2)
            print(f"Point in Frame 1 selected: {point_in_frame1}")

            # Compute the epipolar line in the second image
            epilines_in_frame2 = cv.computeCorrespondEpilines(point_in_frame1, 1, F)
            epilines_in_frame2 = epilines_in_frame2.reshape(-1, 3)
            line = epilines_in_frame2[0]

            # Draw the point in the first image and the epipolar line in the second image
            point_plot = ax.plot(event.xdata, event.ydata, 'ro', markersize=5)[0]
            x_vals = np.array([img1.shape[1], img1.shape[1] + img2.shape[1]])
            y_vals = -(line[2] + line[0] * (x_vals - img1.shape[1])) / line[1]
            line_plot = ax.plot(x_vals, y_vals, 'b-', linewidth=2)[0]

            fig.canvas.draw()
            click_count = 1

        elif click_count == 1:  # Second click: point in the right image
            # Remove previous plots
            if point_plot:
                point_plot.remove()

            # Use the clicked point directly without snapping
            point_in_frame2 = np.array([x_full - width, y_full], dtype=np.float32).reshape(1, 1, 2)
            print(f"Point in Frame 2 selected: {point_in_frame2}")

            # Perform triangulation
            rp1 = point_in_frame1.reshape(1, 2)
            rp2 = point_in_frame2.reshape(1, 2)

            points_3d_homogenous = cv.triangulatePoints(P1, P2, rp1.T, rp2.T)
            points_3d = cv.convertPointsFromHomogeneous(points_3d_homogenous.T)[:, 0, :]

            # Calculate the distance from the camera to the point
            distance = np.linalg.norm(points_3d)
            print(f"3D Point: {points_3d}, Distance from camera: {distance}")

            # Calculate the distance from the last point to the current point
            if points_3d_last is not None:
                distance_last = np.linalg.norm(points_3d - points_3d_last)
                print(f"Distance from last point to current point: {distance_last}")
            points_3d_last = points_3d

            # Draw the point in the second image
            point_plot = ax.plot(event.xdata, event.ydata, 'ro', markersize=5)[0]

            fig.canvas.draw()
            # Reset for the next pair of points
            click_count = 0

list_of_saved_distances = []

def on_key(event):
    if event.key == ' ':  # Space bar
        if distance_last is not None:
            list_of_saved_distances.append(distance_last)
            print(f"Appended {distance_last} to list_of_saved_distances. Size of list is now {len(list_of_saved_distances)}") 
    elif event.key == 'q':  # Quit with 'q'
        print("Exiting...")
        plt.close(fig)

# Initial display of the images
new_img1 = img1.copy()
new_img2 = img2.copy()
combined_display = np.hstack((new_img1, new_img2))
combined_display = cv.resize(combined_display, (int(combined_display.shape[1] * image_scaling), int(combined_display.shape[0] * image_scaling)))
combined_display_rgb = cv.cvtColor(combined_display, cv.COLOR_BGR2RGB)

fig, ax = plt.subplots()
ax.imshow(combined_display_rgb)
ax.set_title("Stereo Matching")
fig.canvas.mpl_connect('button_press_event', click_event)
fig.canvas.mpl_connect('key_press_event', on_key)
plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
plt.show()
if len(list_of_saved_distances) == 0:
    print("No distances saved.")
else:
    try:
        data_list = np.load('experiment_data.npy')
    except:
        print("No file yet?")
        data_list = np.array([])
    data_list = np.append(data_list, list_of_saved_distances)
    np.save('experiment_data.npy', data_list)
    # Print the saved distances
    print(f"Appended {len(list_of_saved_distances)} distances to the list. List now contains {len(data_list)} distances.")

try:
    data_list = np.load('experiment_data.npy')
except:
    print("No file yet?")
    
if SHOW_STATISTICS and len(data_list) > 0:
# Calculate the mean and standard deviation of the distances
    real_length_beam = 1000 #mm
    corrected_data = data_list - real_length_beam
    
    mu, std = norm.fit(corrected_data) 
    print(f"Mean: {mu}, Standard Deviation: {std}")

    # Use Freedman-Diaconis rule to calculate the number of bins
    q75, q25 = np.percentile(corrected_data, [75 ,25])
    bin_width = 2 * (q75 - q25) / (len(corrected_data) ** (1/3))
    num_bins = int((np.max(corrected_data) - np.min(corrected_data)) / bin_width)

    # Plot the histogram and normal distribution

    counts, bins, _ = plt.hist(corrected_data, bins=num_bins, density=True, alpha=0.6, color='b', label='Histogram of Distances')


    # Fit a normal distribution to the data
    xmin, xmax = plt.xlim()
    size_range = xmax - xmin
    x = np.linspace(xmin-size_range*0.1, xmax+size_range*0.1, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2, label='Fitted Normal Distribution')

    # Add labels, title, and legend
    plt.title('Statistical Analysis of Saved Distances', fontsize=18)
    plt.xlabel('Distance', fontsize=16)
    plt.ylabel('Probability Density', fontsize=16)
    plt.axvline(mu, color='r', linestyle='dashed', linewidth=1.5, label=f'Mean: {mu:.2f}')
    plt.axvline(mu + std, color='b', linestyle='dashed', linewidth=1.5, label=f'+1 Std Dev: {mu + std:.2f}')
    plt.axvline(mu - std, color='b', linestyle='dashed', linewidth=1.5, label=f'-1 Std Dev: {mu - std:.2f}')
    plt.legend(fontsize=14)
    plt.grid(True)

    # Show the plot
    plt.show()

    # Additional statistics
    print(f"Minimum distance: {np.min(corrected_data)}")
    print(f"Maximum distance: {np.max(corrected_data)}")
    print(f"Median distance: {np.median(corrected_data)}")
    print(f"25th percentile: {np.percentile(corrected_data, 25)}")
    print(f"75th percentile: {np.percentile(corrected_data, 75)}")

    # Boxplot for visualizing the spread of distances
    plt.figure(figsize=(10, 6))
    plt.boxplot(corrected_data, vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    plt.title('Boxplot of Saved Distances', fontsize=18)
    plt.xlabel('Distance', fontsize=16)
    plt.grid(True)

    # Show the boxplot
    plt.show()