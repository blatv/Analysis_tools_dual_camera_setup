import cv2 as cv
import glob
import numpy as np

#Parameters:
width = 5496 #image width
height = 3672 #image height
rows = 6 #number of checkerboard rows.
columns = 9 #number of checkerboard columns.
world_scaling = 103.7 # World Scale (in mm) of the checkerboard squares.


def calibrate_camera(images_folder):
    images = glob.glob(images_folder)
 
    #criteria used by checkerboard pattern detector.
    #Change this if the code can't find the checkerboard
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:columns,0:rows].T.reshape(-1,2)
 
    #Pixel coordinates of checkerboards
    imgpoints = [] # 2d points in image plane.
 
    #coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space
    
    succesful_image_path = []
    
    
    success_count = 0
    counter = 0
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
     
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (columns,rows), cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE
            + cv.CALIB_CB_FAST_CHECK);
        counter=counter+1
        print(f"Checked image {counter}")
        # If found, add object points, image points (after refining them)
        if ret == True:
            
            objpoints.append(objp)
     
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            
            # Draw and display the corners
            cv.drawChessboardCorners(img, (columns,rows), corners2, ret)
            success_count=success_count+1
            print(f"Succesfully found in img {counter}. Success Count = {success_count}")
            succesful_image_path.append(fname)
            
            img = cv.resize(img, (int(width*0.25), int(height*0.25)))
            cv.imshow('img', img)
            cv.waitKey(1)
        else:
            print(f"No checkerboard found in: {fname}")
        
    cv.destroyAllWindows()
    
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
    if ret:
        print('rmse:', ret)
        print('camera matrix:\n', mtx)
        print('distortion coeffs:', dist)
        print('Rs:\n', rvecs)
        print('Ts:\n', tvecs)
        
        print(f"Focal length in mm is fx:{mtx[0][0]*0.0024},fy:{mtx[1][1]*0.0024}")
        return mtx, dist
    else:
        print("Calibration Failed.")
        exit()
        
def stereo_calibrate(mtx1, dist1, mtx2, dist2, frames_folder_left, frames_folder_right):
    #read the synched frames
    images_names_left = glob.glob(frames_folder_left)
    images_names_right = glob.glob(frames_folder_right)
 
    c1_images = []
    c2_images = []
    for im1, im2 in zip(images_names_left, images_names_right):
        if im1[-7:] != im2[-7:]:
            print("Something strange happend with the order of images. They are not allighned..")
            exit()
        _im = cv.imread(im1, 1)
        c1_images.append(_im)
 
        _im = cv.imread(im2, 1)
        c2_images.append(_im)
 
    #change this if stereo calibration not good.
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:columns,0:rows].T.reshape(-1,2)
    objp = world_scaling* objp
 
    #frame dimensions. Frames should be the same size.
    width = c1_images[0].shape[1]
    height = c1_images[0].shape[0]
 
    #Pixel coordinates of checkerboards
    imgpoints_left = [] # 2d points in image plane.
    imgpoints_right = []
 
    #coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space
 
    for frame1, frame2 in zip(c1_images, c2_images):
        gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        c_ret1, corners1 = cv.findChessboardCorners(gray1, (columns,rows), cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE
            + cv.CALIB_CB_FAST_CHECK);
        c_ret2, corners2 = cv.findChessboardCorners(gray2, (columns,rows), cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE
            + cv.CALIB_CB_FAST_CHECK);
 
        if c_ret1 == True and c_ret2 == True:
            corners1 = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
 
            cv.drawChessboardCorners(frame1, (columns,rows), corners1, c_ret1)
            frame1 = cv.resize(frame1, (int(width*0.18), int(height*0.18)))
            cv.imshow('img', frame1)
 
            cv.drawChessboardCorners(frame2, (columns,rows), corners2, c_ret2)
            frame2 = cv.resize(frame2, (int(width*0.18), int(height*0.18)))
            cv.imshow('img2', frame2)
            cv.waitKey(500)
 
            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)
 
    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    ret, CM1, dist1, CM2, dist2, R, T, E, F = cv.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx1, dist1,
                                                                 mtx2, dist2, (width, height), criteria = criteria, flags = stereocalibration_flags)
 
    print(ret)
    return R, T, E, F

 
 
mtx1, dist1 = calibrate_camera(images_folder = 'Left/*')
mtx2, dist2 = calibrate_camera(images_folder = 'Right/*')
image_width, image_height = 5496, 3672
np.save('mtx1.npy', mtx1)
np.save('dist1.npy', dist1)
np.save('mtx2.npy', mtx2)
np.save('dist2.npy', dist2)

R, T, E, F = stereo_calibrate(mtx1, dist1, mtx2, dist2, 'Overlap/Left/*','Overlap/Right/*')
np.save('R.npy', R)
np.save('T.npy', T)
np.save('E.npy', E)
np.save('F.npy', F)
print("R: ", R)
print("T: ", T)
print("E: ", E)
print("F: ", F)
