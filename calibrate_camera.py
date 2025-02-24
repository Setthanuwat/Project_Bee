import cv2
import numpy as np
import os  # เพิ่มเพื่อใช้จัดการไดเรกทอรี
import glob

# Define checkerboard size and square size in millimeters
Ch_Dim = (8, 6)  # Corners of the checkerboard (number of internal corners per a chessboard row and column)
Sq_size = 24  # Size of one square in millimeters

# Termination criteria for cornerSubPix
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points for 3D world coordinates
obj_3D = np.zeros((Ch_Dim[0] * Ch_Dim[1], 3), np.float32)
obj_3D[:, :2] = np.mgrid[0:Ch_Dim[0], 0:Ch_Dim[1]].T.reshape(-1, 2) * Sq_size

# Arrays to store object points and image points from all the images
obj_points_3D = []  # 3D points in real world space
img_points_2D = []  # 2D points in image plane

# Load images from folder
image_files = glob.glob("test12.jpg")

for image_file in image_files:
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, Ch_Dim, None)

    if ret:
        # If corners are found, refine the corners
        obj_points_3D.append(obj_3D)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        img_points_2D.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, Ch_Dim, corners2, ret)

# Calibrate camera
ret, mtx, dist_coeff, R_vecs, T_vecs = cv2.calibrateCamera(obj_points_3D, img_points_2D, gray.shape[::-1], None, None)

# Create the directory if it doesn't exist
calib_data_path = 'calibration_data'
if not os.path.exists(calib_data_path):
    os.makedirs(calib_data_path)

# Save the calibration results
np.savez(
    f"{calib_data_path}/CalibrationMatrix_college_cpt",
    Camera_matrix=mtx,
    distCoeff=dist_coeff,
    RotationalV=R_vecs,
    TranslationV=T_vecs
)

print("Calibration done and saved.")
