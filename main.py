import cv2
import cv2.aruco as aruco
import numpy as np
from collections import deque

#Camera Calibration
camera_matrix = np.array([
    [953.51712353, 0., 480.83479687],
    [0., 961.72274422, 637.84382984],
    [0., 0., 1.]
], dtype=float)
dist_coeffs = np.array([[0.17966515, -0.44242866, 0.00344937, -0.00931432, 0.33388515]])

# Scale Correction Factors (X, Y, Z)
scale_correction = np.array([-0.497 / 0.550, -0.497 / 0.555, 0.91], dtype=np.float32)

# Tool Marker Corner Coordinates (cm)
tool_marker_corners_3d = {
    5: np.array([[-1.3, -1.3, 8.8], [1.3, -1.3, 8.8], [1.3, 1.3, 8.8], [-1.3, 1.3, 8.8]], dtype=np.float32),
    6: np.array([[-1.3, 2.9, 7.8], [1.3, 2.9, 7.8], [1.3, 4.3, 5.2], [-1.3, 4.3, 5.2]], dtype=np.float32),
    7: np.array([[2.9, 1.3, 7.8], [2.9, -1.3, 7.8], [4.3, -1.3, 5.2], [4.3, 1.3, 5.2]], dtype=np.float32),
    8: np.array([[1.3, -2.9, 7.8], [-1.3, -2.9, 7.8], [-1.3, -4.3, 5.2], [1.3, -4.3, 5.2]], dtype=np.float32),
    9: np.array([[-2.9, -1.3, 7.8], [-2.9, 1.3, 7.8], [-4.3, 1.3, 5.2], [-4.3, -1.3, 5.2]], dtype=np.float32)
}
tool_tip_local = np.array([[0, 0, -11.0]], dtype=np.float32)  # tip position in cm

#ArUco Settings 
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()

#Kalman Filter Setup
kalman = cv2.KalmanFilter(6, 3)
kalman.measurementMatrix = np.eye(3, 6, dtype=np.float32)
kalman.transitionMatrix = np.eye(6, dtype=np.float32)
kalman.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-4
kalman.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1e-2

# Trail Setup
tip_trail = deque(maxlen=50)  # last 50 points

# Video Stream 
cap = cv2.VideoCapture(0)

def rotation_matrix_to_euler_angles(R):
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    return np.degrees([x, y, z])

print("Tracking tool tip... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
     

    if ids is not None:


        ids = ids.flatten()
        object_points = []
        image_points = []

        for i, id in enumerate(ids):
            if id in tool_marker_corners_3d:
                object_points.append(tool_marker_corners_3d[id])
                image_points.append(corners[i][0])

        # Pose estimation for tool
        if object_points:
            object_points = np.vstack(object_points).astype(np.float32)
            image_points = np.vstack(image_points).astype(np.float32)
            success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
            if success:
                R, _ = cv2.Rodrigues(rvec)
                tip_world = R @ tool_tip_local.T + tvec
                tip_world = tip_world.flatten() / 100.0  # cm to meters

                #Apply scale correction BEFORE Kalman filtering
                tip_world *= scale_correction

                # Kalman filter update
                measurement = np.array(tip_world, dtype=np.float32)
                kalman.correct(measurement)
                predicted = kalman.predict()
                tip_filtered = predicted[:3]

                tip_trail.append(tuple(tip_filtered))

                # Euler angles
                euler = rotation_matrix_to_euler_angles(R)
                euler_str = f"Roll={euler[0]:.1f}, Pitch={euler[1]:.1f}, Yaw={euler[2]:.1f}"

                # Display info
                cv2.putText(
                    frame,
                    f"Tip: X={float(tip_filtered[0]):.3f} Y={float(tip_filtered[1]+0.3):.3f} Z={float(tip_filtered[2]):.3f} (m)",
                    (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )

                cv2.putText(frame, euler_str, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

                # Draw trail
                for pt in tip_trail:
                    x, y, z = pt
                    pt2d = (int(50 + x * 300), int(250 - z * 300))  # pseudo projection
                    cv2.circle(frame, pt2d, 3, (255, 0, 255), -1)

        # Show reference marker positions (IDs 0–4)
        for i, id in enumerate(ids):
            if id <= 4:
                corner = corners[i][0]
                center = corner.mean(axis=0).astype(int)
                cv2.circle(frame, tuple(center), 4, (0, 255, 0), -1)
                cv2.putText(frame, f"ID {id}", (center[0] + 5, center[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show 3D positions of center of reference markers (IDs 0–3)
    ref_marker_size = 0.052  # 5.2 cm

    if ids is not None:
        for i, marker_id in enumerate(ids):
            if marker_id in [0, 1, 2, 3]:
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                    [corners[i]], ref_marker_size, camera_matrix, dist_coeffs
                )
                tvec = tvecs[0][0]  # shape (3,)

                label = f"ID {marker_id}: X={tvec[0]:.3f} Y={tvec[1]:.3f} Z={tvec[2]:.3f} m"
                center_px = corners[i][0].mean(axis=0).astype(int)

                cv2.putText(frame, label, (center_px[0] + 5, center_px[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Tool Tip Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

