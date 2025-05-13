# Surgical Tool Tracking (Computer Vision Project)

This project implements a real-time computer vision system to **track the position of a 3D-printed surgical tool** with millimetric accuracy. The system was designed for operating room environments, providing surgeons with precise visual feedback on tool location.

## Objective

To develop a robust visual tracking system using only a monocular camera, leveraging feature detection and pose estimation techniques to estimate the 6-DoF position of a custom-designed surgical tool.

---

## Demo

 [Watch the Project Demo](https://youtu.be/kUrA0y90PfI)

---

## Technologies Used

- **Python**
- **OpenCV** – Computer vision processing
- **NumPy** – Matrix operations
- **3D Printing** – Custom surgical tool for visual tracking
- **Camera Calibration Tools** – Accurate intrinsic parameter estimation

---

## How It Works

The system uses a **calibrated RGB camera** to continuously track a **custom 3D-printed marker-based surgical tool**. The tool has a known geometric marker (a fixed pattern of colored or black circles), allowing reliable pose estimation.

### Key Components:

#### 1. **Camera Calibration**
- Intrinsic and distortion parameters are estimated using a standard chessboard calibration.
- `cv2.findChessboardCorners` and `cv2.calibrateCamera` are used to compute camera matrix and distortion coefficients.

#### 2. **Marker Detection**
- The tool contains precisely placed **2D circular patterns**.
- Detected using:
  - `cv2.findCirclesGrid()` with `cv2.CALIB_CB_SYMMETRIC_GRID`
  - Blob detection setup with `cv2.SimpleBlobDetector`
- The known 3D coordinates of the markers are matched with the 2D pixel coordinates.

#### 3. **Pose Estimation**
- Using `cv2.solvePnP()` and `cv2.projectPoints()`, the system estimates the 3D position and orientation (rvec and tvec) of the tool relative to the camera frame.
- Visual feedback is rendered via:
  - `cv2.drawFrameAxes()` – to show 3D axes over the image.
  - `cv2.putText()` – for on-screen position display.

#### 4. **Real-Time Tracking**
- Frames are captured via a webcam and processed in real-time.
- Stability and precision are achieved through controlled lighting and a fixed camera setup.

---

## Example Output

Translation vector (cm): [x: 2.35, y: -1.12, z: 10.48]
Rotation vector (Rodrigues): [0.12, 0.03, -0.09]


With proper calibration, positional error remains within **±1 mm**, making it suitable for surgical research and prototyping environments.

---

## Requirements

Install dependencies using pip:

```bash
pip install -r requirements.txt
```
Ensure you have a webcam and a printed marker pattern. Calibration data should be generated beforehand using a chessboard.

---

## Repository Structure

Surgical Tool Tracking/

├── main.py

├── README.md

├── .gitignore

├── requirements.txt

└── venv/

---

## Notes

- This project only uses computer vision for tracking.
- The tool geometry and marker layout must be consistent with the predefined 3D model.
- Assumes the camera remains static during tracking.

---

## Acknowledgments

Special thanks to the academic mentors and lab assistants who supported the physical prototyping and system validation.

---

## License

This project is open-source under the MIT License.
