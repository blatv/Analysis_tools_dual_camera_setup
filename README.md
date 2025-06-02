# Analysis_tools_dual_camera_setup

The analysis tools for the dual camera setup - Wilddrone

---

## Overview

This project provides a workflow for:
- **Calibrating two cameras** using chessboard images.
- **Stereo calibration** to compute the relative pose between cameras.
- **Visualizing epipolar lines** and performing 3D triangulation from stereo images.
- **Collecting and analyzing experimental distance measurements.**

---

## Repository Structure

```
.
├── calib.py           # Camera and stereo calibration
├── Final.py           # Main analysis, epipolar geometry, 3D point triangulation, statistics
├── removeLast.py      # Utility to remove last entries from experiment data
├── CalibrationData/   # Directory for calibration .npy files
├── Datasets/          # (Expected) Directory for stereo image datasets (Removed for Privacy Reasons. Add your own data.)
├── experiment_data.npy# (Generated) Numpy file with saved experiment distances
└── README.md          # This file
```

---

## Installation

1. **Clone the repository:**
   ```sh
   git clone <your-repo-url>
   cd Analysis_tools_dual_camera_setup
   ```

2. **Install dependencies:**
   ```sh
   pip install numpy opencv-python matplotlib scipy
   ```

3. **Prepare your data:**
   - Place your calibration images in `Left/` and `Right/` folders.
   - Place your stereo pairs for overlap in `Overlap/Left/` and `Overlap/Right/`.

---

## Usage

### 1. Camera Calibration (`calib.py`)

**Purpose:**  
Calibrate each camera individually and then perform stereo calibration to obtain intrinsic and extrinsic parameters.

**How to run:**
```sh
python calib.py
```

**What it does:**
- Finds chessboard corners in calibration images.
- Computes camera matrices and distortion coefficients.
- Performs stereo calibration to compute rotation (R), translation (T), essential (E), and fundamental (F) matrices.
- Saves calibration results as `.npy` files in the working directory.

**Expected folders:**
- `Left/` and `Right/` for single camera calibration images.
- `Overlap/Left/` and `Overlap/Right/` for stereo calibration images.

---

### 2. Stereo Analysis and Epipolar Geometry (`Final.py`)

**Purpose:**  
Visualize epipolar lines, perform 3D triangulation from stereo images, and analyze measurement statistics.

**How to run:**
```sh
python Final.py
```

**Features:**
- Loads calibration data and stereo images.
- Visualizes epipolar lines for selected or random points.
- Allows interactive selection of points for 3D triangulation.
- Computes and saves distances between 3D points.
- Provides statistical analysis and visualization (histogram, boxplot) of measured distances.

**Configuration:**
- Edit the `CALIBRATION_FILE_PATH`, `IMAGE_FILE_PATH`, and `IMAGE_NAME` variables at the top of `Final.py` to point to your data.

---

### 3. Data Management (`removeLast.py`)

**Purpose:**  
Remove the last N entries from the experimental distance data file (`experiment_data.npy`).

**How to run:**
```sh
python removeLast.py
```

**Instructions:**
- Enter the number of entries to remove when prompted.
- Confirm the deletion.

---

## Data Files

- **Calibration files:**  
  Saved as `.npy` files (e.g., `mtx1.npy`, `dist1.npy`, `R.npy`, `T.npy`, etc.) after running `calib.py`.

- **Experiment data:**  
  `experiment_data.npy` stores the list of measured distances, appended by `Final.py` and managed by `removeLast.py`.

---

## Troubleshooting

- **Images not found:**  
  Ensure your image folders and file paths match those specified in the scripts.
- **Chessboard not detected:**  
  Adjust the `rows`, `columns`, or detection criteria in `calib.py`.
- **Calibration fails:**  
  Use more/better quality calibration images with a clearly visible chessboard.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

**Contact:**  
For questions or contributions, please open an issue or pull request on GitHub.

---

**_This README file was generated using CoPilot._
