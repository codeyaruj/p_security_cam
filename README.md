# p_security_cam
A production-ready Python application for intelligent face recognition using your PC's webcam. The system detects and identifies faces in real-time, displaying green bounding boxes for registered individuals and red boxes for unknown persons.

## Features
✅ **Real-time face detection and recognition**
- Detects multiple faces simultaneously
- Green bounding boxes for registered faces
- Red bounding boxes for unknown faces
✅ **Intelligent capture system** (anti-spam)
- Captures unknown face **only once** when first detected
- Additional captures (max 2-3) only triggered by:
  - Significant head rotation (yaw/pitch change)
  - Face leaving and re-entering the frame
- Built-in cooldown prevents duplicate captures
- Automatic cleanup of stale tracking data
✅ **Performance optimized**
- Frame downscaling for faster processing
- Efficient encoding comparisons
- Smooth real-time display
✅ **Statistics and monitoring**
- Live frame counter
- Registered vs. unknown face counts
- Total images captured
- Press 's' during runtime for detailed stats

## Prerequisites
- Python 3.7 or higher
- Webcam (built-in or USB)
- Linux, macOS, or Windows
- 
## Installation

### 1. Install Dependencies

```bash
pip install opencv-python face_recognition numpy pillow --break-system-packages
```

**Note for Linux users:** You may need to install additional system libraries:

```bash
# Ubuntu/Debian
sudo apt-get install python3-opencv libopencv-dev cmake

# Fedora
sudo dnf install python3-opencv cmake
```

**Note for macOS users with Apple Silicon (M1/M2):**

```bash
# Use conda for better compatibility
conda install -c conda-forge dlib opencv face_recognition
```

### 2. Create Directory Structure

```bash
mkdir registered_faces unknown_faces
```

### 3. Add Registered Faces

Add photos of people you want to register to the `registered_faces` folder:

```
registered_faces/
├── john_doe.jpg
├── jane_smith.jpg
├── bob_johnson.jpg
└── alice_williams.jpg
```

**Important tips for registered face images:**
- Use clear, front-facing photos
- Good lighting, no shadows
- One person per image
- Filename format: `firstname_lastname.jpg` (underscores will be converted to spaces)
- Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`

## Usage

### Basic Usage

```bash
python face_recognition_system.py
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit the application |
| `s` | Show detailed statistics |

### What Happens

1. **System starts** and loads all registered faces from `registered_faces/`
2. **Webcam opens** and displays live feed
3. **Faces are detected** in each frame:
   - **Green box + name**: Registered person detected
   - **Red box + "Unknown Face"**: Unrecognized person
4. **Unknown faces are captured intelligently**:
   - First appearance: 1 image saved immediately
   - Angle changes: Up to 2-3 more images (if head rotates significantly)
   - Same stationary face: NOT captured repeatedly
5. **Images saved** to `unknown_faces/` with timestamp

## How the Intelligent Capture Works

### The Problem
Without intelligent capture, a system would save hundreds or thousands of images of the same person standing still, filling up your disk.

### The Solution
This system uses a **face tracking algorithm** with several safeguards:

1. **Face ID Generation**: Each face gets a unique ID based on its encoding
2. **First Capture**: When a face first appears, capture once immediately
3. **Pose Tracking**: Monitor head rotation (yaw and pitch angles)
4. **Angle-Based Capture**: Only capture again if head rotates >15° (configurable)
5. **Cooldown Period**: Minimum 3 seconds between captures (configurable)
6. **Capture Limit**: Maximum 3 total captures per unique face (configurable)
7. **Automatic Cleanup**: Old face trackers removed after 300 frames of absence

### Example Scenario

```
Frame 1:   Unknown face appears → CAPTURE ✓ (first appearance)
Frame 2-100: Face stationary → no capture
Frame 101: Face turns left 20° → CAPTURE ✓ (angle change)
Frame 102-200: Face stationary → no capture
Frame 201: Face turns right 25° → CAPTURE ✓ (angle change)
Frame 202-300: Face stationary → no capture
Frame 301: Face turns again → NO CAPTURE (limit reached: 3/3)
```

## Configuration

You can adjust parameters in the `main()` function at the bottom of `face_recognition_system.py`:

```python
config = {
    'face_match_threshold': 0.6,      # Lower = stricter matching (0.4-0.7)
    'angle_change_threshold': 15.0,   # Degrees to trigger new capture
    'capture_cooldown_seconds': 3.0,  # Min seconds between captures
    'max_captures_per_face': 3        # Max total captures per face
}
```

### Parameter Guide

| Parameter | Recommended Range | Effect |
|-----------|------------------|---------|
| `face_match_threshold` | 0.4 - 0.7 | Lower = stricter face matching |
| `angle_change_threshold` | 10 - 30 degrees | Higher = fewer angle-based captures |
| `capture_cooldown_seconds` | 2 - 5 seconds | Higher = longer wait between captures |
| `max_captures_per_face` | 2 - 5 images | Higher = more variation captured |

## Troubleshooting

### "Could not open webcam"
- Check if another application is using the camera
- Try changing camera index: `cv2.VideoCapture(1)` instead of `cv2.VideoCapture(0)`
- On Linux, ensure you have camera permissions

### "No face detected in: person.jpg"
- Make sure the registered face image shows a clear, front-facing face
- Try a different photo with better lighting
- Ensure the face is not too small in the image

### Face recognition is slow
- Reduce camera resolution in the code (already set to 640x480)
- Process every 2nd or 3rd frame instead of every frame
- Consider using a GPU-accelerated version of dlib

### False matches or misidentifications
- Lower the `face_match_threshold` (e.g., from 0.6 to 0.5)
- Add more photos of registered people from different angles
- Ensure good lighting in both registered photos and live feed

### Too many/too few captures of unknown faces
- Adjust `angle_change_threshold`: Higher = fewer captures
- Adjust `capture_cooldown_seconds`: Higher = longer wait
- Adjust `max_captures_per_face`: Lower = fewer total captures

## File Structure

```
face_recognition_system/
├── face_recognition_system.py  # Main script
├── registered_faces/           # Put registered face photos here
│   ├── john_doe.jpg
│   └── jane_smith.jpg
└── unknown_faces/              # Auto-generated captures (timestamped)
    ├── unknown_20240202_143025_123456.jpg
    └── unknown_20240202_143045_789012.jpg
```

## Code Architecture

The system is organized into modular components:

### Main Classes

1. **`FaceRecognitionSystem`**: Core system managing:
   - Face database loading
   - Frame processing pipeline
   - Recognition logic
   - Intelligent capture decisions

2. **`FaceTracker`**: Tracks individual unknown faces:
   - First/last seen timestamps
   - Capture count and history
   - Pose (angle) history
   - Cooldown management

### Key Methods

- `_load_registered_faces()`: Loads and encodes registered faces
- `_estimate_head_pose()`: Calculates yaw/pitch from landmarks
- `_should_capture_unknown_face()`: Implements anti-spam logic
- `_capture_unknown_face()`: Saves face image to disk
- `process_frame()`: Main processing pipeline for each frame

## Performance Notes

- **Processing speed**: ~15-30 FPS on a modern laptop (depends on CPU)
- **Memory usage**: ~100-200 MB for typical usage
- **Disk usage**: Minimal (3-10 MB per unknown person detected)

## Privacy & Ethics

⚠️ **Important considerations:**
- This system captures images of people without explicit notification
- Ensure compliance with local privacy laws and regulations
- Use only in appropriate contexts (home security, authorized access control)
- Do not deploy in public spaces without proper signage and consent
- Handle captured images responsibly and securely

## Extending the System

### Add Audio Alerts
```python
import winsound  # Windows
# or
import os  # Linux/Mac

# In process_frame(), add:
if name == "Unknown Face" and should_capture:
    winsound.Beep(1000, 200)  # Windows
    # os.system('say "Unknown person detected"')  # Mac
```

### Log to CSV
```python
import csv
from datetime import datetime

# Add to _capture_unknown_face():
with open('capture_log.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([datetime.now(), filename, current_pose])
```

### Database Integration
Replace file-based storage with SQL:

```python
import sqlite3

# Create database for unknown faces
conn = sqlite3.connect('faces.db')
# Store encodings, timestamps, etc.
```

## Support
For issues or questions:
1. Check the Troubleshooting section above
2. Verify all dependencies are correctly installed
3. Review configuration parameters
4. Test with different lighting conditions and camera angles

## Credits
Built using:
- OpenCV for video capture and display
- face_recognition library (dlib-based) for face detection and encoding
- NumPy for numerical operations
