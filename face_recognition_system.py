#!/usr/bin/env python3
"""
Real-Time Face Recognition System
==================================

SETUP INSTRUCTIONS:
-------------------
1. Install required dependencies:
   pip install opencv-python face_recognition numpy pillow --break-system-packages

2. Create a 'registered_faces' folder in the same directory as this script
3. Add photos of registered people to the folder (one face per image)
4. Name files as: person_name.jpg (e.g., "john_doe.jpg", "jane_smith.jpg")
5. Run the script: python face_recognition_system.py

CONTROLS:
---------
- Press 'q' to quit
- Press 's' to show statistics
- Captured unknown faces are saved to 'unknown_faces' folder

FEATURES:
---------
- Real-time face detection and recognition
- Green boxes for registered faces, red boxes for unknown faces
- Intelligent capture: only saves unknown faces when:
  * Face first appears (1 capture)
  * Face angle changes significantly (max 2 more captures)
  * Face re-enters after leaving frame
- Anti-spam cooldown to prevent folder pollution
"""

import cv2
import face_recognition
import numpy as np
import os
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import math


class FaceRecognitionSystem:
    """Main class for real-time face recognition with intelligent capture."""
    
    def __init__(self, 
                 registered_faces_dir="registered_faces",
                 unknown_faces_dir="unknown_faces",
                 face_match_threshold=0.6,
                 angle_change_threshold=15.0,
                 capture_cooldown_seconds=3.0,
                 max_captures_per_face=3):
        """
        Initialize the face recognition system.
        
        Args:
            registered_faces_dir: Directory containing registered face images
            unknown_faces_dir: Directory to save unknown face captures
            face_match_threshold: Lower = stricter matching (0.0-1.0)
            angle_change_threshold: Degrees of rotation to trigger new capture
            capture_cooldown_seconds: Minimum seconds between captures
            max_captures_per_face: Maximum captures per unique unknown face
        """
        self.registered_faces_dir = Path(registered_faces_dir)
        self.unknown_faces_dir = Path(unknown_faces_dir)
        self.face_match_threshold = face_match_threshold
        self.angle_change_threshold = angle_change_threshold
        self.capture_cooldown_seconds = capture_cooldown_seconds
        self.max_captures_per_face = max_captures_per_face
        
        # Create directories if they don't exist
        self.registered_faces_dir.mkdir(exist_ok=True)
        self.unknown_faces_dir.mkdir(exist_ok=True)
        
        # Storage for registered face encodings and names
        self.known_face_encodings = []
        self.known_face_names = []
        
        # Tracking for unknown faces to prevent spam
        # Key: face_id (based on encoding), Value: FaceTracker object
        self.unknown_face_trackers = {}
        
        # Statistics
        self.stats = {
            'registered_detected': 0,
            'unknown_detected': 0,
            'images_captured': 0,
            'frames_processed': 0
        }
        
        # Load registered faces
        self._load_registered_faces()
        
    def _load_registered_faces(self):
        """Load and encode all registered faces from the directory."""
        print(f"Loading registered faces from {self.registered_faces_dir}...")
        
        if not self.registered_faces_dir.exists():
            print(f"Warning: {self.registered_faces_dir} does not exist. Creating it.")
            return
            
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = [f for f in self.registered_faces_dir.iterdir() 
                      if f.suffix.lower() in supported_formats]
        
        if not image_files:
            print(f"Warning: No face images found in {self.registered_faces_dir}")
            print(f"Please add images with names like 'john_doe.jpg'")
            return
        
        for image_path in image_files:
            try:
                # Load image and get face encoding
                image = face_recognition.load_image_file(str(image_path))
                encodings = face_recognition.face_encodings(image)
                
                if encodings:
                    self.known_face_encodings.append(encodings[0])
                    # Use filename without extension as name
                    name = image_path.stem.replace('_', ' ').title()
                    self.known_face_names.append(name)
                    print(f"  ✓ Loaded: {name}")
                else:
                    print(f"  ✗ No face detected in: {image_path.name}")
                    
            except Exception as e:
                print(f"  ✗ Error loading {image_path.name}: {e}")
        
        print(f"\nTotal registered faces: {len(self.known_face_encodings)}\n")
    
    def _estimate_head_pose(self, face_landmarks):
        """
        Estimate head pose (yaw and pitch) from face landmarks.
        
        This uses a simplified approach based on facial landmark positions.
        Returns approximate angles in degrees.
        """
        if not face_landmarks:
            return 0.0, 0.0
        
        # Get key landmarks
        nose_tip = face_landmarks['nose_tip'][2]  # Center of nose tip
        nose_bridge = face_landmarks['nose_bridge'][0]  # Top of nose
        left_eye = np.mean(face_landmarks['left_eye'], axis=0)
        right_eye = np.mean(face_landmarks['right_eye'], axis=0)
        chin = face_landmarks['chin'][8]  # Center of chin
        
        # Calculate yaw (left-right rotation) based on eye symmetry
        eye_center = (left_eye + right_eye) / 2
        eye_distance = np.linalg.norm(right_eye - left_eye)
        nose_to_eye_center = nose_tip[0] - eye_center[0]
        
        # Normalize and convert to degrees (approximate)
        yaw = (nose_to_eye_center / eye_distance) * 60  # Rough calibration
        
        # Calculate pitch (up-down rotation) based on nose-to-chin distance
        nose_to_chin_y = chin[1] - nose_tip[1]
        nose_bridge_to_tip_y = nose_tip[1] - nose_bridge[1]
        
        if nose_bridge_to_tip_y != 0:
            pitch_ratio = nose_to_chin_y / nose_bridge_to_tip_y
            pitch = (pitch_ratio - 2.5) * 30  # Rough calibration
        else:
            pitch = 0.0
        
        return yaw, pitch
    
    def _get_face_id(self, face_encoding):
        """
        Generate a unique ID for a face based on its encoding.
        
        This allows tracking the same face across frames even if it's unknown.
        """
        # Use a hash of the encoding (rounded to reduce noise)
        rounded = np.round(face_encoding, decimals=2)
        return hash(rounded.tobytes())
    
    def _should_capture_unknown_face(self, face_id, current_pose, frame_number):
        """
        Determine if an unknown face should be captured based on tracking history.
        
        This implements the intelligent capture logic:
        - First appearance: capture immediately
        - Subsequent appearances: only if angle changed significantly
        - Respect cooldown and max captures limit
        """
        current_time = time.time()
        
        # If this face hasn't been seen before, create tracker and capture
        if face_id not in self.unknown_face_trackers:
            self.unknown_face_trackers[face_id] = FaceTracker(
                first_seen_frame=frame_number,
                first_seen_time=current_time,
                initial_pose=current_pose
            )
            return True  # First appearance - capture
        
        tracker = self.unknown_face_trackers[face_id]
        
        # Update that we've seen this face again
        tracker.last_seen_frame = frame_number
        tracker.last_seen_time = current_time
        
        # Check if we've hit the capture limit
        if tracker.capture_count >= self.max_captures_per_face:
            return False
        
        # Check cooldown period
        time_since_last_capture = current_time - tracker.last_capture_time
        if time_since_last_capture < self.capture_cooldown_seconds:
            return False
        
        # Check if angle has changed significantly
        yaw_change = abs(current_pose[0] - tracker.last_captured_pose[0])
        pitch_change = abs(current_pose[1] - tracker.last_captured_pose[1])
        
        if yaw_change > self.angle_change_threshold or pitch_change > self.angle_change_threshold:
            return True  # Significant angle change - capture
        
        return False
    
    def _capture_unknown_face(self, frame, face_location, face_id, current_pose):
        """Save an unknown face image to disk and update tracker."""
        top, right, bottom, left = face_location
        
        # Add some padding around the face
        padding = 20
        height, width = frame.shape[:2]
        top = max(0, top - padding)
        left = max(0, left - padding)
        bottom = min(height, bottom + padding)
        right = min(width, right + padding)
        
        # Extract face region
        face_image = frame[top:bottom, left:right]
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"unknown_{timestamp}.jpg"
        filepath = self.unknown_faces_dir / filename
        
        # Save image
        cv2.imwrite(str(filepath), face_image)
        
        # Update tracker
        tracker = self.unknown_face_trackers[face_id]
        tracker.capture_count += 1
        tracker.last_capture_time = time.time()
        tracker.last_captured_pose = current_pose
        tracker.captured_files.append(filename)
        
        self.stats['images_captured'] += 1
        
        print(f"📸 Captured unknown face: {filename} "
              f"(capture {tracker.capture_count}/{self.max_captures_per_face})")
    
    def _cleanup_stale_trackers(self, current_frame, stale_frame_threshold=300):
        """
        Remove trackers for faces that haven't been seen recently.
        
        This prevents memory buildup from old faces.
        """
        stale_ids = []
        for face_id, tracker in self.unknown_face_trackers.items():
            if current_frame - tracker.last_seen_frame > stale_frame_threshold:
                stale_ids.append(face_id)
        
        for face_id in stale_ids:
            del self.unknown_face_trackers[face_id]
        
        if stale_ids:
            print(f"🧹 Cleaned up {len(stale_ids)} stale face tracker(s)")
    
    def process_frame(self, frame, frame_number):
        """
        Process a single frame: detect faces, recognize them, and handle captures.
        
        Returns the frame with annotations drawn on it.
        """
        # Resize frame for faster processing (face_recognition can be slow)
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces and get encodings
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame, face_locations)
        
        # Scale back up face locations
        face_locations = [(top*4, right*4, bottom*4, left*4) 
                         for (top, right, bottom, left) in face_locations]
        
        # Process each detected face
        for face_encoding, face_location, face_landmarks in zip(
            face_encodings, face_locations, face_landmarks_list):
            
            # Check if face matches any registered face
            matches = face_recognition.compare_faces(
                self.known_face_encodings, 
                face_encoding,
                tolerance=self.face_match_threshold
            )
            
            name = "Unknown Face"
            color = (0, 0, 255)  # Red for unknown
            
            # If match found, use the best match
            if True in matches:
                face_distances = face_recognition.face_distance(
                    self.known_face_encodings, face_encoding
                )
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                    color = (0, 255, 0)  # Green for registered
                    self.stats['registered_detected'] += 1
            
            # If unknown, handle intelligent capture
            if name == "Unknown Face":
                self.stats['unknown_detected'] += 1
                
                # Get face ID and pose
                face_id = self._get_face_id(face_encoding)
                current_pose = self._estimate_head_pose(face_landmarks)
                
                # Check if we should capture this frame
                if self._should_capture_unknown_face(face_id, current_pose, frame_number):
                    self._capture_unknown_face(frame, face_location, face_id, current_pose)
            
            # Draw bounding box and label
            top, right, bottom, left = face_location
            
            # Draw rectangle
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Draw label background
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            
            # Draw label text
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)
        
        # Cleanup stale trackers periodically
        if frame_number % 100 == 0:
            self._cleanup_stale_trackers(frame_number)
        
        self.stats['frames_processed'] = frame_number
        
        return frame
    
    def display_stats(self):
        """Print current statistics."""
        print("\n" + "="*50)
        print("STATISTICS")
        print("="*50)
        print(f"Frames processed:        {self.stats['frames_processed']}")
        print(f"Registered faces seen:   {self.stats['registered_detected']}")
        print(f"Unknown faces seen:      {self.stats['unknown_detected']}")
        print(f"Images captured:         {self.stats['images_captured']}")
        print(f"Active trackers:         {len(self.unknown_face_trackers)}")
        print("="*50 + "\n")
    
    def run(self):
        """Main loop: open camera, process frames, display results."""
        print("Starting face recognition system...")
        print("Press 'q' to quit, 's' to show statistics\n")
        
        # Open webcam
        video_capture = cv2.VideoCapture(0)
        
        if not video_capture.isOpened():
            print("Error: Could not open webcam")
            return
        
        # Set camera properties for better performance
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        frame_number = 0
        
        try:
            while True:
                # Capture frame
                ret, frame = video_capture.read()
                
                if not ret:
                    print("Error: Failed to capture frame")
                    break
                
                frame_number += 1
                
                # Process frame
                annotated_frame = self.process_frame(frame, frame_number)
                
                # Add info overlay
                info_text = f"Frame: {frame_number} | Registered: {len(self.known_face_encodings)} | " \
                           f"Captured: {self.stats['images_captured']}"
                cv2.putText(annotated_frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Display result
                cv2.imshow('Face Recognition System', annotated_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                elif key == ord('s'):
                    self.display_stats()
        
        finally:
            # Cleanup
            video_capture.release()
            cv2.destroyAllWindows()
            print("\nFinal statistics:")
            self.display_stats()


class FaceTracker:
    """Tracks an unknown face across frames to manage intelligent capture."""
    
    def __init__(self, first_seen_frame, first_seen_time, initial_pose):
        self.first_seen_frame = first_seen_frame
        self.first_seen_time = first_seen_time
        self.last_seen_frame = first_seen_frame
        self.last_seen_time = first_seen_time
        
        # Capture tracking
        self.capture_count = 0
        self.last_capture_time = 0  # Will be set on first capture
        self.last_captured_pose = initial_pose
        self.captured_files = []


def main():
    """Entry point for the face recognition system."""
    
    # Configuration parameters (adjust these as needed)
    config = {
        'registered_faces_dir': 'registered_faces',
        'unknown_faces_dir': 'unknown_faces',
        'face_match_threshold': 0.6,  # Lower = stricter (0.4-0.7 recommended)
        'angle_change_threshold': 15.0,  # Degrees of rotation to trigger capture
        'capture_cooldown_seconds': 3.0,  # Minimum seconds between captures
        'max_captures_per_face': 3  # Maximum captures per unique unknown face
    }
    
    print("="*70)
    print("REAL-TIME FACE RECOGNITION SYSTEM")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Registered faces directory: {config['registered_faces_dir']}")
    print(f"  Unknown faces directory:    {config['unknown_faces_dir']}")
    print(f"  Match threshold:            {config['face_match_threshold']}")
    print(f"  Angle change threshold:     {config['angle_change_threshold']}°")
    print(f"  Capture cooldown:           {config['capture_cooldown_seconds']}s")
    print(f"  Max captures per face:      {config['max_captures_per_face']}")
    print()
    
    # Create and run the system
    system = FaceRecognitionSystem(**config)
    system.run()


if __name__ == "__main__":
    main()
