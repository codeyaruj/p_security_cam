#!/usr/bin/env python3
"""
Setup Verification Script
=========================

This script checks if all dependencies are installed and if the webcam is accessible.
Run this before using the main face recognition system.
"""

import sys

def check_imports():
    """Check if all required packages can be imported."""
    print("Checking Python packages...")
    print("-" * 50)
    
    results = {}
    
    # Check each package
    packages = [
        ('cv2', 'opencv-python'),
        ('face_recognition', 'face_recognition'),
        ('numpy', 'numpy'),
        ('PIL', 'pillow')
    ]
    
    for import_name, package_name in packages:
        try:
            __import__(import_name)
            print(f"✓ {package_name:20} - Installed")
            results[package_name] = True
        except ImportError as e:
            print(f"✗ {package_name:20} - Missing")
            print(f"  Install with: pip install {package_name} --break-system-packages")
            results[package_name] = False
    
    print("-" * 50)
    return all(results.values())

def check_camera():
    """Check if webcam is accessible."""
    print("\nChecking webcam access...")
    print("-" * 50)
    
    try:
        import cv2
        
        # Try to open the default camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("✗ Could not open webcam (index 0)")
            print("  Troubleshooting:")
            print("  - Check if another application is using the camera")
            print("  - Try a different camera index (e.g., 1, 2)")
            print("  - On Linux, check camera permissions")
            return False
        
        # Try to read a frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("✗ Could not read frame from webcam")
            return False
        
        height, width = frame.shape[:2]
        print(f"✓ Webcam accessible")
        print(f"  Resolution: {width}x{height}")
        print(f"  Camera index: 0")
        
    except Exception as e:
        print(f"✗ Error accessing webcam: {e}")
        return False
    
    print("-" * 50)
    return True

def check_directories():
    """Check if required directories exist."""
    print("\nChecking directories...")
    print("-" * 50)
    
    from pathlib import Path
    
    dirs = {
        'registered_faces': 'Directory for registered face images',
        'unknown_faces': 'Directory where unknown faces will be saved'
    }
    
    all_exist = True
    for dir_name, description in dirs.items():
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"✓ {dir_name:20} - Exists")
            
            # Count files if it's registered_faces
            if dir_name == 'registered_faces':
                image_files = list(dir_path.glob('*.jpg')) + \
                             list(dir_path.glob('*.jpeg')) + \
                             list(dir_path.glob('*.png'))
                print(f"  Found {len(image_files)} image(s)")
                
                if len(image_files) == 0:
                    print(f"  ⚠ Warning: No registered face images found")
                    print(f"  Add images like 'john_doe.jpg' to this directory")
        else:
            print(f"✗ {dir_name:20} - Missing")
            print(f"  Creating directory...")
            dir_path.mkdir(exist_ok=True)
            print(f"  ✓ Created {dir_name}/")
            all_exist = False
    
    print("-" * 50)
    return True  # Always return True since we create missing dirs

def check_python_version():
    """Check if Python version is adequate."""
    print("\nChecking Python version...")
    print("-" * 50)
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if version.major >= 3 and version.minor >= 7:
        print(f"✓ Python {version_str} (adequate)")
    else:
        print(f"✗ Python {version_str} (too old)")
        print("  This system requires Python 3.7 or higher")
        return False
    
    print("-" * 50)
    return True

def main():
    """Run all checks."""
    print("=" * 50)
    print("FACE RECOGNITION SYSTEM - SETUP VERIFICATION")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_imports),
        ("Webcam", check_camera),
        ("Directories", check_directories)
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n✗ Error during {name} check: {e}")
            results[name] = False
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} - {name}")
    
    print("=" * 50)
    
    if all(results.values()):
        print("\n✓ All checks passed! You're ready to run the face recognition system.")
        print("  Run: python face_recognition_system.py")
    else:
        print("\n✗ Some checks failed. Please fix the issues above before proceeding.")
        sys.exit(1)

if __name__ == "__main__":
    main()
