import face_recognition
import cv2
import numpy as np
import os

print("=" * 50)
print("FACE RECOGNITION INSTALLATION TEST")
print("=" * 50)

print("\n✓ face_recognition version:", face_recognition.__version__)
print("✓ OpenCV version:", cv2.__version__)
print("✓ NumPy version:", np.__version__)

# Test 1: Face detection with webcam
print("\n" + "=" * 50)
print("TEST 1: Webcam Face Detection")
print("=" * 50)

try:
    video_capture = cv2.VideoCapture(0)
    
    if not video_capture.isOpened():
        print("✗ ERROR: Cannot open webcam")
    else:
        print("✓ Webcam opened successfully")
        
        # Capture one frame
        ret, frame = video_capture.read()
        
        if ret:
            print("✓ Frame captured successfully")
            print(f"  Frame shape: {frame.shape}")
            print(f"  Frame dtype: {frame.dtype}")
            
            # Convert to RGB and make contiguous
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame = np.ascontiguousarray(rgb_frame)
            
            print("\n  Testing face detection...")
            face_locations = face_recognition.face_locations(rgb_frame, model='hog')
            
            print(f"✓ Face detection working!")
            print(f"  Detected {len(face_locations)} face(s) in frame")
            
            if len(face_locations) > 0:
                print("\n  Testing face encoding...")
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                print(f"✓ Face encoding working!")
                print(f"  Generated {len(face_encodings)} face encoding(s)")
            else:
                print("\n  ⚠ No faces detected in frame (this is OK if you're not in front of camera)")
        else:
            print("✗ Failed to capture frame")
        
        video_capture.release()
        
except Exception as e:
    print(f"✗ Error during webcam test: {e}")

# Test 2: Load and process an image file (if available)
print("\n" + "=" * 50)
print("TEST 2: Image File Processing")
print("=" * 50)

test_image_path = "test_face.jpg"

if os.path.exists(test_image_path):
    try:
        print(f"Loading test image: {test_image_path}")
        image = face_recognition.load_image_file(test_image_path)
        
        print(f"✓ Image loaded successfully")
        print(f"  Image shape: {image.shape}")
        print(f"  Image dtype: {image.dtype}")
        
        face_locations = face_recognition.face_locations(image)
        print(f"✓ Detected {len(face_locations)} face(s)")
        
        if len(face_locations) > 0:
            face_encodings = face_recognition.face_encodings(image, face_locations)
            print(f"✓ Generated {len(face_encodings)} face encoding(s)")
    except Exception as e:
        print(f"✗ Error processing image: {e}")
else:
    print(f"⚠ Test image not found: {test_image_path}")
    print("  (This is optional - webcam test is sufficient)")

# Test 3: Create a simple colored test image
print("\n" + "=" * 50)
print("TEST 3: Synthetic Image Processing")
print("=" * 50)

try:
    # Create a proper RGB image
    test_img = np.ones((480, 640, 3), dtype=np.uint8) * 128  # Gray image
    test_img = np.ascontiguousarray(test_img)
    
    print("✓ Created synthetic test image")
    print(f"  Shape: {test_img.shape}")
    print(f"  Dtype: {test_img.dtype}")
    print(f"  Is contiguous: {test_img.flags['C_CONTIGUOUS']}")
    
    face_locations = face_recognition.face_locations(test_img)
    print(f"✓ Face detection completed (found {len(face_locations)} faces)")
    print("  (Empty image expected to have 0 faces)")
    
except Exception as e:
    print(f"✗ Error with synthetic image: {e}")

# Summary
print("\n" + "=" * 50)
print("INSTALLATION TEST SUMMARY")
print("=" * 50)
print("✓ All core libraries installed successfully!")
print("\nRecommendation:")
if np.__version__.startswith('2.'):
    print("⚠ NumPy 2.x detected. Consider downgrading to 1.24.3 for better compatibility:")
    print("  pip uninstall numpy")
    print("  pip install numpy==1.24.3")
else:
    print("✓ NumPy version is compatible")

print("\nYou can now run the face recognition system!")
print("=" * 50)
