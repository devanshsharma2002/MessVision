import face_recognition
import cv2
import numpy as np

print("✓ face_recognition version:", face_recognition.__version__)
print("✓ OpenCV version:", cv2.__version__)
print("✓ NumPy version:", np.__version__)

# Test face detection
print("\nTesting face detection...")
test_image = np.zeros((100, 100, 3), dtype=np.uint8)
try:
    face_locations = face_recognition.face_locations(test_image)
    print("✓ Face detection working!")
except Exception as e:
    print("✗ Error:", e)

print("\nAll libraries installed successfully!")
