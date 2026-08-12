import face_recognition
import cv2
import pickle
import numpy as np
import os
from pathlib import Path

class MessStudentDatabase:
    """Manages student enrollment and database operations"""
    
    def __init__(self, db_file='mess_students.pkl'):
        self.db_file = db_file
        self.students = {}
        self.load_database()
    
    def load_database(self):
        """Load existing student database"""
        if os.path.exists(self.db_file):
            with open(self.db_file, 'rb') as f:
                self.students = pickle.load(f)
            print(f"Loaded {len(self.students)} students from database")
        else:
            print("No existing database found. Starting fresh.")
    
    def save_database(self):
        """Save student database to file"""
        with open(self.db_file, 'wb') as f:
            pickle.dump(self.students, f)
        print(f"Database saved with {len(self.students)} students")
    
    def enroll_student(self, roll_no, name, department, image_path):
        """
        Enroll a new student with their face encoding
        
        Args:
            roll_no: Roll number in format '2022bit050', '2023ece001'
            name: Student's full name
            department: Department name
            image_path: Path to student's photo
        """
        # Load and encode the face
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)
        
        if len(face_encodings) == 0:
            print(f"ERROR: No face detected in {image_path}")
            return False
        elif len(face_encodings) > 1:
            print(f"WARNING: Multiple faces detected in {image_path}. Using first face.")
        
        # Store student information
        self.students[roll_no] = {
            'name': name,
            'department': department,
            'roll_no': roll_no,
            'encoding': face_encodings[0]
        }
        
        print(f"✓ Enrolled: {name} ({roll_no}) - {department}")
        return True
    
    def get_all_encodings(self):
        """Get all face encodings and corresponding roll numbers"""
        if not self.students:
            return [], []
        
        encodings = [student['encoding'] for student in self.students.values()]
        roll_nos = list(self.students.keys())
        return encodings, roll_nos


class FaceRecognitionSystem:
    """Real-time face recognition system for mess management"""
    
    def __init__(self, database):
        self.database = database
        self.known_encodings, self.known_roll_nos = database.get_all_encodings()
        
        # Colors for drawing boxes (BGR format)
        self.RECOGNIZED_COLOR = (0, 255, 0)  # Green
        self.UNRECOGNIZED_COLOR = (0, 0, 255)  # Red
        
        # Recognition parameters for high accuracy
        self.tolerance = 0.5  # Lower = stricter matching (range: 0.4-0.6)
        self.model = 'cnn'  # Use 'cnn' for better accuracy (requires GPU) or 'hog' for speed
    
    def recognize_faces(self, frame):
        """
        Detect and recognize faces in a frame
        
        Args:
            frame: Video frame from camera
            
        Returns:
            Processed frame with bounding boxes and labels
        """
        # Resize frame for faster processing (optional)
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces and get encodings
        face_locations = face_recognition.face_locations(rgb_small_frame, model='hog')
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        face_data = []
        
        # Process each detected face
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Scale back up face locations
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2
            
            # Compare with known faces
            matches = face_recognition.compare_faces(
                self.known_encodings, 
                face_encoding, 
                tolerance=self.tolerance
            )
            
            # Calculate face distances for better matching
            face_distances = face_recognition.face_distance(
                self.known_encodings, 
                face_encoding
            )
            
            name = "Not Recognised"
            roll_no = ""
            is_recognized = False
            
            # Find best match
            if len(face_distances) > 0 and True in matches:
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index]:
                    roll_no = self.known_roll_nos[best_match_index]
                    student = self.database.students[roll_no]
                    name = student['name']
                    is_recognized = True
            
            face_data.append({
                'box': (left, top, right, bottom),
                'name': name,
                'roll_no': roll_no,
                'recognized': is_recognized
            })
        
        # Draw boxes and labels
        processed_frame = self.draw_labels(frame, face_data)
        return processed_frame
    
    def draw_labels(self, frame, face_data):
        """Draw bounding boxes and labels on frame"""
        for face in face_data:
            left, top, right, bottom = face['box']
            color = self.RECOGNIZED_COLOR if face['recognized'] else self.UNRECOGNIZED_COLOR
            
            # Draw rectangle around face
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            if face['recognized']:
                # Draw name above the box
                name_bg_height = 30
                cv2.rectangle(
                    frame, 
                    (left, top - name_bg_height), 
                    (right, top), 
                    color, 
                    cv2.FILLED
                )
                cv2.putText(
                    frame, 
                    face['name'], 
                    (left + 6, top - 10), 
                    cv2.FONT_HERSHEY_DUPLEX, 
                    0.6, 
                    (255, 255, 255), 
                    1
                )
                
                # Draw roll number below the box
                roll_bg_height = 30
                cv2.rectangle(
                    frame, 
                    (left, bottom), 
                    (right, bottom + roll_bg_height), 
                    color, 
                    cv2.FILLED
                )
                cv2.putText(
                    frame, 
                    face['roll_no'], 
                    (left + 6, bottom + 20), 
                    cv2.FONT_HERSHEY_DUPLEX, 
                    0.6, 
                    (255, 255, 255), 
                    1
                )
            else:
                # Draw "Not Recognised" label
                label_bg_height = 30
                cv2.rectangle(
                    frame, 
                    (left, top - label_bg_height), 
                    (right, top), 
                    color, 
                    cv2.FILLED
                )
                cv2.putText(
                    frame, 
                    "Not Recognised", 
                    (left + 6, top - 10), 
                    cv2.FONT_HERSHEY_DUPLEX, 
                    0.6, 
                    (255, 255, 255), 
                    1
                )
        
        return frame
    
    def start_recognition(self):
        """Start real-time face recognition from webcam"""
        print("Starting face recognition system...")
        print(f"Loaded {len(self.known_encodings)} students")
        print("Press 'q' to quit")
        
        video_capture = cv2.VideoCapture(0)
        
        # Set camera resolution for better quality
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        while True:
            ret, frame = video_capture.read()
            
            if not ret:
                print("Failed to grab frame")
                break
            
            # Process frame
            processed_frame = self.recognize_faces(frame)
            
            # Display instructions
            cv2.putText(
                processed_frame, 
                "Press 'q' to quit", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 255, 255), 
                2
            )
            
            # Show frame
            cv2.imshow('Mess Face Recognition System', processed_frame)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        video_capture.release()
        cv2.destroyAllWindows()
        print("System stopped")


def main():
    """Main function to demonstrate enrollment and recognition"""
    
    # Initialize database
    db = MessStudentDatabase('mess_students.pkl')
    
    # Example: Enroll students
    print("\n=== ENROLLMENT MODE ===")
    print("Enrolling students...")
    
    # Create a sample enrollment directory structure if needed
    # Format: enrollment_photos/roll_no.jpg
    
    # Example enrollments (replace with actual paths)
    students_to_enroll = [
        {
            'roll_no': '2022bit050',
            'name': 'Rajesh Kumar',
            'department': 'BIT',
            'image_path': 'enrollment_photos/2022bit050.jpg'
        },
        {
            'roll_no': '2023ece001',
            'name': 'Priya Sharma',
            'department': 'ECE',
            'image_path': 'enrollment_photos/2023ece001.jpg'
        },
        # Add more students here
    ]
    
    # Enroll each student
    for student in students_to_enroll:
        if os.path.exists(student['image_path']):
            db.enroll_student(
                student['roll_no'],
                student['name'],
                student['department'],
                student['image_path']
            )
    
    # Save database after enrollment
    db.save_database()
    
    # Start recognition system
    print("\n=== RECOGNITION MODE ===")
    recognition_system = FaceRecognitionSystem(db)
    recognition_system.start_recognition()


if __name__ == "__main__":
    main()
