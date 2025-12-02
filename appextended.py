import face_recognition
import cv2
import pickle
import numpy as np
import os
from pathlib import Path
from datetime import datetime

class StudentDatabase:
    """Manages college and mess student enrollment"""
    
    def __init__(self, college_db='college_students.pkl', mess_db='mess_students.pkl'):
        self.college_db_file = college_db
        self.mess_db_file = mess_db
        self.college_students = {}
        self.mess_students = {}
        self.load_databases()
        
        # Create directories for saving unknown faces
        self.create_save_directories()
    
    def create_save_directories(self):
        """Create folders for saving detected faces"""
        directories = [
            'detected_faces/outsiders',
            'detected_faces/college_non_mess'
        ]
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Created directory: {directory}")
    
    def load_databases(self):
        """Load existing databases"""
        # Load college database
        if os.path.exists(self.college_db_file):
            with open(self.college_db_file, 'rb') as f:
                self.college_students = pickle.load(f)
            print(f"Loaded {len(self.college_students)} college students")
        else:
            print("No college database found. Starting fresh.")
        
        # Load mess database
        if os.path.exists(self.mess_db_file):
            with open(self.mess_db_file, 'rb') as f:
                self.mess_students = pickle.load(f)
            print(f"Loaded {len(self.mess_students)} mess students")
        else:
            print("No mess database found. Starting fresh.")
    
    def save_databases(self):
        """Save both databases"""
        with open(self.college_db_file, 'wb') as f:
            pickle.dump(self.college_students, f)
        
        with open(self.mess_db_file, 'wb') as f:
            pickle.dump(self.mess_students, f)
        
        print(f"Databases saved:")
        print(f"  - College: {len(self.college_students)} students")
        print(f"  - Mess: {len(self.mess_students)} students")
    
    def enroll_student(self, roll_no, name, department, image_path, is_mess_student=False):
        """
        Enroll a student in college and optionally in mess
        
        Args:
            roll_no: Roll number (e.g., '2022bit050')
            name: Student's full name
            department: Department name
            image_path: Path to student's photo
            is_mess_student: True if student is enrolled in mess
        """
        # Load and encode the face
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)
        
        if len(face_encodings) == 0:
            print(f"ERROR: No face detected in {image_path}")
            return False
        elif len(face_encodings) > 1:
            print(f"WARNING: Multiple faces in {image_path}. Using first face.")
        
        student_data = {
            'name': name,
            'department': department,
            'roll_no': roll_no,
            'encoding': face_encodings[0]
        }
        
        # Add to college database
        self.college_students[roll_no] = student_data
        
        # Add to mess database if enrolled
        if is_mess_student:
            self.mess_students[roll_no] = student_data
            print(f"✓ Enrolled: {name} ({roll_no}) - {department} [MESS STUDENT]")
        else:
            print(f"✓ Enrolled: {name} ({roll_no}) - {department} [COLLEGE ONLY]")
        
        return True
    
    def get_all_encodings(self):
        """Get all encodings and roll numbers from both databases"""
        college_encodings = [s['encoding'] for s in self.college_students.values()]
        college_roll_nos = list(self.college_students.keys())
        
        mess_encodings = [s['encoding'] for s in self.mess_students.values()]
        mess_roll_nos = list(self.mess_students.keys())
        
        return {
            'college': (college_encodings, college_roll_nos),
            'mess': (mess_encodings, mess_roll_nos)
        }


class EnhancedFaceRecognitionSystem:
    """Three-tier face recognition: Mess / College / Outsider"""
    
    def __init__(self, database):
        self.database = database
        self.encodings = database.get_all_encodings()
        
        # Colors for three categories (BGR format)
        self.MESS_COLOR = (0, 255, 0)           # Green
        self.COLLEGE_COLOR = (0, 165, 255)      # Orange
        self.OUTSIDER_COLOR = (0, 0, 255)       # Red
        
        # Recognition parameters
        self.tolerance = 0.5
        
        # Track saved faces to avoid duplicates
        self.saved_faces = set()
    
    def recognize_faces(self, frame):
        """Detect and classify faces into three categories"""
        if frame is None or frame.size == 0:
            return frame
        
        # Resize for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        rgb_small_frame = np.ascontiguousarray(rgb_small_frame)
        
        try:
            face_locations = face_recognition.face_locations(rgb_small_frame, model='hog')
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        except Exception as e:
            print(f"Error during face detection: {e}")
            return frame
        
        face_data = []
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Scale back up
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2
            
            category, name, roll_no = self.classify_face(face_encoding)
            
            # Save face if outsider or college non-mess
            if category in ['outsider', 'college']:
                self.save_detected_face(frame, (left, top, right, bottom), category)
            
            face_data.append({
                'box': (left, top, right, bottom),
                'name': name,
                'roll_no': roll_no,
                'category': category
            })
        
        processed_frame = self.draw_labels(frame, face_data)
        return processed_frame
    
    def classify_face(self, face_encoding):
        """
        Classify face into three categories
        Returns: (category, name, roll_no)
        """
        # First check mess database
        mess_encodings, mess_roll_nos = self.encodings['mess']
        
        if len(mess_encodings) > 0:
            matches = face_recognition.compare_faces(
                mess_encodings, face_encoding, tolerance=self.tolerance
            )
            face_distances = face_recognition.face_distance(mess_encodings, face_encoding)
            
            if len(face_distances) > 0 and True in matches:
                best_match_idx = np.argmin(face_distances)
                if matches[best_match_idx]:
                    roll_no = mess_roll_nos[best_match_idx]
                    student = self.database.mess_students[roll_no]
                    return ('mess', student['name'], roll_no)
        
        # Check college database
        college_encodings, college_roll_nos = self.encodings['college']
        
        if len(college_encodings) > 0:
            matches = face_recognition.compare_faces(
                college_encodings, face_encoding, tolerance=self.tolerance
            )
            face_distances = face_recognition.face_distance(college_encodings, face_encoding)
            
            if len(face_distances) > 0 and True in matches:
                best_match_idx = np.argmin(face_distances)
                if matches[best_match_idx]:
                    roll_no = college_roll_nos[best_match_idx]
                    student = self.database.college_students[roll_no]
                    return ('college', student['name'], roll_no)
        
        # Not found in any database
        return ('outsider', 'Outsider', 'UNKNOWN')
    
    def save_detected_face(self, frame, box, category):
        """Save detected face to appropriate folder with padding"""
        left, top, right, bottom = box
    
        # Create unique identifier for this face position
        face_id = f"{left}_{top}_{right}_{bottom}"
    
    # Avoid saving the same face multiple times in same session
        if face_id in self.saved_faces:
            return
    
    # Add padding around the face (adjustable)
        padding_percent = 0.4  # 40% padding (increase for more zoom out)
    
    # Calculate face dimensions
        face_width = right - left
        face_height = bottom - top
    
    # Calculate padding in pixels
        pad_x = int(face_width * padding_percent)
        pad_y = int(face_height * padding_percent)
    
    # Get frame dimensions
        frame_height, frame_width = frame.shape[:2]
    
    # Apply padding with boundary checks
        left_padded = max(0, left - pad_x)
        top_padded = max(0, top - pad_y)
        right_padded = min(frame_width, right + pad_x)
        bottom_padded = min(frame_height, bottom + pad_y)
    
    # Crop face with padding
        face_crop = frame[top_padded:bottom_padded, left_padded:right_padded]
    
        if face_crop.size == 0:
            return
    
    # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
        if category == 'outsider':
            filename = f"detected_faces/outsiders/outsider_{timestamp}.jpg"
        else:  # college non-mess
            filename = f"detected_faces/college_non_mess/student_{timestamp}.jpg"
    
    # Save the face with padding
        cv2.imwrite(filename, face_crop)
        self.saved_faces.add(face_id)
        print(f"📸 Saved {category} face: {filename}")

    
    def draw_labels(self, frame, face_data):
        """Draw colored boxes and labels based on category"""
        for face in face_data:
            left, top, right, bottom = face['box']
            category = face['category']
            
            # Select color based on category
            if category == 'mess':
                color = self.MESS_COLOR
                label_top = face['name']
                label_bottom = face['roll_no']
            elif category == 'college':
                color = self.COLLEGE_COLOR
                label_top = face['name']
                label_bottom = f"{face['roll_no']} (NO MESS)"
            else:  # outsider
                color = self.OUTSIDER_COLOR
                label_top = "OUTSIDER"
                label_bottom = "NOT AUTHORIZED"
            
            # Draw rectangle
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Draw top label (name)
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
                label_top,
                (left + 6, top - 10),
                cv2.FONT_HERSHEY_DUPLEX,
                0.6,
                (255, 255, 255),
                1
            )
            
            # Draw bottom label (roll number / status)
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
                label_bottom,
                (left + 6, bottom + 20),
                cv2.FONT_HERSHEY_DUPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        return frame
    
    def start_recognition(self):
        """Start real-time recognition system"""
        print("\n" + "="*60)
        print("MESS FACE RECOGNITION SYSTEM - THREE-TIER CLASSIFICATION")
        print("="*60)
        print(f"✓ Mess Students: {len(self.encodings['mess'][0])}")
        print(f"✓ College Students: {len(self.encodings['college'][0])}")
        print("\nColor Legend:")
        print("  🟢 GREEN  = Mess Student (Authorized)")
        print("  🟠 ORANGE = College Student (No Mess)")
        print("  🔴 RED    = Outsider (Not Authorized)")
        print("\nPress 'q' to quit")
        print("="*60 + "\n")
        
        video_capture = cv2.VideoCapture(0)
        
        if not video_capture.isOpened():
            print("ERROR: Could not open webcam")
            return
        
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        frame_count = 0
        
        while True:
            ret, frame = video_capture.read()
            
            if not ret:
                print("Failed to grab frame")
                break
            
            frame_count += 1
            processed_frame = self.recognize_faces(frame)
            
            # Display stats
            info_text = f"Mess: {len(self.encodings['mess'][0])} | College: {len(self.encodings['college'][0])} | Press 'q' to quit"
            cv2.putText(
                processed_frame,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            cv2.imshow('Mess Recognition System - 3-Tier Classification', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        video_capture.release()
        cv2.destroyAllWindows()
        print(f"\nSystem stopped. Processed {frame_count} frames")
        print(f"Saved faces: {len(self.saved_faces)}")


def main():
    """Main enrollment and recognition function"""
    
    # Initialize database
    db = StudentDatabase()
    
    print("\n" + "="*60)
    print("ENROLLMENT MODE")
    print("="*60)
    
    # Create enrollment directory if needed
    enrollment_dir = 'enrollment_photos'
    if not os.path.exists(enrollment_dir):
        os.makedirs(enrollment_dir)
        print(f"Created '{enrollment_dir}' directory")
    
    # ENROLLMENT LIST - Update this with your students
    students_to_enroll = [
        # Example 1: Mess student
        {
            'roll_no': '2022bit050',
            'name': 'devansh',
            'department': 'IT',
            'image_path': 'enrollment_photos/2022bit050.jpg',
            'is_mess_student': False  # Enrolled in mess
        },
        # Example 2: College student (no mess)
        {
            'roll_no': '2022ece010',
            'name': 'amul Sharma',
            'department': 'IT',
            'image_path': 'enrollment_photos/2022ece010s.jpg',
            'is_mess_student': False  # College only, no mess
        },
        # Example 3: Another mess student
        {
            'roll_no': '2023ece001',
            'name': 'Priya Singh',
            'department': 'ECE',
            'image_path': 'enrollment_photos/2023ece001.jpg',
            'is_mess_student': True  # Enrolled in mess
        },
        # Add more students here...
    ]
    
    if len(students_to_enroll) == 0:
        print("⚠ No students configured for enrollment")
    else:
        print(f"\nEnrolling {len(students_to_enroll)} students...\n")
        enrolled = 0
        
        for student in students_to_enroll:
            if os.path.exists(student['image_path']):
                if db.enroll_student(
                    student['roll_no'],
                    student['name'],
                    student['department'],
                    student['image_path'],
                    student.get('is_mess_student', False)
                ):
                    enrolled += 1
            else:
                print(f"✗ File not found: {student['image_path']}")
        
        print(f"\n✓ Successfully enrolled {enrolled}/{len(students_to_enroll)} students")
    
    # Save databases
    db.save_databases()
    
    # Start recognition
    print("\n" + "="*60)
    print("STARTING RECOGNITION SYSTEM")
    print("="*60)
    
    recognition_system = EnhancedFaceRecognitionSystem(db)
    recognition_system.start_recognition()


if __name__ == "__main__":
    main()
