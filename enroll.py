import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import face_recognition
import pickle
import os

class EnrollmentGUI:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Mess Student Enrollment System")
        self.window.geometry("1000x750")
        self.window.configure(bg='#f0f0f0')
        
        # Variables
        self.camera_active = False
        self.cap = None
        self.current_frame = None
        self.selected_image_path = None
        
        # Database paths
        self.college_db = 'college_students.pkl'
        self.mess_db = 'mess_students.pkl'
        
        # Create UI
        self.create_ui()
        
    def create_ui(self):
        # Title
        title_label = tk.Label(
            self.window, 
            text="Student Enrollment System", 
            font=("Arial", 24, "bold"),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        title_label.pack(pady=20)
        
        # Top buttons for database management
        top_btn_frame = tk.Frame(self.window, bg='#f0f0f0')
        top_btn_frame.pack(pady=5)
        
        view_btn = tk.Button(
            top_btn_frame,
            text="👥 View Students",
            command=self.view_students,
            font=("Arial", 10, "bold"),
            bg='#3498db',
            fg='white',
            padx=10,
            pady=5,
            cursor='hand2'
        )
        view_btn.grid(row=0, column=0, padx=5)
        
        clear_btn = tk.Button(
            top_btn_frame,
            text="🗑️ Clear All Data",
            command=self.clear_all_data,
            font=("Arial", 10, "bold"),
            bg='#e74c3c',
            fg='white',
            padx=10,
            pady=5,
            cursor='hand2'
        )
        clear_btn.grid(row=0, column=1, padx=5)
        
        stats_btn = tk.Button(
            top_btn_frame,
            text="📊 Show Stats",
            command=self.show_stats,
            font=("Arial", 10, "bold"),
            bg='#f39c12',
            fg='white',
            padx=10,
            pady=5,
            cursor='hand2'
        )
        stats_btn.grid(row=0, column=2, padx=5)
        
        # Main container
        main_frame = tk.Frame(self.window, bg='#f0f0f0')
        main_frame.pack(pady=10, padx=20)
        
        # Left side - Form
        form_frame = tk.LabelFrame(
            main_frame, 
            text="Student Details", 
            font=("Arial", 14, "bold"),
            bg='#ffffff',
            padx=20,
            pady=20
        )
        form_frame.grid(row=0, column=0, padx=10, sticky='nsew')
        
        # Roll Number
        tk.Label(form_frame, text="Roll Number:", font=("Arial", 12), bg='#ffffff').grid(row=0, column=0, sticky='w', pady=10)
        self.roll_entry = tk.Entry(form_frame, font=("Arial", 12), width=20)
        self.roll_entry.grid(row=0, column=1, pady=10, padx=5)
        
        # Name
        tk.Label(form_frame, text="Full Name:", font=("Arial", 12), bg='#ffffff').grid(row=1, column=0, sticky='w', pady=10)
        self.name_entry = tk.Entry(form_frame, font=("Arial", 12), width=20)
        self.name_entry.grid(row=1, column=1, pady=10, padx=5)
        
        # Department
        tk.Label(form_frame, text="Department:", font=("Arial", 12), bg='#ffffff').grid(row=2, column=0, sticky='w', pady=10)
        self.dept_entry = tk.Entry(form_frame, font=("Arial", 12), width=20)
        self.dept_entry.grid(row=2, column=1, pady=10, padx=5)
        
        # Mess Status
        tk.Label(form_frame, text="Mess Enrollment:", font=("Arial", 12), bg='#ffffff').grid(row=3, column=0, sticky='w', pady=10)
        self.mess_var = tk.BooleanVar(value=True)
        mess_check = tk.Checkbutton(
            form_frame, 
            text="Enrolled in Mess", 
            variable=self.mess_var,
            font=("Arial", 11),
            bg='#ffffff'
        )
        mess_check.grid(row=3, column=1, sticky='w', pady=10)
        
        # Right side - Photo capture
        photo_frame = tk.LabelFrame(
            main_frame, 
            text="Student Photo", 
            font=("Arial", 14, "bold"),
            bg='#ffffff',
            padx=20,
            pady=20
        )
        photo_frame.grid(row=0, column=1, padx=10, sticky='nsew')
        
        # Canvas for image display
        self.canvas = tk.Canvas(photo_frame, width=400, height=300, bg='#e0e0e0', highlightthickness=1)
        self.canvas.pack(pady=10)
        
        # Placeholder text
        self.canvas.create_text(200, 150, text="No photo", font=("Arial", 14), fill='gray', tags='placeholder')
        
        # Button frame
        button_frame = tk.Frame(photo_frame, bg='#ffffff')
        button_frame.pack(pady=10)
        
        # Capture from webcam button
        self.camera_btn = tk.Button(
            button_frame,
            text="📷 Start Camera",
            command=self.toggle_camera,
            font=("Arial", 11, "bold"),
            bg='#3498db',
            fg='white',
            padx=15,
            pady=8,
            cursor='hand2'
        )
        self.camera_btn.grid(row=0, column=0, padx=5, pady=5)
        
        # Upload image button
        upload_btn = tk.Button(
            button_frame,
            text="📁 Upload Image",
            command=self.upload_image,
            font=("Arial", 11, "bold"),
            bg='#9b59b6',
            fg='white',
            padx=15,
            pady=8,
            cursor='hand2'
        )
        upload_btn.grid(row=0, column=1, padx=5, pady=5)
        
        # Capture snapshot button
        self.snapshot_btn = tk.Button(
            button_frame,
            text="📸 Take Photo",
            command=self.capture_snapshot,
            font=("Arial", 11, "bold"),
            bg='#27ae60',
            fg='white',
            padx=15,
            pady=8,
            cursor='hand2',
            state='disabled'
        )
        self.snapshot_btn.grid(row=1, column=0, columnspan=2, pady=5)
        
        # Enroll button
        enroll_btn = tk.Button(
            self.window,
            text="✓ ENROLL STUDENT",
            command=self.enroll_student,
            font=("Arial", 14, "bold"),
            bg='#2ecc71',
            fg='white',
            padx=40,
            pady=15,
            cursor='hand2'
        )
        enroll_btn.pack(pady=20)
        
        # Status label
        self.status_label = tk.Label(
            self.window,
            text="Ready to enroll students",
            font=("Arial", 11),
            bg='#f0f0f0',
            fg='#7f8c8d'
        )
        self.status_label.pack(pady=10)
        
    def toggle_camera(self):
        if not self.camera_active:
            self.start_camera()
        else:
            self.stop_camera()
    
    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            self.camera_active = True
            self.camera_btn.config(text="⏹ Stop Camera", bg='#e74c3c')
            self.snapshot_btn.config(state='normal')
            self.update_camera()
            self.status_label.config(text="Camera active - Click 'Take Photo' when ready", fg='#3498db')
        else:
            messagebox.showerror("Error", "Cannot access webcam!")
    
    def stop_camera(self):
        if self.cap:
            self.camera_active = False
            self.cap.release()
            self.camera_btn.config(text="📷 Start Camera", bg='#3498db')
            self.snapshot_btn.config(state='disabled')
            self.status_label.config(text="Camera stopped", fg='#7f8c8d')
    
    def update_camera(self):
        if self.camera_active and self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                
                # Convert to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize to canvas size
                frame_resized = cv2.resize(frame_rgb, (400, 300))
                
                # Convert to PIL and then to ImageTk
                img = Image.fromarray(frame_resized)
                photo = ImageTk.PhotoImage(image=img)
                
                # Display on canvas
                self.canvas.delete('all')
                self.canvas.create_image(200, 150, image=photo)
                self.canvas.photo = photo
            
            self.window.after(30, self.update_camera)
    
    def capture_snapshot(self):
        if self.current_frame is not None:
            # Stop camera
            self.stop_camera()
            
            # Save temp file
            temp_path = "temp_capture.jpg"
            cv2.imwrite(temp_path, self.current_frame)
            self.selected_image_path = temp_path
            
            # Display captured frame
            frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (400, 300))
            img = Image.fromarray(frame_resized)
            photo = ImageTk.PhotoImage(image=img)
            
            self.canvas.delete('all')
            self.canvas.create_image(200, 150, image=photo)
            self.canvas.photo = photo
            
            self.status_label.config(text="✓ Photo captured! Fill details and enroll", fg='#27ae60')
            messagebox.showinfo("Success", "Photo captured successfully!")
    
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Student Photo",
            filetypes=[("Image files", "*.png *.jpg *.jpeg")]
        )
        
        if file_path:
            # Stop camera if active
            if self.camera_active:
                self.stop_camera()
            
            self.selected_image_path = file_path
            
            # Load and display image
            img = Image.open(file_path)
            img = img.resize((400, 300))
            photo = ImageTk.PhotoImage(image=img)
            
            self.canvas.delete('all')
            self.canvas.create_image(200, 150, image=photo)
            self.canvas.photo = photo
            
            self.status_label.config(text="✓ Image uploaded! Fill details and enroll", fg='#27ae60')
    
    def enroll_student(self):
        # Validate inputs
        roll_no = self.roll_entry.get().strip()
        name = self.name_entry.get().strip()
        department = self.dept_entry.get().strip()
        
        if not roll_no or not name or not department:
            messagebox.showerror("Error", "Please fill all fields!")
            return
        
        if not self.selected_image_path:
            messagebox.showerror("Error", "Please capture or upload a photo!")
            return
        
        try:
            # Load and encode face
            self.status_label.config(text="Processing...", fg='#f39c12')
            self.window.update()
            
            image = face_recognition.load_image_file(self.selected_image_path)
            face_encodings = face_recognition.face_encodings(image)
            
            if len(face_encodings) == 0:
                messagebox.showerror("Error", "No face detected! Try again.")
                return
            
            if len(face_encodings) > 1:
                messagebox.showwarning("Warning", "Multiple faces detected. Using first.")
            
            # Load databases
            college_students = {}
            mess_students = {}
            
            if os.path.exists(self.college_db):
                with open(self.college_db, 'rb') as f:
                    college_students = pickle.load(f)
            
            if os.path.exists(self.mess_db):
                with open(self.mess_db, 'rb') as f:
                    mess_students = pickle.load(f)
            
            # Check duplicate
            if roll_no in college_students:
                if not messagebox.askyesno("Exists", f"{roll_no} already enrolled. Update?"):
                    return
            
            # Create record
            student_data = {
                'name': name,
                'department': department,
                'roll_no': roll_no,
                'encoding': face_encodings[0]
            }
            
            # Add to databases
            college_students[roll_no] = student_data
            
            is_mess = self.mess_var.get()
            if is_mess:
                mess_students[roll_no] = student_data
            
            # Save databases
            with open(self.college_db, 'wb') as f:
                pickle.dump(college_students, f)
            
            with open(self.mess_db, 'wb') as f:
                pickle.dump(mess_students, f)
            
            # Save photo
            if not os.path.exists('enrollment_photos'):
                os.makedirs('enrollment_photos')
            
            enrollment_path = f"enrollment_photos/{roll_no}.jpg"
            
            if self.selected_image_path == "temp_capture.jpg":
                os.rename("temp_capture.jpg", enrollment_path)
            else:
                img = Image.open(self.selected_image_path)
                img.save(enrollment_path)
            
            # Success
            mess_status = "MESS" if is_mess else "COLLEGE ONLY"
            messagebox.showinfo(
                "Success", 
                f"✓ {name} enrolled!\n\nRoll: {roll_no}\nStatus: {mess_status}\n\nTotal: {len(college_students)} students"
            )
            
            # Clear form
            self.clear_form()
            self.status_label.config(text=f"✓ {name} enrolled! Ready for next.", fg='#27ae60')
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed:\n{str(e)}")
            self.status_label.config(text="Enrollment failed", fg='#e74c3c')
    
    def view_students(self):
        """View and delete enrolled students"""
        view_window = tk.Toplevel(self.window)
        view_window.title("Enrolled Students")
        view_window.geometry("600x500")
        view_window.configure(bg='#f0f0f0')
        
        # Load databases
        college_students = {}
        mess_students = {}
        
        if os.path.exists(self.college_db):
            with open(self.college_db, 'rb') as f:
                college_students = pickle.load(f)
        
        if os.path.exists(self.mess_db):
            with open(self.mess_db, 'rb') as f:
                mess_students = pickle.load(f)
        
        # Title
        tk.Label(
            view_window,
            text="Enrolled Students",
            font=("Arial", 18, "bold"),
            bg='#f0f0f0'
        ).pack(pady=10)
        
        # Stats
        tk.Label(
            view_window,
            text=f"Total College: {len(college_students)} | Mess: {len(mess_students)}",
            font=("Arial", 12),
            bg='#f0f0f0',
            fg='#7f8c8d'
        ).pack(pady=5)
        
        # Frame for listbox and scrollbar
        list_frame = tk.Frame(view_window)
        list_frame.pack(pady=10, padx=20, fill='both', expand=True)
        
        # Scrollbar
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side='right', fill='y')
        
        # Listbox
        student_list = tk.Listbox(
            list_frame,
            font=("Courier", 10),
            yscrollcommand=scrollbar.set,
            selectmode='single'
        )
        student_list.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=student_list.yview)
        
        # Populate list
        for roll_no, data in sorted(college_students.items()):
            mess_tag = "✓ MESS" if roll_no in mess_students else "✗ NO MESS"
            student_list.insert('end', f"{roll_no} | {data['name']:20s} | {data['department']:5s} | {mess_tag}")
        
        # Delete button
        def delete_selected():
            selection = student_list.curselection()
            if not selection:
                messagebox.showwarning("Warning", "Please select a student to delete!")
                return
            
            selected_text = student_list.get(selection[0])
            roll_no = selected_text.split('|')[0].strip()
            
            if messagebox.askyesno("Confirm", f"Delete student {roll_no}?"):
                # Remove from databases
                if roll_no in college_students:
                    del college_students[roll_no]
                if roll_no in mess_students:
                    del mess_students[roll_no]
                
                # Save databases
                with open(self.college_db, 'wb') as f:
                    pickle.dump(college_students, f)
                with open(self.mess_db, 'wb') as f:
                    pickle.dump(mess_students, f)
                
                # Delete photo
                photo_path = f"enrollment_photos/{roll_no}.jpg"
                if os.path.exists(photo_path):
                    os.remove(photo_path)
                
                messagebox.showinfo("Success", f"Student {roll_no} deleted!")
                view_window.destroy()
                self.view_students()  # Refresh
        
        btn_frame = tk.Frame(view_window, bg='#f0f0f0')
        btn_frame.pack(pady=10)
        
        delete_btn = tk.Button(
            btn_frame,
            text="🗑️ Delete Selected",
            command=delete_selected,
            font=("Arial", 11, "bold"),
            bg='#e74c3c',
            fg='white',
            padx=20,
            pady=10
        )
        delete_btn.pack()
    
    def clear_all_data(self):
        """Clear all enrollment data"""
        if messagebox.askyesno(
            "DANGER", 
            "This will DELETE ALL student data!\n\nAre you absolutely sure?"
        ):
            if messagebox.askyesno(
                "FINAL WARNING",
                "This action CANNOT be undone!\n\nDelete everything?"
            ):
                # Delete database files
                if os.path.exists(self.college_db):
                    os.remove(self.college_db)
                if os.path.exists(self.mess_db):
                    os.remove(self.mess_db)
                
                # Delete enrollment photos
                if os.path.exists('enrollment_photos'):
                    for file in os.listdir('enrollment_photos'):
                        os.remove(os.path.join('enrollment_photos', file))
                
                messagebox.showinfo("Success", "All data cleared!")
                self.status_label.config(text="All data cleared. Ready to start fresh.", fg='#e74c3c')
    
    def show_stats(self):
        """Show enrollment statistics"""
        college_students = {}
        mess_students = {}
        
        if os.path.exists(self.college_db):
            with open(self.college_db, 'rb') as f:
                college_students = pickle.load(f)
        
        if os.path.exists(self.mess_db):
            with open(self.mess_db, 'rb') as f:
                mess_students = pickle.load(f)
        
        # Count by department
        dept_count = {}
        for data in college_students.values():
            dept = data['department']
            dept_count[dept] = dept_count.get(dept, 0) + 1
        
        dept_str = "\n".join([f"  {dept}: {count}" for dept, count in sorted(dept_count.items())])
        
        messagebox.showinfo(
            "Enrollment Statistics",
            f"📊 Database Statistics\n\n"
            f"Total College Students: {len(college_students)}\n"
            f"Total Mess Students: {len(mess_students)}\n"
            f"College Only (No Mess): {len(college_students) - len(mess_students)}\n\n"
            f"By Department:\n{dept_str if dept_str else '  None'}"
        )
    
    def clear_form(self):
        self.roll_entry.delete(0, tk.END)
        self.name_entry.delete(0, tk.END)
        self.dept_entry.delete(0, tk.END)
        self.mess_var.set(True)
        self.selected_image_path = None
        
        # Clear canvas
        self.canvas.delete('all')
        self.canvas.create_text(200, 150, text="No photo", font=("Arial", 14), fill='gray', tags='placeholder')
        self.canvas.photo = None
    
    def run(self):
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.mainloop()
    
    def on_closing(self):
        if self.camera_active:
            self.stop_camera()
        if self.cap is not None:
            self.cap.release()
        self.window.destroy()

if __name__ == "__main__":
    app = EnrollmentGUI()
    app.run()
