# 🚀 MessVision

**AI-Powered Computer Vision Based Mess Entry System**

MessVision is an automated mess/canteen entry management system that uses **facial recognition** to identify registered students and detect unauthorized individuals. It replaces traditional physical ID-card verification with a camera-based computer vision system.

The system uses **Python, OpenCV, face_recognition, NumPy, and Tkinter** to provide real-time face detection, student enrollment, and access verification.

---

## ✨ Features

- 🎥 **Real-Time Face Recognition**  
  Detects and identifies registered students using a live camera feed.

- 👤 **Automated Student Enrollment**  
  Provides a simple GUI for registering students and generating their facial encodings.

- 🚪 **Automated Access Verification**  
  Grants access to recognized students and flags unknown individuals.

- 🛡️ **Outsider Detection**  
  Individuals whose facial encoding does not match the enrolled database are classified as outsiders.

- ⚡ **Lightweight Processing**  
  Designed to run on a standard laptop/PC with a webcam without requiring a dedicated GPU.

- 💾 **Local Data Storage**  
  Student face encodings are stored locally for fast recognition and lookup.

---

## 🧠 How It Works

MessVision follows a simple enrollment → recognition → verification workflow.

### 1. Student Enrollment

The student is registered using the enrollment application.

```text
Student
   ↓
Webcam Capture
   ↓
Face Detection
   ↓
Face Encoding Generation
   ↓
Student Data + Encoding
   ↓
Local Storage
```

The generated facial encoding is stored and associated with the student's identity.

### 2. Real-Time Recognition

During mess entry, the webcam continuously captures frames.

```text
Live Camera Feed
       ↓
   Face Detection
       ↓
 Face Encoding
       ↓
Compare with Stored Encodings
       ↓
 ┌───────────────┴───────────────┐
 ↓                               ↓
Match Found                  No Match
 ↓                               ↓
Access Granted              Outsider Detected
```

### 3. Access Decision

The system compares the live face encoding against registered encodings using facial distance.

The current implementation uses:

- **Distance < 0.6** → Access Granted
- **Distance ≥ 0.6** → Outsider Detected

> **Note:** `0.6` is the configured threshold used by this implementation. Recognition performance can vary depending on lighting, camera quality, facial pose, and enrollment quality.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| **Python 3.10** | Core programming language |
| **OpenCV** | Camera access, image processing and face detection pipeline |
| **face_recognition** | Face encoding and facial similarity comparison |
| **NumPy** | Numerical operations and encoding manipulation |
| **pickle** | Local serialization of face encodings |
| **Tkinter** | Student enrollment GUI |

---

## 📁 Project Structure

```text
MessVision/
│
├── src/
│   ├── app.py                  # Main facial recognition application
│   └── enroll.py               # Student enrollment application
│
├── data/
│   ├── detected_faces/         # Faces captured during recognition
│   └── enrollment_photos/      # Student enrollment images
│
├── tests/
│   └── test.py                 # Test scripts
│
├── assets/
│   └── websiteface/            # Website integration/static assets
│
├── .gitignore
├── requirements.txt
└── README.md
```

> The virtual environment (`.venv/`) should normally be excluded from Git using `.gitignore`.

---

## 🚀 Getting Started

### Prerequisites

Make sure you have:

- Python 3.10
- A working webcam
- Windows, Linux, or macOS
- Git

---

### 1. Clone the Repository

```bash
git clone https://github.com/devanshsharma2002/MessVision.git
cd MessVision
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Virtual Environment

#### Windows

```bash
.venv\Scripts\activate
```

#### macOS / Linux

```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### Step 1 — Enroll a Student

Run:

```bash
python src/enroll.py
```

The enrollment application will:

1. Open the webcam.
2. Detect the student's face.
3. Capture the enrollment image.
4. Generate the corresponding face encoding.
5. Store the student's information locally.

Run the enrollment process once for each student.

---

### Step 2 — Start the Recognition System

Run:

```bash
python src/app.py
```

The application will open the webcam and begin recognizing faces.

Example output:

```text
[+] Face detected: Devansh Sharma
[+] Access GRANTED
```

For an unknown individual:

```text
[+] Face detected: Unknown
[!] OUTSIDER DETECTED
```

---

## 📊 Performance

The current implementation was tested under controlled conditions using a standard laptop webcam.

| Metric | Result |
|---|---|
| Recognition Speed | ~0.8 seconds/frame |
| GPU Required | No |
| Memory Usage | < 500 MB |
| Configured Recognition Threshold | 0.6 |

### Recognition Accuracy

The system demonstrated reliable recognition during testing, including detection of individuals who were not present in the enrolled dataset.

However, accuracy can vary depending on:

- Lighting conditions
- Camera quality
- Distance from camera
- Face angle
- Facial occlusion
- Quality of enrollment images
- Number of registered students

Therefore, accuracy figures should be interpreted as **test-environment results rather than guaranteed real-world performance**.

---

## 🧩 Challenges Addressed

### 💡 Lighting Variations

Different lighting conditions can affect facial recognition performance.

MessVision uses image preprocessing and facial encodings to improve recognition consistency under varying conditions.

### ⚡ Real-Time Performance

Processing full-resolution frames can be computationally expensive.

The recognition pipeline uses resized frames and optimized encoding comparisons to reduce processing overhead.

### 💾 Data Persistence

Face encodings need to be available every time the application starts.

MessVision serializes the locally generated encoding data so it can be loaded quickly during recognition.

### 👨‍💻 User Experience

A Tkinter-based enrollment interface simplifies the process of registering students without requiring command-line interaction.

---

## 🔐 Security Considerations

MessVision is designed as a prototype access-control system.

The current implementation stores face encodings locally and uses a configurable distance threshold for verification.

For production deployment, additional security mechanisms should be considered, including:

- Encrypted database storage
- Authentication and authorization
- Secure access to enrollment functionality
- Liveness/anti-spoofing detection
- Audit logs
- Role-based administration
- Secure backup and recovery
- Proper handling and protection of biometric data

> Facial recognition involves biometric information. Any real-world deployment should comply with applicable privacy and data-protection requirements and obtain appropriate user consent.

---

## 🔮 Future Enhancements

- [ ] Replace local `pickle` storage with **SQLite/PostgreSQL**
- [ ] Add **multi-face detection and recognition**
- [ ] Implement **liveness / anti-spoofing detection**
- [ ] Add student attendance and meal-entry history
- [ ] Create an administrator dashboard
- [ ] Add role-based authentication
- [ ] Generate daily/monthly mess reports
- [ ] Add mobile application support
- [ ] Deploy the system on cloud infrastructure
- [ ] Integrate AWS/GCP-based services
- [ ] Improve recognition performance for different lighting and poses

---

## 🎯 Use Cases

MessVision can be adapted for:

- 🏫 College messes
- 🍽️ University cafeterias
- 🏢 Office cafeterias
- 🏭 Industrial canteens
- 🎓 Hostel dining facilities
- 🔐 Restricted-access environments

---

## 🧪 Testing

Testing scripts are available in:

```text
tests/
└── test.py
```

Run the test script using:

```bash
python tests/test.py
```

---

## ⚠️ Limitations

The current version is a prototype and has several limitations:

- Recognition performance depends on camera and lighting conditions.
- Face encodings are stored locally.
- There is currently no dedicated database management system.
- The system does not yet implement liveness detection.
- A simple distance threshold is used for access decisions.
- Production-level authentication and auditing are not yet implemented.

---

## 📌 Project Status

**Current Status:** Prototype / Working Implementation

The core facial recognition, student enrollment, and outsider detection functionality is implemented and operational.

---

## 👨‍💻 Author

**Devansh Sharma**

B.Tech — Information Technology

---

## 📄 License

This project is intended for **educational and demonstration purposes**.

If you plan to deploy the system in a real-world environment, review the applicable privacy, biometric-data, and security requirements first.