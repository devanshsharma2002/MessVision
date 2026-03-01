# 🚀 MessVision
### AI-Powered Computer Vision Mess Entry System

MessVision is an automated facial recognition system that replaces physical ID cards for mess (canteen) access using Python, OpenCV, and deep learning.

It enables fast, secure, and real-time identity verification using a standard laptop webcam — no GPU required.

---

## 🎯 Features

- ⚡ Real-time Facial Recognition (< 1 second per detection)
- 👤 One-time Face Enrollment for each student
- 🚫 Outsider Detection (Blocks unauthorized access)
- 💻 Lightweight system (Runs on standard laptop webcam)
- 🧠 Deep learning-based face encodings

---

## 🛠 Tech Stack

Python 3.10  
OpenCV  
face_recognition  
pickle  
Tkinter  

---

## 📋 How It Works

### 1️⃣ Enrollment Phase
- Capture student face
- Generate 128D face encoding
- Store encoding in database (pickle file)

### 2️⃣ Recognition Phase
- Capture live video feed
- Detect face
- Generate encoding
- Compare with stored encodings

### 3️⃣ Access Decision
- If distance < 0.6 → ✅ Access Granted
- Else → ⚠️ Outsider Detected

---

## 🚀 Quick Demo

### Enrollment (Run once per student)

```bash
python enroll.py
```

### Live Recognition

```bash
python app.py
```

### Sample Output

```
[+] Face detected: Devansh Sharma - Access GRANTED
[+] Face detected: Unknown - OUTSIDER DETECTED ⚠️
```

---

## 📁 Project Structure

```
MessVision/
│
├── app.py               # Main recognition app
├── test.py              # Main system test app
├── enroll.py            # Student enrollment script
├── requirements.txt     # Dependencies
├── detected_faces/      # Captured images during recognition
└── enrollment_photos/   # Stored enrollment photos
```

---

## ⚙️ Setup & Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/devanshsharma2002/MessVision.git
cd MessVision
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Enroll Students

```bash
python enroll.py
```

### 4️⃣ Run Recognition

```bash
python app.py
```

---

## 📊 Performance

| Metric | Value |
|--------|--------|
| Recognition Speed | ~0.8s per frame |
| Outsider Detection | 100% (10 test subjects) |
| False Acceptance Rate | 0% (Test Dataset) |
| Memory Usage | <500MB |

---

## 🤔 Challenges Solved

- 🌗 Lighting Variations  
  Implemented preprocessing and robust face encodings.

- ⚡ Real-time Performance  
  Optimized using frame resizing and efficient encoding comparison.

- 💾 Data Persistence  
  Used pickle serialization for fast student lookup.

- 🖥 GUI Design  
  Built enrollment interface using Tkinter.

---

## 🔮 Future Enhancements

- Database integration (SQLite / PostgreSQL)
- Multi-face simultaneous detection
- Attendance logging system
- Mobile app integration
- Cloud deployment (AWS / GCP)

---

## 👨‍💻 Author

Devansh Sharma  
B.Tech IT  
Email: devanshsharma2002@gmail.com  

---

⭐ If you found this project useful, consider giving it a star!
