# 🎯 Face Recognition Attendance System Web Application

## 📘 Introduction  
The Face Recognition Attendance System is a real-time web application designed to automate attendance tracking using facial recognition technology. It leverages computer vision and machine learning to eliminate manual processes and streamline secure, efficient attendance logging. The application provides a clean and intuitive interface for face registration, real-time recognition, and attendance history viewing — all within a Flask-powered web platform.

## 🎯 Objectives  
- Automate attendance tracking using real-time face recognition  
- Replace manual attendance methods with secure, contactless authentication  
- Enable live camera streaming within a web interface  
- Allow secure registration of new faces through the browser  
- Log attendance with timestamps in daily CSV files  
- Provide access to attendance logs and registered user information  

## 🧰 Technologies Used  
**Programming Language:**  
Python  

**Frameworks/Libraries:**  
- Flask (Web Framework)  
- OpenCV (Face Detection)  
- scikit-learn (KNN Classifier)  
- NumPy (Array Processing)  
- HTML/CSS (Web Interface)  
- Pickle (Data Serialization)  

**Tools:** VS Code, Jupyter Notebook  
**Storage:** CSV files for attendance logs, Pickle files for face data  

## 🔁 Workflow  

### 🔹 User Registration  
- Capture 50 face samples through webcam  
- Save face data and corresponding user names using Pickle  

### 🔹 Live Face Recognition  
- Access webcam stream in browser  
- Detect and recognize faces using Haar cascades and KNN classifier  
- Log attendance to a CSV file with current timestamp  

### 🔹 Web Dashboard  
- Home page with options to start attendance, register users, and view logs  
- Display list of registered users  
- Browse and download attendance history by date  

## 📊 Result Interpretation  
- Attendance records saved in date-wise CSV files  
- Each entry logs the user name and timestamp  
- History can be viewed through the web dashboard  
- Unrecognized faces are flagged as “Unknown”  

## ✅ Results  
- **Face Recognition Accuracy:** ~90% under good lighting  
- **Logging Latency:** < 1 second per face  
- **Supports** multiple users with unique identities  
- **User Interface:** Clean, intuitive, and accessible  

## 🔮 Future Work  
- Upgrade to Deep Learning (FaceNet or DeepFace)  
- Add user/admin authentication  
- Store logs in databases (e.g., PostgreSQL, MongoDB)  
- Send attendance alerts via email or SMS  
- Deploy as a Docker container or mobile app  

## 🧾 Conclusion  
This project demonstrates how real-time face recognition can modernize attendance systems. Using open-source tools like OpenCV, Flask, and scikit-learn, it delivers a secure, modular, and user-friendly solution for institutions and workplaces.

## 📚 References  
- [OpenCV Haar Cascade Documentation](https://docs.opencv.org/)  
- [Flask Documentation](https://flask.palletsprojects.com/)  
- [scikit-learn KNN Classifier](https://scikit-learn.org/)  
- [Python Pickle Docs](https://docs.python.org/3/library/pickle.html)  

## 📦 Clone Repository  
To clone this repository locally, run:

```bash
git clone https://github.com/adnan-saif/Real_Time_Face_Attendance_System.git
