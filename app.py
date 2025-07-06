from flask import Flask, render_template, Response, request, redirect
import cv2
import numpy as np
import os
import pickle
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime
import csv

app = Flask(__name__)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
os.makedirs("model_data", exist_ok=True)
os.makedirs("attendance", exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        data = []
        cap = cv2.VideoCapture(0)
        count = 0

        while True:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for x, y, w, h in faces:
                face = frame[y:y+h, x:x+w]
                face = cv2.resize(face, (100, 100))
                data.append(face)
                count += 1
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"{count}/50", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Registering - Press Q", frame)
            if cv2.waitKey(1) == ord('q') or count >= 50:
                break

        cap.release()
        cv2.destroyAllWindows()

        data = np.array(data)
        labels = np.array([name] * 50)

        if os.path.exists("model_data/data_face.pkl"):
            with open("model_data/data_face.pkl", "rb") as f:
                old_data = pickle.load(f)
            data = np.append(old_data, data, axis=0)

        with open("model_data/data_face.pkl", "wb") as f:
            pickle.dump(data, f)

        if os.path.exists("model_data/label_name.pkl"):
            with open("model_data/label_name.pkl", "rb") as f:
                old_labels = pickle.load(f)
            labels = np.append(old_labels, labels, axis=0)

        with open("model_data/label_name.pkl", "wb") as f:
            pickle.dump(labels, f)

        return render_template("success.html", name=name)

    return render_template('register.html')


@app.route('/attendance')
def attendance():
    return render_template("live_attendance.html")


def generate_frames():
    if not os.path.exists("model_data/data_face.pkl"):
        yield b"--frame\r\n\r\n"
        return

    with open("model_data/data_face.pkl", "rb") as f:
        faces = pickle.load(f)
    with open("model_data/label_name.pkl", "rb") as f:
        labels = pickle.load(f)

    X = faces.reshape(len(faces), -1)
    y = labels
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X, y)

    date_str = datetime.now().strftime("%Y-%m-%d")
    attendance_file = f"attendance/attendance-{date_str}.csv"

    if not os.path.exists(attendance_file):
        with open(attendance_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "DateTime"])

    marked = set()
    with open(attendance_file, "r") as f:
        next(f)
        for line in f:
            marked.add(line.split(",")[0])

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_rect = face_cascade.detectMultiScale(gray, 1.3, 5)

        for x, y, w, h in faces_rect:
            face_img = frame[y:y+h, x:x+w]
            resized = cv2.resize(face_img, (100, 100)).reshape(1, -1)

            pred = model.predict(resized)
            dist, _ = model.kneighbors(resized)

            if dist[0][0] > 8500:
                label = "Unknown"
                color = (0, 0, 255)
            else:
                label = pred[0]
                color = (0, 255, 0)
                if label not in marked:
                    with open(attendance_file, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([label, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
                    marked.add(label)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/attendance_log', methods=["GET", "POST"])
def attendance_log():
    files = os.listdir("attendance")
    dates = [f.replace("attendance-", "").replace(".csv", "") for f in files if f.endswith(".csv")]
    selected_date = request.form.get("date") if request.method == "POST" else None
    records = []

    if selected_date:
        file_path = f"attendance/attendance-{selected_date}.csv"
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                reader = csv.reader(f)
                next(reader)
                records = list(reader)

    return render_template("attendance_log.html", dates=sorted(dates, reverse=True),
                           selected_date=selected_date, records=records)

@app.route('/users')
def users():
    if not os.path.exists("model_data/label_name.pkl"):
        return render_template("users.html", users=[])

    with open("model_data/label_name.pkl", "rb") as f:
        labels = pickle.load(f)

    # Unique names and their counts
    from collections import Counter
    name_count = Counter(labels)
    users = [{"name": name, "count": count} for name, count in name_count.items()]

    return render_template("users.html", users=users)

if __name__ == '__main__':
    app.run(debug=True)
