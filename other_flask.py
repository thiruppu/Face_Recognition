import cv2
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, render_template, Response
import os

app = Flask(__name__)

# Load the face detection model and recognizer
face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read("recognizer/trainingData.yml")

# Attendance file
attendance_file = 'attendance.csv'
standard_checkin_time = "12:00:00"  # 12 PM
faces_folder = 'static/faces'  # Directory to save face images

# Mapping of IDs to Names
name_mapping = {
    2: "Katty",
    1: "Thiru",
    8: "Sadhana"
}

# Function to update attendance in CSV, save face image, and calculate lateness
def mark_attendance(ide, face_img):
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d %H:%M:%S")
    
    # Check if the attendance file exists, create one if it doesn't
    if not os.path.exists(attendance_file):
        df = pd.DataFrame(columns=['ID', 'Name', 'DateTime', 'Lateness', 'Status'])
        df.to_csv(attendance_file, index=False)

    # Load existing attendance data
    df = pd.read_csv(attendance_file)
    
    # Check if the person is already marked for today
    if not ((df['ID'] == ide) & (df['DateTime'].str.contains(now.strftime("%Y-%m-%d")))).any():
        # Calculate lateness in minutes
        current_time = now.strftime("%H:%M:%S")
        standard_time = datetime.strptime(standard_checkin_time, "%H:%M:%S")
        current_time_dt = datetime.strptime(current_time, "%H:%M:%S")
        lateness_minutes = (current_time_dt - standard_time).seconds // 60

        lateness = f"{lateness_minutes} minutes late" if lateness_minutes > 0 else "On time"

        # Save the detected face as an image
        face_image_path = f"{faces_folder}/{name_mapping[ide]}_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(face_image_path, face_img)

        # Add attendance record
        new_record = {'ID': ide, 'Name': name_mapping[ide], 'DateTime': date_time, 'Lateness': lateness, 'Status': 'Present', 'FaceImage': face_image_path}
        df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
        df.to_csv(attendance_file, index=False)
        print(f"Attendance marked for {name_mapping[ide]} at {date_time} ({lateness})")

        return name_mapping[ide], date_time, lateness, face_image_path  # Return attendance info for display
    return None, None, None, None

# Function to stream video and handle face recognition
def generate_frames():
    while True:
        ret, img = cam.read()  # Capture frame-by-frame
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detect.detectMultiScale(gray, 1.3, 5)

        recognized = None
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            ide, conf = rec.predict(gray[y:y+h, x:x+w])

            if conf < 60 and ide in name_mapping:
                name = name_mapping[ide]
                face_img = img[y:y+h, x:x+w]  # Extract face region
                recognized = mark_attendance(ide, face_img)  # Mark attendance and get details
            else:
                name = "Unknown"

            cv2.putText(img, name, (x, y+h), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Encode the frame for streaming
        ret, buffer = cv2.imencode('.jpg', img)
        img = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')

# Route to fetch attendance data and render it
@app.route('/')
def index():
    # Load attendance data from CSV
    if os.path.exists(attendance_file):
        attendance_df = pd.read_csv(attendance_file)
    else:
        attendance_df = pd.DataFrame(columns=['ID', 'Name', 'DateTime', 'Lateness', 'Status', 'FaceImage'])

    # Convert the attendance data to a list of dictionaries to send to HTML
    attendance_list = attendance_df.to_dict(orient='records')
    return render_template('index.html', attendance_list=attendance_list)

# Route for video streaming
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
