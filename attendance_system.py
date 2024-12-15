import tkinter as tk
from tkinter import ttk
from ttkthemes import ThemedTk
from tkinter import messagebox
import cv2
import numpy as np
import requests 
import os
import urllib.request
import time
from datetime import datetime
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
import mysql.connector

mydb = mysql.connector.connect(
    host = 'localhost',
    user='root',
    password=""
)
print(mydb)

# mycursor = mydb.cursor()
# mycursor.execute("CREATE DATABASE Attendance_DB")

mydb = mysql.connector.connect(
    host = "localhost",
    user = "root",
    password = "",
    database = "Attendance_DB"
)
mycursor = mydb.cursor()
# mycursor.execute("CREATE TABLE student_table(ID int primary key, Roll_no varchar (10), Name varchar(50), Department varchar(50))")

mycursor.execute("SHOW TABLES")
for i in mycursor:
    print(i)

# for i in range(1,31):
    # mycursor.execute(f"ALTER TABLE student_table ADD Dec_{i}_2024 varchar(5);")

url='http://192.168.7.8:8080//shot.jpg'

def train_model():
    data_dir = "C:\\Users\\justa\\OneDrive\\Desktop\\Programs\\AIML\\Attendance System\\data"
    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)] 
    faces = []
    ids = []
     
    for image in path:
        img = Image.open(image).convert('L')
        imageNp = np.array(img, 'uint8')
        id = os.path.split(image)[1].split(".")[1]
        id = int(id.replace('-',''))
        faces.append(imageNp)
        ids.append(id)
         
    ids = np.array(ids)
     
    # Train and save classifier
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces,ids)
    clf.write("classifier.xml")
    messagebox.showinfo('Training Complete', "Call me a python 'cause I remember your face now.")
    print("Training model...")

def register_student():
    if(roll_entry.get()=="" or name_entry.get()=="" or dept_entry.get()==""):
        messagebox.showinfo('Error!', 'Please Fill All The Fields.')
    else:

        mydb = mysql.connector.connect(
        host = "localhost",
        user = "root",
        password = "",
        database = "Attendance_DB"
        )
        mycursor = mydb.cursor()
        mycursor.execute("SELECT * FROM student_table")
        result = mycursor.fetchall()
        id = 1
        for x in result:
            id+=1
        sql = "INSERT INTO student_table(ID, Roll_no, Name, Department) values(%s, %s, %s, %s)"
        val = (id, roll_entry.get(), name_entry.get(), dept_entry.get())
        mycursor.execute(sql,val)
        mydb.commit()

        face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        def face_cropped(img):
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, 1.3, 5)
            # scaling factor = 1.3
            # minimum neighbor = 5
            
            if faces is ():
                return None
            for (x,y,w,h) in faces:
                cropped_face = img[y:y+h,x:x+w]
            return cropped_face
        
        cap = cv2.VideoCapture(0)
        img_id = 0
        
        while True:
            ret, frame = cap.read()
            if face_cropped(frame) is not None:
                img_id+=1
                face = cv2.resize(face_cropped(frame), (200,200))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                file_name_path = "data/user."+str(id)+"."+str(img_id)+".jpg"
                cv2.imwrite(file_name_path, face)
                cv2.putText(face, str(img_id), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
                
                cv2.imshow("Cropped face", face)
                
            if cv2.waitKey(1)==13 or int(img_id)==200: #13 is the ASCII character of Enter
                break
                
        cap.release()
        cv2.destroyAllWindows()
        messagebox.showinfo('Result', 'Face Scan Completed.')
    print("Registering student...")

def take_attendance():
    def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
        
        for (x,y,w,h) in features:

            cv2.rectangle(img, (x,y), (x+w,y+h), color, 2 )
            id, pred = clf.predict(gray_img[y:y+h,x:x+w])
            confidence = int(100*(1-pred/300))

            mydb = mysql.connector.connect(
            host = "localhost",
            user = "root",
            password = "",
            database = "Attendance_DB"
            )

            mycursor= mydb.cursor()
            mycursor.execute("SELECT Name FROM student_table WHERE ID="+str(id))
            s=mycursor.fetchone()
            s = ''+''.join(s)
            
            if confidence>70:
                    date = datetime.today().strftime('%d')
                    cv2.putText(img, s, (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
                    print(f"UPDATE student_table SET Nov_{date}_2024 = 'P' WHERE ID="+str(id))
                    mycursor.execute(f"UPDATE student_table SET Nov_{date}_2024 = 'P' WHERE ID="+str(id))
            else:
                cv2.putText(img, "UNKNOWN", (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1, cv2.LINE_AA)
        
        return img
    
    # loading classifier
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")
    
    
    while True:
        imgResp=urllib.request.urlopen(url)
        imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
        img=cv2.imdecode(imgNp,-1)
        img = draw_boundary(img, faceCascade, 1.3, 6, (255,255,255), "Face", clf)
        cv2.imshow("face Detection", img)
        
        if cv2.waitKey(1)==13:
            break
    video_capture.release()
    cv2.destroyAllWindows()
    print("Taking attendance...")

# Create the main window with a theme
window = ThemedTk(theme="arc")
window.title("Student Information System")

# Customize the window's background color
window.configure(bg="#f0f0f0")

# Create a frame for the form fields
form_frame = ttk.Frame(window, padding=10)
form_frame.grid(row=0, column=0, sticky="nsew")

# Customize the label and entry colors
style = ttk.Style()
style.configure("TLabel", background="#f0f0f0", foreground="#333")
style.configure("TEntry", background="#fff", foreground="#333")

# Create labels and entry fields for roll number, name, and department
roll_label = ttk.Label(form_frame, text="Roll No:")
roll_label.grid(row=0, column=0, sticky="w")
roll_entry = ttk.Entry(form_frame)
roll_entry.grid(row=0, column=1, sticky="ew")

name_label = ttk.Label(form_frame, text="Name:")
name_label.grid(row=1, column=0, sticky="w")
name_entry = ttk.Entry(form_frame)
name_entry.grid(row=1, column=1, sticky="ew")

dept_label = ttk.Label(form_frame, text="Department:")
dept_label.grid(row=2, column=0, sticky="w")
dept_entry = ttk.Entry(form_frame)
dept_entry.grid(row=2, column=1, sticky="ew")

# Create a frame for the buttons
button_frame = ttk.Frame(window, padding=10)
button_frame.grid(row=1, column=0, sticky="nsew")

# Create the buttons
train_button = ttk.Button(button_frame, text="Train Model", command=train_model)
train_button.grid(row=0, column=0, sticky="ew")

register_button = ttk.Button(button_frame, text="Register Student", command=register_student)
register_button.grid(row=0, column=1, sticky="ew")

attendance_button = ttk.Button(button_frame, text="Take Attendance", command=take_attendance)
attendance_button.grid(row=0, column=2, sticky="ew")

# Configure the window layout
window.columnconfigure(0, weight=1)
window.rowconfigure(0, weight=1)
form_frame.columnconfigure(1, weight=1)
button_frame.columnconfigure(0, weight=1)
button_frame.columnconfigure(1, weight=1)
button_frame.columnconfigure(2, weight=1)

window.mainloop()