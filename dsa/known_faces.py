import cv2
import os
import face_recognition
import csv
known_faces_folder = os.listdir(r"C:\Users\hp\Downloads\final dsa\Individual images")
known_folder_path = r"C:\Users\hp\Downloads\final dsa\Individual images"

known_faces_encodings = []
for imagepath in known_faces_folder:
    images = cv2.imread(f'{known_folder_path}/{imagepath}')
    images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
    known_face_encoding = face_recognition.face_encodings(images)
    print("Success") 
    if len(known_face_encoding) > 0:
        known_faces_encodings.append(known_face_encoding[0])
print("Success")       
file_path = r"C:\Users\hp\Downloads\final dsa\known_face_encodings.csv"
with open(file_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(known_faces_encodings) 
    
    
print("done")