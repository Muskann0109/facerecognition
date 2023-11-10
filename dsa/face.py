import cv2
import matplotlib.pyplot as plt
import face_recognition
from mtcnn import MTCNN
import os
import csv
import numpy as np
# Replace with the path to your image
image_path = r"C:\Users\hp\Downloads\final dsa\class images folder\test4.jpg"

# Path to the folder containing known faces
known_faces_folder = os.listdir(r"C:\Users\hp\Downloads\final dsa\Individual images")
known_folder_path = r"C:\Users\hp\Downloads\final dsa\Individual images"
csv_file_path = r"C:\Users\hp\Downloads\final dsa\known_face_encodings.csv"

my_list = []
classNames = []
images = []
present_students = []
myList = os.listdir(known_folder_path)

for cl in myList:
    curImg = cv2.imread(f'{known_folder_path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
# Load the image
image = cv2.imread(image_path)

known_faces_encodings = []
with open(csv_file_path, 'r') as file:
    reader = csv.reader(file)
    known_faces_encodings = [row for row in reader]
    
known_faces_encodings = [[float(item) for item in inner_list] for inner_list in known_faces_encodings]
    
print(len(known_faces_encodings))
# Check if the image was loaded successfully
if image is not None:
    print("Image loaded successfully.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for displaying with matplotlib

    # Initialize the MTCNN detector
    detector = MTCNN()

    # Detect faces in the image
    faces = detector.detect_faces(image)

    # List to store face encodings of detected faces
    face_encodings = []
    
            
            
    
    # Display the detected faces
    for result in faces:
        x, y, width, height = result['box']

        # Extract the face from the image
        face_image = image[y:y + height, x:x + width]
        cv2.imwrite("chota_image.jpg",face_image)
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        # Generate face encoding
        print(type(known_faces_encodings[0][0]))

        face_encoding = face_recognition.face_encodings(face_image)
        #print(type(face_encoding[0][0]))
        if len(face_encoding) > 0:
            face_encodings.append(face_encoding[0])
            # Match the face with known faces
            matches = face_recognition.compare_faces(known_faces_encodings, face_encoding[0], 0.8)
            faceDistances = face_recognition.face_distance(known_faces_encodings, face_encoding[0])
            matchIndex = np.argmin(faceDistances)
            name = "Unknown"

            if matches[matchIndex]:
                #first_match_index = matches.index(True)
                #name = list(known_faces_encodings.keys())[first_match_index]
                name = classNames[matchIndex].upper()
            # Draw rectangle around the face and print the file name
                present_students.append(name)
                cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
                cv2.putText(image, name, (x , y - 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2)
           # plt.text(x, y, name, color='red', fontsize=12, weight='bold')
    print(present_students)
    # Display the image with bounding boxes around the detected faces
    plt.imshow(image)
    plt.show()
    print("writing is started")
    attendance_file_path = r"C:\Users\hp\Downloads\final dsa\attendance.csv"
    with open(attendance_file_path, 'w', newline='') as file:
        #writer = csv.writer(file)
        file.writelines(f"{num}\n"for num in present_students)
    # Now, face_encodings contains the encodings of detected faces
    #print("Face encodings:", face_encodings)
    print("attendance done")
else:
    print(f"Failed to load the image at {image_path}. Please check the file path.")