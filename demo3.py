# Importing all the modules...
import numpy as np
import face_recognition
import cv2
import os
import csv
from datetime import datetime

path='photo' # It will define the folder name from where we are going to collect all the images
images=[] # List of images in form of numpy array.
imageNames=[] # It is the name of each image
myList=os.listdir(path) # List of all the images inside photo folder
myList.pop(0)
myList
for ele in myList:
    print(ele.split('.')[0])


for imageName in myList:
    currentImage=cv2.imread(f'{path}/{imageName}')
    images.append(currentImage)
    imageNames.append(os.path.splitext(imageName)[0])
    print(os.path.splitext(imageName)[0])
print(imageNames)

# Let's define a function that encodes all the images.
def findEncodings(images):
    encodeList=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


# Let's define a function that will mark the attendence in a csv file...
def markAttendance(name):
    now=datetime.now()
    current_date=now.strftime("%Y-%m-%d")
    current_time=now.strftime("%H-%M-%S")
    f=open(current_date+'.csv','a+',newline="")
    lineWriter=csv.writer(f)
    lineWriter.writerow([name,current_time])
    f.close()


known_image_encodings=findEncodings(images)
print("Encoding Completed....!")
        
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(known_image_encodings, encodeFace)
        faceDis = face_recognition.face_distance(known_image_encodings, encodeFace)

        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = imageNames[matchIndex].upper()
            print(name)

            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2),
                          (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()