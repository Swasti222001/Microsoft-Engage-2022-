import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# from PIL import ImageGrab

path = 'Training_images'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    imgHeight = curImg.shape[0]
    imgWidth = curImg.shape[1]
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncodings(images):
    encodeList = []

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAttendance(name):
    already_in_file = set()
    with open('Attendance.csv', "r") as g:       # just read
        for line in g:
            already_in_file.add(line.split(",")[0])

# process your current entry:
        if name not in already_in_file:
            with open('Attendance.csv', "a") as g:   # append
                now = datetime.now()
                dtString = now.strftime('%d-%B-%Y')
                tString = now.strftime('%H:%M:%S')
                g.writelines(f'\n{name},{dtString},{tString}')
    # with open('Attendance.csv', 'r') as f:
    #     myDataList = f.readlines()

    #     nameList = []
    #     for line in myDataList:
    #         entry = line.split(',')
    #         nameList.append(entry[0])
    #         if name not in nameList:
    #             now = datetime.now()
    #             dtString = now.strftime('%H:%M:%S')
    #             f.writelines(f'\n{name},{dtString}')


# FOR CAPTURING SCREEN RATHER THAN WEBCAM
# def captureScreen(bbox=(300,300,690+300,530+300)):
#     capScr = np.array(ImageGrab.grab(bbox))
#     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
#     return capScr
encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1000)


while True:
    if cap.isOpened():

        success, img = cap.read()
        if success:

            # img = captureScreen()
            imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

            facesCurFrame = face_recognition.face_locations(imgS)
            encodesCurFrame = face_recognition.face_encodings(
                imgS, facesCurFrame)

            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                matches = face_recognition.compare_faces(
                    encodeListKnown, encodeFace, tolerance=0.50)
                faceDis = face_recognition.face_distance(
                    encodeListKnown, encodeFace)
        # print(faceDis)
                matchIndex = np.argmin(faceDis)
                name = "unknown"

                if matches[matchIndex]:
                    name = classNames[matchIndex].upper()
        # print(name)
                fontScale = (imgWidth * imgHeight) / (1400*1100)
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2),
                              (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6),
                            cv2.FONT_HERSHEY_COMPLEX, fontScale, (255, 255, 255), 2)
                markAttendance(name)

            cv2.imshow('Webcam', img)
            if cv2.waitKey(1) & 0xFF == ord('c'):
                break
        else:
            print("Error : Failed to capture frame")

    # print error if the connection with camera is unsuccessful
    else:
        print("Cannot open camera")
        # cv2.waitKey(1)
