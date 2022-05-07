import cv2
from cv2 import VideoCapture

count = 0
name = "person"
cap = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier(
    "assets/haarcascade_frontalface_default.xml")
cap.set(3, 1920)
cap.set(4, 1080)
while True:
    success, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for(x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]
        cv2.imwrite(str(count)+' .jpg', face)
        # count += 1
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 3)
        cv2.putText(img, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 255, 0), 3)

    cv2.imshow("Video", img)
    if cv2.waitKey(1) & 0xFF == ord('c'):
        break

cap.release()
cv2.destroyAllWindows()
