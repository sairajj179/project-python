import cv2
import os

# Ensure output folder exists
os.makedirs("auth\\samples", exist_ok=True)

cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Could not open camera")
    exit()

cam.set(3, 640)
cam.set(4, 480)

detector = cv2.CascadeClassifier('auth\\haarcascade_frontalface_default.xml')
if detector.empty():
    print("Failed to load Haar cascade. Check the file path.")
    exit()

face_id = input("Enter a Numeric user ID here: ")
print("Taking samples, look at the camera...")

count = 0

while True:
    ret, img = cam.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1

        cv2.imwrite(f"auth\\samples\\face.{face_id}.{count}.jpg", gray[y:y+h, x:x+w])
        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff
    if k == 27 or count >= 100:
        break

print("Samples taken. Closing the program.")
cam.release()
cv2.destroyAllWindows()
