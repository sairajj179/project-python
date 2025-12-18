import cv2
import os

def AuthenticateFace():
    flag = 0

    if not os.path.exists('auth\\trainer\\trainer.yml'):
        print("Trained model not found.")
        return 0

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('auth\\trainer\\trainer.yml')

    cascadePath = "auth\\haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)
    if faceCascade.empty():
        print("Failed to load Haar cascade.")
        return 0

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Could not open camera.")
        return 0

    cam.set(3, 640)
    cam.set(4, 480)

    font = cv2.FONT_HERSHEY_SIMPLEX
    names = ['', '', 'Ramjee']
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    while True:
        ret, img = cam.read()
        if not ret:
            print("Failed to grab frame.")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            id, accuracy = recognizer.predict(gray[y:y+h, x:x+w])
            if accuracy < 100:
                id_text = names[id]
                acc_text = f"  {round(100 - accuracy)}%"
                flag = 1
            else:
                id_text = "unknown"
                acc_text = f"  {round(100 - accuracy)}%"
                flag = 0

            cv2.putText(img, str(id_text), (x+5, y-5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, acc_text, (x+5, y+h-5), font, 1, (255, 255, 0), 1)

        cv2.imshow('camera', img)

        k = cv2.waitKey(10) & 0xff
        if k == 27 or flag == 1:
            break

    cam.release()
    cv2.destroyAllWindows()
    return flag
AuthenticateFace()
