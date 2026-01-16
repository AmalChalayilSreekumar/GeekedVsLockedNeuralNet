import cv2
import time
import os
import PredictionTest

''''
***Major Credit to haarcascade on GitHub***
For more information go to README file

'''

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

if face_cascade.empty():
    print("Error loading cascade files")
    exit()

cap = cv2.VideoCapture(0)
count = 0
label = "Locked In" #Initialize as locked in

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)

    if not ret:
        print("Failed to grab frame")
        break
        
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Face box size increment 
        incScale = 30 

        x1 = max(x - incScale, 0)
        y1 = max(y - incScale, 0)
        x2 = min(x + w + incScale, img.shape[1])
        y2 = min(y + h + incScale, img.shape[0])


        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)


        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        text_x = x
        text_y = y - 10
        if text_y - th < 0:
            text_y = y + th + 10  # move inside the box if above-screen

        cv2.rectangle(
            img,
            (text_x, text_y - th - baseline),
            (text_x + tw, text_y + baseline),
            (255, 255, 0),
            -1
        )


        # Text on top of background
        cv2.putText(img, label, (text_x, text_y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

    count+=1
    if count % 25 == 0 and len(faces) > 0:
        # Crop the expanded face rectangle
        face_crop = img[y1:y2, x1:x2]
        label = PredictionTest.predict(face_crop)



    cv2.imshow('img', img)

    if (cv2.waitKey(30) & 0xff) == 27:
        break

cap.release()
cv2.destroyAllWindows()
