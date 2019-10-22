import cv2
import pyttsx3
import keyboard
import ImageToSpeech as ITS
import numpy as np
from keras.models import load_model

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
textEngine = pyttsx3.init()

stroke = 2
color = (255, 255, 255)
model = load_model('model_CNN_V2.h5')
font = cv2.FONT_HERSHEY_SIMPLEX
Gender = {0: "Male", 1: "Female"}

def normalizeImage(img):
    IMG_SIZ = 120
    new_img = cv2.resize(img, (IMG_SIZ, IMG_SIZ))
    image = new_img.reshape((120, 120, 1))
    image = image.astype('float32') / 255
    return image

i=0
while(True):
    i+=1
    ret, frame = cap.read()
    if keyboard.is_pressed('i'):
        cv2.imwrite('C:/Users/Jitender kumar/Desktop/Intel HackFury2/Image To String Data/{index}.jpg'.format(index=i), frame)
        ITS.convertImageToString('C:/Users/Jitender kumar/Desktop/Intel HackFury2/Image To String Data/{index}.jpg'.format(index=i))
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame_gray, 1.2, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = frame_gray[y:y + h, x:x + w]     # region of interest
        roi_color = frame[y:y + h, x:x + w]
        img = np.array(roi_gray)
        img = normalizeImage(img)
        prediction = model.predict([[img]]).argmax()
        gender = Gender.get(prediction)
        text = "Some {} is coming towards you.".format(gender)
        if keyboard.is_pressed('p'):
            textEngine.say(text)
            textEngine.runAndWait()
            textEngine.stop()
        cv2.putText(frame, gender, (x, y), font, 1, color, stroke, cv2.LINE_AA)

    cv2.imshow('Eye For Blindness', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
