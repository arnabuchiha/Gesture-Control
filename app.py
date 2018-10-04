import numpy as np
from keras.models import load_model
import cv2
from keras.preprocessing.image import img_to_array

cap = cv2.VideoCapture(0)
classes = 'NONE ONE TWO THREE FOUR FIVE'.split()
model=load_model('model_6cat.h5')
x0, y0, width = 200, 220, 300

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    # Our operations on the frame come here
    cv2.imshow('Gesture Recognition',frame)
    img=cv2.resize(frame,(128,128))
    # img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.GaussianBlur(img, (7,7), 3)
    # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    # ret, new = cv2.threshold(img, 25, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    img= img.astype("float") / 255.0
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    # Display the resulting frame
    
    pred=classes[np.argmax(model.predict(img))]
    print(pred)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()