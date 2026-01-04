from os import listdir
import cv2
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import load_model

cap = cv2.VideoCapture(0)

# Định nghĩa class
class_name = ['Conor', 'Faker', 'SonTung']

# Load model đã train
my_model = load_model("vggmodel-002.h5")
my_model.load_weights("weights-41-0.93-001.keras")

while (True):
    ret, image_org = cap.read()
    if not ret:
        continue

    image_org = cv2.resize(image_org, dsize=None, fx=0.5, fy=0.5)

    image = image_org.copy()
    image = cv2.resize(image, dsize=(300, 300))
    image = image.astype('float') * 1. / 255

    # Chuyển thành tensor
    image = np.expand_dims(image, axis=0)

    # Predict
    predict = my_model.predict(image)
    print("This picture is: ", class_name[np.argmax(predict[0])], (predict[0]))

    print("--")
    print(predict[0])
    print(np.argmax(predict[0]))
    print(class_name[np.argmax(predict[0])])

    print(np.max(predict[0], axis=0))
    if np.max(predict) >= 0.8:
        # Show image
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 50)
        fontScale = 1.5
        color = (0, 255, 0)
        thickness = 2

        cv2.putText(image_org, class_name[np.argmax(predict)], org, font,
                fontScale, color, thickness, cv2.LINE_AA)

    cv2.imshow("Picture", image_org)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()