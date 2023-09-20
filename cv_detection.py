import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model("fire_model.h5")
path = "test_fire.mp4"

cap = cv2.VideoCapture(path)

while True:
	ret, frame = cap.read()
	img = np.asarray(frame)
	img = cv2.resize(img, (224, 224))
	img = img / 255
	img = img.reshape(1, 224, 224, 3)
	predicted = model.predict(img, verbose=0)
	pred = np.argmax(predicted[0])
	prob = predicted[0][pred]
	prob = "{:.2f}%".format(prob * 100)
	if pred == 1:
		label = "Fire"
	else:
		label = "No Fire!"

	cv2.putText(frame, label, (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 125, 120), 2)
	cv2.putText(frame, prob, (35, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 125, 120), 2)
	cv2.imshow("frame", frame)

	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

cap.release()
cv2.destroyAllWindows()

