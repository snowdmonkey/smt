import cv2
import numpy as np

from soccer.locate import ThresholdFieldDetector

detector = ThresholdFieldDetector()
# frame = cv2.imread("data/snap.png")
# field_mask, _ = detector.detect(frame)

cap = cv2.VideoCapture("data/1.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    field_mask, _ = detector.detect(frame)
    frame[field_mask==0] = 0

    cv2.imshow("out", cv2.resize(frame, dsize=None, fx=0.5, fy=0.5))
    if cv2.waitKey(1)==ord("q"):
        break

cv2.destroyAllWindows()