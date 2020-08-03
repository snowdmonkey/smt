import json
from pathlib import Path

import numpy as np
import cv2

from soccer.track import Sort
from soccer.detect import Box


box_home = Path("data/boxes_home_away")
cap = cv2.VideoCapture("/Users/liuxuefe/PycharmProjects/smt/data/1.mp4")
out = cv2.VideoWriter("demo/track/track2.avi", cv2.VideoWriter_fourcc(*'XVID'), 25, (1920, 1080))

tracker = Sort(max_age=50, iou_threshold=0.1)

for i in range(1, 10000):

    ret, frame = cap.read()

    if i > 3000:

        with (box_home / f"{i}.json").open("r") as f:
            d = json.load(f)
        boxes = [Box(**x) for x in d["away"]]

        if len(boxes) == 0:
            continue

        dets = np.array([[box.x1, box.y1, box.x2, box.y2] for box in boxes])

        res = tracker.update(dets)

        for row in res:
            row = [int(x) for x in row]
            frame = cv2.rectangle(frame, (row[0], row[1]), (row[2], row[3]), color=(0, 0, 255), thickness=2)
            frame = cv2.putText(frame, str(row[4]), (row[0], row[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        out.write(frame)
        cv2.imshow("result", cv2.resize(frame, dsize=None, fx=0.5, fy=0.5))
        if cv2.waitKey(40) == ord("q"):
            break

cv2.destroyAllWindows()
out.release()
cap.release()