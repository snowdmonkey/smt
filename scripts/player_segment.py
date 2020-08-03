import json
import logging
from pathlib import Path
from typing import Tuple, List

from soccer.detect import PersonSegmentorWithMaskRCNN, Box
from soccer.label import ColorBasedRoleAssigner, Role
from soccer.locate import ThresholdFieldDetector
import cv2
import numpy as np

logger = logging.getLogger(__name__)

COLOR = {
    Role.HOME: (0, 0, 255),
    Role.AWAY: (255, 0, 0),
    Role.REFEREE: (255, 255, 0)
}

# frame = cv2.imread("data/snap.png")

person_detector = PersonSegmentorWithMaskRCNN()
field_detector = ThresholdFieldDetector()

role_assigner = ColorBasedRoleAssigner(home_hue_range=(5, 15), away_hue_range=(110, 120), min_ratio=0.5)


def get_role(box: Box, mask: np.ndarray, frame: np.ndarray):
    # person_frame = np.zeros_like(frame)
    # person_frame[mask] = frame[mask]
    frame = frame[box.y1:box.y2, box.x1:box.x2, :]
    mask = mask[box.y1:box.y2, box.x1:box.x2]
    role = role_assigner.get_role(frame, mask)
    return role


def detect_frame(frame: np.ndarray) -> Tuple[np.ndarray, List[Box], List[Box]]:

    field_mask, _ = field_detector.detect(frame)

    temp_frame = np.copy(frame)
    temp_frame[field_mask==0] = 0

    boxes, masks, scores = person_detector.detect(temp_frame)

    home_boxes, away_boxes = list(), list()

    for box, mask in zip(boxes, masks):
        mask = np.array((mask > 0.7) * 255, dtype=np.uint8)

        role = get_role(box, mask, frame)

        if role in (Role.HOME, Role.AWAY):
            blue_frame = np.zeros_like(frame)
            color = COLOR[role]
            blue_frame[:, :, 0] = color[0]
            blue_frame[:, :, 1] = color[1]
            blue_frame[:, :, 2] = color[2]
            frame_add_blue = cv2.addWeighted(frame, 0.5, blue_frame, 0.5, 0)
            frame[mask == 255] = frame_add_blue[mask == 255]
            frame = cv2.rectangle(frame, (box.x1, box.y1), (box.x2, box.y2), color, 1, cv2.LINE_AA)

        if role == Role.HOME:
            home_boxes.append(box)
        elif role == Role.AWAY:
            away_boxes.append(box)

    return frame, home_boxes, away_boxes


def main():
    cap = cv2.VideoCapture("data/1.mp4")
    out = cv2.VideoWriter("demo/detect/detect.avi", cv2.VideoWriter_fourcc(*'XVID'), 25, (1920, 1080))

    json_output_folder = Path("data/boxes_home_away")
    json_output_folder.mkdir(parents=True, exist_ok=True)

    i = 0

    while True:
        ret, frame = cap.read()
        if ret is not True:
            break
        out_frame, home_boxes, away_boxes = detect_frame(frame)
        out.write(out_frame)
        with (json_output_folder / f"{i}.json").open("w") as f:
            json.dump({
                "home": [x.__dict__ for x in home_boxes],
                "away": [x.__dict__ for x in away_boxes]
            }, f)

        if i % 100 == 0:
            logger.info(f"processed {i:,} frames")

        i += 1

    out.release()
    cap.release()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s")
    main()