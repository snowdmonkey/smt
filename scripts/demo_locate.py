from typing import Tuple, List

import cv2
import yaml
import numpy as np

from soccer.locate import AnchorBasedPositioner, ThresholdFieldDetector
from soccer.detect import PersonDetectorWithFasterRCNN, PersonSegmentorWithMaskRCNN, Box
from soccer.label import Role, ColorBasedRoleAssigner

with open("config/pitch.yml", "r") as f:
    pitch_anchors = yaml.safe_load(f)["anchors"]

frame_anchors = {
    "D2": (682, 312),
    "C1": (682, 580),
    "C0": (682, 712),
    "C3": (682, 890),
    "C2": (1394, 712),
    "C4": (-8, 712)
}

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


def detect_frame(frame: np.ndarray) -> Tuple[np.ndarray, List[Box], List[Box], List[Box]]:

    field_mask, _ = field_detector.detect(frame)

    temp_frame = np.copy(frame)
    temp_frame[field_mask==0] = 0

    boxes, masks, scores = person_detector.detect(temp_frame)

    home_boxes, away_boxes, referee_boxes = list(), list(), list()

    for box, mask in zip(boxes, masks):
        mask = np.array((mask > 0.7) * 255, dtype=np.uint8)

        role = get_role(box, mask, frame)

        if role in (Role.HOME, Role.AWAY, Role.REFEREE):
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
        elif role == Role.REFEREE:
            referee_boxes.append(box)

    return frame, home_boxes, away_boxes, referee_boxes


frame = cv2.imread("demo/locate/snap.png")
pitch_map = cv2.imread("demo/locate/field.png")

positioner = AnchorBasedPositioner(map_2d_anchors=pitch_anchors)
positioner.calibrate(frame_anchors)
# player_detector = PersonDetectorWithFasterRCNN()

out_frame, home_boxes, away_boxes, referee_boxes = detect_frame(frame)

home_player_positions = [positioner.map(pixel_location=((box.x1+box.x2)/2, box.y2)) for box in home_boxes]
away_player_positions = [positioner.map(pixel_location=((box.x1+box.x2)/2, box.y2)) for box in away_boxes]
referee_player_positions = [positioner.map(pixel_location=((box.x1+box.x2)/2, box.y2)) for box in referee_boxes]

transformed_frame = cv2.warpPerspective(frame, positioner.perceptive_matrix, (pitch_map.shape[1], pitch_map.shape[0]))

out_perceptive = cv2.addWeighted(pitch_map, 0.5, transformed_frame, 0.5, 0)

out_position = np.copy(pitch_map)

for position in home_player_positions:
    out_position = cv2.circle(out_position, center=position, radius=5, color=COLOR[Role.HOME], thickness=1,
                              lineType=cv2.LINE_AA)

for position in away_player_positions:
    out_position = cv2.circle(out_position, center=position, radius=5, color=COLOR[Role.AWAY], thickness=1,
                              lineType=cv2.LINE_AA)

for position in referee_player_positions:
    out_position = cv2.circle(out_position, center=position, radius=5, color=COLOR[Role.REFEREE], thickness=1,
                              lineType=cv2.LINE_AA)

# out_detect = np.copy(frame)
# for box in boxes:
#     out_detect = cv2.rectangle(out_detect, (box.x1, box.y1), (box.x2, box.y2), color=(0, 0, 255), thickness=2,
#                                lineType=cv2.LINE_AA)

cv2.imwrite("demo/locate/perceived_frame.png", out_perceptive)
cv2.imwrite("demo/locate/position_frame.png", out_position)
cv2.imwrite("demo/locate/detect_frame.png", out_frame)

cv2.imshow("perceptive", out_perceptive)
cv2.imshow("position", out_position)
cv2.imshow("detect", out_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
