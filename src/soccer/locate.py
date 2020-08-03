from abc import ABC, abstractmethod
from typing import Tuple, Dict

import cv2
import numpy as np

from .detect import PersonDetectorWithFasterRCNN


class FieldDetector(ABC):

    @abstractmethod
    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """detect soccer field and line from a frame

        :param frame: a frame of image
        :return: two masks, the first is mask of field, the second is mask of lines
        """
        pass


class ThresholdFieldDetector(FieldDetector):

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        field_mask = self._get_field_mask(frame)
        field_frame = cv2.bitwise_and(frame, frame, mask=field_mask)
        line_mask = self._get_line_mask(field_frame)
        return field_mask, line_mask

    def _get_line_mask(self, frame: np.ndarray) -> np.ndarray:
        mask = cv2.inRange(frame, lowerb=np.array([150, 150, 150]), upperb=np.array([255, 255, 255]))
        return mask

    def _get_field_mask(self, frame: np.ndarray) -> np.ndarray:
        hsl = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        mask = cv2.inRange(hsl, np.array([30, 0, 0]), np.array([90, 256, 256]))

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours.sort(key=lambda x: cv2.contourArea(x))
        hull = cv2.convexHull(contours[-1])

        field_mask = np.zeros(shape=(frame.shape[0], frame.shape[1]), dtype=np.uint8)
        field_mask = cv2.drawContours(field_mask, [hull], -1, 255, -1)
        return field_mask


class Positioner(ABC):

    @abstractmethod
    def map(self, pixel_location: Tuple[int, int]) -> Tuple[int, int]:
        """map pixel coordinates to physical coordinates in form of (x, y)

        :param pixel_location: coordinates on an image, in form of (x, y)
        :return: physical coordinates on 2d map, in form of (x, y)
        """
        pass


class AnchorBasedPositioner(Positioner):

    def __init__(self, map_2d_anchors: Dict[str, Tuple[int, int]]):
        super(AnchorBasedPositioner, self).__init__()
        self._2d_map_anchors = map_2d_anchors
        self._perceptive_matrix = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])

    @property
    def perceptive_matrix(self):
        return self._perceptive_matrix

    def calibrate(self, frame_anchors: Dict[str, Tuple]):
        """Generate and set perceptive matrix

        :param frame_anchors: anchors pixel positions in a frame. It has to be collected manually
        """
        src_points, dst_points = list(), list()
        for k in frame_anchors:
            src_points.append(frame_anchors[k])
            dst_points.append(self._2d_map_anchors[k])
        m, _ = cv2.findHomography(np.array(src_points), np.array(dst_points))
        self._perceptive_matrix = m

    def map(self, pixel_location: Tuple[float, float]) -> Tuple[int, int]:
        h = np.array([pixel_location[0], pixel_location[1], 1.0], dtype=np.float32)
        m = self._perceptive_matrix
        transformed_anchor = m[:2, :] @ h
        transformed_anchor /= m[2] @ h
        return int(transformed_anchor[0]), int(transformed_anchor[1])


if __name__ == '__main__':
    img = cv2.imread("/Users/liuxuefe/PycharmProjects/smt/data/snap.png")
    filed_detector = ThresholdFieldDetector()
    player_detector = PersonDetectorWithFasterRCNN()

    boxes = player_detector.detect(img)
    player_mask = np.zeros(shape=(img.shape[0], img.shape[1]), dtype=np.uint8)
    for box in boxes:
        player_mask = cv2.rectangle(player_mask, (box.x1, box.y1), (box.x2, box.y2), color=255, thickness=-1)

    field_mask, line_mask = filed_detector.detect(img)

    line_mask[player_mask == 255] = 0

    cv2.imwrite("/Users/liuxuefe/PycharmProjects/smt/data/line.png", line_mask)

    cv2.imshow("field", cv2.resize(field_mask, dsize=None, fx=0.5, fy=0.5))
    cv2.imshow("line", cv2.resize(line_mask, dsize=None, fx=0.5, fy=0.5))

    cv2.waitKey(0)
    cv2.destroyAllWindows()
