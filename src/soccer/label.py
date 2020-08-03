"""this module provide functionality to predict the role (e.g., player, referee, gatekeeper) of a detected person"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple

import cv2
import numpy as np


class Role(Enum):
    HOME = 0
    AWAY = 1
    REFEREE = 2
    HOME_GATEKEEPER = 3
    AWAY_GATEKEEPER = 4
    UNKNOWN = 5


class RoleAssigner(ABC):

    @abstractmethod
    def get_role(self, frame: np.ndarray, mask: np.ndarray) -> Role:
        pass


class ColorBasedRoleAssigner(RoleAssigner):

    def __init__(self, home_hue_range: Tuple[int, int], away_hue_range: Tuple[int, int], min_ratio = 0.2):
        super(ColorBasedRoleAssigner, self).__init__()
        self._home_hue_range = home_hue_range
        self._away_hue_range = away_hue_range
        self._min_ratio = 0.2

    def get_role(self, frame: np.ndarray, mask: np.ndarray) -> Role:
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        home_ratio = self._get_in_range_ratio(hls, mask, self._home_hue_range)
        away_ratio = self._get_in_range_ratio(hls, mask, self._away_hue_range)
        if (home_ratio < self._min_ratio) and (away_ratio < self._min_ratio):
            return Role.REFEREE
        elif home_ratio > away_ratio:
            return Role.HOME
        else:
            return Role.AWAY

    def _get_in_range_ratio(self, frame_hls: np.ndarray, mask: np.ndarray, color_range: Tuple[int, int]):
        l, h = color_range
        color_mask = cv2.inRange(frame_hls, lowerb=np.array([l, 0, 0]), upperb=np.array([h, 255, 255]))
        positive_mask = cv2.bitwise_and(color_mask, mask)
        return np.sum(positive_mask == 255)/np.sum(mask == 255)