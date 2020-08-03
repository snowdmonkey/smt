import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
import argparse

import torch
import torchvision
import cv2
import numpy as np
from torch import Tensor


logger = logging.getLogger(__name__)


@dataclass
class Box:
    x1: int
    x2: int
    y1: int
    y2: int


def convert_opencv_to_torch(img: np.ndarray) -> Tensor:
    """output of opencv imread of of size HxWxC with channel order BGR, torch expect tensor of size CxHxW with channel
    order RGB"""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.tensor(img) / 255.0
    img = img.permute(2, 0, 1)
    return img


class PersonDetector(ABC):

    def __init__(self):
        self._use_cuda = torch.cuda.is_available()

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Box]:
        """detect persons from a frame

        :param frame: an img from opencv
        :return: list of bounding boxes
        """
        pass


class PersonSegmentor(ABC):

    def __init__(self):
        self._use_cuda = torch.cuda.is_available()

    @abstractmethod
    def detect(self, frame: np.ndarray) -> Tuple[List[Box], List[np.ndarray], List[float]]:
        """instance segmentation for players
        :param frame: input frame form opencv
        :return: list of (bbox, instance_mask, score)
        """
        pass


class PersonSegmentorWithMaskRCNN(PersonSegmentor):

    def __init__(self, min_score: float = 0.5):
        super().__init__()

        self._min_score = min_score
        self._segmentor = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, max_size=1920).eval()

        if self._use_cuda:
            self._segmentor = self._segmentor.cuda()

    @torch.no_grad()
    def detect(self, frame: np.ndarray) -> Tuple[List[Box], List[np.ndarray], List[float]]:
        frame = convert_opencv_to_torch(frame)
        if self._use_cuda:
            frame = frame.cuda()

        predictions = self._segmentor([frame])[0]
        filter = (predictions["labels"] == 1) & (predictions["scores"] > self._min_score)
        boxes = [Box(x1=int(x[0]), y1=int(x[1]), x2=int(x[2]), y2=int(x[3])) for x in predictions["boxes"][filter]]
        masks = [x.cpu().numpy()[0] for x in predictions["masks"][filter]]
        scores = [x for x in predictions["scores"][filter]]
        return boxes, masks, scores


class PersonDetectorWithFasterRCNN(PersonDetector):

    def __init__(self, min_score: float = 0.5):
        super().__init__()

        self._min_score = min_score
        self._detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).eval()
        if self._use_cuda:
            self._detector = self._detector.cuda()

    @torch.no_grad()
    def detect(self, frame: np.ndarray) -> List[Box]:
        frame = convert_opencv_to_torch(frame)
        if self._use_cuda:
            frame = frame.cuda()
        predictions = self._detector([frame])[0]
        boxes = predictions["boxes"][(predictions["labels"] == 1) & (predictions["scores"] > self._min_score)]
        return [Box(x1=int(x[0]), y1=int(x[1]), x2=int(x[2]), y2=int(x[3])) for x in boxes]


# class PersonDetectorWithYolo5(PersonDetector):
#
#     def __init__(self):
#         pass


def process_img(img_path: Path):
    detector = PersonDetectorWithFasterRCNN()
    img = cv2.imread(str(img_path))
    boxes = detector.detect(img)
    for box in boxes:
        img = cv2.rectangle(img, (box.x1, box.y1), (box.x2, box.y2), (0, 0, 255), 2)
    cv2.imshow("detect", cv2.resize(img, dsize=None, fx=0.5, fy=0.5))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_video(video_path: Path):
    detector = PersonDetectorWithFasterRCNN()
    cap = cv2.VideoCapture(str(video_path))
    output_path = video_path.with_name(
        video_path.stem+datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M-%S")
    ).with_suffix(
        video_path.suffix
    )
    json_output_folder = video_path.with_name(video_path.stem+"-"+datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M-%S"))
    json_output_folder.mkdir()
    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'XVID'), 25, (960, 540))

    i = 0

    while True:
        ret, frame = cap.read()
        if ret is not True:
            break
        boxes = detector.detect(frame)
        for box in boxes:
            frame = cv2.rectangle(frame, (box.x1, box.y1), (box.x2, box.y2), (0, 0, 255), 2)

        out.write(cv2.resize(frame, dsize=(960, 540)))
        with (json_output_folder / f"{i}.json").open("w") as f:
            json.dump([x.__dict__ for x in boxes], f)

        if i % 100 == 0:
            logger.info(f"processed {i:,} frames")

        i += 1

    out.release()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("file_type")
    parser.add_argument("file_path")

    args = parser.parse_args()

    if args.file_type == "image":
        process_img(Path(args.file_path))
    elif args.file_type == "video":
        process_video(Path(args.file_path))
    else:
        raise ValueError(args.file_type)
