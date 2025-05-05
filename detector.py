from io import BytesIO
import math
import os
import face_alignment
import cv2
from scipy.spatial import distance as dist
import dataclasses
import json
from PIL import Image
import numpy as np
from typing import Union, List
import torch

images_directory = "images/"


@dataclasses.dataclass
class Response:
    face_polygon: List[tuple[int, int]]
    left_eye: list[tuple[int, int]]
    right_eye: list[tuple[int, int]]
    nose: list[tuple[int, int]]
    face_location: Union[List[int], None] = None
    image_size: Union[List[int], None] = None


@dataclasses.dataclass
class Request:
    filename: str
    task_id: str


pred_types = {
    "face": slice(0, 17),
    "eyebrow1": slice(17, 22),
    "eyebrow2": slice(22, 27),
    "nose": slice(27, 31),
    "nostril": slice(31, 36),
    "eye1": slice(36, 42),
    "eye2": slice(42, 48),
    "lips": slice(48, 60),
    "teeth": slice(60, 68),
}

if os.getenv("CPU"):
    detector = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        device="cpu",
        flip_input=True,
        face_detector="blazeface",
        face_detector_kwargs={
            "min_score_thresh": 0.8,
            "min_suppression_threshold": 0.5,
        },
    )
else:
    detector = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        device="cuda",
        dtype=torch.bfloat16,
        flip_input=True,
        face_detector="blazeface",
    )


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


def _get_eye(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    eye = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return eye


def _define_glasses(image_buffer: BytesIO, landmarks: np.ndarray):
    xmin = min(landmarks[pred_types["nostril"]], key=lambda i: i[0])[0]
    xmax = max(landmarks[pred_types["nostril"]], key=lambda i: i[0])[0]
    ymin = min(landmarks[pred_types["face"]], key=lambda i: i[1])[1]
    ymax = min(landmarks[pred_types["nose"]], key=lambda i: i[1])[1]

    img2 = Image.open(image_buffer)
    img2 = img2.crop((xmin, ymin, xmax, ymax))

    img_blur = cv2.GaussianBlur(np.array(img2), (3, 3), sigmaX=0, sigmaY=0)
    edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)

    edges_center = edges.T[(int(len(edges.T) / 2))]

    return 255 in edges_center


def _get_rotation(landmarks) -> float:
    """if abs(rotation) > 0.17, then face is profile. If abs(rotation) > 0.045, then face is half-profile"""
    face_points = landmarks[pred_types["face"]]
    left = face_points[0]
    right = face_points[-1]
    width = (
        max(landmarks, key=lambda i: i[0])[0] - min(landmarks, key=lambda i: i[0])[0]
    )

    distance = ((left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2) ** 0.5
    return 1 - distance / width


def clockwiseangle_and_distance(point, origin, refvec):
    vector = [point[0] - origin[0], point[1] - origin[1]]
    lenvector = math.hypot(vector[0], vector[1])
    if lenvector == 0:
        return -math.pi, 0

    normalized = [vector[0] / lenvector, vector[1] / lenvector]
    dotprod = normalized[0] * refvec[0] + normalized[1] * refvec[1]
    diffprod = refvec[1] * normalized[0] - refvec[0] * normalized[1]
    angle = math.atan2(diffprod, dotprod)

    if angle < 0:
        return 2 * math.pi + angle, lenvector
    return angle, lenvector


def calculate_forehead_coordinates(brow_left, brow_right, eye_left, eye_right, chin, nose_tip) -> list[tuple[int, int]]:
    # Вычисляем среднюю точку бровей
    brow_mid = np.mean([brow_left, brow_right], axis=0)

    # Вычисляем высоту лба как расстояние между кончиком носа и подбородком
    height_nose_to_chin = np.linalg.norm(nose_tip - chin)

    # Вычисляем координаты лба
    forehead_x = brow_mid[0]
    forehead_y = brow_mid[1] + height_nose_to_chin  # Поднимаем на высоту

    return [(eye_right[0], forehead_y), (forehead_x, forehead_y), (eye_left[0], forehead_y)]


def recognize(image: Image.Image) -> List[Response]:
    responses = []
    img = np.array(image)
    image_height, image_width, _ = img.shape

    face_landmarks_list = detector.get_landmarks(img)

    for landmark in face_landmarks_list:
        left = min(landmark, key=lambda i: i[0])
        right = max(landmark, key=lambda i: i[0])
        top = min(landmark, key=lambda i: i[1])
        bottom = max(landmark, key=lambda i: i[1])

        face_polygon = (
            landmark[pred_types["face"]].tolist()
            + landmark[pred_types["eyebrow1"]].tolist()
            + landmark[pred_types["eyebrow2"]].tolist()
            + calculate_forehead_coordinates(
                min(landmark[pred_types["eyebrow1"]], key=lambda i: i[0]),
                max(landmark[pred_types["eyebrow2"]], key=lambda i: i[0]),
                min(landmark[pred_types["eye1"]], key=lambda i: i[0]),
                max(landmark[pred_types["eye2"]], key=lambda i: i[0]),
                bottom,
                max(landmark[pred_types["nose"]], key=lambda i: i[1])
            )
        )
        origin = face_polygon[0]
        refvec = [0, 1]
        face_polygon = map(lambda i: (int(i[0]), int(i[1])), face_polygon)
        face_polygon = list(
            sorted(
                face_polygon,
                key=lambda p: clockwiseangle_and_distance(p, origin, refvec),
            )
        )[1:]

        responses.append(
            Response(
                face_location=[
                    int(left[0]),
                    int(top[1]),
                    int(right[0]),
                    int(bottom[1]),
                ],
                left_eye=landmark[pred_types["eye1"]].tolist(),
                right_eye=landmark[pred_types["eye2"]].tolist(),
                nose=landmark[pred_types["nostril"]].tolist(),
                face_polygon=face_polygon,
                image_size=img.shape[:2][::-1],
            )
        )
    return responses
