from io import BytesIO
import math
import os
import cv2
from scipy.spatial import distance as dist
import dataclasses
import json
from PIL import Image
import numpy as np
from typing import Union, List
import torch
import mediapipe as mp

images_directory = "images/"
mp_face_mesh = mp.solutions.face_mesh


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


face_oval_connections = [(10, 338), (338, 297), (297, 332), (332, 284),
        (284, 251), (251, 389), (389, 356), (356, 454),
        (454, 323), (323, 361), (361, 288), (288, 397),
        (397, 365), (365, 379), (379, 378), (378, 400),
        (400, 377), (377, 152), (152, 148), (148, 176),
        (176, 149), (149, 150), (150, 136), (136, 172),
        (172, 58), (58, 132), (132, 93), (93, 234),
        (234, 127), (127, 162), (162, 21), (21, 54),
        (54, 103), (103, 67), (67, 109), (109, 10)]


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


def recognize(image: Image.Image) -> List[Response]:
    responses = []
    img = np.array(image)
    image_height, image_width, _ = img.shape

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=5,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:

        results = face_mesh.process(img)

    for face_mesh in results.multi_face_landmarks:
        face_mesh = map(lambda i: (min(math.floor(i.x * image_width), image_width - 1), min(math.floor(i.y * image_height), image_height - 1)), face_mesh.landmark)
        face_mesh = list(face_mesh)
        landmark = []
        for i, j in face_oval_connections:
            landmark.append(face_mesh[j])

        left = min(landmark, key=lambda i: i[0])
        right = max(landmark, key=lambda i: i[0])
        top = min(landmark, key=lambda i: i[1])
        bottom = max(landmark, key=lambda i: i[1])

        face_polygon = (
            landmark
            # + landmark[pred_types["eyebrow1"]].tolist()
            # + landmark[pred_types["eyebrow2"]].tolist()
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
                left_eye=[],
                right_eye=[],
                nose=[],
                face_polygon=face_polygon,
                image_size=img.shape[:2][::-1],
            )
        )
    return responses
