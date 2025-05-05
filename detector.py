from io import BytesIO
import math
import os
import dataclasses
import json
from PIL import Image
import numpy as np
from typing import Union, List
import dlib


@dataclasses.dataclass
class Response:
    face_polygon: list[tuple[int, int]]
    face_location: list[int]


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_81_face_landmarks.dat')


def _clockwiseangle_and_distance(point, origin, refvec):
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


def _extract_face_polygon(shape) -> list[tuple[int, int]]:
    face_polygon = []

    for num in list(range(0, 16 + 1)) + list(range(69, 80 + 1)):  # Facial contour indexes
        face_polygon.append((shape.parts()[num].x, shape.parts()[num].y))

    origin = face_polygon[0]
    refvec = [0, 1]
    face_polygon = list(
        sorted(
            face_polygon,
            key=lambda p: _clockwiseangle_and_distance(p, origin, refvec),
        )
    )

    return face_polygon


def recognize(image: Image.Image) -> list[Response]:
    responses = []
    image_numpy = np.array(image)
    image_numpy = np.fliplr(image)
    detected_faces = detector(image_numpy, 0)

    for k, d in enumerate(detected_faces):
        shape = predictor(image_numpy, d)
        face_polygon = _extract_face_polygon(shape)

        left = min(face_polygon, key=lambda i: i[0])
        right = max(face_polygon, key=lambda i: i[0])
        top = min(face_polygon, key=lambda i: i[1])
        bottom = max(face_polygon, key=lambda i: i[1])

        responses.append(
            Response(
                face_location=[
                    int(left[0]),
                    int(top[1]),
                    int(right[0]),
                    int(bottom[1]),
                ],
                face_polygon=face_polygon,
            )
        )
    return responses

