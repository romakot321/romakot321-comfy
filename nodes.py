import torchvision.transforms as T
import torch.nn.functional as nnf
import torch
import math
from . import detector
from PIL import Image, ImageDraw


class FacesDetectorNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {}),
            }
        }

    CROPPED_FACES_COUNT = 4  # Изменить это для увелечения кол-ва выходных лиц

    RETURN_TYPES = ("IMAGE",) * (CROPPED_FACES_COUNT * 2)
    RETURN_NAMES = tuple("masked_image_" + str(i) for i in range(1, CROPPED_FACES_COUNT + 1)) + tuple(f"image_face_{i}" for i in range(1, CROPPED_FACES_COUNT + 1))
    FUNCTION = "recognize"
    CATEGORY = "Roman"
    DESCRIPTION = """
    Detect faces on image
    """

    def resize_tensor(self, tensor, leftoprightbottom: tuple[int, int, int, int]):
        cropped = T.ToPILImage()(torch.squeeze(tensor).permute(2, 0, 1)).crop(
            leftoprightbottom
        )
        return T.ToTensor()(cropped).permute(1, 2, 0).unsqueeze(3)

    def draw_masked_image(
        self, transformed_image: Image.Image, face: detector.Response
    ) -> Image.Image:
        transformed_image.paste(
            (0, 0, 0), (0, 0, transformed_image.size[0], transformed_image.size[1])
        )
        drawer = ImageDraw.Draw(transformed_image)
        drawer.polygon(face.face_polygon, fill=(255, 255, 255))
        return transformed_image

    def recognize(self, image: torch.Tensor) -> list[torch.Tensor]:
        transformed_image = T.ToPILImage()(torch.squeeze(image).permute(2, 0, 1))
        faces = detector.recognize(transformed_image)

        samples = image.movedim(-1, 1)
        H, W = samples.shape[3], samples.shape[2]

        tensors = []
        image_faces = []
        for face in faces:
            masked_image = self.draw_masked_image(transformed_image.copy(), face)
            masked_image_tensor = T.PILToTensor()(masked_image).permute(1, 2, 0)
            tensors.append(torch.unsqueeze(masked_image_tensor, dim=0))

            left, top, right, bottom = face.face_location
            image_face_tensor = image[:, top:bottom, left:right, :]
            image_face_tensor = nnf.interpolate(image_face_tensor, size=(512, 512), mode='bicubic', align_corners=False)
            image_faces.append(image_face_tensor)

        for _ in range(self.CROPPED_FACES_COUNT - len(image_faces)):
            r = torch.full([1, 64, 64, 1], 0)
            g = torch.full([1, 64, 64, 1], 0)
            b = torch.full([1, 64, 64, 1], 0)
            image_faces.append(torch.cat((r, g, b), dim=-1))
            tensors.append(torch.cat((r, g, b), dim=-1))

        return tensors + image_faces
