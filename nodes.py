import torchvision.transforms as T
import torch
from . import detector


class FacesDetectorNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_face",)
    FUNCTION = "recognize"
    CATEGORY = "Roman/FacesDetect"
    DESCRIPTION = """
    Detect faces on image
    """

    def recognize(self, image: torch.Tensor) -> torch.Tensor:
        transform = T.ToPILImage()
        transformed_image = transform(image)
        faces = detector.recognize(transformed_image)
        croped_image = transformed_image.crop(faces[0].face_location)

        transform = T.Compose([ T.PILToTensor() ])
        return transform(croped_image)
