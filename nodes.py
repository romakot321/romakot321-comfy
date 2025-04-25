import torchvision.transforms as T
import torch
from . import detector
from PIL import Image, ImageDraw


class FilterMaskSize:
    """
    Separates a mask into multiple contiguous components. Returns the individual masks created as well as a MASK_MAPPING which can be used in other nodes when dealing with batches.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"mask": ("IMAGE",), "min_size": ("INT",)},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "separate"

    CATEGORY = "Roman"

    def separate(self, mask, min_size):
        _, H, W = mask.shape
        validated = []
        for i in range(len(mask)):
            if sum(sum(mask[i])).item() > min_size:
                validated.append(mask[i])
        result = torch.zeros([len(validated), H, W])
        for i in range(len(validated)):
            result[i] = validated[i]
        return (result,)


class FacesDetectorNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("masked_image",)
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
        self, transformed_image: Image, face: detector.Response
    ) -> Image:
        transformed_image.paste(
            (0, 0, 0), (0, 0, transformed_image.size[0], transformed_image.size[1])
        )
        drawer = ImageDraw.Draw(transformed_image)
        drawer.polygon(face.face_polygon, fill=(255, 255, 255))
        return transformed_image

    def recognize(self, image: torch.Tensor) -> torch.Tensor:
        transformed_image = T.ToPILImage()(torch.squeeze(image).permute(2, 0, 1))
        faces = detector.recognize(transformed_image)

        samples = image.movedim(-1, 1)
        H, W = samples.shape[3], samples.shape[2]

        tensors = []
        for face in faces:
            masked_image = self.draw_masked_image(transformed_image.copy(), face)
            masked_image_tensor = T.PILToTensor()(masked_image).permute(1, 2, 0)
            tensors.append(masked_image_tensor)

        result = torch.stack(tensors)

        return (result,)
