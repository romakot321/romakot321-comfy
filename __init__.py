from .nodes import FacesDetectorNode, FilterMaskSize

NODE_CLASS_MAPPINGS = {
    "Faces Crop&Mask": FacesDetectorNode,
    "FilterMaskSize": FilterMaskSize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Faces Crop&Mask": "Faces Crop&Mask",
    "FilterMaskSize": "Filer masks by size",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
