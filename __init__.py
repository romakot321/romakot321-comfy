from .nodes import FacesDetectorNode, FilterMaskSize

NODE_CLASS_MAPPINGS = {
    "FacesDetectorNode": FacesDetectorNode,
    "FilterMaskSize": FilterMaskSize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FacesDetectorNode": "Detect Faces",
    "FilterMaskSize": "Filer masks by size",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
