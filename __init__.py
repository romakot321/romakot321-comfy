from .nodes import FacesDetectorNode

NODE_CLASS_MAPPINGS = { "FacesDetectorNode": FacesDetectorNode }

NODE_DISPLAY_NAME_MAPPINGS = { "FacesDetectorNode": "Detect Faces" }

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
