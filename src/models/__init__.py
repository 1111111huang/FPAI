"""Model abstractions and implementations for FPAI."""

from .base_model import FPAIBaseModel, LRModel
from .model_manager import ModelManager

__all__ = ["FPAIBaseModel", "LRModel", "ModelManager"]
