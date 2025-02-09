
from .cleaning import DataCleaningPipeline
from .transforming import DataTransformationPipeline
from .unskew import SkewCorrectionPipeline
from .feature_eng import FeatureEngineeringPipeline

__version__ = "0.1.0"
__all__ = [
    "DataCleaningPipeline",
    "DataTransformationPipeline",
    "SkewCorrectionPipeline",
    "FeatureEngineeringPipeline",
]


def get_version():
    """Return current library version"""
    return __version__
