"""RagOps — a retrieval pipeline that evolves stage by stage, with a quality gate per stage."""

from ragops.config import PipelineConfig
from ragops.pipeline import RagPipeline

__all__ = ["PipelineConfig", "RagPipeline"]
__version__ = "0.1.0"
