import logging
from nodes import NODE_CLASS_MAPPINGS

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("ComfyUI-Dreamlight package loaded")
logger.info(f"Available nodes: {list(NODE_CLASS_MAPPINGS.keys())}")

__all__ = ['NODE_CLASS_MAPPINGS']
