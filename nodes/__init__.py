import logging
from .dreamlight_node import NODE_CLASS_MAPPINGS

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("DreamLight nodes loaded successfully")
logger.info(f"Registered nodes: {list(NODE_CLASS_MAPPINGS.keys())}")

__all__ = ['NODE_CLASS_MAPPINGS']
