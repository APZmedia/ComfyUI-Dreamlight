import logging

NODE_CLASS_MAPPINGS = {}

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from .dreamlight_node import NODE_CLASS_MAPPINGS as node_mappings
    NODE_CLASS_MAPPINGS.update(node_mappings)
    logger.info("DreamLightNode imported successfully")
except ImportError as e:
    logger.error(f"Failed to import DreamLightNode: {e}")
    logger.error("Node will not be available - check dependencies")
except Exception as e:
    logger.error(f"Unexpected error importing DreamLightNode: {e}")

logger.info("DreamLight nodes loaded successfully")
logger.info(f"Final node mappings: {list(NODE_CLASS_MAPPINGS.keys())}")

__all__ = ['NODE_CLASS_MAPPINGS']
