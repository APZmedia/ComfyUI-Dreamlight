import logging
from nodes import NODE_CLASS_MAPPINGS

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("ComfyUI-Dreamlight package loaded")
nodes_list = list(NODE_CLASS_MAPPINGS.keys())
logger.info(f"Available nodes: {nodes_list}")

if 'DreamLightNode' in nodes_list:
    logger.info("Successfully registered DreamLightNode in image/postprocessing category")
else:
    logger.warning("DreamLightNode not found in mappings - check implementation")

__all__ = ['NODE_CLASS_MAPPINGS']
