import os
import logging
from dotenv import load_dotenv
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

# Spherical harmonics functions (from utils/env_lighting.py)
def calculate_spherical_harmonics(env_map, order=2):
    """
    Calculate spherical harmonics coefficients from an environment map.
    
    Args:
        env_map: PIL Image or numpy array of the environment map
        order: Order of spherical harmonics (default=2)
        
    Returns:
        coeffs: Spherical harmonics coefficients
    """
    if isinstance(env_map, Image.Image):
        ldr_image = np.array(env_map).astype(np.float32) / 255.0
    else:
        ldr_image = env_map.astype(np.float32) / 255.0
    
    # Apply gamma correction
    gamma = 2.2
    linear_env = np.power(ldr_image, gamma)
    
    # Calculate spherical harmonics coefficients
    # (Implementation based on DreamLight's approach)
    height, width, _ = linear_env.shape
    num_samples = 10000
    u = np.random.rand(num_samples)
    v = np.random.rand(num_samples)
    
    # Convert to spherical coordinates
    theta = 2 * np.arccos(np.sqrt(1 - u))
    phi = 2 * np.pi * v
    
    # Sample environment map
    x = (phi / (2 * np.pi)) * width
    y = (theta / np.pi) * height
    x = np.clip(x, 0, width - 1).astype(int)
    y = np.clip(y, 0, height - 1).astype(int)
    
    samples = linear_env[y, x]
    
    # Compute spherical harmonics basis functions
    # (Simplified implementation - actual SH calculation would be more complex)
    coeffs
