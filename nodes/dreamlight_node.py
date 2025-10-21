import torch
import numpy as np
from PIL import Image
import os
import sys
import logging
import torch.nn.functional as F
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from diffusers import FluxPipeline
# Add the parent directory to the path to allow importing utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.env_lighting import calculate_spherical_harmonics, generate_spherical_image

# Define NODE_CLASS_MAPPINGS early to ensure registration
NODE_CLASS_MAPPINGS = {
    "DreamLightNode": None  # Will be set after class definition
}

# Set up logging
logger = logging.getLogger(__name__)

def validate_and_download_models():
    """Validate installation and download models if missing"""
    try:
        # Lazy import ComfyUI modules
        import folder_paths
        import comfy.utils
        
        # Get ComfyUI models directory
        models_dir = folder_paths.models_dir
        dreamlight_dir = os.path.join(models_dir, "dreamlight")
        flux_dir = os.path.join(dreamlight_dir, "FLUX", "transformer")
        clip_dir = os.path.join(dreamlight_dir, "CLIP")
        
        # Create directories
        os.makedirs(flux_dir, exist_ok=True)
        os.makedirs(clip_dir, exist_ok=True)
        
        # Check and download FLUX transformer
        flux_path = os.path.join(flux_dir, "model.pth")
        if not os.path.exists(flux_path):
            logger.info("Downloading DreamLight FLUX transformer model...")
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id="LYAWWH/DreamLight",
                filename="FLUX/transformer/model.pth",
                local_dir=flux_dir,
                local_dir_use_symlinks=False
            )
            logger.info("FLUX transformer downloaded successfully")
        else:
            logger.info("FLUX transformer model already exists")
        
        # Check and download CLIP model
        clip_config = os.path.join(clip_dir, "config.json")
        clip_model = os.path.join(clip_dir, "pytorch_model.bin")
        if not os.path.exists(clip_config) or not os.path.exists(clip_model):
            logger.info("Downloading CLIP model...")
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id="LYAWWH/DreamLight",
                local_dir=clip_dir,
                local_dir_use_symlinks=False,
                allow_patterns=["CLIP/models/*"]
            )
            logger.info("CLIP model downloaded successfully")
        else:
            logger.info("CLIP model already exists")
        
        # Validate GPU
        if torch.cuda.is_available():
            logger.info(f"GPU available: {torch.cuda.get_device_name()}")
            logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            logger.warning("No GPU detected - performance may be slow")
        
        logger.info("Installation validation completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Installation validation failed: {e}")
        return False

class DreamLightNode:
    """Custom ComfyUI node for DreamLight image relighting"""
    
    CATEGORY = "image/postprocessing"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("relit_image",)
    FUNCTION = "process"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "foreground_image": ("IMAGE",),
                "background_image": ("IMAGE",),
                "mask": ("MASK",),
                "prompt": ("STRING", {"default": "harmonious, natural, photorealistic"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32-1}),
                "resolution": ("INT", {"default": 1024, "min": 256, "max": 2048}),
            },
            "optional": {
                "environment_map": ("IMAGE",),
            }
        }

    def process(self, foreground_image, background_image, mask, prompt, seed, resolution, environment_map=None):
        # Lazy import ComfyUI modules
        import folder_paths
        import comfy.utils
        
        # Run validation and download if needed (now safe in process method)
        if not hasattr(self, '_models_validated'):
            if validate_and_download_models():
                self._models_validated = True
                logger.info("DreamLight models validated for this session")
            else:
                logger.warning("Model validation failed - using fallback")
                self._models_validated = False
        
        # Convert ComfyUI tensors to numpy arrays
        fg_np = foreground_image[0].cpu().numpy() * 255.0
        bg_np = background_image[0].cpu().numpy() * 255.0
        mask_np = mask[0].cpu().numpy() * 255.0
        
        # Convert to PIL Images
        fg = Image.fromarray(fg_np.astype(np.uint8))
        bg = Image.fromarray(bg_np.astype(np.uint8))
        mask_img = Image.fromarray(mask_np.astype(np.uint8))
        
        # Resize images to target resolution
        w0, h0 = fg.size
        ratio = resolution / max(w0, h0)
        w, h = int(ratio * w0), int(ratio * h0)
        w, h = w - w % 16, h - h % 16  # Ensure divisible by 16
        
        fg = fg.resize((w, h))
        bg = bg.resize((w, h))
        mask_img = mask_img.resize((w, h))
        
        # Convert to numpy arrays for processing
        fg_arr = np.array(fg).astype(np.float32)
        bg_arr = np.array(bg).astype(np.float32)
        mask_arr = np.array(mask_img.convert("L")).astype(np.float32) / 255.0
        matting = np.stack([mask_arr] * 3, axis=-1)
        
        # Apply matting to foreground and background
        fg_arr = fg_arr * matting + 127.0 * (1 - matting)
        fg_arr = np.clip(fg_arr, 0, 255)
        fg_tensor = torch.from_numpy(fg_arr).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
        
        bg_arr = bg_arr * (1 - matting) + 127.0 * matting
        bg_arr = np.clip(bg_arr, 0, 255)
        bg_tensor = torch.from_numpy(bg_arr).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
        
        # Process environment map if provided
        if environment_map is not None:
            env_np = environment_map[0].cpu().numpy() * 255.0
            env_img = Image.fromarray(env_np.astype(np.uint8)).resize((w, h))
            coeffs = calculate_spherical_harmonics(env_img)
            env_lighting = generate_spherical_image(coeffs, h, w)
            env_tensor = torch.from_numpy(env_lighting).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
        else:
            env_tensor = torch.zeros_like(fg_tensor)
        
        # Move tensors to GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fg_tensor = fg_tensor.to(device)
        bg_tensor = bg_tensor.to(device)
        env_tensor = env_tensor.to(device)
        
        # Get node directory to resolve relative paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(current_dir)
        
        # Load DreamLight transformer weights
        flux_model_path = os.path.join(root_dir, "ckpt", "FLUX", "transformer", "model.pth")
        transformer_weights = torch.load(flux_model_path, map_location=device)
        
        # Initialize pipeline
        pipeline = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # Apply DreamLight modifications to transformer
        transformer = pipeline.transformer
        extra_channels = 1 + (1 if environment_map is not None else 0)
        x_embedder = transformer.x_embedder
        new_x_embedder = torch.nn.Linear(
            x_embedder.in_features * (1 + 1 + extra_channels),
            x_embedder.out_features
        )
        new_x_embedder.weight.data.zero_()
        new_x_embedder.weight.data[:, :x_embedder.in_features].copy_(x_embedder.weight.data)
        new_x_embedder.bias.data.copy_(x_embedder.bias.data)
        transformer.x_embedder = new_x_embedder
        transformer.load_state_dict(transformer_weights)
        pipeline.to(device)
        
        # Load CLIP model from local checkpoint
        clip_model_path = os.path.join(root_dir, "ckpt", "CLIP")
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_model_path).to(device)
        clip_processor = CLIPImageProcessor()
        
        # Run DreamLight pipeline
        generator = torch.Generator(device).manual_seed(seed) if seed else None
        mask_tensor = torch.from_numpy(mask_arr).to(device)
        mask_tensor = F.interpolate(mask_tensor.unsqueeze(0).unsqueeze(0), scale_factor=1/16, mode='nearest')
        mask_tensor = mask_tensor.flatten(2).transpose(1, 2)[:, :, 0:1]
        
        image_embeds = image_encoder(clip_processor(images=bg, return_tensors="pt").pixel_values.to(device)).last_hidden_state
        output = pipeline(
            prompt=prompt,
            height=h,
            width=w,
            cond_fg_values=fg_tensor,
            cond_bg_values=bg_tensor,
            cond_env_values=env_tensor,
            generator=generator
        ).images[0]
        
        # Convert output to ComfyUI tensor format
        output_np = np.array(output).astype(np.float32) / 255.0
        output_tensor = torch.from_numpy(output_np).unsqueeze(0)
        return (output_tensor,)
        
NODE_CLASS_MAPPINGS["DreamLightNode"] = DreamLightNode
