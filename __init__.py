import os
import sys
import logging
from dotenv import load_dotenv
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

# Add the parent directory to the path to allow importing utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.env_lighting import calculate_spherical_harmonics, generate_spherical_image

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

def load_hf_token():
    """Load HuggingFace token from .env file or environment variables"""
    try:
        # Try to load from .env file first
        load_dotenv()
        token = os.getenv("HF_TOKEN")
        
        if token:
            logger.info("✓ Using HuggingFace token from .env file")
            return token
        else:
            logger.warning("✗ No HF_TOKEN found in .env file")
        return None
    except Exception as e:
        logger.warning(f"Could not load .env file: {e}")
        return None

def setup_flux_directory():
    """Download FLUX.1-dev model from HuggingFace"""
    import folder_paths
    from huggingface_hub import snapshot_download, HfApi
    
    models_dir = folder_paths.models_dir
    flux_complete_dir = os.path.join(models_dir, "dreamlight", "flux_complete")
    os.makedirs(flux_complete_dir, exist_ok=True)
    
    logger.info("="*60)
    logger.info("FLIGHT CHECK: Setting up FLUX pipeline directory")
    logger.info(f"Target directory: {flux_complete_dir}")
    logger.info("="*60)
    
    # Check if already downloaded
    if os.path.exists(os.path.join(flux_complete_dir, "model_index.json")):
        logger.info("✓ FLUX model already downloaded")
        return flux_complete_dir
    
    # Get HuggingFace token
    hf_token = load_hf_token()
    if not hf_token:
        raise RuntimeError("No HuggingFace token found. Please set HF_TOKEN in your .env file")
    
    logger.info("Downloading FLUX.1-dev from HuggingFace...")
    logger.info("⚠️  FLUX.1-dev is a GATED model - authentication required!")
    logger.info("Make sure you have:")
    logger.info("1. Requested access to FLUX.1-dev: https://huggingface.co/black-forest-labs/FLUX.1-dev")
    logger.info("2. Set HF_TOKEN in your .env file")
    
    try:
        # First, let's check what files are actually in the repository
        logger.info("Checking repository structure...")
        api = HfApi()
        repo_files = api.list_repo_files(
            repo_id="black-forest-labs/FLUX.1-dev",
            token=hf_token
        )
        
        logger.info(f"Found {len(repo_files)} files in FLUX.1-dev repository:")
        for file in sorted(repo_files):
            if "transformer" in file and ".safetensors" in file:
                logger.info(f"  📁 {file}")
            elif file.endswith(".json"):
                logger.info(f"  📄 {file}")
            elif file.endswith(".safetensors"):
                logger.info(f"  💾 {file}")
        
        # Download the entire FLUX.1-dev repository
        logger.info("Downloading complete repository...")
        snapshot_download(
            repo_id="black-forest-labs/FLUX.1-dev",
            local_dir=flux_complete_dir,
            local_dir_use_symlinks=False,
            token=hf_token,
            resume_download=True
        )
        logger.info("✓ Download complete")
        
        # Verify what we actually got
        logger.info("Verifying downloaded files...")
        transformer_dir = os.path.join(flux_complete_dir, "transformer")
        if os.path.exists(transformer_dir):
            logger.info(f"Transformer directory contents:")
            for file in sorted(os.listdir(transformer_dir)):
                file_path = os.path.join(transformer_dir, file)
                if os.path.isfile(file_path):
                    size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    logger.info(f"  📄 {file} ({size_mb:.1f} MB)")
                else:
                    logger.info(f"  📁 {file}/")
        else:
            logger.warning("Transformer directory not found!")
        
        # Check model_index.json to see what files are expected
        model_index_path = os.path.join(flux_complete_dir, "model_index.json")
        if os.path.exists(model_index_path):
            logger.info("Checking model_index.json for expected files...")
            import json
            with open(model_index_path, 'r') as f:
                model_index = json.load(f)
            
            if "transformer" in model_index:
                transformer_info = model_index["transformer"]
                logger.info(f"Transformer config: {transformer_info}")
                
                # Check if there's a weight_map or similar
                if "weight_map" in transformer_info:
                    logger.info("Weight map found:")
                    for key, value in transformer_info["weight_map"].items():
                        logger.info(f"  {key} -> {value}")
        else:
            logger.warning("model_index.json not found!")
        
        return flux_complete_dir
        
    except Exception as e:
        if "authentication" in str(e).lower() or "token" in str(e).lower() or "gated" in str(e).lower():
            logger.error("✗ Authentication failed!")
            logger.error("This could be due to:")
            logger.error("1. Invalid or expired token")
            logger.error("2. No access granted to FLUX.1-dev model")
            logger.error("3. Token doesn't have required permissions")
            raise RuntimeError("Authentication failed for gated FLUX.1-dev model. Please verify your token and access permissions.")
        else:
            logger.error(f"✗ Download failed: {e}")
            raise

def validate_flux_directory(flux_dir):
    """Simple validation - just check if directory exists"""
    if not os.path.exists(flux_dir):
        logger.error(f"✗ FLUX directory does not exist: {flux_dir}")
        return False

    logger.info("✓ FLUX directory exists")
    return True

def validate_and_download_models():
    """Validate installation and download models if missing"""
    try:
        # Lazy import ComfyUI modules
        import folder_paths
        import comfy.utils
        from huggingface_hub import snapshot_download
        
        logger.info("DreamLight node initialized - validating models...")
        
        # Set up FLUX directory
        logger.info("Setting up FLUX pipeline directory...")
        flux_dir = setup_flux_directory()
        
        # Validate the directory
        if not validate_flux_directory(flux_dir):
            raise RuntimeError("FLUX directory validation failed. Please check the logs for detailed instructions on how to fix this issue.")
        
        # Download CLIP model
        models_dir = folder_paths.models_dir
        dreamlight_dir = os.path.join(models_dir, "dreamlight")
        clip_dir = os.path.join(dreamlight_dir, "CLIP")
        clip_models_dir = os.path.join(clip_dir, "models")
        os.makedirs(clip_models_dir, exist_ok=True)
        
        clip_config = os.path.join(clip_models_dir, "config.json")
        if not os.path.exists(clip_config):
            logger.info("No CLIP model found. Downloading CLIP folder...")
            hf_token = load_hf_token()
            if not hf_token:
                raise RuntimeError("HF_TOKEN required for CLIP download.")
            
            snapshot_download(
                repo_id="LYAWWH/DreamLight",
                local_dir=dreamlight_dir,
                local_dir_use_symlinks=False,
                allow_patterns=["CLIP/**"],
                token=hf_token,
                resume_download=True
            )
            logger.info("CLIP model downloaded successfully")
        else:
            logger.info("CLIP model already exists")
        
        logger.info("Testing FLUX pipeline loading...")
        
        # Test pipeline loading
        try:
            from diffusers import FluxPipeline
            import torch
            
            test_pipeline = FluxPipeline.from_pretrained(
                flux_dir,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            logger.info("✓ FLUX pipeline loaded successfully")
            del test_pipeline  # Clean up
            
        except Exception as e:
            logger.error(f"✗ FLUX pipeline loading failed: {e}")
            raise RuntimeError(f"FLUX pipeline loading failed: {e}")
        
        logger.info("✓ All models validated successfully")
        return flux_dir
        
    except Exception as e:
        logger.error(f"Model validation failed: {e}")
        raise

class DreamLightNode:
    """ComfyUI node for DreamLight relighting"""

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
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("relit_image",)
    FUNCTION = "process"
    CATEGORY = "image/postprocessing"
    
    def __init__(self):
        """Initialize node and validate models immediately"""
        # Run model validation on node creation, not during processing
        self.flux_dir = validate_and_download_models()

    def process(self, foreground_image, background_image, mask, prompt, seed, resolution, environment_map=None):
        # Models are already validated in __init__, so we can proceed directly
        
        try:
            from diffusers import FluxPipeline
            import comfy.utils
            
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
            
            # Get ComfyUI models directory
            import folder_paths
            models_dir = folder_paths.models_dir
            clip_dir = os.path.join(models_dir, "dreamlight", "CLIP")
            
            # Load pipeline from pre-validated directory
            logger.info("Loading FLUX pipeline from pre-validated directory...")
            pipeline = FluxPipeline.from_pretrained(
                self.flux_dir,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            pipeline.to(device)
            
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
            
            # Load CLIP model from local checkpoint
            clip_models_dir = os.path.join(clip_dir, "models")
            if not os.path.exists(clip_models_dir):
                raise FileNotFoundError(f"DreamLight CLIP model directory not found at {clip_models_dir}.")
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_models_dir).to(device)
            clip_processor = CLIPImageProcessor()
            
            # Run DreamLight pipeline
            generator = torch.Generator(device).manual_seed(seed) if seed else None
            mask_tensor = torch.from_numpy(mask_arr).to(device)
            mask_tensor = F.interpolate(mask_tensor.unsqueeze(0).unsqueeze(0), size=(h//16, w//16), mode='nearest')
            mask_tensor = mask_tensor.flatten(2).transpose(1, 2)[:, :, 0:1]
            
            image_embeds = image_encoder(clip_processor(images=bg, return_tensors="pt").pixel_values.to(device)).last_hidden_state
            output = pipeline(
                prompt=prompt,
                height=h,
                width=w,
                cond_fg_values=fg_tensor,
                cond_bg_values=bg_tensor,
                cond_env_values=env_tensor,
                generator=generator,
                num_inference_steps=20,
                guidance_scale=3.5
            ).images[0]
            
            # Convert output to ComfyUI tensor format
            output_np = np.array(output).astype(np.float32) / 255.0
            output_tensor = torch.from_numpy(output_np).unsqueeze(0)
            
            logger.info("✓ DreamLight processing complete")
            return (output_tensor,)
            
        except Exception as e:
            logger.error(f"DreamLight processing failed: {e}")
            raise RuntimeError(f"DreamLight processing failed: {e}")

# Node mapping
NODE_CLASS_MAPPINGS = {
    "DreamLightNode": DreamLightNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DreamLightNode": "DreamLight Node"
}

# Package initialization
logger.info("ComfyUI-Dreamlight package loaded successfully")
logger.info("Registered nodes: ['DreamLightNode']")
logger.info("Successfully registered DreamLightNode in image/postprocessing category")
