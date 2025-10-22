import torch
import numpy as np
from PIL import Image
import os
import logging

# Define NODE_CLASS_MAPPINGS early to ensure registration
NODE_CLASS_MAPPINGS = {
    "DreamLightNode": None  # Will be set after class definition
}

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_hf_token():
    """Load HuggingFace token from .env file or environment variables"""
    try:
        from dotenv import load_dotenv
        import os
        
        # Load .env file if it exists
        load_dotenv()
        
        # Try to get token from environment
        token = os.getenv('HF_TOKEN')
        
        if token:
            logger.info("Using HuggingFace token from .env file")
            return token
        else:
            logger.warning("No HF_TOKEN found in .env file - downloads may fail if authentication is required")
            return None
            
    except ImportError:
        logger.warning("python-dotenv not installed - cannot load .env file")
        return None
    except Exception as e:
        logger.warning(f"Error loading HuggingFace token: {e}")
        return None

def download_flux_configs(target_dir, hf_token=None):
    """Download only config files for FLUX.1-dev model"""
    from huggingface_hub import hf_hub_download
    
    config_files = [
        "model_index.json",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "text_encoder_2/config.json", 
        "tokenizer/tokenizer_config.json",
        "tokenizer/vocab.json",
        "tokenizer/merges.txt",
        "tokenizer_2/tokenizer_config.json",
        "tokenizer_2/vocab.json",
        "tokenizer_2/merges.txt",
        "transformer/config.json",
        "vae/config.json"
    ]
    
    for config_file in config_files:
        try:
            hf_hub_download(
                repo_id="black-forest-labs/FLUX.1-dev",
                filename=config_file,
                local_dir=target_dir,
                local_dir_use_symlinks=False,
                token=hf_token
            )
            logger.info(f"Downloaded {config_file}")
        except Exception as e:
            logger.warning(f"Could not download {config_file}: {e}")

def find_flux_transformer_weights():
    """Find existing FLUX transformer weights in ComfyUI directories"""
    import folder_paths
    
    models_dir = folder_paths.models_dir
    search_paths = [
        os.path.join(models_dir, "unet"),
        os.path.join(models_dir, "diffusers"),
        os.path.join(models_dir, "checkpoints"),
        os.path.join(models_dir, "unet", "FLUX1"),
        os.path.expanduser("~/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev")
    ]
    
    logger.info("Searching for FLUX transformer weights...")
    
    for search_path in search_paths:
        if not os.path.exists(search_path):
            logger.debug(f"  ✗ {search_path} (not found)")
            continue
        
        logger.debug(f"  → Scanning {search_path}")
        
        for root, dirs, files in os.walk(search_path):
            for file in files:
                if file.lower() in ["flux1-dev.safetensors", "diffusion_pytorch_model.safetensors"]:
                    full_path = os.path.join(root, file)
                    size_mb = os.path.getsize(full_path) / (1024 * 1024)
                    logger.info(f"  ✓ Found: {full_path} ({size_mb:.1f} MB)")
                    return full_path
    
    logger.warning("  ✗ No transformer weights found")
    return None

def setup_complete_flux_directory():
    """Create a complete FLUX pipeline directory with local transformer weights"""
    import folder_paths
    from huggingface_hub import snapshot_download
    
    models_dir = folder_paths.models_dir
    
    # Create dedicated directory for complete FLUX pipeline
    flux_complete_dir = os.path.join(models_dir, "dreamlight", "flux_complete")
    os.makedirs(flux_complete_dir, exist_ok=True)
    
    logger.info("="*60)
    logger.info("FLIGHT CHECK: Setting up complete FLUX pipeline directory")
    logger.info(f"Target directory: {flux_complete_dir}")
    logger.info("="*60)
    
    # Check if already set up
    required_components = {
        "model_index.json": os.path.join(flux_complete_dir, "model_index.json"),
        "vae": os.path.join(flux_complete_dir, "vae", "config.json"),
        "text_encoder": os.path.join(flux_complete_dir, "text_encoder", "config.json"),
        "text_encoder_2": os.path.join(flux_complete_dir, "text_encoder_2", "config.json"),
        "scheduler": os.path.join(flux_complete_dir, "scheduler", "scheduler_config.json"),
        "transformer": os.path.join(flux_complete_dir, "transformer", "config.json")
    }
    
    all_exist = all(os.path.exists(path) for path in required_components.values())
    
    if all_exist:
        logger.info("✓ All required components already exist")
        return flux_complete_dir
    
    logger.info("✗ Missing components detected, will download from HuggingFace")
    
    # Find local transformer weights
    transformer_weights_path = find_flux_transformer_weights()
    
    if transformer_weights_path:
        logger.info(f"✓ Found local transformer weights: {transformer_weights_path}")
    else:
        logger.warning("✗ No local transformer weights found")
        logger.info("Will download complete model from HuggingFace")
    
    # Download complete model from HuggingFace
    logger.info("Downloading FLUX.1-dev from HuggingFace...")
    logger.info("⚠️  FLUX.1-dev is a GATED model - authentication required!")
    logger.info("This will download ~23GB, but only non-transformer components will be kept")
    
    hf_token = load_hf_token()
    
    if not hf_token:
        logger.error("✗ No HuggingFace token found!")
        logger.error("FLUX.1-dev is a gated model that requires authentication.")
        logger.error("Please set HF_TOKEN in your .env file:")
        logger.error("1. Get your token from: https://huggingface.co/settings/tokens")
        logger.error("2. Request access to FLUX.1-dev: https://huggingface.co/black-forest-labs/FLUX.1-dev")
        logger.error("3. Add to .env file: HF_TOKEN=your_token_here")
        raise RuntimeError("HuggingFace token required for gated FLUX.1-dev model. Please set HF_TOKEN in .env file and request access to the model.")
    
    logger.info(f"✓ Using HuggingFace token for authentication")
    
    try:
        snapshot_download(
            repo_id="black-forest-labs/FLUX.1-dev",
            local_dir=flux_complete_dir,
            local_dir_use_symlinks=False,
            token=hf_token,
            ignore_patterns=["transformer/*"] if transformer_weights_path else []
        )
        logger.info("✓ Download complete")
    except Exception as e:
        if "authentication" in str(e).lower() or "token" in str(e).lower() or "gated" in str(e).lower():
            logger.error("✗ Authentication failed!")
            logger.error("This could be due to:")
            logger.error("1. Invalid or expired token")
            logger.error("2. No access granted to FLUX.1-dev model")
            logger.error("3. Token doesn't have required permissions")
            logger.error("Please check your token and request access to the model.")
            raise RuntimeError("Authentication failed for gated FLUX.1-dev model. Please verify your token and access permissions.")
        else:
            logger.error(f"✗ Download failed: {e}")
            raise
    
    # If we have local transformer weights, copy/link them
    if transformer_weights_path:
        transformer_dir = os.path.join(flux_complete_dir, "transformer")
        os.makedirs(transformer_dir, exist_ok=True)
        
        # Copy the weights file
        import shutil
        target_weights = os.path.join(transformer_dir, "diffusion_pytorch_model.safetensors")
        if not os.path.exists(target_weights):
            logger.info(f"Copying transformer weights to {target_weights}")
            shutil.copy2(transformer_weights_path, target_weights)
            logger.info("✓ Transformer weights copied")
    
    return flux_complete_dir

def validate_flux_directory(flux_dir):
    """Validate that FLUX directory has all required components"""
    logger.info("="*60)
    logger.info("FLIGHT CHECK: Validating FLUX pipeline directory")
    logger.info(f"Directory: {flux_dir}")
    logger.info("="*60)
    
    checks = []
    
    # Check model_index.json
    model_index_path = os.path.join(flux_dir, "model_index.json")
    if os.path.exists(model_index_path):
        logger.info("✓ model_index.json found")
        checks.append(True)
    else:
        logger.error("✗ model_index.json NOT found")
        checks.append(False)
    
    # Check each component
    components = {
        "vae": ["config.json", "diffusion_pytorch_model.safetensors"],
        "text_encoder": ["config.json", "model.safetensors"],
        "text_encoder_2": ["config.json", "model.safetensors"],
        "transformer": ["config.json", "diffusion_pytorch_model.safetensors"],
        "tokenizer": ["tokenizer_config.json"],
        "tokenizer_2": ["tokenizer_config.json"],
        "scheduler": ["scheduler_config.json"]
    }
    
    for component, required_files in components.items():
        component_dir = os.path.join(flux_dir, component)
        if os.path.exists(component_dir):
            logger.info(f"✓ {component}/ directory found")
            for req_file in required_files:
                file_path = os.path.join(component_dir, req_file)
                if os.path.exists(file_path):
                    size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    logger.info(f"  ✓ {req_file} ({size_mb:.1f} MB)")
                    checks.append(True)
                else:
                    logger.error(f"  ✗ {req_file} NOT found")
                    checks.append(False)
        else:
            logger.error(f"✗ {component}/ directory NOT found")
            checks.append(False)
    
    logger.info("="*60)
    if all(checks):
        logger.info("✓ ALL FLIGHT CHECKS PASSED")
        logger.info("="*60)
        return True
    else:
        logger.error(f"✗ FLIGHT CHECKS FAILED ({sum(checks)}/{len(checks)} passed)")
        logger.info("="*60)
        return False


def validate_and_download_models():
    """Validate installation and download models if missing"""
    try:
        # Lazy import ComfyUI modules
        import folder_paths
        import comfy.utils
        
        # Get ComfyUI models directory
        models_dir = folder_paths.models_dir
        dreamlight_dir = os.path.join(models_dir, "dreamlight")
        flux_dir = os.path.join(dreamlight_dir, "FLUX", "DreamLight-FLUX", "transformer")
        clip_dir = os.path.join(dreamlight_dir, "CLIP")
        
        # Create directories
        os.makedirs(flux_dir, exist_ok=True)
        os.makedirs(clip_dir, exist_ok=True)
        
        # Check and download FLUX transformer
        # Download entire FLUX folder and scan for available model files
        logger.info("Checking FLUX model files...")
        
        # Scan for any model files recursively in FLUX directory
        def scan_for_model_files(directory):
            model_files = []
            if os.path.exists(directory):
                for root, dirs, files in os.walk(directory):
                    for file in files:
                        if file.endswith(('.pth', '.safetensors', '.bin')):
                            model_files.append(os.path.join(root, file))
            return model_files
        
        flux_model_files = scan_for_model_files(flux_dir)
        
        if not flux_model_files:
            logger.info("No FLUX model files found. Downloading FLUX folder...")
            from huggingface_hub import snapshot_download, hf_hub_download
            import time
            
            download_success = False
            for attempt in range(3):
                try:
                    logger.info(f"FLUX download attempt {attempt + 1}/3...")
                    
                    # Try to download the specific model.pth file first
                    try:
                        hf_hub_download(
                            repo_id="LYAWWH/DreamLight",
                            filename="FLUX/DreamLight-FLUX/transformer/model.pth",
                            local_dir=dreamlight_dir,
                            local_dir_use_symlinks=False,
                            resume_download=True
                        )
                        logger.info("FLUX model.pth downloaded successfully")
                        download_success = True
                        break
                    except Exception as e:
                        logger.warning(f"Specific file download failed: {e}")
                        logger.info("Falling back to downloading entire FLUX folder...")
                        
                        # Fallback to downloading entire FLUX folder
                        snapshot_download(
                            repo_id="LYAWWH/DreamLight",
                            local_dir=dreamlight_dir,
                            local_dir_use_symlinks=False,
                            allow_patterns=["FLUX/**"],
                            resume_download=True
                        )
                    
                    # Scan again after download
                    flux_model_files = scan_for_model_files(flux_dir)
                    if flux_model_files:
                        logger.info(f"FLUX model downloaded successfully. Found {len(flux_model_files)} model files:")
                        for f in flux_model_files:
                            size_mb = os.path.getsize(f) / (1024 * 1024)
                            logger.info(f"  - {os.path.relpath(f, dreamlight_dir)} ({size_mb:.1f} MB)")
                        download_success = True
                        break
                    else:
                        logger.warning("Download completed but no model files found")
                        
                except Exception as e:
                    logger.warning(f"FLUX download attempt {attempt + 1} failed: {e}")
                    if attempt < 2:
                        time.sleep(5)
            
            if not download_success:
                logger.error("FLUX model download failed. Please check your internet connection and try again.")
                return False
        else:
            logger.info(f"FLUX model files already exist ({len(flux_model_files)} files found)")
            for f in flux_model_files:
                size_mb = os.path.getsize(f) / (1024 * 1024)
                logger.info(f"  - {os.path.relpath(f, dreamlight_dir)} ({size_mb:.1f} MB)")
        
        # Check and download CLIP model
        clip_models_dir = os.path.join(clip_dir, "models")
        clip_config = os.path.join(clip_models_dir, "config.json")
        clip_model = os.path.join(clip_models_dir, "pytorch_model.bin")
        clip_safetensors = os.path.join(clip_models_dir, "model.safetensors")
        
        # Check if any CLIP model files exist
        clip_files_exist = any(os.path.exists(f) for f in [clip_config, clip_model, clip_safetensors])
        
        if not clip_files_exist:
            logger.info("No CLIP model files found. Downloading CLIP folder...")
            from huggingface_hub import snapshot_download
            import time
            
            download_success = False
            for attempt in range(3):
                try:
                    logger.info(f"CLIP download attempt {attempt + 1}/3...")
                    snapshot_download(
                        repo_id="LYAWWH/DreamLight",
                        local_dir=dreamlight_dir,
                        local_dir_use_symlinks=False,
                        allow_patterns=["CLIP/**"],
                        resume_download=True
                    )
                    
                    # Check if any CLIP model files were downloaded
                    clip_files_exist = any(os.path.exists(f) for f in [clip_config, clip_model, clip_safetensors])
                    if clip_files_exist:
                        logger.info("CLIP model downloaded successfully")
                        download_success = True
                        break
                    else:
                        logger.warning("Download completed but no CLIP model files found")
                        
                except Exception as e:
                    logger.warning(f"CLIP download attempt {attempt + 1} failed: {e}")
                    if attempt < 2:
                        time.sleep(3)
            
            if not download_success:
                logger.error("CLIP model download failed. Please check your internet connection.")
                return False
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

import torch.nn.functional as F
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from diffusers import FluxPipeline
from .utils.env_lighting import calculate_spherical_harmonics, generate_spherical_image

class DreamLightNode:
    """Custom ComfyUI node for DreamLight image relighting"""
    
    CATEGORY = "image/postprocessing"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("relit_image",)
    FUNCTION = "process"
    
    # Class-level flag to track if models have been validated
    _models_validated = False

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
    
    def __init__(self):
        """Initialize node and validate models immediately"""
        # Run model validation on node creation, not during processing
        if not DreamLightNode._models_validated:
            logger.info("DreamLight node initialized - validating models...")
            
            # Setup complete FLUX directory with flight checks
            logger.info("Setting up FLUX pipeline directory...")
            flux_dir = setup_complete_flux_directory()
            
            # Validate FLUX directory before proceeding
            if not validate_flux_directory(flux_dir):
                raise RuntimeError("FLUX directory validation failed. See logs for details.")
            
            # Test FLUX pipeline loading
            logger.info("Testing FLUX pipeline loading...")
            try:
                from diffusers import FluxPipeline
                test_pipeline = FluxPipeline.from_pretrained(
                    flux_dir,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                logger.info("✓ FLUX pipeline loaded successfully")
                del test_pipeline  # Clean up test pipeline
            except Exception as e:
                logger.error(f"✗ FLUX pipeline loading failed: {e}")
                raise RuntimeError(f"FLUX pipeline loading failed: {e}")
            
            # Validate DreamLight models
            if validate_and_download_models():
                DreamLightNode._models_validated = True
                logger.info("✓ All DreamLight models validated successfully")
            else:
                logger.error("✗ DreamLight model validation failed")
                raise RuntimeError("DreamLight models are not available. Please check the logs for download errors and ensure you have a stable internet connection.")

    def process(self, foreground_image, background_image, mask, prompt, seed, resolution, environment_map=None):
        # Models are already validated in __init__, so we can proceed directly
        
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
        dreamlight_dir = os.path.join(models_dir, "dreamlight")
        flux_dir = os.path.join(dreamlight_dir, "FLUX", "DreamLight-FLUX", "transformer")
        clip_dir = os.path.join(dreamlight_dir, "CLIP")
        
        
        # Get the pre-validated FLUX directory (setup during __init__)
        import folder_paths
        models_dir = folder_paths.models_dir
        flux_dir = os.path.join(models_dir, "dreamlight", "flux_complete")
        
        # Load pipeline (already validated during __init__)
        logger.info("Loading FLUX pipeline from pre-validated directory...")
        pipeline = FluxPipeline.from_pretrained(
            flux_dir,
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
        pipeline.to(device)
        
        # Load CLIP model from local checkpoint
        clip_models_dir = os.path.join(clip_dir, "models")
        if not os.path.exists(clip_models_dir):
            raise FileNotFoundError(f"DreamLight CLIP model directory not found at {clip_models_dir}. Please ensure models are downloaded.")
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_models_dir).to(device)
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

logger.info("ComfyUI-Dreamlight package loaded successfully")
logger.info(f"Registered nodes: {list(NODE_CLASS_MAPPINGS.keys())}")
if 'DreamLightNode' in NODE_CLASS_MAPPINGS:
    logger.info("Successfully registered DreamLightNode in image/postprocessing category")

__all__ = ['NODE_CLASS_MAPPINGS']
