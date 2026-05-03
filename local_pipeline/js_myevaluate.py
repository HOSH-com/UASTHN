"""
UASTHN Inference Script with Uncertainty Estimation & Detailed Logging
Based on UASTHN paper (DC=512m, WS=1536px settings)

Enhanced features:
- CropTTA uncertainty estimation (data uncertainty)
- Detailed per-image logging with confidence scores
- GPU memory monitoring
- Performance timing breakdown
- Uncertainty rejection statistics
- Progress tracking with ETA
"""

import argparse
import os
import time
import sys
import logging
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm
from model.network import UASTHN
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*autocast.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*meshgrid.*")


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# COLOR CODES FOR TERMINAL OUTPUT
# ============================================================
class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    GRAY = '\033[90m'


def colored_print(text, color=Colors.END, bold=False):
    """Print colored text to console"""
    prefix = Colors.BOLD if bold else ""
    print(f"{prefix}{color}{text}{Colors.END}")


def setup_logging(save_dir=None):
    """Setup logging to both console and file"""
    logger = logging.getLogger("UASTHN")
    logger.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (if save_dir provided)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(save_dir, f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        )
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


def log_gpu_memory(logger, label=""):
    """Log GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)    # GB
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        free = total - allocated
        
        msg = (f"[GPU] {label} | "
               f"Used: {allocated:.2f}GB | "
               f"Reserved: {reserved:.2f}GB | "
               f"Free: {free:.2f}GB | "
               f"Total: {total:.2f}GB")
        logger.info(msg)
        return allocated, free
    return 0, 0


def _strip_module_prefix(state_dict):
    """Remove 'module.' prefix from DataParallel state dicts"""
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned[key] = value
    return cleaned


def _resolve_path(path):
    """Resolve path relative to project root"""
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def _looks_like_ihn_state_dict(state_dict):
    """Check if state dict looks like an IHN model"""
    if not isinstance(state_dict, dict) or len(state_dict) == 0:
        return False
    if not all(isinstance(key, str) for key in state_dict.keys()):
        return False
    return any(
        key.startswith("fnet1.") or key.startswith("update_block_4.")
        for key in state_dict.keys()
    )


def _extract_sub_state(checkpoint, key_name):
    """Extract sub-state dict from checkpoint"""
    if not isinstance(checkpoint, dict):
        return None

    if key_name in checkpoint and isinstance(checkpoint[key_name], dict):
        return checkpoint[key_name]

    if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        return _extract_sub_state(checkpoint["state_dict"], key_name)

    prefix = key_name + "."
    prefixed_state = {
        key[len(prefix):]: value
        for key, value in checkpoint.items()
        if isinstance(key, str) and key.startswith(prefix)
    }
    if len(prefixed_state) > 0:
        return prefixed_state

    return None


def _build_runtime_args(cli_args):
    """
    Build runtime arguments with UASTHN best parameters.
    
    Best parameters from paper:
    - DC = 512m (search radius)
    - WS = 1536px (satellite image size)
    - WR = 256px (resize width)
    - K1 = 6, K2 = 6 (iterations)
    - o_c = 32px (crop offset)
    - N_C = 5 (number of crops)
    - Merge function: max
    - STD method: all
    """
    args = argparse.Namespace()

    # Dataset settings
    args.dataset_name = "UASTHN_Inference"
    args.resize_width = cli_args.resize_width
    args.database_size = cli_args.database_size
    args.lev0 = True
    args.arch = "IHN"
    
    # Iteration settings
    args.iters_lev0 = cli_args.iters_lev0
    args.iters_lev1 = cli_args.iters_lev1
    args.corr_level = cli_args.corr_level
    args.fine_padding = cli_args.fine_padding
    
    # Model settings
    args.detach = False
    args.augment_two_stages = 0
    args.identity = False

    # Device settings
    if cli_args.device is None:
        device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device_str = cli_args.device
    args.device = torch.device(device_str)

    # Performance settings
    args.mixed_precision = args.device.type == "cuda" and not cli_args.force_fp32
    args.gpuid = [
        args.device.index 
        if args.device.type == "cuda" and args.device.index is not None 
        else 0
    ]
    
    # Two-stage settings
    args.two_stages = True
    
    # ============================================================
    # UNCERTAINTY ESTIMATION SETTINGS (ALWAYS SET THESE!)
    # ============================================================
    args.first_stage_ue = cli_args.enable_uncertainty
    args.ue_method = "augment" if cli_args.enable_uncertainty else "single"
    args.ue_aug_method = "shift"                                          # ADDED
    args.crop_width = args.resize_width - cli_args.ue_shift               # ADDED: 256 - 32 = 224
    args.ue_num_crops = cli_args.ue_num_crops if cli_args.enable_uncertainty else 1
    args.ue_shift_crops_types = cli_args.ue_shift_crops_types if cli_args.enable_uncertainty else "random"
    args.ue_shift = cli_args.ue_shift if cli_args.enable_uncertainty else 32
    args.ue_std_method = cli_args.ue_std_method if cli_args.enable_uncertainty else "all"
    args.ue_agg = cli_args.ue_agg if cli_args.enable_uncertainty else "zero"
    args.ue_seed = cli_args.ue_seed if cli_args.enable_uncertainty else 0
    args.check_step = cli_args.check_step if cli_args.enable_uncertainty else -1
    args.ue_combine = cli_args.ue_combine if cli_args.enable_uncertainty else "max"
    args.ue_rej_std = cli_args.ue_rej_std if cli_args.enable_uncertainty else [float('inf')]
    
    # Outlier settings (used by ue_aggregation)
    args.ue_outlier_method = "none"      # ADD THIS
    args.ue_outlier_num = 0              # ADD THIS

    # Ensemble settings (optional)
    args.ue_ensemble_load_models = cli_args.ensemble_models_file if cli_args.enable_uncertainty else None
    
    # Other settings
    args.weight = False
    args.fnet_cat = False
    args.restore_ckpt = None
    args.finetune = False
    args.vis_all = False
    args.si_min = -2.0
    args.generate_test_pairs = False

    # Model paths
    args.eval_model = _resolve_path(cli_args.eval_model)
    args.eval_model_fine = _resolve_path(cli_args.eval_model_fine) if cli_args.eval_model_fine else None
    
    return args


def _load_model_weights(model, args, logger):
    """Load model weights with detailed logging"""
    logger.info("=" * 60)
    logger.info("LOADING MODEL WEIGHTS")
    logger.info("=" * 60)
    
    if not os.path.exists(args.eval_model):
        raise FileNotFoundError(f"Checkpoint not found: {args.eval_model}")
    
    logger.info(f"Coarse model path: {args.eval_model}")
    
    # Load coarse model
    start_time = time.time()
    model_ckpt = torch.load(args.eval_model, map_location=args.device)
    load_time = time.time() - start_time
    logger.info(f"Checkpoint loaded in {load_time:.2f}s")
    
    # Log checkpoint info
    if isinstance(model_ckpt, dict):
        logger.info(f"Checkpoint keys: {list(model_ckpt.keys())}")
        if 'epoch' in model_ckpt:
            logger.info(f"Trained epoch: {model_ckpt.get('epoch', 'N/A')}")
        if 'step' in model_ckpt:
            logger.info(f"Trained step: {model_ckpt.get('step', 'N/A')}")

    # Extract coarse state
    coarse_state = _extract_sub_state(model_ckpt, "netG")
    if coarse_state is None and _looks_like_ihn_state_dict(model_ckpt):
        coarse_state = model_ckpt
        logger.info("Using direct IHN state dict for coarse model")

    if coarse_state is None:
        raise KeyError(
            "Could not find coarse stage weights. "
            "Expected 'netG' or a direct IHN state dict."
        )
    
    # Load coarse weights
    missing, unexpected = model.netG.load_state_dict(
        _strip_module_prefix(coarse_state), strict=False
    )
    logger.info(f"Coarse model - Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    if missing:
        logger.debug(f"Missing keys: {missing[:5]}...")
    if unexpected:
        logger.debug(f"Unexpected keys: {unexpected[:5]}...")

    # Load fine stage if two-stage
    if args.two_stages:
        logger.info("Loading fine stage model...")
        
        if args.eval_model_fine is not None:
            logger.info(f"Fine model path: {args.eval_model_fine}")
            fine_ckpt = torch.load(args.eval_model_fine, map_location=args.device)
            fine_state = _extract_sub_state(fine_ckpt, "netG")
            if fine_state is None:
                fine_state = _extract_sub_state(fine_ckpt, "netG_fine")
            if fine_state is None and _looks_like_ihn_state_dict(fine_ckpt):
                fine_state = fine_ckpt
        else:
            logger.info("Using same checkpoint for fine stage")
            fine_state = _extract_sub_state(model_ckpt, "netG_fine")

        if fine_state is None:
            raise KeyError(
                "Could not find fine stage weights. "
                "Expected 'netG_fine' or a separate fine checkpoint."
            )
        
        missing_f, unexpected_f = model.netG_fine.load_state_dict(
            _strip_module_prefix(fine_state), strict=False
        )
        logger.info(f"Fine model - Missing keys: {len(missing_f)}, Unexpected keys: {len(unexpected_f)}")

    # Setup and eval mode
    model.setup()
    model.netG.eval()
    if args.two_stages:
        model.netG_fine.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.netG.parameters())
    trainable_params = sum(p.numel() for p in model.netG.parameters() if p.requires_grad)
    logger.info(f"Coarse model parameters: {total_params:,} (trainable: {trainable_params:,})")
    
    if args.two_stages:
        total_params_f = sum(p.numel() for p in model.netG_fine.parameters())
        logger.info(f"Fine model parameters: {total_params_f:,}")
    
    logger.info(f"Uncertainty estimation: {'ENABLED' if args.first_stage_ue else 'DISABLED'}")
    if args.first_stage_ue:
        logger.info(f"  Method: {args.ue_method}")
        logger.info(f"  Num crops: {args.ue_num_crops}")
        logger.info(f"  Crop offset: {args.ue_shift}px")
        logger.info(f"  Crop type: {args.ue_shift_crops_types}")
        logger.info(f"  STD method: {args.ue_std_method}")
        logger.info(f"  Aggregate: {args.ue_agg}")
        logger.info(f"  Rejection thresholds: {args.ue_rej_std}")

    logger.info("Model loaded successfully!")
    log_gpu_memory(logger, "After model loading")
    return model


def _create_reference_points(args):
    """Create reference corner coordinates"""
    four_point_org_single = torch.zeros((1, 2, 2, 2), device=args.device)
    four_point_org_single[:, :, 0, 0] = torch.tensor([0, 0], device=args.device)
    four_point_org_single[:, :, 0, 1] = torch.tensor([args.resize_width - 1, 0], device=args.device)
    four_point_org_single[:, :, 1, 0] = torch.tensor([0, args.resize_width - 1], device=args.device)
    four_point_org_single[:, :, 1, 1] = torch.tensor([args.resize_width - 1, args.resize_width - 1], device=args.device)
    return four_point_org_single


def _predict_four_points(model, img1, img2, args):
    """
    Predict 4-corner displacement.
    Handles models that return variable number of values from netG.
    """
    img1 = img1.to(args.device)
    img2 = img2.to(args.device)

    dummy_flow = torch.zeros(
        (img1.shape[0], 2, args.resize_width, args.resize_width), 
        device=args.device
    )

    model.set_input(img1, img2, dummy_flow)
    
    # Monkey-patch netG to handle variable return values
    # This fixes: "ValueError: too many values to unpack (expected 2)"
    original_netG = model.netG
    
    def wrapped_netG(**kwargs):
        result = original_netG(**kwargs)
        if isinstance(result, tuple):
            if len(result) >= 2:
                # Take first two values (four_preds_list, four_pred)
                return result[0], result[1]
            elif len(result) == 1:
                return [], result[0]
        return [], result
    
    # Temporarily replace netG
    model.netG = wrapped_netG
    
    try:
        model.forward(for_test=True)
    finally:
        # Restore original netG
        model.netG = original_netG
    
    # Get four_pred from model
    if hasattr(model, 'four_pred'):
        four_pred = model.four_pred.detach()
    else:
        raise AttributeError("Model does not have 'four_pred' attribute after forward()")
    
    # Get uncertainty if available
    uncertainty = None
    ue_mask = None
    
    if args.first_stage_ue:
        try:
            if hasattr(model, 'std_four_pred_five_crops'):
                uncertainty = model.std_four_pred_five_crops.detach()
                uncertainty_mean = uncertainty.view(uncertainty.shape[0], -1).mean(dim=1)
                
                ue_mask = torch.ones(len(uncertainty_mean), dtype=torch.bool)
                for threshold in args.ue_rej_std:
                    if threshold < float('inf'):
                        ue_mask = ue_mask & (uncertainty_mean <= threshold)
        except Exception:
            pass
    
    return four_pred, uncertainty, ue_mask

def _save_results(df, output_path, logger):
    """Save results with logging"""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    ext = os.path.splitext(output_path)[1].lower()
    if ext == ".csv":
        df.to_csv(output_path, index=False)
    else:
        df.to_excel(output_path, index=False)
    
    logger.info(f"Results saved to: {output_path}")
    logger.info(f"Total predictions: {len(df)}")
    
    # Log statistics
    if 'uncertainty' in df.columns:
        valid_unc = df['uncertainty'].dropna()
        if len(valid_unc) > 0:
            logger.info(f"Uncertainty - Mean: {valid_unc.mean():.4f}, "
                       f"Median: {valid_unc.median():.4f}, "
                       f"Max: {valid_unc.max():.4f}")
    
    if 'accepted' in df.columns:
        accepted_count = df['accepted'].sum() if df['accepted'].dtype == bool else (df['accepted'] == 1).sum()
        logger.info(f"Accepted predictions: {accepted_count}/{len(df)} "
                   f"({100*accepted_count/len(df):.1f}%)")


def run_js_loop(cli_args):
    """Main inference loop with uncertainty estimation and detailed logging"""
    
    # Setup logging
    logger = setup_logging(cli_args.log_dir)
    
    # Print header
    colored_print("\n" + "=" * 70, Colors.CYAN, bold=True)
    colored_print("UASTHN INFERENCE WITH UNCERTAINTY ESTIMATION", Colors.CYAN, bold=True)
    colored_print("=" * 70 + "\n", Colors.CYAN, bold=True)
    
    # Log system info
    logger.info("SYSTEM INFORMATION")
    logger.info(f"Python: {sys.version.split()[0]}")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"cuDNN version: {torch.backends.cudnn.version()}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logger.warning("CUDA not available, using CPU")
    logger.info(f"CPU threads: {torch.get_num_threads()}")
    
    # Apply Jetson defaults if needed
    if cli_args.jetson_mode and cli_args.device is None:
        cli_args.device = "cuda:0"
        logger.info("Jetson mode enabled, using cuda:0")

    # Build arguments
    logger.info("\nBuilding runtime arguments...")
    args = _build_runtime_args(cli_args)
    
    # Log configuration
    logger.info("=" * 60)
    logger.info("CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Resize width: {args.resize_width}px")
    logger.info(f"Database size: {args.database_size}px")
    logger.info(f"Scale factor: {args.database_size / args.resize_width:.1f}x")
    logger.info(f"Stage 1 iterations: {args.iters_lev0}")
    logger.info(f"Stage 2 iterations: {args.iters_lev1}")
    logger.info(f"Correlation level: {args.corr_level}")
    logger.info(f"Fine padding: {args.fine_padding}px")
    logger.info(f"Device: {args.device}")
    logger.info(f"Mixed precision: {args.mixed_precision}")
    logger.info(f"Two-stage: {args.two_stages}")
    logger.info(f"Uncertainty: {'ENABLED' if args.first_stage_ue else 'DISABLED'}")
    logger.info(f"Model path: {args.eval_model}")

    # Performance optimizations
    if cli_args.cpu_threads > 0:
        torch.set_num_threads(cli_args.cpu_threads)
        logger.info(f"CPU threads set to: {cli_args.cpu_threads}")

    if args.device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True
        logger.info("cuDNN benchmark: enabled")

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
        logger.info("Float32 matmul precision: high")

    # Initialize model
    logger.info("\n" + "=" * 60)
    logger.info("INITIALIZING MODEL")
    logger.info("=" * 60)
    
    log_gpu_memory(logger, "Before model init")
    
    # Create UASTHN model
    logger.info("Creating UASTHN model...")
    try:
        model = UASTHN(args)
        logger.info("Model created successfully")
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        raise
    
    # Load weights
    model = _load_model_weights(model, args, logger)

    # Setup transforms
    thermal_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((args.resize_width, args.resize_width)),
        transforms.ToTensor(),
    ])

    four_point_org_single = _create_reference_points(args)
    scale = args.database_size / args.resize_width

    # Setup paths
    satellite_dir = _resolve_path(cli_args.satellite_dir)
    thermal_dir = _resolve_path(cli_args.thermal_dir)
    output_excel = _resolve_path(cli_args.output_excel)
    
    logger.info(f"\nSatellite directory: {satellite_dir}")
    logger.info(f"Thermal directory: {thermal_dir}")
    logger.info(f"Output file: {output_excel}")

    # Validate input files
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATING INPUT FILES")
    logger.info("=" * 60)
    
    missing_files = []
    existing_files = 0
    
    for i in range(cli_args.num_samples):
        sat_idx = i // cli_args.tiles_per_satellite + 1
        th_idx = i % cli_args.tiles_per_satellite + 1
        img1_path = os.path.join(satellite_dir, f"{sat_idx}.tif")
        img2_path = os.path.join(thermal_dir, f"{sat_idx}_{th_idx}.tif")
        
        if os.path.exists(img1_path) and os.path.exists(img2_path):
            existing_files += 1
        else:
            if not os.path.exists(img1_path):
                missing_files.append(img1_path)
            if not os.path.exists(img2_path):
                missing_files.append(img2_path)
    
    logger.info(f"Total expected samples: {cli_args.num_samples}")
    logger.info(f"Existing file pairs: {existing_files}")
    logger.info(f"Missing files: {len(missing_files)}")
    
    if len(missing_files) > 0 and len(missing_files) <= 10:
        for f in missing_files:
            logger.warning(f"  Missing: {f}")

    # ============================================================
    # MAIN INFERENCE LOOP
    # ============================================================
    # ============================================================
    # MAIN INFERENCE LOOP
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("STARTING INFERENCE")
    logger.info("=" * 60)
    
    all_corners = []
    times = []
    uncertainties = []
    accepted_flags = []
    skipped_count = 0
    error_count = 0
    successful_count = 0
    
    loop_iter = range(cli_args.num_samples)
    
    if not cli_args.disable_tqdm:
        iterator = tqdm(loop_iter, desc="Inference", unit="img")
    else:
        iterator = loop_iter

    with torch.inference_mode():
        for i in iterator:
            sat_idx = i // cli_args.tiles_per_satellite + 1   # satellite index (1-based)
            th_idx = i % cli_args.tiles_per_satellite + 1    # thermal tile index (1-based)

            img1_path = os.path.join(satellite_dir, f"{sat_idx}.tif")
            img2_path = os.path.join(thermal_dir, f"{sat_idx}_{th_idx}.tif")

            if not os.path.exists(img1_path) or not os.path.exists(img2_path):
                skipped_count += 1
                # ADDED: sat_idx and th_idx in the skipped row
                all_corners.append([i] + [np.nan]*8 + [img1_path, img2_path, sat_idx, th_idx, np.nan, 0])
                continue

            try:
                # Load images
                img1 = TF.to_tensor(Image.open(img1_path).convert("RGB")).unsqueeze(0)
                img2 = thermal_transform(Image.open(img2_path)).unsqueeze(0)

                # Time the inference
                if args.device.type == "cuda":
                    torch.cuda.synchronize(args.device)
                start_time = time.perf_counter()

                # Use _predict_four_points which has the monkey-patch fix for netG
                four_pred, uncertainty, ue_mask = _predict_four_points(model, img1, img2, args)

                if args.device.type == "cuda":
                    torch.cuda.synchronize(args.device)
                elapsed = time.perf_counter() - start_time
                
                times.append(elapsed)

                # Process predictions
                four_point_1 = four_pred + four_point_org_single
                four_point_1 = four_point_1.flatten(2).permute(0, 2, 1).contiguous()
                four_point_1 = four_point_1 * scale

                points = four_point_1.squeeze(0).cpu().tolist()
                flat_points = [coord for point in points for coord in point]

                # Uncertainty
                unc_value = np.nan
                is_accepted = 1
                
                if uncertainty is not None:
                    unc_value = uncertainty.view(-1).mean().cpu().item()
                    if ue_mask is not None and len(ue_mask) > 0:
                        is_accepted = int(ue_mask[0].item())
                    uncertainties.append(unc_value)
                    accepted_flags.append(is_accepted)

                # Store results - WITH sat and th columns
                all_corners.append([
                    i,                    # 1. image_index
                    flat_points[0],      # 2. x1
                    flat_points[1],      # 3. y1
                    flat_points[2],      # 4. x2
                    flat_points[3],      # 5. y2
                    flat_points[4],      # 6. x3
                    flat_points[5],      # 7. y3
                    flat_points[6],      # 8. x4
                    flat_points[7],      # 9. y4
                    img1_path,           # 10. satellite_path
                    img2_path,           # 11. thermal_path
                    img1_path,           # 12. sat
                    img2_path,           # 13. th
                    unc_value,           # 14. uncertainty
                    is_accepted          # 15. accepted
                ])
                successful_count += 1

                if i % 10 == 0 or i == 0:
                    center_x = (flat_points[0] + flat_points[2] + flat_points[4] + flat_points[6]) / 4
                    center_y = (flat_points[1] + flat_points[3] + flat_points[5] + flat_points[7]) / 4
                    status = "OK" if is_accepted else "REJ"
                    logger.info(
                        f"Img {i+1:4d}/{cli_args.num_samples} [{status}] | "
                        f"Ctr: ({center_x:7.1f}, {center_y:7.1f}) | "
                        f"T: {elapsed:.3f}s | "
                        f"Sat{sat_idx}_T{th_idx}"
                    )

            except Exception as e:
                error_count += 1
                if error_count <= 3:
                    logger.error(f"Error image {i}: {e}")
                    import traceback
                    traceback.print_exc()
                # ADDED: sat_idx and th_idx in the error row
                # For skip case (8 NaN for coordinates + img1_path + img2_path + img1_path + img2_path + NaN + 0)
                all_corners.append([i] + [np.nan]*8 + [img1_path, img2_path, img1_path, img2_path, np.nan, 0])

                # For error case
                all_corners.append([i] + [np.nan]*8 + [img1_path, img2_path, img1_path, img2_path, np.nan, 0])

    # ============================================================
    # PERFORMANCE SUMMARY
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("PERFORMANCE SUMMARY")
    logger.info("=" * 60)
    
    logger.info(f"Total expected: {cli_args.num_samples}")
    logger.info(f"Successful: {successful_count}")
    logger.info(f"Skipped (missing files): {skipped_count}")
    logger.info(f"Errors: {error_count}")
    
    if times:
        valid_times = times[cli_args.warmup_skip:] if len(times) > cli_args.warmup_skip else times
        
        if valid_times:
            avg_time = np.mean(valid_times)
            median_time = np.median(valid_times)
            min_time = np.min(valid_times)
            max_time = np.max(valid_times)
            fps = 1.0 / avg_time if avg_time > 0 else 0.0
            
            logger.info(f"\nTiming (excluding {cli_args.warmup_skip} warmup):")
            logger.info(f"  Avg:  {avg_time*1000:.1f}ms | Med: {median_time*1000:.1f}ms")
            logger.info(f"  Min: {min_time*1000:.1f}ms | Max: {max_time*1000:.1f}ms")
            logger.info(f"  FPS: {fps:.2f} | Total: {sum(times):.1f}s")
    
    # Uncertainty statistics
    if uncertainties:
        valid_unc = [u for u in uncertainties if not np.isnan(u)]
        if valid_unc:
            logger.info(f"\nUncertainty: Mean={np.mean(valid_unc):.4f} Med={np.median(valid_unc):.4f}")
            if accepted_flags:
                accepted_count = sum(accepted_flags)
                logger.info(f"Accepted: {accepted_count}/{len(accepted_flags)} ({100*accepted_count/len(accepted_flags):.1f}%)")

    # ============================================================
    # SAVE RESULTS
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("SAVING RESULTS")
    logger.info("=" * 60)
    
    # UPDATED: Added "sat" and "th" columns
    columns = [
        "image_index",       # 1
        "x1", "y1",          # 2,3
        "x2", "y2",          # 4,5
        "x3", "y3",          # 6,7
        "x4", "y4",          # 8,9
        "satellite_path",    # 10
        "thermal_path",      # 11
        "sat",               # 12
        "th",                # 13
        "uncertainty",       # 14
        "accepted",          # 15
    ]
    
    df = pd.DataFrame(all_corners, columns=columns)
    _save_results(df, output_excel, logger)

    # Final summary
    colored_print("\n" + "=" * 70, Colors.GREEN, bold=True)
    colored_print("INFERENCE COMPLETE", Colors.GREEN, bold=True)
    colored_print("=" * 70, Colors.GREEN, bold=True)
    logger.info(f"Results: {output_excel}")
    logger.info(f"OK: {successful_count} | Skip: {skipped_count} | Err: {error_count}")
    if times:
        logger.info(f"Avg: {avg_time:.3f}s/img ({fps:.2f} FPS)")

def parse_cli_args():
    """Parse command line arguments with uncertainty estimation options"""
    parser = argparse.ArgumentParser(
        description="UASTHN Inference with Uncertainty Estimation (DC=512m, WS=1536px)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic inference (no uncertainty)
  python js_myevaluate.py --eval_model model.pth --satellite_dir data/satellite --thermal_dir data/thermal
  
  # With uncertainty estimation
  python js_myevaluate.py --eval_model model.pth --enable_uncertainty --satellite_dir data/satellite --thermal_dir data/thermal
  
  # With custom rejection thresholds
  python js_myevaluate.py --eval_model model.pth --enable_uncertainty --ue_rej_std 0.5 1.0 2.0
        """
    )

    # Model paths
    parser.add_argument("--eval_model", type=str, default="model.pt", 
                       help="Path to coarse/fused checkpoint")
    parser.add_argument("--eval_model_fine", type=str, default=None, 
                       help="Optional path to fine-stage checkpoint")
    parser.add_argument("--device", type=str, default=None, 
                       help="Device like cuda:0 or cpu")

    # Model architecture parameters (from UASTHN paper best settings)
    parser.add_argument("--resize_width", type=int, default=256, 
                       help="Resize width (WR=256 from paper)")
    parser.add_argument("--database_size", type=int, default=1536, 
                       help="Database/satellite image size (WS=1536 from paper)")
    parser.add_argument("--iters_lev0", type=int, default=6, 
                       help="Stage 1 iterations (K1=6 from paper)")
    parser.add_argument("--iters_lev1", type=int, default=6, 
                       help="Stage 2 iterations (K2=6 from paper)")
    parser.add_argument("--corr_level", type=int, default=4, 
                       help="Correlation pyramid levels")
    parser.add_argument("--fine_padding", type=int, default=32, 
                       help="Fine stage padding (32px from paper)")

    # Input/Output paths
    parser.add_argument("--satellite_dir", type=str, default="js_datasets/Dehat/satellite", 
                       help="Directory containing satellite images")
    parser.add_argument("--thermal_dir", type=str, default="js_datasets/Dehat/thermal", 
                       help="Directory containing thermal images")
    parser.add_argument("--output_excel", type=str, default="js_excels/UASTHN_results.xlsx", 
                       help="Output Excel/CSV file path")
    parser.add_argument("--log_dir", type=str, default="logs/inference", 
                       help="Directory for log files")

    # Data parameters
    parser.add_argument("--num_samples", type=int, default=108, 
                       help="Number of samples to process")
    parser.add_argument("--tiles_per_satellite", type=int, default=9, 
                       help="Number of thermal tiles per satellite image")
    parser.add_argument("--warmup_skip", type=int, default=1, 
                       help="Number of initial inferences to skip for timing stats")
    parser.add_argument("--disable_tqdm", action="store_true", 
                       help="Disable tqdm progress bar")

    # ============================================================
    # UNCERTAINTY ESTIMATION PARAMETERS
    # ============================================================
    parser.add_argument("--enable_uncertainty", action="store_true", 
                       help="Enable CropTTA uncertainty estimation")
    
    parser.add_argument("--ue_num_crops", type=int, default=5, 
                       help="Number of crops for uncertainty (N_C=5 from paper)")
    parser.add_argument("--ue_shift", type=int, default=32, 
                       help="Crop offset in pixels (o_c=32 from paper)")
    parser.add_argument("--ue_shift_crops_types", type=str, default="random", 
                       choices=["random", "grid"], 
                       help="Crop sampling method (random from paper)")
    parser.add_argument("--ue_std_method", type=str, default="all", 
                       choices=["any", "all", "mean"], 
                       help="Uncertainty aggregation method (all from paper)")
    parser.add_argument("--ue_agg", type=str, default="zero", 
                       choices=["zero", "mean"], 
                       help="Prediction aggregation method (zero=use original)")
    parser.add_argument("--ue_combine", type=str, default="max", 
                       choices=["min", "max", "add"], 
                       help="Merge function for combined uncertainty (max from paper)")
    parser.add_argument("--ue_seed", type=int, default=0, 
                       help="Random seed for crop sampling")
    parser.add_argument("--ue_rej_std", type=float, nargs="+", default=[0.5, 1.0, 2.0], 
                       help="Uncertainty rejection thresholds (in pixels)")
    parser.add_argument("--check_step", type=int, default=-1, 
                       help="Early stopping check step (-1=disabled)")
    parser.add_argument("--ensemble_models_file", type=str, 
                       default="./local_pipeline/ensembles/ensemble_512_IHN.txt", 
                       help="Path to ensemble models list file")

    # Runtime controls
    parser.add_argument("--jetson_mode", action="store_true", 
                       help="Apply sensible defaults for Jetson")
    parser.add_argument("--force_fp32", action="store_true", 
                       help="Disable mixed precision")
    parser.add_argument("--cpu_threads", type=int, default=0, 
                       help="Set torch CPU threads; 0 keeps default")

    return parser.parse_args()


if __name__ == "__main__":
    run_js_loop(parse_cli_args())