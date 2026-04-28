import argparse
import os
import time

import pandas as pd
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from model.network import UASTHN


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _strip_module_prefix(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned[key] = value
    return cleaned


def _resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def _looks_like_ihn_state_dict(state_dict):
    if not isinstance(state_dict, dict) or len(state_dict) == 0:
        return False
    if not all(isinstance(key, str) for key in state_dict.keys()):
        return False
    return any(key.startswith("fnet1.") or key.startswith("update_block_4.") for key in state_dict.keys())


def _extract_sub_state(checkpoint, key_name):
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
    args = argparse.Namespace()

    args.dataset_name = "NoName"
    args.resize_width = cli_args.resize_width
    args.database_size = cli_args.database_size
    args.lev0 = True
    args.arch = "IHN"
    args.iters_lev0 = cli_args.iters_lev0
    args.iters_lev1 = cli_args.iters_lev1
    args.corr_level = cli_args.corr_level
    args.fine_padding = cli_args.fine_padding
    args.detach = False
    args.augment_two_stages = 0
    args.identity = False

    if cli_args.device is None:
        device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device_str = cli_args.device
    args.device = torch.device(device_str)

    args.mixed_precision = args.device.type == "cuda" and not cli_args.force_fp32
    args.gpuid = [args.device.index if args.device.type == "cuda" and args.device.index is not None else 0]
    args.two_stages = True
    args.first_stage_ue = False
    args.ue_method = "augment"
    args.ue_ensemble_load_models = "./local_pipeline/ensembles/ensemble_512_IHN.txt"
    args.ue_num_crops = 5
    args.check_step = -1
    args.weight = False
    args.fnet_cat = False
    args.restore_ckpt = None
    args.finetune = False
    args.vis_all = False
    args.si_min = -2.0

    args.eval_model = _resolve_path(cli_args.eval_model)
    args.eval_model_fine = _resolve_path(cli_args.eval_model_fine) if cli_args.eval_model_fine else None
    return args


def _load_model_weights(model, args):
    if not os.path.exists(args.eval_model):
        raise FileNotFoundError(f"Checkpoint not found: {args.eval_model}")

    model_ckpt = torch.load(args.eval_model, map_location=args.device)

    coarse_state = _extract_sub_state(model_ckpt, "netG")
    if coarse_state is None and _looks_like_ihn_state_dict(model_ckpt):
        coarse_state = model_ckpt

    if coarse_state is None:
        raise KeyError("Could not find coarse stage weights. Expected 'netG' or a direct IHN state dict.")

    model.netG.load_state_dict(_strip_module_prefix(coarse_state), strict=False)

    if args.two_stages:
        if args.eval_model_fine is not None:
            fine_ckpt = torch.load(args.eval_model_fine, map_location=args.device)
            fine_state = _extract_sub_state(fine_ckpt, "netG")
            if fine_state is None:
                fine_state = _extract_sub_state(fine_ckpt, "netG_fine")
            if fine_state is None and _looks_like_ihn_state_dict(fine_ckpt):
                fine_state = fine_ckpt
        else:
            fine_state = _extract_sub_state(model_ckpt, "netG_fine")

        if fine_state is None:
            raise KeyError("Could not find fine stage weights. Expected 'netG_fine' or a separate fine checkpoint.")

        model.netG_fine.load_state_dict(_strip_module_prefix(fine_state), strict=False)

    model.setup()
    model.netG.eval()
    if args.two_stages:
        model.netG_fine.eval()
    return model


def _create_reference_points(args):
    four_point_org_single = torch.zeros((1, 2, 2, 2), device=args.device)
    four_point_org_single[:, :, 0, 0] = torch.tensor([0, 0], device=args.device)
    four_point_org_single[:, :, 0, 1] = torch.tensor([args.resize_width - 1, 0], device=args.device)
    four_point_org_single[:, :, 1, 0] = torch.tensor([0, args.resize_width - 1], device=args.device)
    four_point_org_single[:, :, 1, 1] = torch.tensor([args.resize_width - 1, args.resize_width - 1], device=args.device)
    return four_point_org_single


def _predict_four_points(model, img1, img2, args):
    img1 = img1.to(args.device)
    img2 = img2.to(args.device)

    # UASTHN.set_input expects flow_gt, but it is not used for this inference script.
    dummy_flow = torch.zeros((img1.shape[0], 2, args.resize_width, args.resize_width), device=args.device)

    model.set_input(img1, img2, dummy_flow)
    model.forward(for_test=True)
    return model.four_pred.detach()


def _save_results(df, output_path):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    ext = os.path.splitext(output_path)[1].lower()
    if ext == ".csv":
        df.to_csv(output_path, index=False)
    else:
        df.to_excel(output_path, index=False)


def run_js_loop(cli_args):
    if cli_args.jetson_mode and cli_args.device is None:
        cli_args.device = "cuda:0"

    args = _build_runtime_args(cli_args)

    if cli_args.cpu_threads > 0:
        torch.set_num_threads(cli_args.cpu_threads)

    if args.device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
                torch.backends.cuda.matmul.allow_tf32 = True

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    model = UASTHN(args)
    model = _load_model_weights(model, args)

    thermal_transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((args.resize_width, args.resize_width)),
            transforms.ToTensor(),
        ]
    )

    four_point_org_single = _create_reference_points(args)
    scale = args.database_size / args.resize_width

    satellite_dir = _resolve_path(cli_args.satellite_dir)
    thermal_dir = _resolve_path(cli_args.thermal_dir)
    output_excel = _resolve_path(cli_args.output_excel)

    all_corners = []
    times = []

    loop_iter = range(cli_args.num_samples)
    iterator = tqdm(loop_iter, disable=cli_args.disable_tqdm)

    with torch.inference_mode():
        for i in iterator:
            sat_idx = i // cli_args.tiles_per_satellite + 1
            th_idx = i % cli_args.tiles_per_satellite + 1

            img1_path = os.path.join(satellite_dir, f"{sat_idx}.tif")
            img2_path = os.path.join(thermal_dir, f"{sat_idx}_{th_idx}.tif")

            if not os.path.exists(img1_path) or not os.path.exists(img2_path):
                print(f"Skip index {i}: missing image file.")
                continue

            img1 = TF.to_tensor(Image.open(img1_path).convert("RGB")).unsqueeze(0)
            img2 = thermal_transform(Image.open(img2_path)).unsqueeze(0)

            if args.device.type == "cuda":
                torch.cuda.synchronize(args.device)
            start_time = time.perf_counter()

            four_pred = _predict_four_points(model, img1, img2, args)

            if args.device.type == "cuda":
                torch.cuda.synchronize(args.device)
            elapsed = time.perf_counter() - start_time
            times.append(elapsed)

            four_point_1 = four_pred + four_point_org_single
            four_point_1 = four_point_1.flatten(2).permute(0, 2, 1).contiguous()
            four_point_1 = four_point_1 * scale

            points = four_point_1.squeeze(0).cpu().tolist()
            flat_points = [coord for point in points for coord in point]
            all_corners.append([i] + flat_points + [img1_path, img2_path])

            print(f"Done image {i + 1}/{cli_args.num_samples} in {elapsed:.3f} sec")

    if times:
        start_idx = min(cli_args.warmup_skip, len(times))
        valid_times = times[start_idx:] if start_idx < len(times) else times
        avg_time = sum(valid_times) / len(valid_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0.0
        print(f"Average time: {avg_time:.3f} sec/image, FPS: {fps:.2f}")

    columns = [
        "image_index",
        "x1", "y1",
        "x2", "y2",
        "x3", "y3",
        "x4", "y4",
        "sat",
        "th",
    ]
    df = pd.DataFrame(all_corners, columns=columns)
    _save_results(df, output_excel)
    print(f"Saved corners to {output_excel}")


def parse_cli_args():
    parser = argparse.ArgumentParser(description="JS-style UASTHN loop inference (standalone file)")

    parser.add_argument("--eval_model", type=str, default="model.pt", help="Path to coarse/fused checkpoint")
    parser.add_argument("--eval_model_fine", type=str, default=None, help="Optional path to fine-stage checkpoint")
    parser.add_argument("--device", type=str, default=None, help="Device like cuda:0 or cpu")

    parser.add_argument("--resize_width", type=int, default=256)
    parser.add_argument("--database_size", type=int, default=1536)
    parser.add_argument("--iters_lev0", type=int, default=6)
    parser.add_argument("--iters_lev1", type=int, default=6)
    parser.add_argument("--corr_level", type=int, default=4)
    parser.add_argument("--fine_padding", type=int, default=32)

    parser.add_argument("--satellite_dir", type=str, default="js_datasets/Dehat/satellite")
    parser.add_argument("--thermal_dir", type=str, default="js_datasets/Dehat/thermal")
    parser.add_argument("--output_excel", type=str, default="js_excels/new.xlsx")

    parser.add_argument("--num_samples", type=int, default=108)
    parser.add_argument("--tiles_per_satellite", type=int, default=9)
    parser.add_argument("--warmup_skip", type=int, default=1)
    parser.add_argument("--disable_tqdm", action="store_true")

    # Jetson / runtime controls
    parser.add_argument("--jetson_mode", action="store_true", help="Apply sensible defaults for Jetson (uses cuda:0 if device is omitted)")
    parser.add_argument("--force_fp32", action="store_true", help="Disable mixed precision for numerical stability")
    parser.add_argument("--cpu_threads", type=int, default=0, help="Set torch CPU threads; 0 keeps default")

    return parser.parse_args()


if __name__ == "__main__":
    run_js_loop(parse_cli_args())
