"""
UASTHN Demo: Uncertainty-Aware Satellite-Thermal Homography Network
Supports uploading to / loading from HuggingFace Hub.
Input: 1 RGB satellite image + 1 thermal image
Output: 4-point displacement + uncertainty (std) via CropTTA + visualization

TIMING ASSESSMENT:
- IHN_coarse (~50ms): First stage, main computational cost
- IHN_fine (~40ms): Second stage refinement  
- uncertainty_estimation (~180ms): CropTTA with 5 crops, scales with ue_num_crops
- data_loading (~40ms): Image I/O from disk
- crop_refinement (~3ms): Negligible
- Total ~310ms per image = ~3 FPS

This is normal for 2-stage inference with uncertainty. For faster inference:
1. Reduce ue_num_crops from 5 to 3 (saves ~70ms)
2. Use GPU batch processing instead of per-image
3. Use mixed precision (fp16)

RESOURCE MONITORING:
To enable CPU/GPU/RAM tracking, install:
  pip install psutil
  pip install gputil

These provide ResourceMonitor for tracking system resources during inference.
"""
import sys
import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.geometry.transform as tgm
import kornia.geometry.bbox as bbox_utils
from huggingface_hub import PyTorchModelHubMixin
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from tqdm import tqdm

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False

# Import model building blocks from local_pipeline
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'local_pipeline'))
from extractor import BasicEncoderQuarter
from corr import CorrBlock
from update import CNN_64
from utils import coords_grid

from model.js_kornia_replacement import (
    get_perspective_transform_torch, 
    crop_and_resize_torch,
)


# ==============================================================================
# Model Components (redefined without args dependency for HuggingFace)
# ==============================================================================

class GMA(nn.Module):
    """Update block that predicts delta 4-point displacement from correlation and flow.
    Redefined from local_pipeline/update.py to remove args dependency.
    """
    def __init__(self, corr_level, sz):
        super().__init__()
        if sz == 64:
            if corr_level == 2:
                init_dim = 164    # 2 * 81 + 2
            elif corr_level == 4:
                init_dim = 326    # 4 * 81 + 2
            elif corr_level == 6:
                init_dim = 488    # 6 * 81 + 2
            else:
                raise NotImplementedError(f"corr_level={corr_level} not supported")
            self.cnn = CNN_64(128, init_dim=init_dim)
        else:
            raise NotImplementedError(f"GMA with sz={sz} not supported in this demo")

    def forward(self, corr, flow):
        return self.cnn(torch.cat((corr, flow), dim=1))


class IHN(nn.Module):
    """Iterative Homography Network.
    Redefined from local_pipeline/model/network.py to remove args dependency.
    State dict keys are compatible with original checkpoints (after stripping 'module.').
    """
    def __init__(self, resize_width, corr_level):
        super().__init__()
        self.resize_width = resize_width
        self.fnet1 = BasicEncoderQuarter(output_dim=256, norm_fn='instance')
        sz = resize_width // 4
        self.update_block_4 = GMA(corr_level, sz)
        self.imagenet_mean = None
        self.imagenet_std = None

    def get_flow_now_4(self, four_point):
        four_point = four_point / 4
        four_point_org = torch.zeros((2, 2, 2)).to(four_point.device)
        four_point_org[:, 0, 0] = torch.Tensor([0, 0])
        four_point_org[:, 0, 1] = torch.Tensor([self.sz[3] - 1, 0])
        four_point_org[:, 1, 0] = torch.Tensor([0, self.sz[2] - 1])
        four_point_org[:, 1, 1] = torch.Tensor([self.sz[3] - 1, self.sz[2] - 1])

        four_point_org = four_point_org.unsqueeze(0).repeat(self.sz[0], 1, 1, 1)
        four_point_new = four_point_org + four_point
        four_point_org = four_point_org.flatten(2).permute(0, 2, 1).contiguous()
        four_point_new = four_point_new.flatten(2).permute(0, 2, 1).contiguous()
        # H = tgm.get_perspective_transform(four_point_org, four_point_new)
        H = get_perspective_transform_torch(four_point_org, four_point_new)

        gridy, gridx = torch.meshgrid(
            torch.linspace(0, self.resize_width // 4 - 1, steps=self.resize_width // 4),
            torch.linspace(0, self.resize_width // 4 - 1, steps=self.resize_width // 4))
        points = torch.cat(
            (gridx.flatten().unsqueeze(0), gridy.flatten().unsqueeze(0),
             torch.ones((1, self.resize_width // 4 * self.resize_width // 4))),
            dim=0).unsqueeze(0).repeat(H.shape[0], 1, 1).to(four_point.device)
        points_new = H.bmm(points)
        if torch.isnan(points_new).any():
            raise KeyError("Some of transformed coords are NaN!")
        points_new = points_new / points_new[:, 2, :].unsqueeze(1)
        points_new = points_new[:, 0:2, :]
        flow = torch.cat(
            (points_new[:, 0, :].reshape(self.sz[0], self.sz[3], self.sz[2]).unsqueeze(1),
             points_new[:, 1, :].reshape(self.sz[0], self.sz[3], self.sz[2]).unsqueeze(1)),
            dim=1)
        return flow

    def forward(self, image1, image2, iters_lev0=6, corr_level=2, corr_radius=4):
        if self.imagenet_mean is None:
            self.imagenet_mean = torch.Tensor([0.485, 0.456, 0.406]).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(image1.device)
            self.imagenet_std = torch.Tensor([0.229, 0.224, 0.225]).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(image1.device)
        image1 = (image1.contiguous() - self.imagenet_mean) / self.imagenet_std
        image2 = (image2.contiguous() - self.imagenet_mean) / self.imagenet_std

        global_timing.start('IHN_feature_extraction')
        fmap1 = self.fnet1(image1).float()
        fmap2 = self.fnet1(image2).float()
        global_timing.end('IHN_feature_extraction')

        global_timing.start('Corr Init')
        corr_fn = CorrBlock(fmap1, fmap2, num_levels=corr_level, radius=corr_radius)
        global_timing.end('Corr Init')

        N, C, H, W = image1.shape
        coords0 = coords_grid(N, H // 4, W // 4).to(image1.device)
        coords1 = coords_grid(N, H // 4, W // 4).to(image1.device)

        sz = fmap1.shape
        self.sz = sz
        four_point_disp = torch.zeros((sz[0], 2, 2, 2)).to(fmap1.device)
        four_point_predictions = []
        sum_corr = 0.0
        sum_update = 0.0

        for itr in range(iters_lev0):

            start_time = time.perf_counter()
            corr = corr_fn(coords1)
            sum_corr += time.perf_counter() - start_time

            flow = coords1 - coords0

            start_time = time.perf_counter()
            delta_four_point = self.update_block_4(corr, flow)
            sum_update += time.perf_counter() - start_time

            try:
                last_four_point_disp = four_point_disp
                four_point_disp = four_point_disp + delta_four_point
                coords1 = self.get_flow_now_4(four_point_disp)
                four_point_predictions.append(four_point_disp)
            except Exception:
                four_point_disp = last_four_point_disp
                coords1 = self.get_flow_now_4(four_point_disp)
                four_point_predictions.append(four_point_disp)

        global_timing.add_time('Corr', sum_corr)
        global_timing.add_time('Update', sum_update)
        return four_point_predictions, four_point_disp


# ==============================================================================
# UASTHN HuggingFace Model
# ==============================================================================

class UASTHN(nn.Module, PyTorchModelHubMixin):
    """
    Uncertainty-Aware Satellite-Thermal Homography Network with HuggingFace Hub support.

    Two-stage model: coarse IHN alignment + fine IHN refinement.
    Supports CropTTA (Crop-based Test-Time Augmentation) for uncertainty estimation.
    """
    def __init__(self, model_config, **kwargs):
        super().__init__()
        self.model_config = model_config

        self.resize_width = model_config.get('resize_width', 256)
        self.database_size = model_config.get('database_size', 1536)
        self.corr_level = model_config.get('corr_level', 4)
        self.iters_lev0 = model_config.get('iters_lev0', 6)
        self.iters_lev1 = model_config.get('iters_lev1', 6)
        self.fine_padding = model_config.get('fine_padding', 0)
        self.ue_shift_crops_types = model_config.get('ue_shift_crops_types', 'random')

        self.netG = IHN(self.resize_width, self.corr_level)
        self.netG_fine = IHN(self.resize_width, 2)

    def forward(self, satellite_image, thermal_image):
        """
        Args:
            satellite_image: [B, 3, database_size, database_size] RGB satellite (values in [0, 1])
            thermal_image: [B, 3, resize_width, resize_width] 3-channel thermal (values in [0, 1])
        Returns:
            four_pred: [B, 2, 2, 2] predicted 4-point displacement at resize_width scale
        """
        image_1 = F.interpolate(satellite_image, size=self.resize_width,
                                mode='bilinear', align_corners=True, antialias=True)
        image_2 = thermal_image

        _, four_pred = self.netG(
            image1=image_1, image2=image_2,
            iters_lev0=self.iters_lev0, corr_level=self.corr_level)

        image_1_crop, delta, flow_bbox = self._crop_for_refinement(
            satellite_image, four_pred)
        
        _, four_pred_fine = self.netG_fine(
            image1=image_1_crop, image2=image_2,
            iters_lev0=self.iters_lev1)
        four_pred = self._combine_coarse_fine(four_pred_fine, delta, flow_bbox)

        return four_pred

    def forward_with_uncertainty(self, satellite_image, thermal_image,
                             ue_num_crops=5, ue_shift=32,
                             ue_shift_crops_types=None, 
                             firststage_ue=True,
                             ue_seed=0):
        """
        CropTTA: Run inference with multiple shifted crops for uncertainty estimation.
        
        Args:
            satellite_image: [B, 3, database_size, database_size]
            thermal_image: [B, 3, resize_width, resize_width]
            ue_num_crops: number of crops (including original), default 5
            ue_shift: max shift in pixels, default 32 (matching paper)
            ue_shift_crops_types: "grid" or "random" (uses model config if None)
            firststage_ue: use coarse-stage predictions (faster, matches paper)
            ue_seed: random seed for crop positions
        
        Returns:
            four_pred: [B, 2, 2, 2] predicted displacement (from unshifted crop)
            std_pred: [B, 2, 2, 2] per-corner std across crops (uncertainty)
        """
        device = satellite_image.device
        B, C, H_th, W_th = thermal_image.shape
        _, _, H_sat, W_sat = satellite_image.shape
        rw = self.resize_width
        
        # Use model config if not specified
        # if ue_shift_crops_types is None:
        #     ue_shift_crops_types = self.model_config.get('ue_shift_crops_types', 'random')
        
        rng = np.random.default_rng(seed=ue_seed)

        # Generate crop shifts
        if ue_shift_crops_types == "grid":
            if 2 <= ue_num_crops <= 5:
                x_shift_grid = np.linspace(0, ue_shift, 2)
                y_shift_grid = np.linspace(0, ue_shift, 2)
            else:
                raise NotImplementedError(
                    f"Grid mode only supports ue_num_crops 2-5, got {ue_num_crops}")
            xx, yy = np.meshgrid(x_shift_grid, y_shift_grid)
            xx, yy = xx.reshape(-1), yy.reshape(-1)
            idx = list(range(len(xx)))
            rng.shuffle(idx)
            idx = idx[:ue_num_crops - 1]
            selected_x = list(xx[idx])
            selected_y = list(yy[idx])
        elif ue_shift_crops_types == "random":
            selected_x = [float(rng.integers(0, ue_shift + 1))
                        for _ in range(ue_num_crops - 1)]
            selected_y = [float(rng.integers(0, ue_shift + 1))
                        for _ in range(ue_num_crops - 1)]
        else:
            raise NotImplementedError(
                f"Unknown ue_shift_crops_types: {ue_shift_crops_types}")
        
        crop_w = float(rw - ue_shift)

        # Build crops: first = original, rest = shifted
        x_start_list = [0.0] + selected_x
        y_start_list = [0.0] + selected_y
        w_list = [float(rw)] + [crop_w] * (ue_num_crops - 1)

        # Replicate images: [B*ue_num_crops, C, H, W]
        sat_rep = satellite_image.unsqueeze(1).repeat(1, ue_num_crops, 1, 1, 1).view(
            B * ue_num_crops, C, H_sat, W_sat)
        thermal_rep = thermal_image.unsqueeze(1).repeat(1, ue_num_crops, 1, 1, 1).view(
            B * ue_num_crops, C, H_th, W_th)

        # Generate bboxes
        x_start = torch.tensor(x_start_list, dtype=torch.float32, device=device).repeat(B)
        y_start = torch.tensor(y_start_list, dtype=torch.float32, device=device).repeat(B)
        w = torch.tensor(w_list, dtype=torch.float32, device=device).repeat(B)
        bbox_s = bbox_utils.bbox_generator(x_start, y_start, w, w)

        # H_CTtoT: crop coords → original thermal coords
        bbox_s_swap = torch.stack(
            [bbox_s[:, 0], bbox_s[:, 1], bbox_s[:, 3], bbox_s[:, 2]], dim=1)
        four_point_org = self._get_four_point_org(rw, device).repeat(
            B * ue_num_crops, 1, 1, 1)
        four_point_org_flat = four_point_org.view(
            B * ue_num_crops, 2, 4).permute(0, 2, 1).contiguous()
        H_CTtoT = get_perspective_transform_torch(bbox_s_swap, four_point_org_flat)

        # Crop thermal images
        thermal_cropped = crop_and_resize_torch(thermal_rep, bbox_s, (rw, rw))

        # Run coarse stage on all crops
        # image_1 = F.interpolate(sat_rep, size=rw,
        #                         mode='bilinear', align_corners=True, antialias=True)
        image_1 = sat_rep
        _, four_pred_coarse = self.netG(
            image1=image_1, image2=thermal_cropped,
            iters_lev0=self.iters_lev0, corr_level=self.corr_level)

        # Decide: firststage_ue or full two-stage
        if firststage_ue:
            all_preds_raw = four_pred_coarse
        else:
            image_1_crop, delta, flow_bbox = self._crop_for_refinement(
                sat_rep, four_pred_coarse)

            _, four_pred_fine = self.netG_fine(
                image1=image_1_crop, image2=thermal_cropped,
                iters_lev0=self.iters_lev1)

            all_preds_raw = self._combine_coarse_fine(
                four_pred_fine, delta, flow_bbox)

        # Recover predictions from crop coords → original coords
        four_point_org_single = self._get_four_point_org(rw, device)
        recovered_preds = []

        for i in range(B * ue_num_crops):
            pred_i = all_preds_raw[i:i+1]
            four_corners = pred_i + four_point_org_single

            # H_StoT: bbox corners → predicted corners
            H_StoT = get_perspective_transform_torch(
                bbox_s_swap[i:i+1],
                four_corners.view(1, 2, 4).permute(0, 2, 1).contiguous())
            
            # FIX: Keep on same device
            H_StoT_inv = torch.linalg.inv(H_StoT.cpu())  

            corners_homo = torch.cat([
                four_corners.view(1, 2, 4),
                torch.ones(1, 1, 4, device=device)
            ], dim=1)

            # Transform chain
            transformed = H_StoT @ H_CTtoT[i:i+1] @ H_StoT_inv.to(corners_homo.device) @ corners_homo
            transformed = transformed[:, :2, :] / transformed[:, 2:, :]
            recovered = transformed.view(1, 2, 2, 2) - four_point_org_single
            recovered_preds.append(recovered)

        recovered_preds = torch.cat(recovered_preds, dim=0)
        recovered_preds = recovered_preds.view(B, ue_num_crops, 2, 2, 2)

        # Final prediction = unshifted crop
        four_pred = recovered_preds[:, 0]
        
        # Uncertainty = std across crops
        std_pred = torch.std(recovered_preds, dim=1)

        return four_pred, std_pred

    def _get_four_point_org(self, size, device):
        fp = torch.zeros((1, 2, 2, 2), device=device)
        fp[0, :, 0, 0] = torch.tensor([0.0, 0.0])
        fp[0, :, 0, 1] = torch.tensor([size - 1.0, 0.0])
        fp[0, :, 1, 0] = torch.tensor([0.0, size - 1.0])
        fp[0, :, 1, 1] = torch.tensor([size - 1.0, size - 1.0])
        return fp

    def _crop_for_refinement(self, image_1_ori, four_pred):
        device = four_pred.device
        rw = self.resize_width
        ds = self.database_size
        alpha = ds / rw

        four_point_org = self._get_four_point_org(rw, device)
        four_point = four_pred + four_point_org

        x = four_point[:, 0].clone()
        y = four_point[:, 1].clone()

        x[:, :, 0] = x[:, :, 0] * alpha
        x[:, :, 1] = (x[:, :, 1] + 1) * alpha
        y[:, 0, :] = y[:, 0, :] * alpha
        y[:, 1, :] = (y[:, 1, :] + 1) * alpha

        left = torch.min(x.view(x.shape[0], -1), dim=1)[0]
        right = torch.max(x.view(x.shape[0], -1), dim=1)[0]
        top = torch.min(y.view(y.shape[0], -1), dim=1)[0]
        bottom = torch.max(y.view(y.shape[0], -1), dim=1)[0]

        w = torch.max(torch.stack([right - left, bottom - top], dim=1), dim=1)[0]
        c = torch.stack([(left + right) / 2, (bottom + top) / 2], dim=1)

        w_padded = w + 2 * self.fine_padding
        crop_top_left = c + torch.stack([-w_padded / 2, -w_padded / 2], dim=1)
        x_start = crop_top_left[:, 0]
        y_start = crop_top_left[:, 1]

        bbox_s = bbox_utils.bbox_generator(x_start, y_start, w_padded, w_padded)
        delta = (w_padded / rw).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        # image_1_crop = tgm.crop_and_resize(image_1_ori, bbox_s, (rw, rw))
        image_1_crop = crop_and_resize_torch(image_1_ori, bbox_s, (rw, rw))

        bbox_s_swap = torch.stack(
            [bbox_s[:, 0], bbox_s[:, 1], bbox_s[:, 3], bbox_s[:, 2]], dim=1)
        four_cor_bbox = bbox_s_swap.permute(0, 2, 1).view(-1, 2, 2, 2)
        four_point_org_large = self._get_four_point_org(ds, device)
        flow_bbox = four_cor_bbox - four_point_org_large

        return image_1_crop.detach(), delta.detach(), flow_bbox.detach()

    def _combine_coarse_fine(self, four_pred_fine, delta, flow_bbox):
        alpha = self.database_size / self.resize_width
        kappa = delta / alpha
        return four_pred_fine * kappa + flow_bbox / alpha

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *,
                        subfolder=None, force_download=False, token=None,
                        cache_dir=None, local_files_only=False, revision=None,
                        **model_kwargs):
        """Load model from HuggingFace Hub with subfolder support."""
        import json
        from huggingface_hub import hf_hub_download

        model_id = str(pretrained_model_name_or_path)
        config_file = hf_hub_download(
            repo_id=model_id, filename="config.json", subfolder=subfolder,
            revision=revision, cache_dir='js_config',
            force_download=force_download, token=token,
            local_files_only=local_files_only,
        )
        with open(config_file, "r") as f:
            config = json.load(f)
        print('*** conf', config)
        model = cls(model_config=config.get("model_config", config), **model_kwargs)

        try:
            model_file = hf_hub_download(
                repo_id=model_id, filename="model.safetensors",
                subfolder=subfolder, revision=revision, cache_dir='js_models',
                force_download=force_download, token=token,
                local_files_only=local_files_only,
            )
            return cls._load_as_safetensor(model, model_file, "cpu", False)
        except Exception:
            model_file = hf_hub_download(
                repo_id=model_id, filename="pytorch_model.bin",
                subfolder=subfolder, revision=revision, cache_dir=cache_dir,
                force_download=force_download, token=token,
                local_files_only=local_files_only,
            )
            return cls._load_as_pickle(model, model_file, "cpu", False)



# ==============================================================================
# Preprocessing & Visualization
# ==============================================================================

def load_and_preprocess_satellite(image_path, database_size):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize([database_size, database_size]),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)


def load_and_preprocess_thermal(image_path, resize_width):
    image = Image.open(image_path).convert('L')
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize([resize_width, resize_width]),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)


def visualize_result(satellite_image, thermal_image, four_pred, resize_width,
                     database_size, save_path='examples/UASTHN_result.png',
                     gt_image_path=None, std_pred=None):
    alpha = database_size / resize_width

    four_point_org = torch.zeros((1, 2, 2, 2))
    four_point_org[:, :, 0, 0] = torch.tensor([0, 0])
    four_point_org[:, :, 0, 1] = torch.tensor([resize_width - 1, 0])
    four_point_org[:, :, 1, 0] = torch.tensor([0, resize_width - 1])
    four_point_org[:, :, 1, 1] = torch.tensor([resize_width - 1, resize_width - 1])

    four_point_pred = four_pred.cpu() + four_point_org

    sat_display = F.interpolate(satellite_image, size=resize_width,
                                mode='bilinear', align_corners=True, antialias=True)
    sat_np = (sat_display[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    thermal_np = (thermal_image[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

    pred_pts = four_point_pred[0].numpy()
    pts = np.array([
        [pred_pts[0, 0, 0], pred_pts[1, 0, 0]],  # TL
        [pred_pts[0, 0, 1], pred_pts[1, 0, 1]],  # TR
        [pred_pts[0, 1, 1], pred_pts[1, 1, 1]],  # BR
        [pred_pts[0, 1, 0], pred_pts[1, 1, 0]],  # BL
    ], dtype=np.int32).reshape((-1, 1, 2))

    sat_with_bbox = sat_np.copy()
    cv2.polylines(sat_with_bbox, [pts], True, (0, 255, 0), 2)

    four_point_org_flat = four_point_org.flatten(2).permute(0, 2, 1).contiguous()
    four_point_pred_flat = four_point_pred.flatten(2).permute(0, 2, 1).contiguous()
    # H = tgm.get_perspective_transform(four_point_org_flat, four_point_pred_flat)
    H = get_perspective_transform_torch(four_point_org_flat, four_point_pred_flat)
    warped_thermal = tgm.warp_perspective(thermal_image.cpu(), H,
                                          (resize_width, resize_width))
    warped_np = (warped_thermal[0].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)

    has_gt = gt_image_path is not None and os.path.exists(gt_image_path)
    ncols = 5 if has_gt else 4
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))

    axes[0].imshow(sat_np)
    axes[0].set_title('Satellite Image')
    axes[0].axis('off')

    axes[1].imshow(thermal_np, cmap='gray')
    axes[1].set_title('Thermal Image')
    axes[1].axis('off')

    axes[2].imshow(sat_with_bbox)
    axes[2].set_title('Predicted Alignment (green bbox)')
    axes[2].axis('off')

    axes[3].imshow(sat_np)
    axes[3].imshow(warped_np, alpha=0.5)
    axes[3].set_title('Overlay')
    axes[3].axis('off')

    if has_gt:
        gt_img = np.array(Image.open(gt_image_path).convert('RGB'))
        axes[4].imshow(gt_img)
        axes[4].set_title('Ground Truth')
        axes[4].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to {save_path}")

    disp = four_pred[0].cpu()
    disp_scaled = disp * alpha
    print(f"\n4-Point Displacement (pixels at {resize_width}x{resize_width} scale):")
    print(f"  Top-Left:     dx={disp[0, 0, 0]:.2f}, dy={disp[1, 0, 0]:.2f}")
    print(f"  Top-Right:    dx={disp[0, 0, 1]:.2f}, dy={disp[1, 0, 1]:.2f}")
    print(f"  Bottom-Left:  dx={disp[0, 1, 0]:.2f}, dy={disp[1, 1, 0]:.2f}")
    print(f"  Bottom-Right: dx={disp[0, 1, 1]:.2f}, dy={disp[1, 1, 1]:.2f}")
    print(f"\n4-Point Displacement (scaled to {database_size}x{database_size}):")
    print(f"  Top-Left:     dx={disp_scaled[0, 0, 0]:.2f}, dy={disp_scaled[1, 0, 0]:.2f}")
    print(f"  Top-Right:    dx={disp_scaled[0, 0, 1]:.2f}, dy={disp_scaled[1, 0, 1]:.2f}")
    print(f"  Bottom-Left:  dx={disp_scaled[0, 1, 0]:.2f}, dy={disp_scaled[1, 1, 0]:.2f}")
    print(f"  Bottom-Right: dx={disp_scaled[0, 1, 1]:.2f}, dy={disp_scaled[1, 1, 1]:.2f}")

    if std_pred is not None:
        std = std_pred[0].cpu()
        std_scaled = std * alpha
        print(f"\nUncertainty Std (pixels at {resize_width}x{resize_width} scale):")
        print(f"  Top-Left:     sx={std[0, 0, 0]:.2f}, sy={std[1, 0, 0]:.2f}")
        print(f"  Top-Right:    sx={std[0, 0, 1]:.2f}, sy={std[1, 0, 1]:.2f}")
        print(f"  Bottom-Left:  sx={std[0, 1, 0]:.2f}, sy={std[1, 1, 0]:.2f}")
        print(f"  Bottom-Right: sx={std[0, 1, 1]:.2f}, sy={std[1, 1, 1]:.2f}")
        print(f"\nUncertainty Std (scaled to {database_size}x{database_size}):")
        print(f"  Top-Left:     sx={std_scaled[0, 0, 0]:.2f}, sy={std_scaled[1, 0, 0]:.2f}")
        print(f"  Top-Right:    sx={std_scaled[0, 0, 1]:.2f}, sy={std_scaled[1, 0, 1]:.2f}")
        print(f"  Bottom-Left:  sx={std_scaled[0, 1, 0]:.2f}, sy={std_scaled[1, 1, 0]:.2f}")
        print(f"  Bottom-Right: sx={std_scaled[0, 1, 1]:.2f}, sy={std_scaled[1, 1, 1]:.2f}")
        print(f"\n  Mean Std: {std.mean():.2f} (at {resize_width}x{resize_width}), "
              f"{std_scaled.mean():.2f} (at {database_size}x{database_size})")


# ==============================================================================
# Resource Monitor (GPU, CPU, RAM)
# ==============================================================================

class ResourceMonitor:
    """Monitor GPU, CPU, and RAM usage"""
    def __init__(self):
        self.cpu_usage = []
        self.ram_usage = []
        self.gpu_usage = []
        self.gpu_memory = []
    
    def sample(self):
        """Sample current resource usage"""
        if HAS_PSUTIL:
            self.cpu_usage.append(psutil.cpu_percent(interval=0.1))
            self.ram_usage.append(psutil.virtual_memory().percent)
        
        if HAS_GPUTIL and torch.cuda.is_available():
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    self.gpu_usage.append(gpus[0].load * 100)
                    self.gpu_memory.append(gpus[0].memoryUsed / gpus[0].memoryTotal * 100)
            except:
                pass
    
    def summary(self):
        """Print resource usage summary"""
        print(f"\n{'='*70}")
        print("System Resource Usage")
        print(f"{'='*70}")
        
        if self.cpu_usage:
            print(f"{'CPU':<28} | Avg: {np.mean(self.cpu_usage):6.2f}% | "
                  f"[{np.min(self.cpu_usage):6.2f}% - {np.max(self.cpu_usage):6.2f}%]")
        
        if self.ram_usage:
            print(f"{'RAM':<28} | Avg: {np.mean(self.ram_usage):6.2f}% | "
                  f"[{np.min(self.ram_usage):6.2f}% - {np.max(self.ram_usage):6.2f}%]")
        
        if self.gpu_usage:
            print(f"{'GPU Utilization':<28} | Avg: {np.mean(self.gpu_usage):6.2f}% | "
                  f"[{np.min(self.gpu_usage):6.2f}% - {np.max(self.gpu_usage):6.2f}%]")
        
        if self.gpu_memory:
            print(f"{'GPU Memory':<28} | Avg: {np.mean(self.gpu_memory):6.2f}% | "
                  f"[{np.min(self.gpu_memory):6.2f}% - {np.max(self.gpu_memory):6.2f}%]")
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"{'PyTorch GPU':<28} | Used: {allocated:.2f}GB | "
                  f"Reserved: {reserved:.2f}GB | Total: {total:.2f}GB")
        
        print(f"{'='*70}")


# ==============================================================================
# Timing Tracker
# ==============================================================================

class TimingTracker:
    """Simple timing tracker for module profiling"""
    def __init__(self):
        self.timers = {}
        self.accumulated = {}
    
    def start(self, name):
        self.timers[name] = time.perf_counter()
    
    def end(self, name):
        if name in self.timers:
            elapsed = time.perf_counter() - self.timers[name]
            if name not in self.accumulated:
                self.accumulated[name] = []
            self.accumulated[name].append(elapsed)

    def add_time(self, name, elapsed):
        if name not in self.accumulated:
            self.accumulated[name] = []
        self.accumulated[name].append(elapsed)
    
    def summary(self):
        """Return dict with mean/min/max/count timing for each module"""
        summary = {}
        for name, times in self.accumulated.items():
            summary[name] = {
                'mean': np.mean(times),
                'min': np.min(times),
                'max': np.max(times),
                'std': np.std(times),
                'count': len(times),
                'total': np.sum(times)
            }
        return summary
    
    def print_summary(self, num_images):
        """Print detailed timing summary"""
        print(f"\n{'='*70}")
        print("Timing Details - Per Module Per Image")
        print(f"{'='*70}")
        print(f"{'Module':<28} | {'Count':<6} | {'Avg (ms)':<10} | {'Total (s)':<10} | {'%':<6}")
        print(f"{'-'*70}")
        
        timing_summary = self.summary()
        total_time = sum(s['total'] for s in timing_summary.values())
        
        for module in timing_summary.keys():
            stats = timing_summary[module]
            percent = (stats['total'] / (total_time + 1e-6)) * 100
            print(f"{module:<28} | {stats['count']:<6.0f} | {stats['mean']*1000:<10.2f} | {stats['total']:<10.2f} | {percent:<6.1f}%")
        
        print(f"{'-'*70}")
        print(f"{'TOTAL':<28} | {'':<6} | {'':<10} | {total_time:<10.2f} | {'100.0%':<6}")
        print(f"\nAvg per image: {total_time/num_images:.3f}s ({num_images/total_time:.2f} FPS)")
        print(f"{'='*70}")

# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":
    os.system('sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches')
    
    import time
    import pandas as pd
    from tqdm import tqdm
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Load Model from HuggingFace Hub ----
    print("Loading model...")
    repo_id = 'xjh19972/UASTHN'
    model = UASTHN.from_pretrained(repo_id)
    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}\n")

    # ---- Configuration ----
    satellite_dir = 'js_datasets/Dehat/satellite'
    thermal_dir = 'js_datasets/Dehat/thermal'
    output_excel = 'js_excels/batch_results.xlsx'
    num_samples = 108
    tiles_per_satellite = 9

    # ---- Collect valid image pairs ----
    image_pairs = []
    for i in range(num_samples):
        sat_idx = i // tiles_per_satellite + 1
        th_idx = i % tiles_per_satellite + 1
        sat_path = os.path.join(satellite_dir, f"{sat_idx}.tif")
        th_path = os.path.join(thermal_dir, f"{sat_idx}_{th_idx}.tif")
        
        if os.path.exists(sat_path) and os.path.exists(th_path):
            image_pairs.append((sat_idx, th_idx, sat_path, th_path))

    print(f"Found {len(image_pairs)} valid image pairs\n")
    print(f"Processing {'='*40}\n")

    # ---- Batch inference ----
    results = []
    global_timing = TimingTracker()
    resource_monitor = ResourceMonitor()
    count = 0
    alpha = model.database_size / model.resize_width

    with torch.inference_mode():
        for sat_idx, th_idx, sat_path, th_path in tqdm(image_pairs, desc="Inference"):
            resource_monitor.sample()
            
            try:
                if count != 0:
                    global_timing.start('data_loading')
                satellite = load_and_preprocess_satellite(sat_path,model.resize_width).to(device)
                thermal = load_and_preprocess_thermal(th_path, model.resize_width).to(device)
                if count != 0:
                    global_timing.end('data_loading')

                # ---- Forward with uncertainty ----
                four_pred, std_pred = model.forward_with_uncertainty(
                    satellite, thermal,
                    ue_num_crops=5, ue_shift=32,
                    ue_shift_crops_types=model.ue_shift_crops_types,
                    firststage_ue=True)
                
                
                # ---- Extract results ----
                four_point_org = torch.zeros((1, 2, 2, 2), device=device)
                four_point_org[:, :, 0, 0] = torch.tensor([0, 0])
                four_point_org[:, :, 0, 1] = torch.tensor([model.resize_width - 1, 0])
                four_point_org[:, :, 1, 0] = torch.tensor([0, model.resize_width - 1])
                four_point_org[:, :, 1, 1] = torch.tensor([model.resize_width - 1, model.resize_width - 1])
                
                corners = (four_pred[0] + four_point_org[0]).cpu().numpy() * alpha
                corners_flat = np.array([
                    corners[0, 0, 0], corners[1, 0, 0],
                    corners[0, 0, 1], corners[1, 0, 1],
                    corners[0, 1, 0], corners[1, 1, 0],
                    corners[0, 1, 1], corners[1, 1, 1],
                ])
                
                uncertainty = std_pred[0].cpu().mean().item() * alpha
                
                results.append({
                    'image_index': count,
                    'sat_idx': sat_idx,
                    'th_idx': th_idx,
                    'x1': corners_flat[0], 'y1': corners_flat[1],
                    'x2': corners_flat[2], 'y2': corners_flat[3],
                    'x3': corners_flat[4], 'y3': corners_flat[5],
                    'x4': corners_flat[6], 'y4': corners_flat[7],
                    'sat': sat_path,
                    'th': th_path,
                    'uncertainty': uncertainty,
                })
                count += 1
                
            except Exception as e:
                print(f"Error {sat_idx}_{th_idx}: {e}")
                continue

    # ---- Save results (with division-by-zero protection) ----
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_excel), exist_ok=True)
    df.to_excel(output_excel, index=False)
    
    n_processed = len(df)
    print(f"\n{'='*60}")
    print(f"Results saved: {output_excel}")
    print(f"Processed: {n_processed}/{len(image_pairs)} images")
    print(f"{'='*60}")

    # ---- Timing Summary (with zero-check) ----
    if n_processed > 0:
        global_timing.print_summary(n_processed)
    else:
        print("No images processed - check errors above.")

    # ---- Resource Summary ----
    resource_monitor.summary()

    # ---- Uncertainty Summary ----
    if 'uncertainty' in df.columns:
        unc = df['uncertainty'].values
        print(f"\nUncertainty Statistics:")
        print(f"{'-'*60}")
        print(f"{'Mean':25s} | {unc.mean():.4f}")
        print(f"{'Median':25s} | {np.median(unc):.4f}")
        print(f"{'Std':25s} | {unc.std():.4f}")
        print(f"{'Min - Max':25s} | {unc.min():.4f} - {unc.max():.4f}")
    
    print(f"{'='*60}\n")
