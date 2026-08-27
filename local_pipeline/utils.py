import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import interpolate
from skimage import io
import random
import sys
import torch.optim as optim
from PIL import Image
import logging
import wandb
import matplotlib.pyplot as plt
from datasets_4cor_img import inv_base_transforms
import torchvision
import cv2
import time

def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    # img (b*crops*64*64, 1, 64/2^i, 64/2^i)
    # coords (b*crops*64*64, 9, 9, 2) / 2^i
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    # print('@@@ xgrid bef', xgrid.shape, xgrid)
    # print('@@@ ygrid bef', ygrid.shape, ygrid)
    xgrid = 2 * xgrid / (W - 1) - 1  # normalize indices for grid_sample function (-1,+1)
    ygrid = 2 * ygrid / (H - 1) - 1
    # print('@@@ xgrid aft', xgrid.shape, xgrid)
    # print('@@@ ygrid aft', ygrid.shape, ygrid)

    grid = torch.cat([xgrid, ygrid], dim=-1)  # b*crops*64*64, 9, 9, 2
    # print('@@@ grid', grid.shape, grid)
    img = F.grid_sample(img, grid, align_corners=True)  # b*crops*64*64, 1, 9, 9

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].expand(batch, -1, -1, -1)


def save_img(img, path):
    img = inv_base_transforms(img.detach().cpu())
    img.save(path)


def save_overlap_img(img1, img2, path):
    img1 = inv_base_transforms(img1.detach().cpu())
    img1 = np.array(img1)
    img2 = inv_base_transforms(img2.detach().cpu())
    img2 = np.array(img2)
    plt.figure(figsize=(50, 10), dpi=200)
    plt.axis('off')
    plt.imshow(img2)
    plt.imshow(img1, alpha=0.25)
    plt.savefig(path, bbox_inches='tight')
    plt.close()

def save_overlap_bbox_img(img1, img2, path, four_point_gt, four_point_pred, crop_bbox=None, ue_mask=None):
    four_point_gt = np.round(four_point_gt.cpu().numpy())
    four_point_pred = np.round(four_point_pred.cpu().numpy())
    plt.figure(figsize=(50, 10), dpi=200)
    plt.axis('off')
    img1_list = np.empty((img1.shape[0], img1.shape[2], img1.shape[3], img1.shape[1]))
    img2_list = np.empty((img1.shape[0], img1.shape[2], img1.shape[3], img1.shape[1]))
    for i in range(len(img2)):
        image1 = inv_base_transforms(img1[i].detach().cpu())
        image1 = np.array(image1)
        image2 = inv_base_transforms(img2[i].detach().cpu())
        image2 = np.array(image2)
        four_point_gt_single = np.int32(four_point_gt[i]).reshape((-1,1,2))
        temp = four_point_gt_single[2].copy()
        four_point_gt_single[2] = four_point_gt_single[3]
        four_point_gt_single[3] = temp
        four_point_pred_single = np.int32(four_point_pred[i]).reshape((-1,1,2))
        temp = four_point_pred_single[2].copy()
        four_point_pred_single[2] = four_point_pred_single[3]
        four_point_pred_single[3] = temp
        image2=cv2.polylines(image2,[four_point_gt_single],True,(0,255,0),2)
        image2=cv2.polylines(image2,[four_point_pred_single],True,(255,0,0),1)
        if crop_bbox is not None:
            crop_bbox_single = np.int32(crop_bbox[i]).reshape((-1,1,2))
            temp = crop_bbox_single[2].copy()
            crop_bbox_single[2] = crop_bbox_single[3]
            crop_bbox_single[3] = temp
            image2=cv2.polylines(image2,[crop_bbox_single],True,(0,0,255),1)
        img1_list[i] = image1
        img2_list[i] = image2
    img1_tensor = torch.from_numpy(img1_list).permute(0, 3, 1, 2)
    img2_tensor = torch.from_numpy(img2_list).permute(0, 3, 1, 2)
    img1_tensor = torchvision.utils.make_grid(img1_tensor, nrow=16, padding = 0, pad_value=255)
    img2_tensor = torchvision.utils.make_grid(img2_tensor, nrow=16, padding = 0, pad_value=255)
    img1 = np.array(img1_tensor.permute(1, 2, 0)).astype(np.uint8)
    img2 = np.array(img2_tensor.permute(1, 2, 0)).astype(np.uint8)
    plt.imshow(img2)
    plt.imshow(img1, alpha=0.25)
    if ue_mask is not None and ue_mask[0, 3] == False:
        path = path.split('.')[0]+"_rej.png"
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.to(x.device)
    vgrid = torch.autograd.Variable(grid) + flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(x, vgrid, align_corners=True)
    mask = torch.autograd.Variable(torch.ones(x.size())).to(x.device)
    mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)

    mask[mask < 0.999] = 0
    mask[mask > 0] = 1

    return output * mask


def sequence_loss(four_preds, four_pred, flow_gt, gamma, args, metrics, four_ue_list=None):
    """ Loss function defined over sequence of flow predictions """

    if args.first_stage_ue and args.ue_method == "augment":
        flow_4cor = torch.zeros((four_preds[0].shape[0]//args.ue_num_crops, 2, 2, 2)).to(four_preds[0].device)
        flow_4cor[:, :, 0, 0] = flow_gt[:, :, 0, 0]
        flow_4cor[:, :, 0, 1] = flow_gt[:, :, 0, -1]
        flow_4cor[:, :, 1, 0] = flow_gt[:, :, -1, 0]
        flow_4cor[:, :, 1, 1] = flow_gt[:, :, -1, -1]
        flow_4cor_repeat = torch.zeros((four_preds[0].shape[0]//args.ue_num_crops, args.ue_num_crops, 2, 2, 2)).to(four_preds[0].device)
        _, C, H, W= flow_gt.shape
        flow_gt_repeat = flow_gt.view(four_preds[0].shape[0]//args.ue_num_crops, 1, C, H, W).repeat(1, args.ue_num_crops, 1, 1, 1)
        flow_4cor_repeat[:, :, :, 0, 0] = flow_gt_repeat[:, :, :, 0, 0]
        flow_4cor_repeat[:, :, :, 0, 1] = flow_gt_repeat[:, :, :, 0, -1]
        flow_4cor_repeat[:, :, :, 1, 0] = flow_gt_repeat[:, :, :, -1, 0]
        flow_4cor_repeat[:, :, :, 1, 1] = flow_gt_repeat[:, :, :, -1, -1]
    else:
        flow_4cor = torch.zeros((four_preds[0].shape[0], 2, 2, 2)).to(four_preds[0].device)
        flow_4cor[:, :, 0, 0] = flow_gt[:, :, 0, 0]
        flow_4cor[:, :, 0, 1] = flow_gt[:, :, 0, -1]
        flow_4cor[:, :, 1, 0] = flow_gt[:, :, -1, 0]
        flow_4cor[:, :, 1, 1] = flow_gt[:, :, -1, -1]
    ce_loss = 0.0

    if args.first_stage_ue and args.ue_method == "single":
        assert four_ue_list is not None
        for i in range(args.iters_lev0):
            i_weight = gamma ** (args.iters_lev0 - i - 1)
            i4cor_loss = (four_preds[i] - flow_4cor)**2*torch.exp(-four_ue_list[i])/2 + four_ue_list[i]/2
            ce_loss += i_weight * (i4cor_loss).mean()
    elif args.first_stage_ue and args.ue_method == "augment":
        for i in range(args.iters_lev0):
            i_weight = gamma ** (args.iters_lev0 - i - 1)
            four_pred_reshape = four_preds[i].view(four_preds[i].shape[0]//args.ue_num_crops, args.ue_num_crops, 2, 2, 2)
            i4cor_loss_ori = args.ue_lambda_tta * (four_pred_reshape[:, :1] - flow_4cor_repeat[:, :1]).abs()
            i4cor_loss = args.ue_lambda_tta * (four_pred_reshape[:, 1:] - flow_4cor_repeat[:, 1:]).abs()
            ce_loss += i_weight * (torch.cat([i4cor_loss_ori, i4cor_loss], dim=1)).mean()
    else:
        for i in range(args.iters_lev0):
            i_weight = gamma ** (args.iters_lev0 - i - 1)
            i4cor_loss = (four_preds[i] - flow_4cor).abs()
            ce_loss += i_weight * (i4cor_loss).mean()

    if args.two_stages:
        for i in range(args.iters_lev0, args.iters_lev1 + args.iters_lev0):
            i_weight = gamma ** (args.iters_lev1 + args.iters_lev0 - i - 1)
            i4cor_loss = (four_preds[i] - flow_4cor).abs()
            ce_loss += i_weight * (i4cor_loss).mean()
        mace = torch.sum((four_preds[-1] - flow_4cor) ** 2, dim=1).sqrt()
    elif args.first_stage_ue and args.ue_method == "augment":
        mace = torch.sum((four_pred - flow_4cor) ** 2, dim=1).sqrt()
    else:
        mace = torch.sum((four_preds[-1] - flow_4cor) ** 2, dim=1).sqrt()

    metrics['ce_loss'] = ce_loss.item()

    metrics['1px'] = (mace < 1).float().mean().item()
    metrics['3px'] = (mace < 3).float().mean().item()
    metrics['mace'] = mace.mean().item()

    return ce_loss, metrics


def single_loss(four_preds, four_pred, flow_gt, gamma, args, metrics, four_ue_list=None):
    """ Loss function defined over sequence of flow predictions """

    if args.first_stage_ue and args.ue_method == "augment":
        flow_4cor = torch.zeros((four_preds[0].shape[0]//args.ue_num_crops, 2, 2, 2)).to(four_preds[0].device)
        flow_4cor[:, :, 0, 0] = flow_gt[:, :, 0, 0]
        flow_4cor[:, :, 0, 1] = flow_gt[:, :, 0, -1]
        flow_4cor[:, :, 1, 0] = flow_gt[:, :, -1, 0]
        flow_4cor[:, :, 1, 1] = flow_gt[:, :, -1, -1]
        flow_4cor_repeat = torch.zeros((four_preds[0].shape[0]//args.ue_num_crops, args.ue_num_crops, 2, 2, 2)).to(four_preds[0].device)
        _, C, H, W= flow_gt.shape
        flow_gt_repeat = flow_gt.view(four_preds[0].shape[0]//args.ue_num_crops, 1, C, H, W).repeat(1, args.ue_num_crops, 1, 1, 1)
        flow_4cor_repeat[:, :, :, 0, 0] = flow_gt_repeat[:, :, :, 0, 0]
        flow_4cor_repeat[:, :, :, 0, 1] = flow_gt_repeat[:, :, :, 0, -1]
        flow_4cor_repeat[:, :, :, 1, 0] = flow_gt_repeat[:, :, :, -1, 0]
        flow_4cor_repeat[:, :, :, 1, 1] = flow_gt_repeat[:, :, :, -1, -1]
    else:
        flow_4cor = torch.zeros((four_preds[0].shape[0], 2, 2, 2)).to(four_preds[0].device)
        flow_4cor[:, :, 0, 0] = flow_gt[:, :, 0, 0]
        flow_4cor[:, :, 0, 1] = flow_gt[:, :, 0, -1]
        flow_4cor[:, :, 1, 0] = flow_gt[:, :, -1, 0]
        flow_4cor[:, :, 1, 1] = flow_gt[:, :, -1, -1]

    if args.first_stage_ue and args.ue_method == "single":
        assert four_ue_list is not None
        ce_loss = ((four_preds[0] - flow_4cor)**2*torch.exp(-four_ue_list[0])/2 + four_ue_list[0]/2).mean()
    elif args.first_stage_ue and args.ue_method == "augment":
        four_pred_reshape = four_preds[0].view(four_preds[0].shape[0]//args.ue_num_crops, args.ue_num_crops, 2, 2, 2)
        i4cor_loss_ori = args.ue_lambda_tta * (four_pred_reshape[:, :1] - flow_4cor_repeat[:, :1]).abs()
        i4cor_loss = args.ue_lambda_tta * (four_pred_reshape[:, 1:] - flow_4cor_repeat[:, 1:]).abs()
        ce_loss = (torch.cat([i4cor_loss_ori, i4cor_loss], dim=1)).mean()
    else:
        ce_loss = (four_preds[0] - flow_4cor).abs().mean()
        
    mace = torch.sum((four_pred - flow_4cor) ** 2, dim=1).sqrt()
    metrics['1px'] = (mace < 1).float().mean().item()
    metrics['3px'] = (mace < 3).float().mean().item()
    metrics['mace'] = mace.mean().item()
    metrics['ce_loss'] = ce_loss.item()

    return ce_loss, metrics

def single_neg_loss(gamma, args, metrics, four_ue_list, four_ue_pred_list=None):
    """ Loss function defined over sequence of flow predictions """

    neg_loss = torch.mean(F.relu(args.neg_margin - four_ue_list[0]))
    metrics['neg_loss'] = neg_loss.item()
    neg_loss = args.neg_loss_lambda * neg_loss
    if four_ue_pred_list is not None:
        raise NotImplementedError()

    return neg_loss, metrics

def sequence_neg_loss(gamma, args, metrics, four_ue_list, four_ue_pred_list=None):
    """ Loss function defined over sequence of flow predictions """
    neg_loss = 0.0
    for i in range(args.iters_lev0):
        i_weight = gamma ** (args.iters_lev0 - i - 1)
        i4cor_loss = F.relu(args.neg_margin - four_ue_list[i])
        neg_loss += i_weight * (i4cor_loss).mean()
    metrics['neg_loss'] = neg_loss.item()
    neg_loss = args.neg_loss_lambda * neg_loss

    if four_ue_pred_list is not None:
        raise NotImplementedError()
    
    return neg_loss, metrics

def fetch_optimizer(args, model_para):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model_para, lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=args.lr, total_steps=args.num_steps + 100,
                                              pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
    return optimizer, scheduler



class ResourceMonitor:
    """Monitor GPU, CPU, and RAM usage with detailed tracking"""
    def __init__(self):
        # Try to import psutil
        try:
            import psutil
            self.HAS_PSUTIL = True
            self.psutil = psutil
        except ImportError:
            self.HAS_PSUTIL = False
            self.psutil = None
        
        # Try to import GPUtil
        try:
            import GPUtil
            self.HAS_GPUTIL = True
            self.GPUtil = GPUtil
        except ImportError:
            self.HAS_GPUTIL = False
            self.GPUtil = None
        
        # Regular samples (for average usage)
        self.cpu_usage = []
        self.ram_usage = []
        self.gpu_usage = []
        self.gpu_memory = []
        
        # Peak tracking
        self.peak_cpu = 0
        self.peak_ram = 0
        self.peak_gpu_util = 0
        self.peak_gpu_memory = 0
        
        # Delta tracking
        self.delta_history = []
        self.baseline = None
        self.last_snapshot = None
        
        # GPU memory history (for trend analysis)
        self.gpu_memory_history = []
        self.cpu_usage_history = []
    
    def get_detailed_gpu_info(self):
        """Get detailed GPU information"""
        info = {
            'utilization': None,
            'memory_used_gb': None,
            'memory_total_gb': None,
            'memory_percent': None,
            'temperature': None,
            'power_usage': None
        }
        
        if self.HAS_GPUTIL and torch.cuda.is_available():
            try:
                gpus = self.GPUtil.getGPUs()
                if gpus:
                    info['utilization'] = gpus[0].load * 100
                    info['memory_used_gb'] = gpus[0].memoryUsed / 1024
                    info['memory_total_gb'] = gpus[0].memoryTotal / 1024
                    info['memory_percent'] = (gpus[0].memoryUsed / gpus[0].memoryTotal) * 100
                    info['temperature'] = gpus[0].temperature
                    info['power_usage'] = getattr(gpus[0], 'power_draw', None)
            except:
                pass
        
        # Add PyTorch-specific info
        if torch.cuda.is_available():
            info['pytorch_allocated_gb'] = torch.cuda.memory_allocated() / (1024**3)
            info['pytorch_reserved_gb'] = torch.cuda.memory_reserved() / (1024**3)
            info['pytorch_max_allocated_gb'] = torch.cuda.max_memory_allocated() / (1024**3)
            torch.cuda.reset_peak_memory_stats()  # Reset for next measurement
        
        return info
    
    def snapshot(self, label=None):
        """Take a snapshot of current resource usage"""
        snapshot_data = {
            'label': label,
            'timestamp': time.time(),
            'cpu_percent': None,
            'ram_percent': None,
            'ram_gb': None,
            'gpu_info': self.get_detailed_gpu_info()
        }
        
        if self.HAS_PSUTIL:
            snapshot_data['cpu_percent'] = self.psutil.cpu_percent(interval=0)
            mem = self.psutil.virtual_memory()
            snapshot_data['ram_percent'] = mem.percent
            snapshot_data['ram_gb'] = mem.used / (1024**3)
        
        return snapshot_data
    
    def measure_delta(self, start_snapshot, end_snapshot, operation_name="Operation"):
        """Calculate delta between two snapshots"""
        delta = {
            'operation': operation_name,
            'duration': end_snapshot['timestamp'] - start_snapshot['timestamp'],
            'cpu_delta': None,
            'ram_delta_gb': None,
            'ram_delta_percent': None,
            'gpu_util_delta': None,
            'gpu_memory_delta_gb': None,
            'gpu_memory_delta_percent': None,
            'pytorch_allocated_delta_gb': None,
            'pytorch_reserved_delta_gb': None
        }
        
        if start_snapshot['cpu_percent'] is not None and end_snapshot['cpu_percent'] is not None:
            delta['cpu_delta'] = end_snapshot['cpu_percent'] - start_snapshot['cpu_percent']
        
        if start_snapshot['ram_gb'] is not None and end_snapshot['ram_gb'] is not None:
            delta['ram_delta_gb'] = end_snapshot['ram_gb'] - start_snapshot['ram_gb']
            delta['ram_delta_percent'] = end_snapshot['ram_percent'] - start_snapshot['ram_percent']
        
        # GPU deltas
        start_gpu = start_snapshot['gpu_info']
        end_gpu = end_snapshot['gpu_info']
        
        if start_gpu['utilization'] is not None and end_gpu['utilization'] is not None:
            delta['gpu_util_delta'] = end_gpu['utilization'] - start_gpu['utilization']
        
        if start_gpu['memory_used_gb'] is not None and end_gpu['memory_used_gb'] is not None:
            delta['gpu_memory_delta_gb'] = end_gpu['memory_used_gb'] - start_gpu['memory_used_gb']
            if start_gpu['memory_percent'] is not None and end_gpu['memory_percent'] is not None:
                delta['gpu_memory_delta_percent'] = end_gpu['memory_percent'] - start_gpu['memory_percent']
        
        if start_gpu.get('pytorch_allocated_gb') is not None and end_gpu.get('pytorch_allocated_gb') is not None:
            delta['pytorch_allocated_delta_gb'] = end_gpu['pytorch_allocated_gb'] - start_gpu['pytorch_allocated_gb']
            delta['pytorch_reserved_delta_gb'] = end_gpu['pytorch_reserved_gb'] - start_gpu['pytorch_reserved_gb']
        
        self.delta_history.append(delta)
        return delta
    
    def start_monitoring(self):
        """Start monitoring with baseline snapshot"""
        self.baseline = self.snapshot(label="Baseline")
        self.last_snapshot = self.baseline
        return self.baseline
    
    def checkpoint(self, label):
        """Take a checkpoint and calculate delta from last checkpoint"""
        current = self.snapshot(label=label)
        if self.last_snapshot is not None:
            delta = self.measure_delta(self.last_snapshot, current, label)
            self.last_snapshot = current
            return delta
        self.last_snapshot = current
        return None
    
    def end_monitoring(self, label="Total"):
        """End monitoring and calculate total delta from baseline"""
        final = self.snapshot(label=label)
        if self.baseline is not None:
            total_delta = self.measure_delta(self.baseline, final, f"Total_{label}")
            return total_delta
        return None
    
    def sample(self):
        """Sample current resource usage and track peaks"""
        if self.HAS_PSUTIL:
            cpu = self.psutil.cpu_percent(interval=0.1)
            ram = self.psutil.virtual_memory().percent
            self.cpu_usage.append(cpu)
            self.ram_usage.append(ram)
            self.cpu_usage_history.append(cpu)
            
            # Update peaks
            self.peak_cpu = max(self.peak_cpu, cpu)
            self.peak_ram = max(self.peak_ram, ram)
        
        if self.HAS_GPUTIL and torch.cuda.is_available():
            try:
                gpus = self.GPUtil.getGPUs()
                if gpus:
                    gpu_util = gpus[0].load * 100
                    gpu_mem = gpus[0].memoryUsed / gpus[0].memoryTotal * 100
                    self.gpu_usage.append(gpu_util)
                    self.gpu_memory.append(gpu_mem)
                    self.gpu_memory_history.append(gpu_mem)
                    
                    # Update peaks
                    self.peak_gpu_util = max(self.peak_gpu_util, gpu_util)
                    self.peak_gpu_memory = max(self.peak_gpu_memory, gpu_mem)
            except:
                pass
    
    def print_delta_summary(self):
        """Print summary of all deltas measured"""
        if not self.delta_history:
            print("\nNo delta measurements recorded.")
            return
        
        print(f"\n{'='*100}")
        print("Resource Usage Delta Analysis")
        print(f"{'='*100}")
        
        for delta in self.delta_history:
            print(f"\n📊 {delta['operation']}:")
            print(f"   Duration: {delta['duration']:.2f}s")
            print(f"{'-'*70}")
            
            if delta['cpu_delta'] is not None:
                print(f"  CPU Change:        {delta['cpu_delta']:+.2f}%")
            
            if delta['ram_delta_gb'] is not None:
                print(f"  RAM Change:        {delta['ram_delta_gb']:+.2f} GB ({delta['ram_delta_percent']:+.2f}%)")
            
            if delta['gpu_util_delta'] is not None:
                print(f"  GPU Util Change:   {delta['gpu_util_delta']:+.2f}%")
            
            if delta['gpu_memory_delta_gb'] is not None:
                print(f"  GPU Memory Change: {delta['gpu_memory_delta_gb']:+.2f} GB ({delta['gpu_memory_delta_percent']:+.2f}%)")
            
            if delta.get('pytorch_allocated_delta_gb') is not None:
                print(f"  PyTorch Allocated: {delta['pytorch_allocated_delta_gb']:+.2f} GB")
                print(f"  PyTorch Reserved:  {delta['pytorch_reserved_delta_gb']:+.2f} GB")
    
    def print_optimization_suggestions(self):
        """Print suggestions based on resource usage patterns"""
        print(f"\n{'='*100}")
        print("Optimization Suggestions")
        print(f"{'='*100}")
        
        # GPU utilization suggestions
        if self.gpu_usage and np.mean(self.gpu_usage) < 50:
            print("\n⚠️  LOW GPU UTILIZATION DETECTED (Avg: {:.1f}%)".format(np.mean(self.gpu_usage)))
            print("   Suggestions:")
            print("   • Increase batch size to better utilize GPU")
            print("   • Use torch.compile() for faster execution")
            print("   • Consider using larger input resolution")
            print("   • Profile data loading pipeline for bottlenecks")
        
        # CPU bottleneck detection
        if self.cpu_usage and np.mean(self.cpu_usage) > 70:
            print("\n⚠️  HIGH CPU USAGE DETECTED (Avg: {:.1f}%)".format(np.mean(self.cpu_usage)))
            print("   Suggestions:")
            print("   • Use num_workers > 0 in DataLoader")
            print("   • Pre-process and cache data if possible")
            print("   • Use pinned memory (pin_memory=True)")
        
        # GPU memory suggestions
        if self.gpu_memory and np.mean(self.gpu_memory) < 30:
            print("\nℹ️  LOW GPU MEMORY USAGE (Avg: {:.1f}%)".format(np.mean(self.gpu_memory)))
            print("   Suggestion: Consider larger model or higher resolution")
        
        # RAM usage suggestions
        if self.ram_usage and np.mean(self.ram_usage) > 80:
            print("\n⚠️  HIGH RAM USAGE DETECTED (Avg: {:.1f}%)".format(np.mean(self.ram_usage)))
            print("   Suggestions:")
            print("   • Process images in smaller batches")
            print("   • Clear cache periodically with torch.cuda.empty_cache()")
        
        # Delta analysis for inference
        inference_delta = None
        for delta in self.delta_history:
            if "Inference" in delta['operation']:
                inference_delta = delta
                break
        
        if inference_delta:
            print(f"\n📈 INFERENCE MEMORY FOOTPRINT:")
            print(f"   GPU Memory Increase: {inference_delta.get('gpu_memory_delta_gb', 0):.2f} GB")
            print(f"   RAM Increase: {inference_delta.get('ram_delta_gb', 0):.2f} GB")
    
    def summary(self):
        """Print resource usage summary with peaks"""
        print(f"\n{'='*70}")
        print("System Resource Usage Summary")
        print(f"{'='*70}")
        
        if self.cpu_usage:
            print(f"\n{'CPU':<30} | Avg: {np.mean(self.cpu_usage):6.2f}% | "
                  f"Peak: {self.peak_cpu:6.2f}% | "
                  f"Range: [{np.min(self.cpu_usage):6.2f}% - {np.max(self.cpu_usage):6.2f}%]")
        
        if self.ram_usage:
            print(f"{'RAM':<30} | Avg: {np.mean(self.ram_usage):6.2f}% | "
                  f"Peak: {self.peak_ram:6.2f}% | "
                  f"Range: [{np.min(self.ram_usage):6.2f}% - {np.max(self.ram_usage):6.2f}%]")
        
        if self.gpu_usage:
            print(f"{'GPU Utilization':<30} | Avg: {np.mean(self.gpu_usage):6.2f}% | "
                  f"Peak: {self.peak_gpu_util:6.2f}% | "
                  f"Range: [{np.min(self.gpu_usage):6.2f}% - {np.max(self.gpu_usage):6.2f}%]")
        
        if self.gpu_memory:
            print(f"{'GPU Memory':<30} | Avg: {np.mean(self.gpu_memory):6.2f}% | "
                  f"Peak: {self.peak_gpu_memory:6.2f}% | "
                  f"Range: [{np.min(self.gpu_memory):6.2f}% - {np.max(self.gpu_memory):6.2f}%]")
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            max_allocated = torch.cuda.max_memory_allocated() / (1024**3)
            print(f"\n{'PyTorch GPU State':<30} | Used: {allocated:.2f}GB | "
                  f"Reserved: {reserved:.2f}GB | Total: {total:.2f}GB")
            print(f"{'PyTorch Peak':<30} | Max Allocated: {max_allocated:.2f}GB")
        
        print(f"{'='*70}")
        
        # Print delta summary if available
        if self.delta_history:
            self.print_delta_summary()
            self.print_optimization_suggestions()
    
    def log_to_logger(self, logger):
        """Log resource summary to logger"""
        logger.info("\n" + "=" * 60)
        logger.info("RESOURCE USAGE SUMMARY")
        logger.info("=" * 60)
        
        if self.cpu_usage:
            logger.info(f"CPU - Avg: {np.mean(self.cpu_usage):.1f}%, Peak: {self.peak_cpu:.1f}%")
        
        if self.ram_usage:
            logger.info(f"RAM - Avg: {np.mean(self.ram_usage):.1f}%, Peak: {self.peak_ram:.1f}%")
        
        if self.gpu_usage:
            logger.info(f"GPU Util - Avg: {np.mean(self.gpu_usage):.1f}%, Peak: {self.peak_gpu_util:.1f}%")
        
        if self.gpu_memory:
            logger.info(f"GPU Memory - Avg: {np.mean(self.gpu_memory):.1f}%, Peak: {self.peak_gpu_memory:.1f}%")
        
        # Log optimization suggestions
        if self.gpu_usage and np.mean(self.gpu_usage) < 50:
            logger.warning(f"Low GPU utilization ({np.mean(self.gpu_usage):.1f}%). Consider larger batch size.")


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
            del self.timers[name]
    
    def add_time(self, name, elapsed):
        """Directly add a time measurement (for when you already have elapsed time)"""
        if name not in self.accumulated:
            self.accumulated[name] = []
        self.accumulated[name].append(elapsed)
    
    def summary(self):
        """Return dict with mean/min/max/count timing for each module"""
        summary = {}
        for name, times in self.accumulated.items():
            if len(times) > 0:
                summary[name] = {
                    'mean': np.mean(times),
                    'min': np.min(times),
                    'max': np.max(times),
                    'std': np.std(times),
                    'count': len(times),
                    'total': np.sum(times)
                }
        return summary
    
    def reset(self):
        """Clear all accumulated timings"""
        self.accumulated = {}
        self.timers = {}
    
    def print_summary(self, num_images, warmup_images=0):
        """Print summary, skipping the first N images"""
        if warmup_images > 0 and num_images > warmup_images:
            # Skip first N records for each module
            filtered_accumulated = {}
            for name, times in self.accumulated.items():
                if len(times) > warmup_images:
                    filtered_accumulated[name] = times[warmup_images:]
            
            # Temporarily replace accumulated for summary
            original_accumulated = self.accumulated
            self.accumulated = filtered_accumulated
            effective_images = num_images - warmup_images
        else:
            effective_images = num_images
        
        print(f"\n{'='*70}")
        print(f"Timing Details - Per Module Per Image (excluding first {warmup_images} warmup)")
        print(f"{'='*70}")
        print(f"{'Module':<50} | {'Count':<6} | {'Avg (ms)':<10} | {'Total (s)':<10} | {'%':<6}")
        print(f"{'-'*70}")
        
        timing_summary = self.summary()
        
        if not timing_summary:
            print("No timing data available.")
            return
        
        total_time = sum(s['total'] for s in timing_summary.values())
        
        for module in timing_summary.keys():
            stats = timing_summary[module]
            percent = (stats['total'] / (total_time + 1e-6)) * 100
            print(f"{module:<50} | {stats['count']:<6.0f} | {stats['mean']*1000:<10.2f} | {stats['total']:<10.2f} | {percent:<6.1f}%")
        
        print(f"{'-'*70}")
        print(f"{'TOTAL':<50} | {'':<6} | {'':<10} | {total_time:<10.2f} | {'100.0%':<6}")
        print(f"\nAvg per image: {total_time/effective_images:.3f}s ({effective_images/total_time:.2f} FPS)")
        print(f"{'='*70}")
        
        # Restore original if we modified it
        if warmup_images > 0 and num_images > warmup_images:
            self.accumulated = original_accumulated