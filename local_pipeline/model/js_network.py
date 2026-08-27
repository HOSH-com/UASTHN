import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.geometry.transform as tgm
import kornia.geometry.bbox as bbox
from update import GMA
from extractor import BasicEncoderQuarter
from corr import CorrBlock
from utils import coords_grid, sequence_loss, single_loss, single_neg_loss, sequence_neg_loss, fetch_optimizer, warp, TimingTracker
import os
import sys
from model.sync_batchnorm import convert_model
import wandb
import torchvision
import random
import time
import logging
from model.baseline import DHN
import datasets_4cor_img as datasets
import numpy as np, math

from model.js_kornia_replacement import (
    get_perspective_transform_torch,
    crop_and_resize_torch,
)

autocast = torch.amp.autocast

class IHN(nn.Module):
    def __init__(self, args, first_stage, ue_method="none", timer = None):
        super().__init__()
        # self.device = torch.device('cuda:' + str(args.gpuid[0]))
        self.device = torch.device(args.device)

        if timer is None:
            self.global_timing = TimingTracker()
        else:
            self.global_timing = timer
        self.args = args
        self.ue_method = ue_method
        self.hidden_dim = 128
        self.context_dim = 128
        self.first_stage = first_stage
        self.fnet1 = BasicEncoderQuarter(output_dim=256, norm_fn='instance')
        if self.args.lev0:
            sz = self.args.resize_width // 4
            self.update_block_4 = GMA(self.args, sz)
            if self.ue_method == "single" and self.first_stage:
                self.ue_update_block_4 = GMA(self.args, sz)
        self.imagenet_mean = None
        self.imagenet_std = None

    def get_flow_now_4(self, four_point):
        four_point = four_point / 4
        four_point_org = torch.zeros((2, 2, 2)).to(four_point.device)
        four_point_org[:, 0, 0] = torch.Tensor([0, 0])
        four_point_org[:, 0, 1] = torch.Tensor([self.sz[3]-1, 0])
        four_point_org[:, 1, 0] = torch.Tensor([0, self.sz[2]-1])
        four_point_org[:, 1, 1] = torch.Tensor([self.sz[3]-1, self.sz[2]-1])

        four_point_org = four_point_org.unsqueeze(0)
        four_point_org = four_point_org.repeat(self.sz[0], 1, 1, 1)
        four_point_new = four_point_org + four_point
        four_point_org = four_point_org.flatten(2).permute(0, 2, 1).contiguous()
        four_point_new = four_point_new.flatten(2).permute(0, 2, 1).contiguous()
        H = tgm.get_perspective_transform(four_point_org, four_point_new)
        # H = get_perspective_transform_torch(four_point_org, four_point_new)
        gridy, gridx = torch.meshgrid(torch.linspace(0, self.args.resize_width//4-1, steps=self.args.resize_width//4), torch.linspace(0, self.args.resize_width//4-1, steps=self.args.resize_width//4))
        points = torch.cat((gridx.flatten().unsqueeze(0), gridy.flatten().unsqueeze(0), torch.ones((1, self.args.resize_width//4 * self.args.resize_width//4))),
                           dim=0).unsqueeze(0).repeat(H.shape[0], 1, 1).to(four_point.device)
        points_new = H.bmm(points)
        if torch.isnan(points_new).any():
            raise KeyError("Some of transformed coords are NaN!")
        points_new = points_new / points_new[:, 2, :].unsqueeze(1)
        points_new = points_new[:, 0:2, :]
        flow = torch.cat((points_new[:, 0, :].reshape(self.sz[0], self.sz[3], self.sz[2]).unsqueeze(1),
                          points_new[:, 1, :].reshape(self.sz[0], self.sz[3], self.sz[2]).unsqueeze(1)), dim=1)
        return flow

    def get_flow_now_2(self, four_point):
        four_point = four_point / 2
        four_point_org = torch.zeros((2, 2, 2)).to(four_point.device)
        four_point_org[:, 0, 0] = torch.Tensor([0, 0])
        four_point_org[:, 0, 1] = torch.Tensor([self.sz[3]-1, 0])
        four_point_org[:, 1, 0] = torch.Tensor([0, self.sz[2]-1])
        four_point_org[:, 1, 1] = torch.Tensor([self.sz[3]-1, self.sz[2]-1])

        four_point_org = four_point_org.unsqueeze(0)
        four_point_org = four_point_org.repeat(self.sz[0], 1, 1, 1)
        four_point_new = four_point_org + four_point
        four_point_org = four_point_org.flatten(2).permute(0, 2, 1).contiguous()
        four_point_new = four_point_new.flatten(2).permute(0, 2, 1).contiguous()
        H = tgm.get_perspective_transform(four_point_org, four_point_new)
        gridy, gridx = torch.meshgrid(torch.linspace(0, self.sz[3]-1, steps=self.sz[3]), torch.linspace(0, self.sz[2]-1, steps=self.sz[2]))
        points = torch.cat((gridx.flatten().unsqueeze(0), gridy.flatten().unsqueeze(0), torch.ones((1, self.sz[3] * self.sz[2]))),
                           dim=0).unsqueeze(0).repeat(self.sz[0], 1, 1).to(four_point.device)
        points_new = H.bmm(points)
        points_new = points_new / points_new[:, 2, :].unsqueeze(1)
        points_new = points_new[:, 0:2, :]
        flow = torch.cat((points_new[:, 0, :].reshape(self.sz[0], self.sz[3], self.sz[2]).unsqueeze(1),
                          points_new[:, 1, :].reshape(self.sz[0], self.sz[3], self.sz[2]).unsqueeze(1)), dim=1)
        return flow

    def initialize_flow_4(self, img):
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//4, W//4).to(img.device)
        coords1 = coords_grid(N, H//4, W//4).to(img.device)

        return coords0, coords1

    def initialize_flow_2(self, img):
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//2, W//2).to(img.device)
        coords1 = coords_grid(N, H//2, W//2).to(img.device)

        return coords0, coords1

    def forward(self, image1, image2, iters_lev0 = 6, iters_lev1=6, corr_level=2, corr_radius=4, early_stop=-1, four_point_disp_init_64=None):
        # image1 = 2 * (image1 / 255.0) - 1.0
        # image2 = 2 * (image2 / 255.0) - 1.

        stage = ""
        if self.first_stage:
            # print('### img1', image1[0].shape, image1[0])
            # print('### img2', image2[0].shape, image2[0])
            stage = "First Stage"
        else:
            stage = "Second Stage"

        if self.imagenet_mean is None:
            self.imagenet_mean = torch.Tensor([0.485, 0.456, 0.406]).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(image1.device)
            self.imagenet_std = torch.Tensor([0.229, 0.224, 0.225]).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(image1.device)
        image1 = (image1.contiguous() - self.imagenet_mean) / self.imagenet_std
        image2 = (image2.contiguous() - self.imagenet_mean) / self.imagenet_std
        # time1 = time.time()
        with autocast(device_type='cuda', enabled=self.args.mixed_precision):
            # fmap1_64, fmap1_128 = self.fnet1(image1)
            # fmap2_64, _ = self.fnet1(image2)

            self.global_timing.start(f"IHN Feature extraction {stage}")

            if not self.args.fnet_cat:
                fmap1_64 = self.fnet1(image1)
                fmap2_64 = self.fnet1(image2)
            else:
                fmap_64 = self.fnet1(torch.cat([image1, image2], dim=0))
                fmap1_64 = fmap_64[:image1.shape[0]]
                fmap2_64 = fmap_64[image1.shape[0]:]
        # time2 = time.time()
        # print("Time for fnet1: " + str(time2 - time1) + " seconds") # 0.004 + # 0.004

        fmap1 = fmap1_64.float()
        fmap2 = fmap2_64.float()
        self.global_timing.end(f"IHN Feature extraction {stage}")

        self. global_timing.start(f"CorrBlock Initialazation {stage}")
        # print(fmap1.shape, fmap2.shape)
        corr_fn = CorrBlock(fmap1, fmap2, num_levels=corr_level, radius=corr_radius)
        # print('### corr_pyramid', len(corr_fn.corr_pyramid))
        # for i in range(len(corr_fn.corr_pyramid)):
        #     print(f'@@@ corr_fn.corr_pyramid[{i}]', corr_fn.corr_pyramid[i].shape, corr_fn.corr_pyramid[i])

        sz = fmap1_64.shape
        self.sz = sz
        coords0, coords1 = self.initialize_flow_4(image1)
        four_point_disp = torch.zeros((sz[0], 2, 2, 2)).to(fmap1.device)
        # print(coords1.shape)
        # print('### coords1 bef\n', coords1.shape, coords1)
        if four_point_disp_init_64 is not None:
            # Global translation applied to the whole quarter-resolution grid,
            # used by the 'ue_sec points' secondary uncertainty pass to start
            # the iterative refinement from an offset location instead of the
            # identity (coords0 == coords1) initialization.
            # coords1 = coords1 + four_point_disp_init_64.to(coords1.device).view(coords1.shape[0], 2, 1, 1)
            # print('### four_point_disp_init_64\n', four_point_disp_init_64.shape, four_point_disp_init_64)
            four_point_disp = four_point_disp_init_64 * 4
            coords1 = self.get_flow_now_4(four_point_disp)
        
        # print('### coords1 aft\n', coords1.shape, coords1)


        if self.args.check_step != -1 and self.first_stage and self.ue_method == "augment":
            B, C, H, W = fmap1.shape
            corr_fn_early = CorrBlock(fmap1.view(B//self.args.ue_num_crops, self.args.ue_num_crops, C, H, W)[:, 0], fmap2.view(B//self.args.ue_num_crops, self.args.ue_num_crops, C, H, W)[:, 0], num_levels=corr_level, radius=corr_radius)
            coords0_early = coords0.view(coords0.shape[0]//self.args.ue_num_crops, self.args.ue_num_crops, coords0.shape[1], coords0.shape[2], coords0.shape[3])[:,0]
        self. global_timing.end(f"CorrBlock Initialazation {stage}")
        # print(coords0.shape, coords1.shape)
        four_point_predictions = []
        if self.ue_method == "single" and self.first_stage:
            four_point_ues = []
        # time1 = time.time()

        sum_corr = 0.0
        sum_update = 0.0
        sum_dlt = 0.0
        # self.global_timing.start(f"for {stage}")
        for itr in range(iters_lev0):
            # if itr == 2: exit()
            # print('\n###', itr, '###\n')
            start_time = time.perf_counter()
            if (self.first_stage and (self.args.check_step == -1 or itr <= self.args.check_step)) or not self.first_stage:
                corr = corr_fn(coords1)
                flow = coords1 - coords0
            elif self.ue_method=="augment":
                corr = corr_fn_early(coords1_early)
                flow = coords1_early - coords0_early
            else:
                corr = corr_fn(coords1)
                flow = coords1 - coords0
            sum_corr += time.perf_counter() - start_time
            # print(corr.shape, flow.shape)
            # print('### flow[0]', flow[0].shape, flow[0])
            # print('### corr[0]', corr[0].shape, corr[0])
            with autocast(device_type='cuda', enabled=self.args.mixed_precision):
                start_time = time.perf_counter()
                if self.args.weight:
                    delta_four_point, weight = self.update_block_4(corr, flow)
                else:
                    delta_four_point = self.update_block_4(corr, flow)
                    if self.ue_method == "single" and self.first_stage:
                        ue_four_point = torch.clamp(self.ue_update_block_4(corr, flow), min=self.args.si_min)
                sum_update += time.perf_counter() - start_time

            try:
                start_time = time.perf_counter()
                last_four_point_disp = four_point_disp
                four_point_disp =  four_point_disp + delta_four_point
                # print('### delta_four_point', delta_four_point.shape, delta_four_point)
                # print('### four_point_disp', four_point_disp.shape, four_point_disp)
                # print('### coords1 bef', coords1.shape, coords1)
                coords1 = self.get_flow_now_4(four_point_disp) # Possible error: Unsolvable H
                # print('### coords1 aft', coords1.shape, coords1)
                four_point_predictions.append(four_point_disp)
                sum_dlt += time.perf_counter() - start_time

                if self.ue_method == "single" and self.first_stage:
                    four_point_ues.append(ue_four_point)
                if itr == self.args.check_step and self.first_stage and self.ue_method == "augment":
                    self.sz = torch.Size([self.sz[0]//self.args.ue_num_crops, self.sz[1], self.sz[2], self.sz[3]])
                    four_point_disp = four_point_disp.view(four_point_disp.shape[0]//self.args.ue_num_crops, self.args.ue_num_crops, 2, 2, 2)[:, 0]
                    coords1_early = coords1.view(coords1.shape[0]//self.args.ue_num_crops, self.args.ue_num_crops, coords1.shape[1], coords1.shape[2], coords1.shape[3])[:, 0]
            except Exception as e:
                logging.debug(e)
                logging.debug("Ignore this delta. Use last disp.")
                four_point_disp = last_four_point_disp
                coords1 = self.get_flow_now_4(four_point_disp) # Possible error: Unsolvable H
                four_point_predictions.append(four_point_disp)
                if self.ue_method == "single" and self.first_stage and self.ue_method == "augment":
                    four_point_ues.append(ue_four_point)
                if itr == self.args.check_step and not self.first_stage:
                    self.sz = torch.Size([self.sz[0]//self.args.ue_num_crops, self.sz[1], self.sz[2], self.sz[3]])
                    four_point_disp = four_point_disp.view(four_point_disp.shape[0]//self.args.ue_num_crops, self.args.ue_num_crops, 2, 2, 2)[:, 0]
                    coords1_early = coords1.view(coords1.shape[0]//self.args.ue_num_crops, self.args.ue_num_crops, coords1.shape[1], coords1.shape[2], coords1.shape[3])[:, 0]

            if early_stop!=-1 and itr==early_stop:
                break
        # self.global_timing.end(f"for {stage}")
        # time2 = time.time()
        # print("Time for iterative: " + str(time2 - time1) + " seconds") # 0.12
        self.global_timing.add_time(f'Corr {stage}', sum_corr)
        self.global_timing.add_time(f'Update {stage}', sum_update)
        self.global_timing.add_time(f'DLT {stage}', sum_dlt)

        # if self.first_stage:
            # print('### fimg1[0]', fmap1[0].shape, fmap1[0])
            # print('### fimg2[0]', fmap2[0].shape, fmap2[0])
            # print('### four_point_disp[0]', four_point_disp[0].shape, four_point_disp[0])
        if self.ue_method == "single" and self.first_stage:
            return four_point_predictions, four_point_disp, four_point_ues
        else:
            return four_point_predictions, four_point_disp

arch_list = {"IHN": IHN,
             "DHN": DHN,
             }

class UASTHN():
    def __init__(self, args, for_training=False):
        super().__init__()
        self.count = 0
        self.args = args
        self.global_timing = TimingTracker()
        self.ue_method = args.ue_method
        self.device = args.device
        self.four_point_org_single = torch.zeros((1, 2, 2, 2)).to(self.device)
        self.four_point_org_single[:, :, 0, 0] = torch.Tensor([0, 0]).to(self.device)
        self.four_point_org_single[:, :, 0, 1] = torch.Tensor([self.args.resize_width - 1, 0]).to(self.device)
        self.four_point_org_single[:, :, 1, 0] = torch.Tensor([0, self.args.resize_width - 1]).to(self.device)
        self.four_point_org_single[:, :, 1, 1] = torch.Tensor([self.args.resize_width - 1, self.args.resize_width - 1]).to(self.device)
        self.four_point_org_large_single = torch.zeros((1, 2, 2, 2)).to(self.device)
        self.four_point_org_large_single[:, :, 0, 0] = torch.Tensor([0, 0]).to(self.device)
        self.four_point_org_large_single[:, :, 0, 1] = torch.Tensor([self.args.database_size - 1, 0]).to(self.device)
        self.four_point_org_large_single[:, :, 1, 0] = torch.Tensor([0, self.args.database_size - 1]).to(self.device)
        self.four_point_org_large_single[:, :, 1, 1] = torch.Tensor([self.args.database_size - 1, self.args.database_size - 1]).to(self.device) # Only to calculate flow so no -1

        if self.args.custom == "satcrop":
            alpha = self.args.database_size / self.args.resize_width
            w_rs = self.args.database_size_large // alpha
            self.four_point_org_xrs_single = torch.zeros((1, 2, 2, 2)).to(self.device)
            self.four_point_org_xrs_single[:, :, 0, 0] = torch.Tensor([0, 0]).to(self.device)
            self.four_point_org_xrs_single[:, :, 0, 1] = torch.Tensor([w_rs - 1, 0]).to(self.device)
            self.four_point_org_xrs_single[:, :, 1, 0] = torch.Tensor([0, w_rs - 1]).to(self.device)
            self.four_point_org_xrs_single[:, :, 1, 1] = torch.Tensor([w_rs - 1, w_rs - 1]).to(self.device)

            self.four_point_org_xs_single = torch.zeros((1, 2, 2, 2)).to(self.device)
            self.four_point_org_xs_single[:, :, 0, 0] = torch.Tensor([0, 0]).to(self.device)
            self.four_point_org_xs_single[:, :, 0, 1] = torch.Tensor([self.args.database_size_large - 1, 0]).to(self.device)
            self.four_point_org_xs_single[:, :, 1, 0] = torch.Tensor([0, self.args.database_size_large - 1]).to(self.device)
            self.four_point_org_xs_single[:, :, 1, 1] = torch.Tensor([self.args.database_size_large - 1, self.args.database_size_large - 1]).to(self.device)

        if self.args.first_stage_ue and self.ue_method == "ensemble":
            self.ensemble_model_names_raw = open(args.ue_ensemble_load_models, "r").readlines()
            self.ensemble_model_names = []
            assert self.args.ue_num_crops <= len(self.ensemble_model_names_raw)
            for i in range(self.args.ue_num_crops):
                self.ensemble_model_names.append(self.ensemble_model_names_raw[i].strip())
            self.netG_list = [arch_list[args.arch](args, True, self.ue_method) for i in range(self.args.ue_num_crops)]
        else:
            self.netG = arch_list[args.arch](args, True, self.ue_method, timer=self.global_timing)
        self.shift_flow_bbox = None
        if args.two_stages:
            corr_level = args.corr_level
            args.corr_level = 2
            self.netG_fine = IHN(args, False, timer=self.global_timing)
            args.corr_level = corr_level
            if args.restore_ckpt is not None and not args.finetune:
                self.set_requires_grad(self.netG, False)
        self.criterionAUX = sequence_loss if self.args.arch == "IHN" else single_loss
        self.criterionNEG = sequence_neg_loss if self.args.arch == "IHN" else single_neg_loss
        if self.args.first_stage_ue:
            self.ue_rng = np.random.default_rng(seed=args.ue_seed)
        if for_training:
            if args.two_stages:
                if args.restore_ckpt is None or args.finetune:
                    self.optimizer_G, self.scheduler_G = fetch_optimizer(args, list(self.netG.parameters()) + list(self.netG_fine.parameters()))
                else:
                    self.optimizer_G, self.scheduler_G = fetch_optimizer(args,list(self.netG_fine.parameters()))
            else:
                self.optimizer_G, self.scheduler_G = fetch_optimizer(args, list(self.netG.parameters()))

    def setup(self):
        if hasattr(self, 'netD'):
            self.netD = self.init_net(self.netD)
        if self.args.first_stage_ue and self.ue_method == "ensemble":
            for i in range(len(self.netG_list)):
                self.netG_list[i] = self.init_net(self.netG_list[i])
        else:
            self.netG = self.init_net(self.netG)
        if hasattr(self, 'netG_fine'):
            self.netG_fine = self.init_net(self.netG_fine)

    def init_net(self, model):
        # model = torch.nn.DataParallel(model)
        # if torch.cuda.device_count() >= 2:
        #     # When using more than 1GPU, use sync_batchnorm for torch.nn.DataParallel
        #     model = convert_model(model)
        #     model = model.to(self.device)
        model = model.to(self.device)
        return model

    def set_input(self, A, B, flow_gt=None, neg_A=None):
        self.image_1_ori = A.to(self.device, non_blocking=True)
        self.image_2 = B.to(self.device, non_blocking=True)
        self.flow_gt = flow_gt.to(self.device, non_blocking=True)
        if self.flow_gt is not None:
            if self.args.vis_all:
                self.real_warped_image_2 = mywarp(datasets.base_transforms(self.image_2), self.flow_gt, self.four_point_org_single) # Comment for performance evaluation
            self.flow_4cor = torch.zeros((self.flow_gt.shape[0], 2, 2, 2)).to(self.flow_gt.device)
            self.flow_4cor[:, :, 0, 0] = self.flow_gt[:, :, 0, 0]
            self.flow_4cor[:, :, 0, 1] = self.flow_gt[:, :, 0, -1]
            self.flow_4cor[:, :, 1, 0] = self.flow_gt[:, :, -1, 0]
            self.flow_4cor[:, :, 1, 1] = self.flow_gt[:, :, -1, -1]
        else:
            self.real_warped_image_2 = None
        self.image_1 = F.interpolate(self.image_1_ori, size=self.args.resize_width, mode='bilinear', align_corners=True, antialias=True)
        if self.args.vis_all:
            self.image_1_show = datasets.base_transforms(self.image_1) # Comment for performance evaluation
        if neg_A is not None:
            self.image_1_neg_ori = neg_A.to(self.device, non_blocking=True)
            self.image_1_neg = F.interpolate(self.image_1_neg_ori, size=self.args.resize_width, mode='bilinear', align_corners=True, antialias=True)
        else:
            self.image_1_neg = None

    def forward(self, for_training=False, for_test=False):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.count += 1

        # time1 = time.time()
        # --- Generate Crops & Augment ---
        if self.args.first_stage_ue and self.ue_method == "augment":
            self.first_stage_ue_generate()

        # --- Run First Stage ---
        if self.args.first_stage_ue:
            if self.ue_method == "ensemble":
                four_preds_list_ensemble = []
                if self.args.ue_method == "augment_ensemble":
                    four_preds_list, _ = self.netG_list[0](image1=self.image_1, image2=self.image_2, iters_lev0=self.args.iters_lev0, corr_level=self.args.corr_level, early_stop=self.args.check_step)
                else:
                    four_preds_list, _ = self.netG_list[0](image1=self.image_1, image2=self.image_2, iters_lev0=self.args.iters_lev0, corr_level=self.args.corr_level)
                four_preds_list_ensemble.append(four_preds_list)
                for i in range(1, len(self.netG_list)):
                    four_preds_list, _ = self.netG_list[i](image1=self.image_1, image2=self.image_2, iters_lev0=self.args.iters_lev0, corr_level=self.args.corr_level, early_stop=self.args.check_step)
                    four_preds_list_ensemble.append(four_preds_list)
                self.four_preds_list, self.four_pred = self.stack_ensemble_results(four_preds_list_ensemble, self.args.check_step)
            elif self.ue_method == "single":
                self.four_preds_list, self.four_pred, self.four_pred_ue_list = self.netG(image1=self.image_1, image2=self.image_2, iters_lev0=self.args.iters_lev0, corr_level=self.args.corr_level)
            else:
                self.four_preds_list, self.four_pred = self.netG(image1=self.image_1, image2=self.image_2, iters_lev0=self.args.iters_lev0, corr_level=self.args.corr_level)
        else:
            self.four_preds_list, self.four_pred = self.netG(image1=self.image_1, image2=self.image_2, iters_lev0=self.args.iters_lev0, corr_level=self.args.corr_level)

        # --- First Stage Aggregation ---
        # self.four_preds_list = [self.four_preds_list[-1]]
        if self.args.first_stage_ue:
            # for i in range(len(self.four_preds_list)): # DEBUG
            #     self.four_preds_list[i] = self.flow_4cor # DEBUG
            # self.four_pred = self.flow_4cor # DEBUG
            # if self.ue_method == "augment":
            #     pass
                # self.fake_warped_image_2_multi_before = mywarp(self.image_2, self.four_preds_list[self.args.check_step], self.four_point_org_single) # Comment for performance evaluation

            if self.ue_method != "single":
                self.four_preds_list, self.four_pred = self.first_stage_ue_aggregation(self.four_preds_list, for_training)

            # --- Secondary uncertainty pass: random-offset starting points ---
            # Must run *before* self.image_1/self.image_2 get sliced back down
            # below, so it sees all `ue_num_crops` CropTTA crop variations
            # combined with the `ue_sec_points_n` random-offset starts
            # (ue_num_crops * ue_sec_points_n predictions per tile).
            ue_sec = getattr(self.args, "ue_sec", "none")
            self.four_preds_list_ue_sec, self.four_pred_ue_sec, self.std_four_pred_ue_sec = None, None, None
            if ue_sec is not 'none' and self.ue_method != "ensemble":
                # TODO IMPLEMENT FOR BATCH GREATER THAN 1 

                if self.std_four_pred_five_crops.shape[0] != 1:
                    raise NotImplementedError("Only batch size 1 is supported for ue_sec!")
                ue = self.std_four_pred_five_crops.view(self.std_four_pred_five_crops.shape[0], -1).mean(dim=1).item()

                if self.args.ue_sec_trigger_range[0] <= ue <= self.args.ue_sec_trigger_range[1]:
                    if ue_sec == "points":
                        # print("!!! ue", ue, self.std_four_pred_five_crops)
                        self.four_preds_list_ue_sec, self.four_pred_ue_sec, self.std_four_pred_ue_sec = self.run_ue_sec_points()

            if self.ue_method == "augment":
                B5, C, H, W = self.image_2.shape
                # image_2_full = self.image_2.view(B5//self.args.ue_num_crops, self.args.ue_num_crops, C, H, W)[:, :1].repeat(1, self.args.ue_num_crops, 1, 1, 1).view(-1, C, H, W) # Comment for performance evaluation
                # self.fake_warped_image_2_multi_after = mywarp(image_2_full, self.four_preds_list[self.args.check_step], self.four_point_org_single) # Comment for performance evaluation
                # self.image_1_multi = self.image_1 # Comment for performance evaluation
                # self.image_2_multi = self.image_2 # Comment for performance evaluation
                self.image_1 = self.image_1.view(B5//self.args.ue_num_crops, self.args.ue_num_crops, C, H, W)[:, 0]
                self.image_2 = self.image_2.view(B5//self.args.ue_num_crops, self.args.ue_num_crops, C, H, W)[:, 0]
            elif self.ue_method == "single":
                self.std_four_pred_five_crops = torch.sqrt(torch.exp(self.four_pred_ue_list[-1]))
        else:
            # No CropTTA active (ue_num_crops == 1): still run the secondary
            # pass, on the plain per-tile batch.
            if getattr(self.args, "ue_sec", "none") == "points":
                # init attr
                self.std_four_pred_sec_points = None

                self.run_ue_sec_points()

        # time2 = time.time()
        # logging.debug("Time for 1st forward pass: " + str(time2 - time1) + " seconds")

        # TODO IMPLEMENT UE REJECTION

        # --- Run Second Stage ---
        if self.args.two_stages and not (self.ue_method == "ensemble" and self.args.ue_method == "augment_ensemble"):
            # self.four_pred = self.flow_4cor # DEBUG
            # self.four_preds_list[-1] = self.four_pred # DEBUG
            # self.four_preds_list[-1] = torch.zeros_like(self.four_pred).to(self.four_pred.device) # DEBUG
            # time1 = time.time()
            self.image_1_crop, delta, self.flow_bbox = self.get_cropped_st_images(self.image_1_ori, self.four_pred, self.args.fine_padding, self.args.detach, self.args.augment_two_stages)
            # time2 = time.time()
            # logging.debug("Time for crop: " + str(time2 - time1) + " seconds")
            # time1 = time.time()
            self.image_2_crop = self.image_2
            self.four_preds_list_fine, self.four_pred_fine = self.netG_fine(image1=self.image_1_crop, image2=self.image_2_crop, iters_lev0=self.args.iters_lev1)
            # time2 = time.time()
            # logging.debug("Time for 2nd forward pass: " + str(time2 - time1) + " seconds")
            # self.four_pred_fine = torch.zeros_like(self.four_pred).to(self.four_pred.device) # DEBUG
            # self.four_preds_list_fine[-1] = self.four_pred_fine # DEBUG
            # print(self.four_pred[0])
            # print(self.four_pred_fine[0])
            self.four_preds_list_fine, self.four_pred = self.combine_coarse_fine(self.four_preds_list, self.four_pred, self.four_preds_list_fine, self.four_pred_fine, delta, self.flow_bbox, for_training)
            # print(self.four_pred[0])
            # raise KeyError()
        if self.args.vis_all:
            self.fake_warped_image_2 = mywarp(datasets.base_transforms(self.image_2), self.four_pred, self.four_point_org_single) # Comment for performance evaluation

    def _sample_ue_sec_points_offsets(self, n):
        """Sample per-repeat global (dx, dy) offsets used to initialize coords1
        away from coords0 for the '--ue_sec points' secondary uncertainty pass.

        The first offset is always (0, 0), matching the normal/primary
        prediction (coords0 == coords1). The remaining (n - 1) offsets are
        drawn so that the overlap between the shifted and un-shifted
        quarter-resolution grids stays at least `ue_sec_points_width / grid_size`
        (e.g. width=48 on a 64-sized grid guarantees >= 3/4 overlap).
        """
        grid_size = self.args.resize_width // 4
        width = self.args.ue_sec_points_width
        max_offset = grid_size - width

        # offsets = torch.zeros((n, 2), device=self.device)
        if n >= 1 and max_offset >= 0:
            if self.args.ue_sec_points_mode != "rand":
                raise NotImplementedError(
                    f"ue_sec_points_mode='{self.args.ue_sec_points_mode}' is not implemented yet"
                )
            # offsets[1:] = (torch.rand((n - 1, 2), device=self.device) * 2 - 1) * max_offset
            offsets = torch.tensor(self.ue_rng.integers(0, max_offset + 1, size=(n, 2)), device=self.device)  # TODO use float
        return offsets

    def run_ue_sec_points(self):
        """Secondary uncertainty pass ('--ue_sec points'): re-run the coarse
        (first) stage `ue_sec_points_n` times for each of the `ue_num_crops`
        CropTTA crop variations, using different global square-bbox offsets.

        coords1_offset has shape (B_total * n, 2, 2, 2):
            [batch, x/y, up/down, left/right]

        The four corners represent displacement of the square bbox relative
        to the constant large frame of size `resize_width // 4`.
        """
        B_total, C, H, W = self.image_1.shape
        num_crops = self.args.ue_num_crops
        B = B_total // num_crops
        n = self.args.ue_sec_points_n - 1

        img1_rep = (self.image_1.unsqueeze(1).repeat(1, n, 1, 1, 1).view(B_total * n, C, H, W))
        img2_rep = (self.image_2.unsqueeze(1).repeat(1, n, 1, 1, 1).view(B_total * n, C, H, W))
        if self.args.custom == 'satcrop':
            self.xrcs_before = self.xrcs_before.unsqueeze(1).repeat(1, n, 1, 1, 1).view(B_total * n, 2, 2, 2)
        else:
            self.xct_before = self.xct_before.unsqueeze(1).repeat(1, n, 1, 1, 1).view(B_total * n, 2, 2, 2)

        # Sample upper-left corner of each square bbox.
        # Shape: (n, 2), where [:, 0] = x and [:, 1] = y.
        offsets_single = self._sample_ue_sec_points_offsets(n)

        frame = self.args.resize_width // 4
        width = self.args.ue_sec_points_width

        # Convert upper-left bbox offsets into displacement of all 4 corners.
        #
        # Shape:
        #   (n, 2, 2, 2)
        #    │  │  │  └── left/right
        #    │  │  └───── up/down
        #    │  └──────── x/y
        #    └─────────── sampled offset
        #
        # The bbox coordinates are measured relative to the large frame,
        # so the displacement of the right/bottom edge is:
        #
        #     offset + width - frame
        #
        dx = offsets_single[:, 0]
        dy = offsets_single[:, 1]

        # corner_disp = torch.zeros((n, 2, 2, 2), device=offsets_single.device, dtype=offsets_single.dtype,)
        corner_disp = torch.zeros((n, 2, 2, 2), device=offsets_single.device, dtype=torch.float32)

        # x displacement
        corner_disp[:, 0, 0, 0] = dx
        corner_disp[:, 0, 0, 1] = dx + width - frame
        corner_disp[:, 0, 1, 0] = dx
        corner_disp[:, 0, 1, 1] = dx + width - frame

        # y displacement
        corner_disp[:, 1, 0, 0] = dy
        corner_disp[:, 1, 0, 1] = dy
        corner_disp[:, 1, 1, 0] = dy + width - frame
        corner_disp[:, 1, 1, 1] = dy + width - frame

        # # x displacement
        # corner_disp[:, 0, 0, 0] = 40.71875
        # corner_disp[:, 0, 0, 1] = -133
        # corner_disp[:, 0, 1, 0] = 41.03125
        # corner_disp[:, 0, 1, 1] = -134.625

        # # y displacement
        # corner_disp[:, 1, 0, 0] = 10.8125
        # corner_disp[:, 1, 0, 1] = 11.328125
        # corner_disp[:, 1, 1, 0] = -163.5
        # corner_disp[:, 1, 1, 1] = -163.875

        # Repeat the same n offsets for every image/crop in the batch.
        # Final shape: (B_total * n, 2, 2, 2)
        start_points_disp = (corner_disp.unsqueeze(0).repeat(B_total, 1, 1, 1, 1).view(B_total * n, 2, 2, 2))

        # Bypass the CropTTA early-stop/ue_num_crops-based batch splitting
        # inside IHN.forward, since this pass has its own batch layout.
        prev_check_step = self.args.check_step
        self.args.check_step = -1

        try:
            four_preds_list, four_pred = self.netG(
                image1=img1_rep,
                image2=img2_rep,
                iters_lev0=self.args.iters_lev0,
                corr_level=self.args.corr_level,
                four_point_disp_init_64=start_points_disp,
            )
        finally:
            self.args.check_step = prev_check_step


        alpha = self.args.database_size / self.args.resize_width
        four_preds_list, _, _, _ = self.ue_aggregation(four_preds_list, alpha, False, self.args.check_step)
        four_pred = four_preds_list[-1]

        # print('@@@ self.four_pred\n', self.four_pred.shape, self.four_pred)
        # print('@@@ four_pred\n' , four_pred.shape, four_pred)

        four_pred = four_pred.view(B, num_crops, n, 2, 2, 2)

        combined = four_pred.reshape(B, num_crops * n, 2, 2, 2)

        std_four_pred = torch.std(combined, dim=1)  # (B, 2, 2, 2)

        # print('@@@ self.four_pred\n', self.std_four_pred_five_crops, self.four_pred.shape, self.four_pred)
        # print('@@@ four_pred\n', std_four_pred, four_pred.shape, four_pred)

        return four_preds_list, four_pred, std_four_pred

    def forward_neg(self, for_training=False):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # time1 = time.time()
        if self.args.first_stage_ue and self.ue_method == "augment":
            self.first_stage_ue_generate(neg_forward=True)
        if self.args.first_stage_ue and self.ue_method == "single":
            four_preds_list_neg, four_pred_neg, self.four_pred_ue_neg_list = self.netG(image1=self.image_1_neg, image2=self.image_2, iters_lev0=self.args.iters_lev0, corr_level=self.args.corr_level)
        else:
            four_preds_list_neg, four_pred_neg = self.netG(image1=self.image_1_neg, image2=self.image_2, iters_lev0=self.args.iters_lev0, corr_level=self.args.corr_level)
        if self.args.first_stage_ue:
            # for i in range(len(self.four_preds_list)): # DEBUG
            #     self.four_preds_list[i] = self.flow_4cor # DEBUG
            # self.four_pred = self.flow_4cor # DEBUG
            if self.ue_method != "single":
                _, _ = self.first_stage_ue_aggregation(four_preds_list_neg, for_training, neg_forward=True)
            if self.ue_method == "augment":
                B5, C, H, W = self.image_2.shape
                self.image_1_neg = self.image_1_neg.view(B5//self.args.ue_num_crops, self.args.ue_num_crops, C, H, W)[:, 0]
                self.image_2 = self.image_2.view(B5//self.args.ue_num_crops, self.args.ue_num_crops, C, H, W)[:, 0]

    def get_cropped_st_images(self, image_1_ori, four_pred, fine_padding, detach=True, augment_two_stages=0):
        '''
        Inputs:
        - image_1_ori: original image_1
        - four_pred: (D rs→rt) IHN1 displacement of resized img2 from resized img1
        - fine_padding: padding for bbox crop

        Return:
        - image_1_crop: padded-squared-resized bbox crop of image_1_ori
        - delta: ratio of bbox to resize_width
        - flow_bbox: (D s→b) Displacement of bbox corners from original corners
        '''
        # D rs→rt to X rt
        if self.args.custom == 'satcrop':
            four_point = four_pred + self.four_point_org_xrs_single
        else:
            four_point = four_pred + self.four_point_org_single

        x = four_point[:, 0]
        y = four_point[:, 1]
        # Make it same scale as image_1_ori
        # X rt to X t
        alpha = self.args.database_size / self.args.resize_width
        x[:, :, 0] = x[:, :, 0] * alpha
        x[:, :, 1] = (x[:, :, 1] + 1) * alpha
        y[:, 0, :] = y[:, 0, :] * alpha
        y[:, 1, :] = (y[:, 1, :] + 1) * alpha
        # Crop
        left = torch.min(x.view(x.shape[0], -1), dim=1)[0]  # B
        right = torch.max(x.view(x.shape[0], -1), dim=1)[0] # B
        top = torch.min(y.view(y.shape[0], -1), dim=1)[0]   # B
        bottom = torch.max(y.view(y.shape[0], -1), dim=1)[0] # B
        if augment_two_stages!=0:
            if self.args.augment_type == "bbox":
                left += (torch.rand(left.shape).to(left.device) * 2 - 1) * augment_two_stages
                right += (torch.rand(right.shape).to(right.device) * 2 - 1) * augment_two_stages
                top += (torch.rand(top.shape).to(top.device) * 2 - 1) * augment_two_stages
                bottom += (torch.rand(bottom.shape).to(bottom.device) * 2 - 1) * augment_two_stages
            w = torch.max(torch.stack([right-left, bottom-top], dim=1), dim=1)[0] # B
            c = torch.stack([(left + right)/2, (bottom + top)/2], dim=1) # B, 2
            if self.args.augment_type == "center":
                w += torch.rand(w.shape).to(w.device) * augment_two_stages # only expand?
                c += (torch.rand(c.shape).to(c.device) * 2 - 1) * augment_two_stages
        else:
            w = torch.max(torch.stack([right-left, bottom-top], dim=1), dim=1)[0] # B
            c = torch.stack([(left + right)/2, (bottom + top)/2], dim=1) # B, 2
        w_padded = w + 2 * fine_padding # same as ori scale
        crop_top_left = c + torch.stack([-w_padded / 2, -w_padded / 2], dim=1) # B, 2 = x, y
        x_start = crop_top_left[:, 0] # B
        y_start = crop_top_left[:, 1] # B
        bbox_s = bbox.bbox_generator(x_start, y_start, w_padded, w_padded)
        delta = (w_padded / self.args.resize_width).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        image_1_crop = tgm.crop_and_resize(image_1_ori, bbox_s, (self.args.resize_width, self.args.resize_width)) # It will be padded when it is out of boundary
        # swap bbox_s
        bbox_s_swap = torch.stack([bbox_s[:, 0], bbox_s[:, 1], bbox_s[:, 3], bbox_s[:, 2]], dim=1)
        # X b
        four_cor_bbox = bbox_s_swap.permute(0, 2, 1). view(-1, 2, 2, 2)

        # X b to D s→b
        if self.args.custom == 'satcrop':
            flow_bbox = four_cor_bbox - self.four_point_org_xs_single
        else:
            flow_bbox = four_cor_bbox - self.four_point_org_large_single

        if detach:
            image_1_crop = image_1_crop.detach()
            delta = delta.detach()
            flow_bbox = flow_bbox.detach()
        return image_1_crop, delta, flow_bbox

    def combine_coarse_fine(self, four_preds_list, four_pred, four_preds_list_fine, four_pred_fine, delta, flow_bbox, for_training):
        '''
        Inputs:
        - four_pred: (D rs→rt) IHN1 displacement of resized img2 from resized img1
        - four_pred_fine: (D rb→rt) IHN2 displacement of resized img2 from resized bbox crop of img1
        - delta: ratio of bbox to resize_width
        - flow_bbox: (D s→b) Displacement of bbox corners from original corners

        Return:
        - four_pred_fine: (D rs→rt) Refined displacement of resized img2 from resized img1
        '''
        alpha = self.args.database_size / self.args.resize_width
        kappa = delta / alpha
        four_preds_list_fine = [four_preds_list_fine_single * kappa + flow_bbox / alpha for four_preds_list_fine_single in four_preds_list_fine]
        # D rs→rt = D rb→rt * kappa + D s→b / alpha
        # D rs→rt = D rb→rt + D rs→rb
        four_pred_fine = four_pred_fine * kappa + flow_bbox / alpha
        four_preds_list = four_preds_list + four_preds_list_fine
        return four_preds_list, four_pred_fine

    def first_stage_ue_generate(self, neg_forward=False):
        B, C, H, W = self.image_2.shape

        self.image_1 = self.image_1.unsqueeze(1).repeat(1, self.args.ue_num_crops, 1, 1, 1).view(B*self.args.ue_num_crops, C, H, W)
        self.image_2 = self.image_2.unsqueeze(1).repeat(1, self.args.ue_num_crops, 1, 1, 1).view(B*self.args.ue_num_crops, C, H, W)
        if self.args.ue_aug_method == "shift":
            bbox_s = self.first_stage_ue_generate_bbox()

            if self.args.custom == "satcrop":
                B, C, H, W = self.image_1_ori.shape
                image_1_ori = self.image_1_ori.unsqueeze(1).repeat(1, self.args.ue_num_crops, 1, 1, 1).view(B*self.args.ue_num_crops, C, H, W)
                self.image_1 = tgm.crop_and_resize(image_1_ori, bbox_s, (self.args.resize_width, self.args.resize_width))
                # print('*** self.image_1', self.image_1.shape)

            else:
                self.image_2 = tgm.crop_and_resize(self.image_2, bbox_s, (self.args.resize_width, self.args.resize_width))
        elif self.args.ue_aug_method == "mask":
            self.image_2 = self.image_2.view(B, self.args.ue_num_crops, C, H, W)
            mask = torch.rand((self.image_2.shape[0], int(self.args.ue_num_crops - 1), 1, self.image_2.shape[3]//self.args.ue_mask_patchsize, self.image_2.shape[4]//self.args.ue_mask_patchsize)).to(self.image_2.device) > self.args.ue_mask_prob
            mask = torch.repeat_interleave(torch.repeat_interleave(mask, self.args.ue_mask_patchsize, dim=3), self.args.ue_mask_patchsize, dim=4)
            self.image_2[:, 1:] = self.image_2[:, 1:] * mask
            self.image_2 = self.image_2.view(B*self.args.ue_num_crops, C, H, W)

    def first_stage_ue_aggregation(self, four_preds_list, for_training, neg_forward=False):
        alpha = self.args.database_size / self.args.resize_width

        four_preds_list, four_pred, self.std_four_preds_list, self.std_four_pred_five_crops = self.ue_aggregation(four_preds_list, alpha, for_training, self.args.check_step)
        # print("Positve UE std: " + str((self.std_four_pred_five_crops[0]))) # Comment for performance evaluation
        return four_preds_list, four_pred

    def first_stage_ue_generate_bbox(self):
        beta = self.args.crop_width / self.args.resize_width
        resized_ue_shift = self.args.ue_shift / beta
        x_start = torch.zeros((self.image_2.shape[0])).to(self.image_2.device)
        y_start = torch.zeros((self.image_2.shape[0])).to(self.image_2.device)
        NUM_CROPS = self.args.ue_num_crops

        if self.args.custom == "satcrop":
            # one center crop and the rest random crops from the large image, with the same size as the center crop
            self.args.ue_shift = self.args.database_size_large - self.args.database_size
            resized_ue_shift = self.args.ue_shift
            # center crop
            half_shift = resized_ue_shift // 2
            x_center_shift = [half_shift]
            y_center_shift = [half_shift]
            MIN_SHIFT = min(half_shift - 1, 30)
            CROP_WIDTH = self.args.database_size
            FRAME_WIDTH = self.args.database_size_large
            FRAME_SUB_CROP = FRAME_WIDTH - CROP_WIDTH


            if self.args.ue_shift_crops_types in ['plus', 'cross', 'plus_cross']:
                x_shift, y_shift = [], []
                cur_crop_type = self.args.ue_shift_crops_types if self.args.ue_shift_crops_types != 'plus_cross' else 'plus'


                for i in range(self.args.ue_num_crops - 1):
                    # TODO for loop slow in inference, replace it
                    # TODO add const or rand choice
                    x_val = self.ue_rng.integers(MIN_SHIFT, half_shift)
                    y_val = self.ue_rng.integers(MIN_SHIFT, half_shift)

                    # print('$$$ forrrr', self.args.ue_shift_crops_types, 'but', cur_crop_type)

                    if cur_crop_type == "plus":
                        # 4 random plus shape crops (up, left, down, right)
                        x_shift.append(half_shift if i % 2 == 0 else half_shift + (-1 if (i // 2) % 2 == 0 else 1) * x_val)
                        y_shift.append(half_shift if i % 2 == 1 else half_shift + (-1 if (i // 2) % 2 == 0 else 1) * y_val)

                        if i % 8 == 3 and self.args.ue_shift_crops_types == 'plus_cross':
                            # print('!!! yesssss')
                            cur_crop_type = 'cross'

                    elif cur_crop_type == "cross":
                        # 4 random cross shape crops (top-left, top-right, bottom-left, bottom-right)
                        x_shift.append(half_shift + (-1 if i % 2 == 0 else 1) * x_val)
                        y_shift.append(half_shift + (-1 if i % 4 > 1 else 1) * y_val)
                        if i % 8 == 7 and self.args.ue_shift_crops_types == 'plus_cross': 
                            # print('@@@ yesssss')
                            cur_crop_type = 'plus'


            elif self.args.ue_shift_crops_types == "random":
                x_shift = [int(self.ue_rng.integers(0, resized_ue_shift)) for i in range(self.args.ue_num_crops - 1)]
                y_shift = [int(self.ue_rng.integers(0, resized_ue_shift)) for i in range(self.args.ue_num_crops - 1)]

            elif self.args.ue_shift_crops_types == 'grid':
                # find root
                grid_x_cells = math.isqrt(NUM_CROPS)
                is_grid_odd = False
                if grid_x_cells * grid_x_cells == NUM_CROPS and grid_x_cells % 2 == 1:  # k^2
                    is_grid_odd = True
                elif grid_x_cells * grid_x_cells != NUM_CROPS - 1 or grid_x_cells % 2 == 1:  # k^2 + 1
                    raise ValueError(f"Invalid NUM_CROPS={NUM_CROPS}: expected k^2 for odd k ,or k^2 + 1 for even k")

                x_shift = np.linspace((FRAME_SUB_CROP - resized_ue_shift) // 2, (FRAME_SUB_CROP + resized_ue_shift) // 2, grid_x_cells)
                y_shift = np.linspace((FRAME_SUB_CROP - resized_ue_shift) // 2, (FRAME_SUB_CROP + resized_ue_shift) // 2, grid_x_cells)
                x_shift, y_shift = np.meshgrid(x_shift, y_shift)
                x_shift = list(x_shift.reshape(-1))
                y_shift = list(y_shift.reshape(-1))
                if is_grid_odd:  # remove centeral cell crop to avoid redundancy with main crop
                    del x_shift[len(x_shift) // 2]
                    del y_shift[len(y_shift) // 2]
                

            w_random = [CROP_WIDTH for i in range(self.args.ue_num_crops)]
            x_shift = torch.tensor(x_center_shift + x_shift).repeat(self.image_2.shape[0]//self.args.ue_num_crops).to(self.image_2.device)
            y_shift = torch.tensor(y_center_shift + y_shift).repeat(self.image_2.shape[0]//self.args.ue_num_crops).to(self.image_2.device)
            w = torch.tensor(w_random, dtype=torch.float).repeat(self.image_2.shape[0]//self.args.ue_num_crops).to(self.image_2.device)

        else:
            if self.args.ue_shift_crops_types == "grid":
                resized_ue_shift_sample = resized_ue_shift
                if self.args.ue_num_crops >= 2 and self.args.ue_num_crops <= 5:
                    x_shift_grid = np.linspace(0, resized_ue_shift_sample, 2) # 1 -> 1 2-4 -> 4 5-9 -> 9
                    y_shift_grid = np.linspace(0, resized_ue_shift_sample, 2)
                else:
                    raise NotImplementedError()
                x_shift_grid, y_shift_grid = np.meshgrid(x_shift_grid, y_shift_grid)
                x_shift_grid = x_shift_grid.reshape(-1)
                y_shift_grid = y_shift_grid.reshape(-1)
                idx = list(range(len(x_shift_grid)))
                self.ue_rng.shuffle(idx)
                idx = idx[:self.args.ue_num_crops-1]
                x_shift_grid_list = list(x_shift_grid[idx])
                y_shift_grid_list = list(y_shift_grid[idx])
                w_grid = [(self.args.resize_width - resized_ue_shift_sample) for i in range(len(x_shift_grid_list))]
                x_shift = torch.tensor([0] + x_shift_grid_list).repeat(self.image_2.shape[0]//self.args.ue_num_crops).to(self.image_2.device) # on 256x256
                y_shift = torch.tensor([0] + y_shift_grid_list).repeat(self.image_2.shape[0]//self.args.ue_num_crops).to(self.image_2.device)
                w = torch.tensor([self.args.resize_width] + w_grid, dtype=torch.float).repeat(self.image_2.shape[0]//self.args.ue_num_crops).to(self.image_2.device)

            elif self.args.ue_shift_crops_types == "random":
                x_shift_random = [int(self.ue_rng.integers(0, resized_ue_shift)) for i in range(self.args.ue_num_crops - 1)]
                y_shift_random = [int(self.ue_rng.integers(0, resized_ue_shift)) for i in range(self.args.ue_num_crops - 1)]
                w_random = [self.args.resize_width - resized_ue_shift for i in range(self.args.ue_num_crops - 1)]
                x_shift = torch.tensor([0] + x_shift_random).repeat(self.image_2.shape[0]//self.args.ue_num_crops).to(self.image_2.device) # on 256x256
                y_shift = torch.tensor([0] + y_shift_random).repeat(self.image_2.shape[0]//self.args.ue_num_crops).to(self.image_2.device)
                w = torch.tensor([self.args.resize_width] + w_random).repeat(self.image_2.shape[0]//self.args.ue_num_crops).to(self.image_2.device)

            elif self.args.ue_shift_crops_types in ['plus', 'cross', 'plus_cross']:
                x_shift, y_shift = [], []
                cur_crop_type = self.args.ue_shift_crops_types

                for i in range(self.args.ue_num_crops - 1):
                    # TODO add const or rand choice
                    x_val = self.ue_rng.integers(MIN_SHIFT, half_shift)
                    y_val = self.ue_rng.integers(MIN_SHIFT, half_shift)

                    if cur_crop_type == "plus":
                        # 4 random plus shape crops (up, left, down, right)
                        x_shift.append(half_shift if i % 2 == 0 else half_shift + (-1 if (i // 2) % 2 == 0 else 1) * x_val)
                        y_shift.append(half_shift if i % 2 == 1 else half_shift + (-1 if (i // 2) % 2 == 0 else 1) * y_val)
                        if self.args.ue_shift_crops_types == 'plus_cross' and i % 8 == 3: 
                            cur_crop_type = 'cross'

                    elif cur_crop_type == "cross":
                        # 4 random cross shape crops (top-left, top-right, bottom-left, bottom-right)
                        x_shift.append(half_shift + (-1 if i % 2 == 0 else 1) * x_val)
                        y_shift.append(half_shift + (-1 if i % 4 > 1 else 1) * y_val)
                        if self.args.ue_shift_crops_types == 'plus_cross' and i % 8 == 7: 
                            cur_crop_type = 'plus'

                w_random = [self.args.resize_width - self.args.ue_shift for i in range(self.args.ue_num_crops - 1)]
                x_shift = torch.tensor([0] + x_shift).repeat(self.image_2.shape[0]//self.args.ue_num_crops).to(self.image_2.device)
                y_shift = torch.tensor([0] + y_shift).repeat(self.image_2.shape[0]//self.args.ue_num_crops).to(self.image_2.device)
                w = torch.tensor([self.args.resize_width] + w_random, dtype=torch.float).repeat(self.image_2.shape[0]//self.args.ue_num_crops).to(self.image_2.device)

            elif self.args.ue_shift_crops_types == "random_relax":
                resized_ue_shift_list = [int(self.ue_rng.integers(1, 2*resized_ue_shift)) for i in range(self.args.ue_num_crops - 1)]
                x_shift_random = [int(self.ue_rng.integers(0, resized_ue_shift_list[i])) for i in range(self.args.ue_num_crops - 1)]
                y_shift_random = [int(self.ue_rng.integers(0, resized_ue_shift_list[i])) for i in range(self.args.ue_num_crops - 1)]
                w_random = [self.args.resize_width - resized_ue_shift_list[i] for i in range(self.args.ue_num_crops - 1)]
                x_shift = torch.tensor([0] + x_shift_random).repeat(self.image_2.shape[0]//self.args.ue_num_crops).to(self.image_2.device) # on 256x256
                y_shift = torch.tensor([0] + y_shift_random).repeat(self.image_2.shape[0]//self.args.ue_num_crops).to(self.image_2.device)
                w = torch.tensor([self.args.resize_width] + w_random, dtype=torch.float).repeat(self.image_2.shape[0]//self.args.ue_num_crops).to(self.image_2.device)
            else:
                raise NotImplementedError()

        x_start += x_shift
        y_start += y_shift
        bbox_s = bbox.bbox_generator(x_start, y_start, w, w)
        bbox_s_swap = torch.stack([bbox_s[:, 0], bbox_s[:, 1], bbox_s[:, 3], bbox_s[:, 2]], dim=1)
        if self.args.custom == "satcrop":
            alpha = self.args.database_size / self.args.resize_width
            self.xrcs_before: torch.Tensor = bbox_s_swap / alpha  # 291 scale
            # print("*** xrcs_before\n", self.xrcs_before)
            self.xrcs_before = self.xrcs_before.permute(0, 2, 1).view(bbox_s_swap.shape[0], 2, 2, 2).contiguous() # B x 2 x 2 x 2
            # print("*** xrcs_before\n", self.xrcs_before)
        else:
            self.xct_before: torch.Tensor = bbox_s_swap
            self.H_CTtoT = tgm.get_perspective_transform(bbox_s_swap, self.four_point_org_single.repeat(bbox_s_swap.shape[0],1,1,1).view(bbox_s_swap.shape[0], 2, 4).permute(0, 2, 1).contiguous())

        return bbox_s

    def ue_aggregation(self, four_preds_list, alpha, for_training, check_step=-1):

        if check_step == -1:
            agg_step = len(four_preds_list)
        else:
            agg_step = check_step + 1

        if self.args.custom == "satcrop":
            four_preds_recovered_list = []
            for i in range(agg_step):
                four_corners = four_preds_list[i] + self.xrcs_before # B x 2 x 2 x 2
                four_point_org_xrs_single_repeat = self.four_point_org_xrs_single.repeat(four_preds_list[i].shape[0],1,1,1)
                four_preds_recovered_single = four_corners - four_point_org_xrs_single_repeat
                four_preds_recovered_list.append(four_preds_recovered_single)
                # print('*** four_preds_list[i]\n', four_preds_list[i])
                # print('*** xrcs_before\n', self.xrcs_before)
                # print('*** four_corners\n', four_corners)
                # print('*** four_point_org_xrs_single_repeat\n', four_point_org_xrs_single_repeat)
                # print('*** four_preds_recovered_single\n', four_preds_recovered_single)

        else:
            if self.ue_method == "augment":
                if self.args.ue_aug_method == "shift":
                    # Recover shift
                    four_preds_recovered_list = []
                    for i in range(agg_step):
                        # Formula 4: X_rct
                        four_point_org_single_repeat = self.four_point_org_single.repeat(four_preds_list[i].shape[0],1,1,1)
                        four_corners = four_preds_list[i] + four_point_org_single_repeat # B x 2 x 2 x 2
                        # Formula 1
                        H_StoT = tgm.get_perspective_transform(self.xct_before, four_corners.view(-1, 2, 4).permute(0, 2, 1).contiguous())
                        # Formula 2: self.H_CTtoT
                        # Formula 3: X_rt
                        H_StoT_inv = torch.linalg.inv(H_StoT)
                        four_corners_aug = torch.cat([four_corners.view(four_corners.shape[0], 2, 4),
                                                    torch.ones((four_corners.shape[0], 1, 4)).to(four_corners.device)], dim=1) # B x 3 x 4
                        x_rct_bef_assumed = torch.bmm(self.H_CTtoT, torch.bmm(H_StoT_inv, four_corners_aug))
                        x_rct_bef_assumed = x_rct_bef_assumed[:,:2,:] / x_rct_bef_assumed[:,2:,:] # B x 2 x 4
                        four_corners = torch.bmm(H_StoT, torch.bmm(self.H_CTtoT, torch.bmm(H_StoT_inv, four_corners_aug))) # B x 3 x 4
                        four_corners = four_corners[:,:2,:] / four_corners[:,2:,:] # B x 2 x 4
                        # Formula 5: D_rs→rt
                        four_preds_recovered_single = four_corners.view(four_corners.shape[0], 2, 2, 2) - four_point_org_single_repeat
                        four_preds_recovered_list.append(four_preds_recovered_single)

        for i in range(agg_step, len(four_preds_list)):
            four_preds_recovered_list.append(four_preds_list[i])

        four_preds_list = four_preds_recovered_list

        four_pred = four_preds_list[check_step]  # Last Iteration Ds

        four_pred_five_crops = None

        if self.ue_method == "ensemble":
            four_pred_five_crops = four_pred.view(four_pred.shape[0]//len(self.netG_list), len(self.netG_list), 2, 2, 2)
        elif self.ue_method == "augment":
            four_pred_five_crops = four_pred.view(four_pred.shape[0]//self.args.ue_num_crops, self.args.ue_num_crops, 2, 2, 2)

        assert four_pred_five_crops is not None

        # UE
        if self.args.ue_outlier_method != "none" and self.args.ue_outlier_num != 0 and not for_training:
            mace_distance = (four_pred_five_crops[:, :1] - four_pred_five_crops)**2  # (D_t - D_ct)^2
            mace_distance = (mace_distance[:, :, 0] + mace_distance[:, :, 1])**0.5  # rad(dx^2 + dy^2)
            mace_distance = mace_distance.mean(dim=2).mean(dim=2)  # mean
            mask = torch.ones((four_pred_five_crops.shape[0], four_pred_five_crops.shape[1])).to(four_pred_five_crops.device)

            if self.args.ue_outlier_method == "max":
                _, max_indices = torch.topk(mace_distance, self.args.ue_outlier_num, dim=1)
                for i in range(self.args.ue_outlier_num):
                    max_indice = max_indices[:, i]
                    mask = mask.scatter_(1,max_indice.unsqueeze(1), 0.)
            elif self.args.ue_outlier_method == "dis":
                mask[mace_distance > self.args.ue_outlier_dis] = False
                for i in range(len(mask)):
                    if torch.count_nonzero(mask[i]) <= self.args.ue_outlier_num:
                        _, min_indices = torch.topk(mace_distance[i], self.args.ue_num_crops - self.args.ue_outlier_num, largest=False)
                        mask[i] = False
                        mask[i] = mask[i].scatter_(0, min_indices, 1.)

            four_pred_five_crops_res = four_pred_five_crops[mask.bool()].view(four_pred_five_crops.shape[0], four_pred_five_crops.shape[1] - self.args.ue_outlier_num, 2, 2, 2)
            std_four_pred_five_crops = torch.std(four_pred_five_crops_res, dim=1)
            print(mace_distance)
        else:
            std_four_pred_five_crops = torch.std(four_pred_five_crops, dim=1)
            std = std_four_pred_five_crops.view(std_four_pred_five_crops.shape[0], -1).mean(dim=1)

        # Aggregate Final Displacement
        if check_step == -1:
            mean_four_pred_five_crops = torch.mean(four_pred_five_crops, dim=1)
            four_pred_agg_list = []
            for i in range(len(four_pred_five_crops)):
                if self.args.ue_agg == "mean":
                    four_pred_agg = mean_four_pred_five_crops[i]
                elif self.args.ue_agg == "zero":
                    four_pred_agg = four_pred_five_crops[i, 0]
                four_pred_agg_list.append(four_pred_agg)
            four_pred_new = torch.stack(four_pred_agg_list)
        else:
            four_pred_new = four_preds_list[-1]

        four_preds_std_list_new = []
        for i in range(len(four_preds_list)):
            if i < agg_step:
                if self.ue_method == "ensemble":
                    four_pred_single = four_preds_list[i].view(four_preds_list[i].shape[0]//len(self.netG_list), len(self.netG_list), 2, 2, 2)
                elif self.ue_method == "augment":
                    four_pred_single = four_preds_list[i].view(four_preds_list[i].shape[0]//self.args.ue_num_crops, self.args.ue_num_crops, 2, 2, 2)
                else:
                    # FIX: Default case
                    four_pred_single = four_preds_list[i].view(four_preds_list[i].shape[0], 1, 2, 2, 2)
            else:
                four_pred_single = four_preds_list[i].view(four_preds_list[i].shape[0], 1, 2, 2, 2)

            std_four_pred_single = torch.std(four_pred_single, dim=1)
            four_preds_std_list_new.append(std_four_pred_single)
        return four_preds_list, four_pred_new, four_preds_std_list_new, std_four_pred_five_crops

    def stack_ensemble_results(self, four_preds_list_ensemble, early_stop):
        four_preds_list = []
        if early_stop == -1:
            agg_step = len(four_preds_list_ensemble[0])
        else:
            agg_step = early_stop + 1
        for i in range(agg_step):
            four_preds_list_single = []
            for j in range(len(four_preds_list_ensemble)):
                four_preds_list_single.append(four_preds_list_ensemble[j][i]) # batch size
            four_preds_list_single = torch.stack(four_preds_list_single, dim=1).view(-1, 2, 2, 2)
            four_preds_list.append(four_preds_list_single)
        for i in range(agg_step, len(four_preds_list_ensemble[0])):
            four_preds_list_single = four_preds_list_ensemble[0][i]
            four_preds_list.append(four_preds_list_single)
        four_pred = four_preds_list[-1]
        return four_preds_list, four_pred

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # Second, G(A) = B
        if self.ue_method == "single":
            self.loss_G_Homo, self.metrics = self.criterionAUX(self.four_preds_list, self.four_pred, self.flow_gt, self.args.gamma, self.args, self.metrics, four_ue_list=self.four_pred_ue_list)
        else:
            self.loss_G_Homo, self.metrics = self.criterionAUX(self.four_preds_list, self.four_pred, self.flow_gt, self.args.gamma, self.args, self.metrics)
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_Homo
        self.metrics["G_loss"] = self.loss_G.cpu().item()
        self.loss_G.backward()

    def backward_D(self):
        """Calculate GAN and L1 loss for the generator"""
        # Second, G(A) = B
        if self.ue_method == "single":
            self.loss_D, self.metrics = self.criterionNEG(self.args.gamma, self.args, self.metrics, self.four_pred_ue_neg_list)
        else:
            self.loss_D, self.metrics = self.criterionNEG(self.args.gamma, self.args, self.metrics, self.std_four_preds_neg_list)
        # combine loss and calculate gradients
        self.metrics["D_loss"] = self.loss_D.cpu().item()
        self.loss_D.backward()

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def optimize_parameters(self):
        self.forward(for_training=True) # Calculate Fake A
        if self.args.neg_training:
            self.forward_neg(for_training=True)
        self.metrics = dict()
        # update G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        if self.args.neg_training:
            self.backward_D()
        if self.args.restore_ckpt is None or self.args.finetune:
            nn.utils.clip_grad_norm_(self.netG.parameters(), self.args.clip)
        if self.args.two_stages:
            nn.utils.clip_grad_norm_(self.netG_fine.parameters(), self.args.clip)
        self.optimizer_G.step()             # update G's weights
        return self.metrics

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        self.scheduler_G.step()

def mywarp(x, flow_pred, four_point_org_single, ue_std=None):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    if not torch.isnan(flow_pred).any():
        if flow_pred.shape[-1] != 2:
            flow_4cor = torch.zeros((flow_pred.shape[0], 2, 2, 2)).to(flow_pred.device)
            flow_4cor[:, :, 0, 0] = flow_pred[:, :, 0, 0]
            flow_4cor[:, :, 0, 1] = flow_pred[:, :, 0, -1]
            flow_4cor[:, :, 1, 0] = flow_pred[:, :, -1, 0]
            flow_4cor[:, :, 1, 1] = flow_pred[:, :, -1, -1]
        else:
            flow_4cor = flow_pred

        four_point_1 = flow_4cor + four_point_org_single

        four_point_org = four_point_org_single.repeat(flow_pred.shape[0],1,1,1).flatten(2).permute(0, 2, 1).contiguous()
        four_point_1 = four_point_1.flatten(2).permute(0, 2, 1).contiguous()
        try:
            H = tgm.get_perspective_transform(four_point_org, four_point_1)
        except Exception:
            logging.debug("No solution")
            H = torch.eye(3).to(four_point_org.device).repeat(four_point_1.shape[0],1,1)
        warped_image = tgm.warp_perspective(x, H, (x.shape[2], x.shape[3]))
    else:
        logging.debug("Output NaN by model error.")
        warped_image = x
    return warped_image
