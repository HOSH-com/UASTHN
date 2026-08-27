import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import *
import time

try:
    import alt_cuda_corr
except:
    pass # alt_cuda_corr is not compiled


class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        corr = CorrBlock.corr(fmap1, fmap2)
        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)  # b*crops*64*64, 1, 64, 64

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)  # b*crops*64*64, 1, 64/2^i, 64/2^i
            self.corr_pyramid.append(corr)

        r = radius
        dx = torch.linspace(-r, r, 2 * r + 1)
        dy = torch.linspace(-r, r, 2 * r + 1)
        self.delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(fmap1.device)

    def __call__(self, coords):
        r = self.radius  # 4
        coords = coords.permute(0, 2, 3, 1)  # b*crops, 64, 64, 2
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):  # 4
            # print('@@@@@@@@', i, '@@@@@@@@')
            corr = self.corr_pyramid[i]  # b*crops*64*64, 1, 64/2^i, 64/2^i
            delta = self.delta  # 9, 9, 2

            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2 ** i  # b*crops*64*64, 1, 1, 2
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)  # 1, 9, 9, 2
            coords_lvl = centroid_lvl + delta_lvl  # b*crops*64*64, 9, 9, 2

            # print('@@@ delta', delta.shape, delta)
            # print('@@@ delta_lvl', delta_lvl.shape, delta_lvl)
            # print('@@@ coords', coords.shape, coords)
            # print('@@@ centroid_lvl', centroid_lvl.shape, centroid_lvl)
            # print('@@@ coords_lvl', coords_lvl.shape, coords_lvl)

            # print('@@@ corr bef', corr.shape, corr)
            corr = bilinear_sampler(corr, coords_lvl)  # b*crops*64*64, 1, 9, 9
            # print('@@@ corr aft', corr.shape, corr)
            corr = corr.view(batch, h1, w1, -1)  # b*crops, 64, 64, 9*9
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)

        corr = torch.relu(torch.matmul(fmap1.transpose(1, 2), fmap2))
        corr = corr.view(batch, ht, wd, 1, ht, wd)

        return corr

class CorrBlockSingleScale(nn.Module):
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        super().__init__()
        self.radius = radius

        corr = CorrBlock.corr(fmap1, fmap2)
        batch, h1, w1, dim, h2, w2 = corr.shape
        self.corr = corr.reshape(batch * h1 * w1, dim, h2, w2)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        corr = self.corr
        dx = torch.linspace(-r, r, 2 * r + 1)
        dy = torch.linspace(-r, r, 2 * r + 1)
        delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)

        centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2)
        delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
        coords_lvl = centroid_lvl + delta_lvl

        corr = bilinear_sampler(corr, coords_lvl)
        out = corr.view(batch, h1, w1, -1)
        out = out.permute(0, 3, 1, 2).contiguous().float()
        return out

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())
