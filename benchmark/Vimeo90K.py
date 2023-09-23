import os
import sys
sys.path.append('.')
import cv2
import math
import torch
import numpy as np
from torch.nn import functional as F
from benchmark.pytorch_msssim import ssim_matlab as SSIM
from models.WaveletVFI import WaveletVFI
from thop import profile

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def convert(param):
    return {k.replace("module.", ""): v for k, v in param.items() if "module." in k}

def PSNR(img_pred, img_gt):
    psnr = -10 * torch.log10(((img_pred - img_gt) * (img_pred - img_gt)).mean())
    return psnr

model = WaveletVFI()
model.load_state_dict(convert(torch.load('./models/waveletvfi_latest.pth', map_location='cpu')))
model.eval()
model.to(device)

th = None

path = '/youtu_action_data/NTIRE/vimeo_triplet/'
f = open(path + 'tri_testlist.txt', 'r')
psnr_list = []
ssim_list = []
flops_list = []
for i in f:
    name = str(i).strip()
    if(len(name) <= 1):
        continue
    print(path + 'sequences/' + name + '/im1.png')
    I0 = cv2.imread(path + 'sequences/' + name + '/im1.png')
    I1 = cv2.imread(path + 'sequences/' + name + '/im2.png')
    I2 = cv2.imread(path + 'sequences/' + name + '/im3.png')
    I0 = (torch.tensor(I0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
    I1 = (torch.tensor(I1.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
    I2 = (torch.tensor(I2.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)

    macs, params, outputs = profile(model, inputs=(I0, I2, I1, False, th), verbose=False, output=True)
    I1_pred = outputs[0]

    psnr = PSNR(I1_pred, I1).detach().cpu().numpy()
    ssim = SSIM(I1_pred, I1).detach().cpu().numpy()

    psnr_list.append(psnr)
    ssim_list.append(ssim)
    flops_list.append(macs / 1e9)
    print("Avg PSNR: {:.3f} SSIM: {:.4f} FLOPs(G): {:.3f}".format(np.mean(psnr_list), np.mean(ssim_list), np.mean(flops_list)))
