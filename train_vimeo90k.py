import os
import math
import time
import random
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from models.WaveletVFI import WaveletVFI
from models.utils import AverageMeter
from datasets import Vimeo90K_Train_Dataset, Vimeo90K_Test_Dataset
import logging


def get_lr(args, iters):
    ratio = 0.5 * (1.0 + np.cos(iters / (args.epochs * args.iters_per_epoch) * math.pi))
    lr = (args.lr_start - args.lr_end) * ratio + args.lr_end
    return lr


def get_tau(args, epoch):
    tau = max(1.0 - epoch / (args.epochs / 2.0), 0.4)
    return tau


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def set_tau(model, tau):
    model.tau = tau


def train(args, ddp_model):
    local_rank = args.local_rank
    print('Distributed Data Parallel Training WaveletVFI on Rank {}'.format(local_rank))

    if local_rank == 0:
        os.makedirs(args.log_path, exist_ok=True)
        log_path = os.path.join(args.log_path, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
        os.makedirs(log_path, exist_ok=True)
        logger = logging.getLogger()
        logger.setLevel('INFO')
        BASIC_FORMAT = '%(asctime)s:%(levelname)s:%(message)s'
        DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
        formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
        chlr = logging.StreamHandler()
        chlr.setFormatter(formatter)
        chlr.setLevel('INFO')
        fhlr = logging.FileHandler(os.path.join(log_path, 'train.log'))
        fhlr.setFormatter(formatter)
        logger.addHandler(chlr)
        logger.addHandler(fhlr)
        logger.info(args)

    dataset_train = Vimeo90K_Train_Dataset('/home/ltkong/Datasets/Vimeo90K/vimeo_triplet', True)
    sampler = DistributedSampler(dataset_train)
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=True, sampler=sampler)
    args.iters_per_epoch = dataloader_train.__len__()
    iters = args.resume_epoch * args.iters_per_epoch

    dataset_val = Vimeo90K_Test_Dataset('/home/ltkong/Datasets/Vimeo90K/vimeo_triplet')
    dataloader_val = DataLoader(dataset_val, batch_size=16, num_workers=16, pin_memory=True, shuffle=False, drop_last=True)

    optimizer = optim.AdamW(ddp_model.parameters(), lr=args.lr_start, weight_decay=0)

    time_stamp = time.time()
    avg_rec = AverageMeter()
    avg_wav = AverageMeter()
    avg_com = AverageMeter()
    best_psnr = 0.0

    for epoch in range(args.resume_epoch, args.epochs):
        sampler.set_epoch(epoch)
        for i, data in enumerate(dataloader_train):
            img0, imgt, img1 = data
            img0, imgt, img1 = img0.to(args.device), imgt.to(args.device), img1.to(args.device)

            data_time_interval = time.time() - time_stamp
            time_stamp = time.time()

            lr = get_lr(args, iters)
            set_lr(optimizer, lr)

            tau = get_tau(args, epoch)
            set_tau(ddp_model, tau)

            optimizer.zero_grad()

            if args.dynamic:
                th = None
            else:
                th = 0.0

            imgt_pred, imgt_merge, flow_t0_pred, flow_t1_pred, occ_t_pred, mask_t_pred, loss_rec, loss_wav, loss_com, thresh = ddp_model(img0, img1, imgt, args.dynamic, th)

            loss = loss_rec + loss_wav + loss_com

            loss.backward()
            optimizer.step()

            avg_rec.update(loss_rec.cpu().data)
            avg_wav.update(loss_wav.cpu().data)
            avg_com.update(loss_com.cpu().data)
            train_time_interval = time.time() - time_stamp

            if (iters+1) % 100 == 0 and local_rank == 0:
                logger.info('epoch:{}/{} iter:{}/{} time:{:.2f}+{:.2f} lr:{:.5e} loss_rec:{:.4e} loss_wav:{:.4e} loss_com:{:.4e}'.format(epoch+1, args.epochs, iters+1, args.epochs * args.iters_per_epoch, data_time_interval, train_time_interval, lr, avg_rec.avg, avg_wav.avg, avg_com.avg))
                avg_rec.reset()
                avg_wav.reset()
                avg_com.reset()

            iters += 1
            time_stamp = time.time()

        if (epoch+1) % args.eval_interval == 0 and local_rank == 0:
            psnr = evaluate(args, ddp_model, dataloader_val, epoch, logger)
            if psnr > best_psnr:
                best_psnr = psnr
                torch.save(ddp_model.module.state_dict(), '{}/waveletvfi_{}.pth'.format(log_path, 'best'))
            torch.save(ddp_model.module.state_dict(), '{}/waveletvfi_{}.pth'.format(log_path, 'latest'))


def evaluate(args, ddp_model, dataloader_val, epoch, logger):
    loss_rec_list = []
    loss_wav_list = []
    loss_com_list = []
    psnr_list = []
    time_stamp = time.time()
    for i, data in enumerate(dataloader_val):
        img0, imgt, img1 = data
        img0, imgt, img1 = img0.to(args.device), imgt.to(args.device), img1.to(args.device)

        if args.dynamic:
            th = None
        else:
            th = 0.0
            
        with torch.no_grad():
            imgt_pred, imgt_merge, flow_t0_pred, flow_t1_pred, occ_t_pred, mask_t_pred, loss_rec, loss_wav, loss_com, thresh = ddp_model(img0, img1, imgt, False, th)

        loss_rec_list.append(loss_rec.cpu().numpy())
        loss_wav_list.append(loss_wav.cpu().numpy())
        loss_com_list.append(loss_com.cpu().numpy())

        for j in range(img0.shape[0]):
            psnr = -10 * math.log10(torch.mean((imgt_pred[j] - imgt[j]) * (imgt_pred[j] - imgt[j])).cpu().data)
            psnr_list.append(psnr)

    eval_time_interval = time.time() - time_stamp

    logger.info('eval epoch:{}/{} time:{:.2f} loss_rec:{:.4e} loss_wav:{:.4e} loss_com:{:.4e} psnr:{:.3f}'.format(epoch+1, args.epochs, eval_time_interval, np.array(loss_rec_list).mean(), np.array(loss_wav_list).mean(), np.array(loss_com_list).mean(), np.array(psnr_list).mean()))
    return np.array(psnr_list).mean()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WaveletVFI')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--world_size', default=4, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--eval_interval', default=1, type=int)
    parser.add_argument('--batch_size', default=6, type=int)
    parser.add_argument('--lr_start', default=1e-4, type=float)
    parser.add_argument('--lr_end', default=1e-5, type=float)
    parser.add_argument('--log_path', default='checkpoint', type=str)
    parser.add_argument('--resume_epoch', default=0, type=int)
    parser.add_argument('--resume_path', default=None, type=str)
    parser.add_argument('--dynamic', default=None, type=str)
    args = parser.parse_args()

    dist.init_process_group(backend='gloo', world_size=args.world_size)
    torch.cuda.set_device(args.local_rank)
    args.device = torch.device('cuda', args.local_rank)
    args.num_workers = args.batch_size
    if args.dynamic != None:
        args.dynamic = True
    else:
        args.dynamic = False

    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    
    model = WaveletVFI().to(args.device)
    # model.load_state_dict(torch.load('./checkpoint/stage_1/waveletvfi_latest.pth'))

    if args.resume_epoch != 0:
        model.load_state_dict(torch.load(args.resume_path))

    ddp_model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    train(args, ddp_model)

    dist.destroy_process_group()
