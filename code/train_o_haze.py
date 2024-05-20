import os

from torch.backends import cudnn
from tqdm import tqdm

from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from model import DM2FNet, DM2FNet_woPhy
from datasets import OHazeDataset
import torch
from utils import *
from torch import optim, nn
from torch.utils.data import DataLoader
import torch.cuda.amp as amp


device = 'cuda' if torch.cuda.is_available() else 'cpu'


cfgs = {
    'use_physical': True,
    'num_epoch': 3,
    'train_batch_size': 1,
    'last_epoch': 0,
    'lr': 2e-4,
    'lr_decay': 0.9,
    'weight_decay': 2e-5,
    'momentum': 0.9,
    # 'snapshot': 'step_14000_loss_0.11376_lr_0.000068',
    'snapshot': '',
    'val_freq': 1,
    'log_freq': 100,    # 100 step per log
    'crop_size': 512,
    'ckpt_path': '../checkpoints',
    'dataset': 'o-haze',
    'log_path': '../log/train_ohaze_log.txt'
}


def validate(net, valid_loader, epoch, optimizer, criterion):
    print('validating...')
    net.eval()

    loss_record = AvgMeter()
    psnr_record, ssim_record = AvgMeter(), AvgMeter()

    with torch.no_grad():
        for data in tqdm(valid_loader, desc=f"[Valid] epoch: {epoch}"):
            haze, gt = data

            haze = haze.to(device)
            gt = gt.to(device)

            dehaze = net(haze)

            loss = criterion(dehaze, gt)
            loss_record.update(loss.item(), haze.size(0))

            for i in range(len(haze)):
                r = dehaze[i].cpu().numpy().transpose([1, 2, 0])  # data range [0, 1]
                g = gt[i].cpu().numpy().transpose([1, 2, 0])
                psnr = peak_signal_noise_ratio(g, r)
                ssim = structural_similarity(g, r, data_range=1, multichannel=True,
                                             gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
                psnr_record.update(psnr)
                ssim_record.update(ssim)

    snapshot_name = 'epoch_%d_loss_%.5f_lr_%.6f' % (epoch + 1, loss_record.avg, optimizer.param_groups[1]['lr'])
    print('[validate]: [epoch %d], [loss %.5f]' % (epoch + 1, loss_record.avg))
    os.makedirs(os.path.join(cfgs['ckpt_path'], cfgs['dataset']), exist_ok=True)
    torch.save(net.state_dict(),
               os.path.join(cfgs['ckpt_path'], cfgs['dataset'], snapshot_name + '.pth'))
    torch.save(optimizer.state_dict(),
               os.path.join(cfgs['ckpt_path'], cfgs['dataset'], snapshot_name + '_optim.pth'))

    net.train()


def train(net, train_loader, valid_loader, optimizer, criterion, log_path=None):
    epoch = cfgs['last_epoch']
    # step = epoch * len(train_loader)
    # total = cfgs['num_epoch'] * len(train_loader)
    step = 0
    total = 40000

    scaler = amp.GradScaler()
    torch.cuda.empty_cache()

    for e in range(epoch, cfgs['num_epoch']):
        train_loss_record = AvgMeter()
        loss_x_jf_record = AvgMeter()
        loss_x_j1_record, loss_x_j2_record = AvgMeter(), AvgMeter()
        loss_x_j3_record, loss_x_j4_record = AvgMeter(), AvgMeter()

        with tqdm(train_loader, desc=f"[Train] epoch: {e}") as bar:
            for data in bar:
                optimizer.param_groups[0]['lr'] = 2 * cfgs['lr'] * (1 - float(step) / total) ** cfgs['lr_decay']
                optimizer.param_groups[1]['lr'] = cfgs['lr'] * (1 - float(step) / total) ** cfgs['lr_decay']

                data = list(data)
                data = data + max(4 - len(data), 0) * [None]
                haze, gt, _, _ = data

                batch_size = haze.size(0)

                haze = haze.to(device)
                gt = gt.to(device)

                optimizer.zero_grad()

                with amp.autocast():
                    x_jf, x_j1, x_j2, x_j3, x_j4 = net(haze)

                    loss_x_jf = criterion(x_jf, gt)
                    loss_x_j1 = criterion(x_j1, gt)
                    loss_x_j2 = criterion(x_j2, gt)
                    loss_x_j3 = criterion(x_j3, gt)
                    loss_x_j4 = criterion(x_j4, gt)

                    loss = loss_x_jf + loss_x_j1 + loss_x_j2 + loss_x_j3 + loss_x_j4

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # update recorder
                train_loss_record.update(loss.item(), batch_size)

                loss_x_jf_record.update(loss_x_jf.item(), batch_size)
                loss_x_j1_record.update(loss_x_j1.item(), batch_size)
                loss_x_j2_record.update(loss_x_j2.item(), batch_size)
                loss_x_j3_record.update(loss_x_j3.item(), batch_size)
                loss_x_j4_record.update(loss_x_j4.item(), batch_size)

                bar.set_postfix(train_loss=train_loss_record.avg, fusion_loss=loss_x_jf_record.avg)

                step += 1

                if (step + 1) % cfgs['log_freq'] == 0:
                    log = '[step %d], [train loss %.5f], [loss_x_fusion %.5f], [loss_x_j1 %.5f], ' \
                          '[loss_x_j2 %.5f], [loss_x_j3 %.5f], [loss_x_j4 %.5f], [lr %.13f]' % \
                          (step + 1, train_loss_record.avg, loss_x_jf_record.avg,
                           loss_x_j1_record.avg, loss_x_j2_record.avg, loss_x_j3_record.avg, loss_x_j4_record.avg,
                           optimizer.param_groups[1]['lr'])
                    # print(log)
                    if log_path:
                        with open(log_path, 'a') as f:
                            f.write(log + '\n')

        if (e + 1) % cfgs['val_freq'] == 0:
            validate(net, valid_loader, e, optimizer, criterion)


if __name__ == '__main__':
    cudnn.benchmark = True

    model = DM2FNet_woPhy().to(device).train()
    opt = optim.Adam([
        {'params': [param for name, param in model.named_parameters()
                    if name[-4:] == 'bias' and param.requires_grad],
         'lr': 2 * cfgs['lr']},
        {'params': [param for name, param in model.named_parameters()
                    if name[-4:] != 'bias' and param.requires_grad],
         'lr': cfgs['lr'],
         'weight_decay': cfgs['weight_decay']}
    ], lr=cfgs['lr'], weight_decay=cfgs['weight_decay'])

    train_hazy_path = '../datas/O-HAZE/train_crop_512/hazy'
    train_gt_path = '../datas/O-HAZE/train_crop_512/gt'
    train_data_loader = OHazeDataset(train_hazy_path, train_gt_path)
    train_data_loader = DataLoader(train_data_loader, batch_size=cfgs['train_batch_size'],
                                   shuffle=True, drop_last=True)

    valid_hazy_path = '../datas/O-HAZE/valid_crop_512/hazy'
    valid_gt_path = '../datas/O-HAZE/valid_crop_512/gt'
    valid_data_loader = OHazeDataset(valid_hazy_path, valid_gt_path, mode='valid')
    valid_data_loader = DataLoader(valid_data_loader, batch_size=cfgs['train_batch_size'])

    if cfgs['snapshot']:
        model.load_state_dict(torch.load(os.path.join(cfgs['ckpt_path'], cfgs['dataset'], cfgs['snapshot']+'.pth')))
        opt.load_state_dict(torch.load(os.path.join(cfgs['ckpt_path'], cfgs['dataset'], cfgs['snapshot']+'_optim.pth')))
        cfgs['last_epoch'] = int(cfgs['snapshot'].split('_')[1])

    loss_fn = nn.L1Loss().to(device)

    train(model, train_data_loader, valid_data_loader, opt, loss_fn, cfgs['log_path'])
