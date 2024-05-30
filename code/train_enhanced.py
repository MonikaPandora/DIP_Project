import gc
import os

from torch.backends import cudnn
from tqdm import tqdm

from model import MLF
from model_enhanced import *
from datasets import OHazeDataset, ITS_Dataset, SotsDataset
import torch
from utils import *
from torch import optim, nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from ReconstructLoss import ReconstructLoss, ContrastLoss

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# cfgs = {
#     'num_features': 128,
#     'num_epoch': 10,
#     'train_batch_size': 8,
#     'last_epoch': 0,
#     'lr': 2e-4,
#     'lr_decay': 0.9,
#     'weight_decay': 2e-5,
#     'snapshot': '',
#     'val_freq': 1,
#     'log_freq': 100,  # 100 step per log
#     'crop_size': 512,
#     'model': DM2FNet_Enhanced_OHaze,
#     'weight_cr': 0.1,
#     'weight_rc': 1.0,
#     'ckpt_path': '../checkpoints/enhanced',
#     'dataset': 'ohaze',
#     'log_path': '../enhanced_results/train_enhanced_ohaze_log.txt',
#     'train_path': '../datas/O-HAZE/train_crop_512',
#     'valid_path': '../datas/O-HAZE/valid',
#     'base_name': 'Base_OHAZE',
#     'helper_network': None
# }


cfgs = {
    'num_features': 128,
    'num_epoch': 16,
    'train_batch_size': 12,
    'last_epoch': 0,
    'lr': 5e-4,
    'lr_decay': 0.9,
    'weight_decay': 0,
    'snapshot': '',
    'val_freq': 2,
    'log_freq': 100,  # 100 step per log
    'crop_size': 256,
    'model': DM2FNet_Enhanced,
    'weight_cr': 0.1,
    'weight_rc': 1.0,
    'ckpt_path': '../checkpoints/enhanced',
    'dataset': 'its',
    'log_path': '../enhanced_results/train_enhanced_its_log.txt',
    'train_path': '../datas/ITS_v2',
    'valid_path': '../datas/SOTS/nyuhaze500',
    'base_name': 'Base_ITS',
    'helper_network': TSNet_t2
}


def validate(net, valid_loader, epoch, optimizer, criterion):
    print('validating...')
    net.eval()

    loss_record = AvgMeter()

    with torch.no_grad():
        for data in tqdm(valid_loader, desc=f"[Valid] epoch: {epoch}"):
            haze, gt = data

            haze = haze.to(device)
            gt = gt.to(device)

            dehaze = net(haze)

            loss = criterion(dehaze, gt)
            loss_record.update(loss.item(), haze.size(0))

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
    step = epoch * len(train_loader)
    total = cfgs['num_epoch'] * len(train_loader)

    recons = ReconstructLoss() if cfgs['weight_rc'] > 0 else None
    cr = ContrastLoss() if cfgs['weight_cr'] > 0 else None

    for e in range(epoch, cfgs['num_epoch']):
        train_loss_record = AvgMeter()
        loss_x_jf_record, loss_x_j0_record = AvgMeter(), AvgMeter()
        loss_x_j1_record, loss_x_j2_record = AvgMeter(), AvgMeter()
        loss_x_j3_record, loss_x_j4_record = AvgMeter(), AvgMeter()
        loss_a_record, loss_t_record = AvgMeter(), AvgMeter()
        loss_recons_record = AvgMeter()

        with tqdm(train_loader, desc=f"[Train] epoch: {e}") as bar:
            for data in bar:
                torch.cuda.empty_cache()
                optimizer.param_groups[0]['lr'] = 2 * cfgs['lr'] * (1 - float(step) / total) ** cfgs['lr_decay']
                optimizer.param_groups[1]['lr'] = cfgs['lr'] * (1 - float(step) / total) ** cfgs['lr_decay']

                data = list(data)
                data = data + max(4 - len(data), 0) * [None]
                haze, gt, gt_ato, gt_trans_map = data

                batch_size = haze.size(0)

                haze = haze.to(device)
                gt = gt.to(device)
                gt_ato = gt_ato.to(device) if gt_ato is not None else None
                gt_trans_map = gt_trans_map.to(device) if gt_ato is not None else None

                optimizer.zero_grad()

                x_jf, x_j0, x_j1, x_j2, x_j3, x_j4, t, a = net(haze)

                loss_x_jf = criterion(x_jf, gt)
                loss_x_j0 = criterion(x_j0, gt)
                loss_x_j1 = criterion(x_j1, gt)
                loss_x_j2 = criterion(x_j2, gt)
                loss_x_j3 = criterion(x_j3, gt)
                loss_x_j4 = criterion(x_j4, gt)

                loss = loss_x_jf + loss_x_j0 + loss_x_j1 + loss_x_j2 + loss_x_j3 + loss_x_j4

                if cr is not None:
                    loss += cr(x_jf, gt, haze) * cfgs['weight_cr']

                if recons is not None:
                    loss_rec = recons(haze, gt, a, t)
                    loss_recons_record.update(loss_rec.item())
                    loss += loss_rec * cfgs['weight_rc']

                if gt_ato is not None:
                    loss_a = criterion(a.view(batch_size, -1), gt_ato)
                    loss_a_record.update(loss_a.item(), batch_size)
                    loss += loss_a

                if gt_trans_map is not None:
                    loss_t = criterion(t, gt_trans_map)
                    loss_t_record.update(loss_t.item(), batch_size)
                    loss += 10 * loss_t

                loss.backward()
                optimizer.step()

                del haze, gt
                gc.collect()

                # update recorder
                train_loss_record.update(loss.item(), batch_size)

                loss_x_jf_record.update(loss_x_jf.item(), batch_size)
                loss_x_j0_record.update(loss_x_j0.item(), batch_size)
                loss_x_j1_record.update(loss_x_j1.item(), batch_size)
                loss_x_j2_record.update(loss_x_j2.item(), batch_size)
                loss_x_j3_record.update(loss_x_j3.item(), batch_size)
                loss_x_j4_record.update(loss_x_j4.item(), batch_size)

                bar.set_postfix(train_loss=train_loss_record.avg, fusion_loss=loss_x_jf_record.avg)

                step += 1

                if (step + 1) % cfgs['log_freq'] == 0:
                    log = '[step %d], [train loss %.5f], [loss_x_fusion %.5f], [loss_x_phy %.5f], ' \
                          '[loss_x_j1 %.5f], [loss_x_j2 %.5f], [loss_x_j3 %.5f], [loss_x_j4 %.5f], [loss_a %.5f], ' \
                          '[loss_t %.5f], [loss_rc %.5f], [lr %.13f]' % \
                          (step, train_loss_record.avg, loss_x_jf_record.avg, loss_x_j0_record.avg,
                           loss_x_j1_record.avg, loss_x_j2_record.avg, loss_x_j3_record.avg, loss_x_j4_record.avg,
                           loss_a_record.avg, loss_t_record.avg, loss_recons_record.avg,
                           optimizer.param_groups[1]['lr'])

                    if log_path:
                        with open(log_path, 'a') as f:
                            f.write(log + '\n')

        if (e + 1) % cfgs['val_freq'] == 0:
            validate(net if not isinstance(net, Learning_with_helper) else net.dehaze_model,
                     valid_loader, e, optimizer, criterion)


if __name__ == '__main__':
    cudnn.benchmark = True

    model = cfgs['model'](num_features=cfgs['num_features'],
                          base=cfgs['base_name']).to(device).train()

    if callable(cfgs['helper_network']):
        model = Learning_with_helper(model, cfgs['helper_network']().to(device).train())

    opt = optim.Adam([
        {'params': [param for name, param in model.named_parameters()
                    if name[-4:] == 'bias' and param.requires_grad],
         'lr': 2 * cfgs['lr']},
        {'params': [param for name, param in model.named_parameters()
                    if name[-4:] != 'bias' and param.requires_grad],
         'lr': cfgs['lr'],
         'weight_decay': cfgs['weight_decay']}
    ], lr=cfgs['lr'], weight_decay=cfgs['weight_decay'])

    if cfgs['dataset'] == 'ohaze':
        train_hazy_path = os.path.join(cfgs['train_path'], 'hazy')
        train_gt_path = os.path.join(cfgs['train_path'], 'gt')
        train_data_loader = OHazeDataset(train_hazy_path, train_gt_path)
        train_data_loader = DataLoader(train_data_loader, batch_size=cfgs['train_batch_size'],
                                       shuffle=True, drop_last=True)

        valid_hazy_path = os.path.join(cfgs['valid_path'], 'hazy')
        valid_gt_path = os.path.join(cfgs['valid_path'], 'gt')
        valid_data_loader = OHazeDataset(valid_hazy_path, valid_gt_path, mode='valid')
        valid_data_loader = DataLoader(valid_data_loader, batch_size=1)
    elif cfgs['dataset'] == 'its':
        train_dataset = ITS_Dataset(cfgs['train_path'], True, cfgs['crop_size'])
        train_data_loader = DataLoader(train_dataset, batch_size=cfgs['train_batch_size'], num_workers=4,
                                       shuffle=True, drop_last=True)

        valid_dataset = SotsDataset(cfgs['valid_path'])
        valid_data_loader = DataLoader(valid_dataset, batch_size=8)
    else:
        raise NotImplementedError

    if cfgs['snapshot']:
        model.load_state_dict(torch.load(os.path.join(cfgs['ckpt_path'], cfgs['dataset'], cfgs['snapshot'] + '.pth')))
        opt.load_state_dict(
            torch.load(os.path.join(cfgs['ckpt_path'], cfgs['dataset'], cfgs['snapshot'] + '_optim.pth')))
        cfgs['last_epoch'] = int(cfgs['snapshot'].split('_')[1])

    loss_fn = nn.L1Loss().to(device)

    train(model, train_data_loader, valid_data_loader, opt, loss_fn, cfgs['log_path'])
