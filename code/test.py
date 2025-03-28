import gc
import os

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
from skimage.color import deltaE_ciede2000, rgb2lab
import torch
from torch import nn
from torchvision import transforms
from tqdm import tqdm

from utils import *
from model import DM2FNet
from model_enhanced import DM2FNet_Enhanced, DM2FNet_Enhanced_OHaze
from datasets import OHazeDataset, MineDataset, HazeRD_Dataset
from torch.utils.data import DataLoader


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# # device = 'cpu'

# args = {
#     'ckpt_path': '../checkpoints/enhanced',
#     'pred_path': '../predictions/enhanced',
#     'dataset_name': 'ohaze',
#     'model': DM2FNet_Enhanced_OHaze,
#     'base_name': 'Base_OHAZE',
#     'num_features': 128,
#     'snapshot': 'epoch_9_loss_0.05189_lr_0.000025'
# }
#
# to_test = {
#     'O-HAZE': '../datas/O-HAZE',
# }

args = {
    'ckpt_path': '../checkpoints/enhanced',
    'pred_path': '../predictions/enhanced',
    'dataset_name': 'its',  # which dataset trained on
    'model': DM2FNet_Enhanced,
    'base_name': 'Base_ITS',
    'num_features': 128,
    'snapshot': 'epoch_14_loss_0.01449_lr_0.000077',
}

to_test = {
    'Mine': '../datas/Mine',
    # 'HazeRD': '../datas/HazeRD'
}

to_pil = transforms.ToPILImage()


def compute_ciede2000(gt, pred):
    return deltaE_ciede2000(rgb2lab(gt), rgb2lab(pred)).mean()


def main():
    criterion = nn.L1Loss().to(device)

    net = args['model'](num_features=args['num_features'],
                        base=args['base_name']).to(device)

    if len(args['snapshot']) > 0:
        print('load snapshot \'%s\' for testing' % args['snapshot'])
        net.load_state_dict(
            torch.load(
                os.path.join(args['ckpt_path'], args['dataset_name'], args['snapshot'] + '.pth'),
                # map_location='cuda' if torch.cuda.is_available() else 'cpu'
            )
        )

    net.eval()

    with torch.no_grad():
        for name, root in to_test.items():
            if 'HazeRD' in name:
                dataset = HazeRD_Dataset(root)
            elif 'O-HAZE' == name:
                dataset = OHazeDataset(os.path.join(root, 'test', 'hazy'),
                                       os.path.join(root, 'test', 'gt'),
                                       mode='test')
            elif name in ['Mine', 'Mine_its']:
                dataset = MineDataset(root)
            else:
                raise NotImplementedError

            dataloader = DataLoader(dataset, batch_size=1)

            mses, psnrs, ssims, ciede2000 = [], [], [], []
            loss_record = AvgMeter()

            for data in tqdm(dataloader, desc='[Pred]'):
                haze, gts, fs = data

                os.makedirs(os.path.join(args['pred_path'], args['dataset_name'], args['snapshot']), exist_ok=True)

                haze = haze.to(device)

                res = net(haze).detach()

                loss = criterion(res, gts.to(device))
                loss_record.update(loss.item(), haze.size(0))

                for i in range(len(fs)):
                    r = res[i].cpu().numpy().transpose([1, 2, 0])
                    gt = gts[i].cpu().numpy().transpose([1, 2, 0])
                    mse = mean_squared_error(gt, r)
                    mses.append(mse)
                    psnr = peak_signal_noise_ratio(gt, r)
                    psnrs.append(psnr)
                    ssim = structural_similarity(gt, r, data_range=1, channel_axis=2,
                                                 gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
                    ssims.append(ssim)
                    ciede = compute_ciede2000(gt, r)
                    ciede2000.append(ciede)
                    print('predicting for {} [{}]: MSE {:.4f}, PSNR {:.4f}, SSIM {:.4f}, CIEDE2000: {:.4f}'
                          .format(name, fs[i], mse, psnr, ssim, ciede))

                os.makedirs(os.path.join(args['pred_path'], name, args['snapshot']), exist_ok=True)
                for r, f in zip(res.cpu(), fs):
                    to_pil(r).save(
                        os.path.join(args['pred_path'], name, args['snapshot'], '%s.jpg' % f))

            print(f"[{name}] L1: {loss_record.avg:.6f}, MSE: {np.mean(mses):.6f}, "
                  f"PSNR: {np.mean(psnrs):.6f}, SSIM: {np.mean(ssims):.6f}, CIEDE2000: {np.mean(ciede2000):.6f}")


if __name__ == '__main__':
    main()
