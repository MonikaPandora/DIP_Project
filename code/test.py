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
from datasets import OHazeDataset
from torch.utils.data import DataLoader

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

args = {
    'ckpt_path': '../checkpoints',
    'pred_path': '../predictions',
    'dataset_name': 'o-haze',
    'base_name': 'Base_OHAZE',
    'num_features': 128,
    'snapshot': 'epoch_3_loss_0.05023_lr_0.000173'
}

to_test = {
    # 'SOTS': TEST_SOTS_ROOT,
    'O-HAZE': '../datas/O-HAZE/test',
}

to_pil = transforms.ToPILImage()


def compute_ciede2000(gt, pred):
    return deltaE_ciede2000(rgb2lab(gt), rgb2lab(pred))


def main():
    with torch.no_grad():
        criterion = nn.L1Loss().to(device)

        for name, root in to_test.items():
            if 'SOTS' in name:
                net = DM2FNet().to(device)
                # dataset = SotsDataset(root)
            elif 'O-HAZE' == name:
                net = model = DM2FNet(num_features=args['num_features'],
                                      base=args['base_name']).to(device)
                dataset = OHazeDataset(os.path.join(root, 'test', 'hazy'),
                                       os.path.join(root, 'test', 'gt'),
                                       mode='test')
            else:
                raise NotImplementedError

            if len(args['snapshot']) > 0:
                print('load snapshot \'%s\' for testing' % args['snapshot'])
                net.load_state_dict(
                    torch.load(
                        os.path.join(args['ckpt_path'], args['dataset_name'], args['snapshot'] + '.pth'),
                        map_location='cuda' if torch.cuda.is_available() else 'cpu'
                    )
                )

            net.eval()
            dataloader = DataLoader(dataset, batch_size=1)

            mses, psnrs, ssims, ciede2000 = [], [], [], []
            loss_record = AvgMeter()

            for data in tqdm(dataloader, desc='[Pred]'):
                # haze_image, _, _, _, fs = data
                haze, gts, fs = data
                # print(haze.shape, gts.shape)

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
                    ssim = structural_similarity(gt, r, data_range=1, multichannel=True,
                                                 gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
                    ssims.append(ssim)
                    ciede = compute_ciede2000(gt, r)
                    ciede2000.append(ciede)
                    print('predicting for {} [{}]: MSE {:.4f}, PSNR {:.4f}, SSIM {:.4f}, CIEDE2000: {:.4f}'
                          .format(name, fs[i], mse, psnr, ssim, ciede))

                for r, f in zip(res.cpu(), fs):
                    to_pil(r).save(
                        os.path.join(args['pred_path'], args['dataset_name'], args['snapshot'], '%s.jpg' % f))

            print(f"[{name}] L1: {loss_record.avg:.6f}, MSE: {np.mean(mses):.6f}, "
                  f"PSNR: {np.mean(psnrs):.6f}, SSIM: {np.mean(ssims):.6f}, CIEDE2000: {np.mean(ciede2000):.6f}")


if __name__ == '__main__':
    main()
