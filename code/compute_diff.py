import os

import torch
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor, ToPILImage


def compute_diff(gt_paths, to_compute_paths, save_to):
    assert len(gt_paths) == len(to_compute_paths)

    os.makedirs(save_to, exist_ok=True)

    for gt_p, tgt_p in zip(gt_paths, to_compute_paths):
        name = os.path.splitext(os.path.split(gt_p)[-1])[0]
        ground_truth = Image.open(gt_p).convert('RGB')
        target = Image.open(tgt_p).convert('RGB')

        ground_truth = ToTensor()(ground_truth)
        target = ToTensor()(target)
        # diff = (target - ground_truth).clamp(min=0, max=1)
        diff = torch.abs(target - ground_truth).clamp(min=0, max=1)
        diff = ToPILImage()(diff)
        diff.save(os.path.join(save_to, name.replace('GT', 'diff') + '.png'))


if __name__ == '__main__':
    gt_root = '../datas/O-HAZE/test/gt'
    gt_names = os.listdir(gt_root)
    gt_paths = [os.path.join(gt_root, n) for n in gt_names]

    tgt_root = '../predictions/O-HAZE/epoch_2_loss_0.08506_lr_0.000074'
    tgt_names = os.listdir(tgt_root)
    tgt_paths = [os.path.join(tgt_root, n) for n in tgt_names]

    save_path = '../difference/pred/epoch_2_loss_0.08506_lr_0.000074'

    compute_diff(gt_paths, tgt_paths, save_path)
