import os
from PIL import Image
import pandas as pd
from math import ceil
from tqdm import tqdm


OHAZE_ROOT = '../datas/O-HAZE'


def crop_for_train_and_valid():
    ohaze_root = OHAZE_ROOT
    crop_size = 512

    ori_haze_root = os.path.join(OHAZE_ROOT, 'hazy')
    ori_gt_root = os.path.join(OHAZE_ROOT, 'GT')

    patch_root = os.path.join(ohaze_root, 'train_crop_{}'.format(crop_size))
    patch_haze_path = os.path.join(patch_root, 'hazy')
    patch_gt_path = os.path.join(patch_root, 'gt')

    os.makedirs(patch_root, exist_ok=True)
    os.makedirs(patch_haze_path, exist_ok=True)
    os.makedirs(patch_gt_path, exist_ok=True)

    # first 35 images for training
    img_names = [img_name for img_name in os.listdir(ori_haze_root)]
    train_list = img_names[:35]

    for idx, img_name in enumerate(tqdm(train_list)):
        img_f_name, img_l_name = os.path.splitext(img_name)
        gt_f_name = '{}GT'.format(img_f_name[: -4])

        img = Image.open(os.path.join(ori_haze_root, img_name))
        gt = Image.open(os.path.join(ori_gt_root, gt_f_name + img_l_name))

        assert img.size == gt.size

        w, h = img.size
        stride = int(crop_size / 3.)
        h_steps = 1 + int(ceil(float(max(h - crop_size, 0)) / stride))
        w_steps = 1 + int(ceil(float(max(w - crop_size, 0)) / stride))

        for h_idx in range(h_steps):
            for w_idx in range(w_steps):
                ws0 = w_idx * stride
                ws1 = crop_size + ws0
                hs0 = h_idx * stride
                hs1 = crop_size + hs0
                if h_idx == h_steps - 1:
                    hs0, hs1 = max(h - crop_size, 0), h
                if w_idx == w_steps - 1:
                    ws0, ws1 = max(w - crop_size, 0), w
                img_crop = img.crop((ws0, hs0, ws1, hs1))
                gt_crop = gt.crop((ws0, hs0, ws1, hs1))

                img_crop.save(os.path.join(patch_haze_path, '{}_h_{}_w_{}.png'.format(img_f_name, h_idx, w_idx)))
                gt_crop.save(os.path.join(patch_gt_path, '{}_h_{}_w_{}.png'.format(gt_f_name, h_idx, w_idx)))


def split_train_and_valid(train_portion=0.7):
    train_gt_path = os.path.join(OHAZE_ROOT, 'train_crop_512/gt')
    train_hazy_path = os.path.join(OHAZE_ROOT, 'train_crop_512/hazy')

    valid_gt_path = os.path.join(OHAZE_ROOT, 'valid_crop_512/gt')
    os.makedirs(valid_gt_path, exist_ok=True)
    valid_hazy_path = os.path.join(OHAZE_ROOT, 'valid_crop_512/hazy')
    os.makedirs(valid_hazy_path, exist_ok=True)

    valid_gt = [os.path.join(train_gt_path, name) for name in os.listdir(train_gt_path)]
    valid_hazy = [os.path.join(train_hazy_path, name) for name in os.listdir(train_hazy_path)]

    valid = list(zip(valid_gt, valid_hazy))
    import random
    random.shuffle(valid)
    valid = valid[int(len(valid) * train_portion):]

    import shutil
    for gt, hazy in valid:
        shutil.move(gt, valid_gt_path)
        shutil.move(hazy, valid_hazy_path)


# def gather_as_csv():
#     train_gt_path = os.path.join(OHAZE_ROOT, 'train_crop_512/gt')
#     train_hazy_path = os.path.join(OHAZE_ROOT, 'train_crop_512/hazy')
#     valid_gt_path = os.path.join(OHAZE_ROOT, 'valid_crop_512/gt')
#     valid_hazy_path = os.path.join(OHAZE_ROOT, 'valid_crop_512/hazy')
#
#     train_gt_names = list(os.listdir(train_gt_path))
#     train_hazy_names = list(os.listdir(train_hazy_path))
#     valid_gt_names = list(os.listdir(valid_gt_path))
#     valid_hazy_names = list(os.listdir(valid_hazy_path))
#
#     assert len(train_hazy_names) == len(train_gt_names)
#     assert len(valid_hazy_names) == len(valid_gt_names)
#
#     for i, pair in enumerate(zip(train_hazy_names, train_gt_names)):
#         haze = Image.open(os.path.join(train_hazy_path, pair[0])).convert('RGB')
#         gt = Image.open(os.path.join(train_gt_path, pair[1])).convert('RGB')


if __name__ == '__main__':
    crop_for_train_and_valid()
    split_train_and_valid(train_portion=0.7)
