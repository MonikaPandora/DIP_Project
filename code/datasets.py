import torch
from PIL import Image
import os
import random
import numpy as np
from torch.utils import data
from torchvision import transforms
from torchvision.transforms import ToTensor

to_tensor = ToTensor()


def random_crop(size, haze, gt, extra=None):
    w, h = haze.size
    assert haze.size == gt.size

    if w < size or h < size:
        haze = transforms.Resize(size)(haze)
        gt = transforms.Resize(size)(gt)
        w, h = haze.size

    x1 = random.randint(0, w - size)
    y1 = random.randint(0, h - size)

    _haze = haze.crop((x1, y1, x1 + size, y1 + size))
    _gt = gt.crop((x1, y1, x1 + size, y1 + size))

    if extra is None:
        return _haze, _gt
    else:
        # extra: trans or predict
        assert haze.size == extra.size
        _extra = extra.crop((x1, y1, x1 + size, y1 + size))
        return _haze, _gt, _extra


class HazeRD_Dataset(data.Dataset):
    def __init__(self, path):
        hazy_names = os.listdir(os.path.join(path, 'simu'))
        self.data_paths = list(zip(
            [os.path.join(path, 'simu', name) for name in hazy_names],
            [os.path.join(path, 'img', '_'.join(name.split('_')[:2] + ['RGB.jpg'])) for name in hazy_names],
        ))

    def __getitem__(self, index):
        haze_path, gt_path = self.data_paths[index]
        name = os.path.splitext(os.path.split(haze_path)[-1])[0]

        img = Image.open(haze_path).convert('RGB')
        gt = Image.open(gt_path).convert('RGB')

        assert img.size == gt.size

        return to_tensor(img), to_tensor(gt), name

    def __len__(self):
        return len(self.data_paths)


class ITS_Dataset(data.Dataset):
    def __init__(self, path, flip=False, crop=None):
        self.flip = flip
        self.crop = crop

        def make_dataset_its(root):
            items = []
            for img_name in os.listdir(os.path.join(root, 'hazy')):
                idx0, idx1, ato = os.path.splitext(img_name)[0].split('_')
                gt = os.path.join(root, 'clear', idx0 + '.png')
                trans = os.path.join(root, 'trans', idx0 + '_' + idx1 + '.png')
                haze = os.path.join(root, 'hazy', img_name)
                items.append([haze, trans, float(ato), gt])

            return items

        self.datas = make_dataset_its(path)

    def __getitem__(self, index):
        haze_path, trans_path, ato, gt_path = self.datas[index]
        name = os.path.splitext(os.path.split(haze_path)[1])[0]

        haze = Image.open(haze_path).convert('RGB')
        trans = Image.open(trans_path).convert('L')
        gt = Image.open(gt_path).convert('RGB')

        assert haze.size == trans.size
        assert trans.size == gt.size

        if self.crop:
            haze, gt, trans = random_crop(self.crop, haze, gt, trans)

        if self.flip and random.random() < 0.5:
            haze = haze.transpose(Image.FLIP_LEFT_RIGHT)
            trans = trans.transpose(Image.FLIP_LEFT_RIGHT)
            gt = gt.transpose(Image.FLIP_LEFT_RIGHT)

        haze = to_tensor(haze)
        trans = to_tensor(trans)
        gt = to_tensor(gt)
        gt_ato = torch.Tensor([ato]).float()

        return haze, gt, gt_ato, trans

    def __len__(self):
        return len(self.datas)


class SotsDataset(data.Dataset):
    def __init__(self, root):
        self.root = root
        self.imgs = [(os.path.join(root, 'hazy', img_name),
                      os.path.join(root, 'trans', img_name),
                      os.path.join(root, 'gt', img_name))
                     for img_name in os.listdir(os.path.join(root, 'hazy'))]

    def __getitem__(self, index):
        haze_path, trans_path, gt_path = self.imgs[index]
        name = os.path.splitext(os.path.split(haze_path)[1])[0]

        haze = Image.open(haze_path).convert('RGB')
        haze = to_tensor(haze)

        idx0 = name.split('_')[0]
        gt = Image.open(os.path.join(self.root, 'gt', idx0 + '.png')).convert('RGB')
        gt = to_tensor(gt)
        if gt.shape != haze.shape:
            # crop the indoor images
            gt = gt[:, 10: 470, 10: 630]

        return haze, gt

    def __len__(self):
        return len(self.imgs)


class MineDataset(data.Dataset):
    def __init__(self, path):
        self.data_paths = list(zip(
            sorted(
                [os.path.join(path, 'hazy', name) for name in os.listdir(os.path.join(path, 'hazy'))]
            ),
            sorted(
                [os.path.join(path, 'gt', name) for name in os.listdir(os.path.join(path, 'gt'))]
            )
        ))

    def __getitem__(self, index):
        haze_path, gt_path = self.data_paths[index]
        name = os.path.splitext(os.path.split(haze_path)[-1])[0]

        img = Image.open(haze_path).convert('RGB')
        gt = Image.open(gt_path).convert('RGB')

        return to_tensor(img), to_tensor(gt), name

    def __len__(self):
        return len(self.data_paths)


class OHazeDataset(data.Dataset):
    def __init__(self, hazy_path, gt_path, mode='train'):
        self.mode = mode

        gt = []
        for name in os.listdir(hazy_path):
            gt_name = name.replace('hazy', 'GT')
            assert os.path.exists(os.path.join(gt_path, gt_name)), name
            gt.append(os.path.join(gt_path, gt_name))

        hazy = [os.path.join(hazy_path, name) for name in os.listdir(hazy_path)]

        self.data_paths = list(zip(hazy, gt))

    def __getitem__(self, index):
        """
        :param index:
        :return: hazy_img, ground_truth, ground_truth_of_atmosphere, ground_truth_of_transmap
        """
        haze_path, gt_path = self.data_paths[index]
        name = os.path.splitext(os.path.split(haze_path)[-1])[0]

        img = Image.open(haze_path).convert('RGB')
        gt = Image.open(gt_path).convert('RGB')

        if self.mode == 'train':
            # img, gt = random_crop(416, img, gt)
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                gt = gt.transpose(Image.FLIP_LEFT_RIGHT)

            rotate_degree = np.random.choice([-90, 0, 90, 180])
            img, gt = img.rotate(rotate_degree, Image.BILINEAR), gt.rotate(rotate_degree, Image.BILINEAR)

        if self.mode != 'test':
            return to_tensor(img), to_tensor(gt)
        else:
            return to_tensor(img), to_tensor(gt), name

    def __len__(self):
        return len(self.data_paths)


if __name__ == '__main__':
    ohaze = OHazeDataset('../datas/O-HAZE/valid_crop_512/hazy',
                         '../datas/O-HAZE/valid_crop_512/gt',
                         mode='train')
    for data in ohaze:
        print(data[0].shape)
