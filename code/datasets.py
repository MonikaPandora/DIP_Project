from PIL import Image
import os
import random
import numpy as np
from torch.utils import data
from torchvision.transforms import ToTensor


to_tensor = ToTensor()


class OHazeDataset(data.Dataset):
    def __init__(self, hazy_path, gt_path, mode='train'):
        self.mode = mode
        hazy = os.listdir(hazy_path)
        gt = os.listdir(gt_path)
        self.data_paths = list(zip(
            [os.path.join(hazy_path, name) for name in hazy],
            [os.path.join(gt_path, name) for name in gt]
        ))

        # abandon some crashed image
        # hazy = []
        # gt = []
        # for h, g in self.data_paths:
        #     try:
        #         hz = Image.open(h).convert('RGB')
        #         t = Image.open(g).convert('RGB')
        #     except :
        #         print(f'delete: ', h, g)
        #         os.remove(h)
        #         os.remove(g)
        #     else:
        #         hazy.append(h)
        #         gt.append(g)
        # self.data_paths = list(zip(
        #     hazy,
        #     gt
        # ))

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
