import torch
from torch.nn.functional import softmax
from torch.nn.functional import interpolate as upsample
import torch.nn as nn

import torchvision.models as models


class Base(nn.Module):
    def __init__(self):
        super(Base, self).__init__()
        self.mean_in = None
        self.std_in = None
        self.mean_out = None
        self.std_out = None

    def norm_in(self, x):
        return (x - self.mean_in) / self.std_in

    def norm_out(self, x):
        return x * self.std_out + self.mean_out


class Base_ITS(Base):
    def __init__(self):
        super(Base, self).__init__()
        rgb_mean = (0.485, 0.456, 0.406)
        self.mean_in = nn.Parameter(torch.Tensor(rgb_mean).view(1, 3, 1, 1), requires_grad=False)
        self.mean_out = self.mean_in
        rgb_std = (0.229, 0.224, 0.225)
        self.std_in = nn.Parameter(torch.Tensor(rgb_std).view(1, 3, 1, 1), requires_grad=False)
        self.std_out = self.std_in


class Base_OHAZE(Base):
    def __init__(self):
        super(Base_OHAZE, self).__init__()
        rgb_mean = (0.47421, 0.50878, 0.56789)
        self.mean_in = nn.Parameter(torch.Tensor(rgb_mean).view(1, 3, 1, 1), requires_grad=False)
        rgb_std = (0.10168, 0.10488, 0.11524)
        self.std_in = nn.Parameter(torch.Tensor(rgb_std).view(1, 3, 1, 1), requires_grad=False)

        rgb_mean = (0.35851, 0.35316, 0.34425)
        self.mean_out = nn.Parameter(torch.Tensor(rgb_mean).view(1, 3, 1, 1), requires_grad=False)
        rgb_std = (0.16391, 0.16174, 0.17148)
        self.std_out = nn.Parameter(torch.Tensor(rgb_std).view(1, 3, 1, 1), requires_grad=False)


class MLF(nn.Module):
    def __init__(self, num_features=64, backbone='resnext101_32x8d'):
        super(MLF, self).__init__()
        self.num_features = num_features

        assert backbone in ['resnet50', 'resnet101',
                            'resnet152', 'resnext50_32x4d', 'resnext101_32x8d']

        # backbone = models.__dict__[backbone](pretrained=True)
        backbone = models.__dict__[backbone]()
        backbone.load_state_dict(torch.load('../resnext101_32x8d-8ba56ff5.pth'))
        del backbone.fc
        self.backbone = backbone

        self.down1 = nn.Sequential(
            nn.Conv2d(256, num_features, kernel_size=1), nn.SELU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(512, num_features, kernel_size=1), nn.SELU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(1024, num_features, kernel_size=1), nn.SELU()
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(2048, num_features, kernel_size=1), nn.SELU()
        )

    def forward(self, x):
        backbone = self.backbone
        layer0 = backbone.conv1(x)
        layer0 = backbone.bn1(layer0)
        layer0 = backbone.relu(layer0)
        layer0 = backbone.maxpool(layer0)

        layer1 = backbone.layer1(layer0)
        layer2 = backbone.layer2(layer1)
        layer3 = backbone.layer3(layer2)
        layer4 = backbone.layer4(layer3)

        down1 = self.down1(layer1)
        down2 = self.down2(layer2)
        down3 = self.down3(layer3)
        down4 = self.down4(layer4)

        down2 = upsample(down2, size=down1.size()[2:], mode='bilinear')
        down3 = upsample(down3, size=down1.size()[2:], mode='bilinear')
        down4 = upsample(down4, size=down1.size()[2:], mode='bilinear')

        mlf = torch.stack((down1, down2, down3, down4), dim=1)
        return mlf


class AFIM(nn.Module):
    def __init__(self, num_features=128):
        super(AFIM, self).__init__()
        self.attention_phy = nn.Sequential(
            nn.Conv2d(num_features * 4, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features * 4, kernel_size=1)
        )

        self.refine = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=1)
        )

    def forward(self, mlf):
        n, _, c, h, w = mlf.size()

        attention_phy = self.attention_phy(mlf.view(n, -1, h, w))
        attention_phy = softmax(attention_phy.view(n, -1, c, h, w), dim=1)

        AMLIF = torch.sum(attention_phy * mlf, dim=1, keepdim=False)
        AMLIF = self.refine(AMLIF) + AMLIF
        return AMLIF


class Attention(nn.Module):
    def __init__(self, num_features=128, num=5):
        super(Attention, self).__init__()
        self.attention_fusion = nn.Sequential(
            nn.Conv2d(num_features * 4, num_features, kernel_size=1), nn.SELU(),
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3 * num, kernel_size=1)
        )

    def forward(self, mlf, ori_h, ori_w):
        n, _, _, h, w = mlf.size()
        concat = torch.concat([mlf[:, i, :, :, :] for i in range(mlf.size()[1])], dim=1)

        attention_fusion = upsample(self.attention_fusion(concat), size=(ori_h, ori_w), mode='bilinear')
        attention_fusion = torch.split(attention_fusion.view(n, 3, -1, ori_h, ori_w), split_size_or_sections=1, dim=1)
        attention_fusion = [softmax(attn, dim=2) for attn in attention_fusion]
        attention_fusion = torch.concat(attention_fusion, dim=1)
        return attention_fusion


class ASModel(nn.Module):
    def __init__(self, num_features=128):
        super(ASModel, self).__init__()
        self.t = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 1, kernel_size=1), nn.Sigmoid()
        )
        self.a = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_features, num_features, kernel_size=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=1), nn.SELU(),
            nn.Conv2d(num_features, 1, kernel_size=1), nn.Sigmoid()
        )

    def forward(self, amlif, img):
        # J0 = (I - A0 * (1 - T0)) / T0
        a = self.a(amlif)
        t = upsample(self.t(amlif), size=img.size()[2:], mode='bilinear').clamp(min=1e-8, max=1.)
        j0 = ((img - a * (1 - t)) / t).clamp(min=0., max=1.)
        return j0, t, a


class SeparationModel(nn.Module):
    def __init__(self, num_features=128):
        super(SeparationModel, self).__init__()
        self.js = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
                nn.Conv2d(num_features // 2, 3, kernel_size=1)
            ) for _ in range(4)
        ])

    def forward(self, amlifs, img, norm, base):
        log_img = torch.log(img.clamp(min=1e-8))
        log_log_inverse = torch.log(torch.log(1 / img.clamp(min=1e-8, max=(1 - 1e-8))))

        # J1 = I * exp(R1)
        r1 = upsample(self.js[0](amlifs[0]), size=log_img.size()[2:], mode='bilinear')
        j1 = torch.exp(log_img + r1).clamp(min=0., max=1.)

        # J2 = I + R2
        r2 = upsample(self.js[1](amlifs[1]), size=img.size()[2:], mode='bilinear')
        j2 = base.norm_out(norm + r2).clamp(min=0., max=1.)

        # J3 = I ^ R3
        r3 = upsample(self.js[2](amlifs[2]), size=img.size()[2:], mode='bilinear')
        j3 = torch.exp(-torch.exp(log_log_inverse + r3)).clamp(min=0., max=1.)

        # J4 = log(1 + I * R4)
        r4 = upsample(self.js[3](amlifs[3]), size=img.size()[2:], mode='bilinear')
        j4 = (torch.log(1 + torch.exp(log_img + r4))).clamp(min=0., max=1.)
        return j1, j2, j3, j4


class DM2FNet(nn.Module):
    def __init__(self, num_features=128, arch='resnext101_32x8d', base='Base'):
        super(DM2FNet, self).__init__()
        self.mlf = MLF(num_features, backbone=arch)
        self.afims = nn.ModuleList([AFIM(num_features) for _ in range(5)])
        self.as_model = ASModel(num_features)
        self.separations = SeparationModel(num_features)
        self.attn = Attention(num_features)

        self.base = eval(base)()

        for m in self.modules():
            if isinstance(m, nn.SELU) or isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, img, img_hd=None):
        if img_hd is not None:
            img = img_hd
        # img_norm = (img - self.mean) / self.std
        img_norm = self.base.norm_in(img)

        n, c, h, w = img.size()

        mlf = self.mlf(img_norm)
        amlifs = [afim(mlf) for afim in self.afims]
        attn = self.attn(mlf, h, w)

        j0, t, a = self.as_model(amlifs[0], img)
        j1, j2, j3, j4 = self.separations(amlifs[1:], img, img_norm, self.base)

        all_res = torch.stack((j0, j1, j2, j3, j4), dim=2)
        # print(attn.shape)
        # print(all_res.shape)
        weighted_rgb = torch.split((all_res * attn), split_size_or_sections=1, dim=1)
        weighted_rgb = [torch.sum(channel, dim=2).clamp(min=0., max=1.) for channel in weighted_rgb]
        fusion = torch.concat(weighted_rgb, dim=1)

        if self.training:
            return fusion, j0, j1, j2, j3, j4, t, a
        else:
            return fusion


if __name__ == '__main__':
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    model = DM2FNet()
    model.to(device)
    model.eval()
    tmp = torch.tensor(list(map(float, range(1 * 3 * 128 * 128)))).view(1, 3, 128, 128).to(device)
    print(tmp.shape)
    import time

    s = time.time()
    with torch.no_grad():
        out = model(tmp)
    e = time.time()
    print(out.shape, e - s)
