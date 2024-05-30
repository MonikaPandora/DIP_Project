from model import *
from TSNet import *


class PixelAttention(nn.Module):
    def __init__(self, num_features):
        super(PixelAttention, self).__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(num_features, num_features // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // 8, num_features, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.attn(x)
        return x * y


class ChannelAttention(nn.Module):
    def __init__(self, num_features):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.attn = nn.Sequential(
            nn.Conv2d(num_features, num_features // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // 8, num_features, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.attn(y)
        return x * y


class MLF_Enhanced(nn.Module):
    def __init__(self, num_features=64, backbone='resnext101_32x8d'):
        super(MLF_Enhanced, self).__init__()
        self.num_features = num_features

        assert backbone in ['resnet50', 'resnet101',
                            'resnet152', 'resnext50_32x4d', 'resnext101_32x8d']

        # backbone = models.__dict__[backbone](pretrained=True)
        backbone = models.__dict__[backbone]()
        backbone.load_state_dict(torch.load('../resnext101_32x8d-8ba56ff5.pth'))
        del backbone.fc
        self.backbone = backbone

        self.down1 = nn.Sequential(
            nn.Conv2d(256, num_features, kernel_size=1), nn.SELU(),
            ChannelAttention(num_features),
            PixelAttention(num_features),
            nn.ReLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(512, num_features, kernel_size=1), nn.SELU(),
            ChannelAttention(num_features),
            PixelAttention(num_features),
            nn.ReLU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(1024, num_features, kernel_size=1), nn.SELU(),
            ChannelAttention(num_features),
            PixelAttention(num_features),
            nn.ReLU()
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(2048, num_features, kernel_size=1), nn.SELU(),
            ChannelAttention(num_features),
            PixelAttention(num_features),
            nn.ReLU()
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


class ReformatedASModel(nn.Module):
    def __init__(self, num_features=128, mix_factor=1.):
        super(ReformatedASModel, self).__init__()
        self.k = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )

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

        self.mix = Mix(m=mix_factor)

    def forward_asm(self, amlif, img):
        # J0 = (I - A0 * (1 - T0)) / T0
        a = self.a(amlif)
        t = upsample(self.t(amlif), size=img.size()[2:], mode='bilinear').clamp(min=1e-8, max=1.)
        j0 = ((img - a * (1 - t)) / t).clamp(min=0., max=1.)
        return j0, t, a

    def forward_reformat_asm(self, amlif, img):
        k = upsample(self.k(amlif), size=img.size()[2:], mode='bilinear')
        return (k * img - k).clamp(min=0., max=1.)

    def forward(self, amlif, img):
        j0, t, a = self.forward_asm(amlif, img)
        j0_ = self.forward_reformat_asm(amlif, img)
        return self.mix(j0, j0_), t, a


class DM2FNet_Enhanced_OHaze(nn.Module):
    def __init__(self, num_features=128, arch='resnext101_32x8d', base='Base'):
        super(DM2FNet_Enhanced_OHaze, self).__init__()
        self.mlf = MLF(num_features, backbone=arch)
        self.afims = nn.ModuleList([AFIM(num_features) for _ in range(5)])
        self.as_model = ReformatedASModel(num_features, mix_factor=0.4)
        self.separations = SeparationModel(num_features)
        self.attn = Attention(num_features, 5)

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


class DM2FNet_Enhanced(nn.Module):
    def __init__(self, num_features=128, arch='resnext101_32x8d', base='Base'):
        super(DM2FNet_Enhanced, self).__init__()
        self.mlf = MLF_Enhanced(num_features, backbone=arch)
        self.afims = nn.ModuleList([AFIM(num_features) for _ in range(5)])
        self.as_model = ASModel(num_features)
        self.separations = SeparationModel(num_features)
        self.attn = Attention(num_features, 5)

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
        weighted_rgb = torch.split((all_res * attn), split_size_or_sections=1, dim=1)
        weighted_rgb = [torch.sum(channel, dim=2).clamp(min=0., max=1.) for channel in weighted_rgb]
        fusion = torch.concat(weighted_rgb, dim=1)

        if self.training:
            return fusion, j0, j1, j2, j3, j4, t, a
        else:
            return fusion


class Learning_with_helper(nn.Module):
    def __init__(self, dehaze_model, helper):
        super(Learning_with_helper, self).__init__()
        self.dehaze_model = dehaze_model
        self.helper = helper

    def forward(self, x):
        dehaze = self.dehaze_model(x)
        if self.training:
            return self.helper(dehaze[0]), *dehaze[1:]
        else:
            return dehaze


if __name__ == '__main__':
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    model = DM2FNet_Enhanced(base='Base_OHAZE')
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
