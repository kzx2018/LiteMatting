import torch
import torch.nn as nn
import torch.nn.functional as F


def GN(x):
    if x % 32 == 0:
        return nn.GroupNorm(32, x)
    if x % 16 == 0:
        return nn.GroupNorm(16, x)
    if x % 8 == 0:
        return nn.GroupNorm(8, x)
    return nn.GroupNorm(4, x)


class MSLPPM(nn.Module):
    def __init__(self, in_channels):
        super(MSLPPM, self).__init__()
        mid_channels = int(in_channels / 2)
        inter_channels = int(mid_channels / 4)
        self.trans = nn.Sequential(nn.Conv2d(in_channels, mid_channels, 1, 1, 0, bias=False),
                                   GN(mid_channels),
                                   nn.ReLU(True),
                                   nn.Conv2d(mid_channels, inter_channels, 1, 1, 0, bias=False),
                                   GN(inter_channels),
                                   nn.ReLU(True)
                                   )
        self.pool1 = nn.AdaptiveAvgPool2d((5, 5))
        self.pool2 = nn.AdaptiveAvgPool2d((13, 13))
        self.pool3 = nn.AdaptiveAvgPool2d((1, None))
        self.pool4 = nn.AdaptiveAvgPool2d((None, 1))
        self.pool5 = nn.AdaptiveAvgPool2d((15, 7))
        self.pool6 = nn.AdaptiveAvgPool2d((7, 15))
        self.pool7 = nn.AdaptiveAvgPool2d((23, 11))
        self.pool8 = nn.AdaptiveAvgPool2d((11, 23))

        self.conv1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 3), 1, (1, 1), bias=False),
                                   GN(inter_channels),
                                   nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 3), 1, (1, 1), bias=False),
                                   GN(inter_channels),
                                   nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False),
                                   GN(inter_channels),
                                   nn.ReLU(True))
        self.conv4 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False),
                                   GN(inter_channels),
                                   nn.ReLU(True))
        self.conv5 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (5, 3), 1, (2, 1), bias=False),
                                   GN(inter_channels),
                                   nn.ReLU(True))
        self.conv6 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 5), 1, (1, 2), bias=False),
                                   GN(inter_channels),
                                   nn.ReLU(True))
        self.conv7 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (7, 3), 1, (3, 1), bias=False),
                                   GN(inter_channels),
                                   nn.ReLU(True))
        self.conv8 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 7), 1, (1, 3), bias=False),
                                   GN(inter_channels),
                                   nn.ReLU(True))
        self.conv = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                  GN(inter_channels),
                                  nn.ReLU(True))

    def forward(self, x):
        """
            out: [1, 160, 32, 32]
        """
        _, _, h, w = x.size()
        x = self.trans(x)
        x1 = F.interpolate(self.conv1(self.pool1(x)), size=(h, w), mode='bilinear', align_corners=False)
        x2 = F.interpolate(self.conv2(self.pool2(x)), size=(h, w), mode='bilinear', align_corners=False)
        x3 = F.interpolate(self.conv3(self.pool3(x)), size=(h, w), mode='bilinear', align_corners=False)
        x4 = F.interpolate(self.conv4(self.pool4(x)), size=(h, w), mode='bilinear', align_corners=False)
        x5 = F.interpolate(self.conv5(self.pool5(x)), size=(h, w), mode='bilinear', align_corners=False)
        x6 = F.interpolate(self.conv6(self.pool6(x)), size=(h, w), mode='bilinear', align_corners=False)
        x7 = F.interpolate(self.conv7(self.pool7(x)), size=(h, w), mode='bilinear', align_corners=False)
        x8 = F.interpolate(self.conv8(self.pool8(x)), size=(h, w), mode='bilinear', align_corners=False)
        s = self.conv(F.relu_(x1 + x2))
        l = self.conv(F.relu_(x3 + x4))
        m = self.conv(F.relu_(x5 + x6))
        n = self.conv(F.relu_(x7 + x8))
        out = torch.cat([s, m, n, l], dim=1)
        return out


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, dilation, expand_ratio, batch_norm):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        BatchNorm2d = batch_norm

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        self.kernel_size = 3
        self.dilation = dilation

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 0, dilation, groups=hidden_dim, bias=False),
                BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 0, dilation, groups=hidden_dim, bias=False),
                BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                BatchNorm2d(oup),
            )

    def fixed_padding(self, inputs, kernel_size, dilation):
        kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
        return padded_inputs

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.kernel_size == (3, 3):
                m.dilation = (dilate, dilate)
                m.padding = (dilate, dilate)

    def forward(self, x):
        x_pad = self.fixed_padding(x, self.kernel_size, dilation=self.dilation)
        if self.use_res_connect:
            return x + self.conv(x_pad)
        else:
            return self.conv(x_pad)


class InvertedResidualLeaky(nn.Module):
    def __init__(self, inp, oup, stride, dilation, expand_ratio, batch_norm):
        super(InvertedResidualLeaky, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        BatchNorm2d = batch_norm

        hidden_dim = round(oup * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        self.kernel_size = 3
        self.dilation = dilation

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 0, dilation, groups=hidden_dim, bias=False),
                BatchNorm2d(hidden_dim),
                nn.LeakyReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                BatchNorm2d(hidden_dim),
                nn.LeakyReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 0, dilation, groups=hidden_dim, bias=False),
                BatchNorm2d(hidden_dim),
                nn.LeakyReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                BatchNorm2d(oup),
            )

    def fixed_padding(self, inputs, kernel_size, dilation):
        kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
        return padded_inputs

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.kernel_size == (3, 3):
                m.dilation = (dilate, dilate)
                m.padding = (dilate, dilate)

    def forward(self, x):
        x_pad = self.fixed_padding(x, self.kernel_size, dilation=self.dilation)
        if self.use_res_connect:
            return x + self.conv(x_pad)
        else:
            return self.conv(x_pad)


class PSPModule(nn.Module):
    def __init__(self, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()
        self.stages = nn.ModuleList([self._make_stage(size) for size in sizes])

    def _make_stage(self, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        return prior

    def forward(self, feats):
        n, c, _, _ = feats.size()
        priors = [stage(feats).view(n, c, -1) for stage in self.stages]
        center = torch.cat(priors, -1)
        return center


class GFNB(nn.Module):

    def __init__(self, low_in_channels, high_in_channels, out_channels, key_channels, value_channels, dropout=0):
        super(GFNB, self).__init__()
        self.in_channels = low_in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels

        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels, kernel_size=1, stride=1, padding=0),
            GN(self.key_channels),
            nn.LeakyReLU(inplace=True)

        )
        self.f_query = nn.Sequential(
            nn.Conv2d(in_channels=high_in_channels, out_channels=self.key_channels, kernel_size=1, stride=1, padding=0),
            GN(self.key_channels),
            nn.LeakyReLU(inplace=True)
        )
        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels, kernel_size=1,
                                 stride=1, padding=0)
        self.W = nn.Conv2d(in_channels=self.value_channels, out_channels=self.out_channels, kernel_size=1, stride=1,
                           padding=0)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)
        self.psp = PSPModule(sizes=(1, 3, 6, 8))

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(out_channels + high_in_channels, out_channels, kernel_size=1, padding=0),
            GN(out_channels),
            nn.Dropout2d(dropout)
        )

    def forward(self, low_feats, high_feats):
        batch_size, h, w = high_feats.size(0), high_feats.size(2), high_feats.size(3)
        value = self.f_value(low_feats)
        value = self.psp(value)
        value = value.permute(0, 2, 1)

        key = self.f_key(low_feats)
        key = self.psp(key)

        query = self.f_query(high_feats).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, *high_feats.size()[2:])
        context = self.W(context)
        output = self.conv_bn_dropout(torch.cat([context, high_feats], 1))
        return output


def conv_bn(inp, oup, k=3, s=1, BatchNorm2d=nn.BatchNorm2d):
    return nn.Sequential(
        nn.Conv2d(inp, oup, k, s, padding=k // 2, bias=False),
        BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class MobileMatting(nn.Module):
    def __init__(self):
        super(MobileMatting, self).__init__()
        output_stride = 32
        BatchNorm2d = GN
        width_mult = 1.
        self.width_mult = width_mult
        block = InvertedResidual
        blockleaky = InvertedResidualLeaky

        initial_channel = 32
        current_stride = 1
        rate = 1

        inverted_residual_setting = [
            # t, p, c, n, s, d
            [1, 32, 32, 1, 1, 1],
            [6, 32, 24, 2, 2, 1],
            [6, 24, 32, 3, 2, 1],
            [6, 32, 64, 4, 2, 1],
            [6, 64, 96, 3, 1, 1],
            [6, 96, 160, 3, 2, 1],
            [6, 240, 320, 1, 1, 1], ]

        initial_channel = int(initial_channel * width_mult)
        self.layerx = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            GN(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            GN(64),
            self._build_layer(block, [4, 64, 64, 4, 1, 1], BatchNorm2d)
        )
        self.layer0 = conv_bn(3 + 3 + 2, initial_channel, 3, 2, BatchNorm2d)

        self.down = nn.Upsample(scale_factor=1 / 32, mode='nearest', align_corners=False)

        current_stride *= 2

        for i, setting in enumerate(inverted_residual_setting):
            s = setting[4]
            if current_stride == output_stride:
                inverted_residual_setting[i][4] = 1  # change stride
                rate *= s
                inverted_residual_setting[i][5] = rate
            else:
                current_stride *= s

        self.layer1 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1), BatchNorm2d(32), nn.ReLU(inplace=True))
        self.layer2 = self._build_layer(block, inverted_residual_setting[1], BatchNorm2d, downsample=True)
        self.layer3 = self._build_layer(block, inverted_residual_setting[2], BatchNorm2d, downsample=True)
        self.layer4 = self._build_layer(block, inverted_residual_setting[3], BatchNorm2d, downsample=True)
        self.layer5 = self._build_layer(block, inverted_residual_setting[4], BatchNorm2d)
        self.layer6 = self._build_layer(block, inverted_residual_setting[5], BatchNorm2d, downsample=True)
        self.layer7 = self._build_layer(block, inverted_residual_setting[6], BatchNorm2d)
        self.ppm = MSLPPM(320)
        self.gfnb = GFNB(64, 160, 80, 80, 80)
        self.dfpool = nn.Sequential(nn.Conv2d(320 + 160, 128, 1, 1, 0), nn.GroupNorm(16, 128), nn.LeakyReLU(True))
        self.uper1 = self._build_layer(blockleaky, [4, 160 + 128, 160, 3, 1, 1], BatchNorm2d)
        self.uper2 = self._build_layer(blockleaky, [4, 256, 128, 3, 1, 1], BatchNorm2d)
        self.uper3 = self._build_layer(blockleaky, [4, 160, 96, 3, 1, 1], BatchNorm2d)
        self.uper4 = self._build_layer(blockleaky, [4, 120, 64, 3, 1, 1], BatchNorm2d)
        self.uper5 = nn.Sequential(nn.Conv2d(96, 48, 3, 1, 1), BatchNorm2d(48), nn.PReLU(48),
                                   nn.Conv2d(48, 32, 3, 1, 1), BatchNorm2d(32), nn.PReLU(32))
        self.out = nn.Sequential(nn.Conv2d(32 + 6, 24, 3, 1, 1), nn.PReLU(24), nn.Conv2d(24, 12, 3, 1, 1), nn.PReLU(12),
                                 nn.Conv2d(12, 1, 3, 1, 1))
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def _build_layer(self, block, layer_setting, batch_norm, downsample=False):
        t, p, c, n, s, d = layer_setting
        input_channel = int(p * self.width_mult)
        output_channel = int(c * self.width_mult)
        layers = []
        for i in range(n):
            if i == 0:
                d0 = d
                if downsample:
                    d0 = d // 2 if d > 1 else 1
                layers.append(block(input_channel, output_channel, s, d0, expand_ratio=t, batch_norm=batch_norm))
            else:
                layers.append(block(input_channel, output_channel, 1, d, expand_ratio=t, batch_norm=batch_norm))
            input_channel = output_channel

        return nn.Sequential(*layers)

    def forward(self, img, tri, sixc):
        '''
               x : (N, 8, H, W)
               l1 : (N, 32, H/2, W/2)
               l2 : (N, 24, H/4, W/4)
               l3 : (N, 32, H/8, W/8)
               l4 : (N, 64, H/16, W/16)
               l5 : (N, 96, H/16, W/16)
               l6 : (N, 160, H/32, W/32)
               l7 : (N, 320, H/32, W/32)
        '''

        input = torch.cat((img * 2. - 1., tri, sixc), dim=1)

        l0 = self.layer0(input)
        l1 = self.layer1(l0)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        l5 = self.layer5(l4)
        l6 = self.layer6(l5)

        _, _, h, w = l6.shape
        unkown = torch.nn.functional.interpolate(tri[:, 1:2] + tri[:, 2:3], size=(h, w), mode='nearest')
        lx = self.layerx(img)
        l6_g = self.gfnb(lx, l6 * unkown)
        l6_c = torch.cat((l6, l6_g), dim=1)

        l7 = self.layer7(l6_c)
        feats = self.ppm(l7)
        l7 = torch.cat([l7, feats], 1)
        l7 = self.dfpool(l7)

        lmid = torch.cat((l6, l7), dim=1)
        lmid = self.uper1(lmid)
        lmid = torch.cat((self.up(lmid), l5), dim=1)
        lmid = self.uper2(lmid)
        lmid = torch.cat((self.up(lmid), l3), dim=1)
        lmid = self.uper3(lmid)
        lmid = torch.cat((self.up(lmid), l2), dim=1)
        lmid = self.uper4(lmid)
        lmid = torch.cat((self.up(lmid), l1), dim=1)
        lmid = self.uper5(lmid)
        lmid = self.up(lmid)
        lmid = torch.cat((lmid, img, tri), dim=1)
        lmid = self.out(lmid)
        lmid = torch.clamp(lmid, 0, 1)
        return lmid


if __name__ == '__main__':

    from fvcore.nn import FlopCountAnalysis, parameter_count_table

    a = torch.randn(1, 3, 1024, 1024)
    b = torch.randn(1, 3, 1024, 1024)
    c = torch.randn(1, 2, 1024, 1024)
    model = MobileMatting()
    model.eval()
    output = model(a, b, c)
    print(output.size())
    # flops = FlopCountAnalysis(model, (a, b, c))
    # print("GFLOPs: ", flops.total() / 1e9)
    # print("Number of parameter: ", parameter_count_table(model))
