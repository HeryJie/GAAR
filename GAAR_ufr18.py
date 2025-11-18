import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from thop import profile
from torch.distributions import Normal
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import timm
from geoseg.models.Hyperbolic_Attention import HyperbolicAttention
import geoopt
import torchvision.models as models
from geoseg.models.Transformer_Eu import EU_MHSA
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6(inplace=False)
        )

class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU6(inplace=False)
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

class WF(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(WF, self).__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        return x


class FeatureRefinementHead(nn.Module):
    def __init__(self, in_channels=64, decode_channels=64):
        super().__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = 1e-8
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

        self.pa = nn.Sequential(nn.Conv2d(decode_channels, decode_channels, kernel_size=3, padding=1, groups=decode_channels),
                                nn.Sigmoid())
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                Conv(decode_channels, decode_channels//16, kernel_size=1),
                                nn.ReLU6(),
                                Conv(decode_channels//16, decode_channels, kernel_size=1),
                                nn.Sigmoid())

        self.shortcut = ConvBN(decode_channels, decode_channels, kernel_size=1)
        self.proj = SeparableConvBN(decode_channels, decode_channels, kernel_size=3)
        self.act = nn.ReLU6()

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        shortcut = self.shortcut(x)
        pa = self.pa(x) * x
        ca = self.ca(x) * x
        x = pa + ca
        x = self.proj(x) + shortcut
        x = self.act(x)
        return x

class AuxHead(nn.Module):

    def __init__(self, in_channels=64, num_classes=8):
        super().__init__()
        self.conv = ConvBNReLU(in_channels, in_channels)
        self.drop = nn.Dropout(0.1)
        self.conv_out = Conv(in_channels, num_classes, kernel_size=1)

    def forward(self, x, h, w):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
        return feat
########################################################################################################################
class GeometricAwareChannelRouter(nn.Module):
    def __init__(self,
                 in_channels=512,
                 hidden_dim=128,
                 init_temp=1.0,
                 min_temp=0.1,
                 anneal_rate=0.00003,
                 lambda_sparse=0.005
                 ):
        super().__init__()
        self.channel_analyzer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, in_channels)
        )
        self.manifold = geoopt.PoincareBall(c=1.0)
        self.temp = init_temp
        self.min_temp = min_temp
        self.anneal_rate = anneal_rate

        self.lambda_sparse = lambda_sparse

        self._initialize_weights()
        self.HDN = HyperbolicAttention(in_channels, in_channels, manifold=self.manifold,num_heads=8)
        self.EuN = EU_MHSA(in_channels, in_channels, num_heads=8)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                nn.init.constant_(m.bias, 0)

    def forward(self, x, current_step, training=False):
        B, C, H, W = x.shape

        g = self.channel_analyzer(x)  # [B, 512, 32 32] -> [B, C]
        g = g.squeeze(-1)  # [B, C]

        p = torch.sigmoid(g)  # [B, C]

        if training:
            noise = torch.rand_like(p).log_().neg_().log_().neg_()
            noisy_logits = (torch.log(p + 1e-8) + noise) / self.temp
            mask = torch.sigmoid(noisy_logits)
        else:
            mask = (p > 0.5).float()

        hyp_feat = x * mask.unsqueeze(-1).unsqueeze(-1)  # [B, C, H, W]
        euc_feat = x * (1 - mask.unsqueeze(-1).unsqueeze(-1))

        hyp_out = self.manifold.logmap0(self.HDN(self.manifold.expmap0(hyp_feat)))
        euc_out = self.EuN(euc_feat)

        fused_out = hyp_out + euc_out
        route_loss = self.lambda_sparse * torch.mean(p * (1 - p))
        if training:
            exp_term = torch.exp(-self.anneal_rate * torch.tensor(current_step, dtype=torch.float32))
            self.temp = max(
                self.temp * exp_term.item(),
                self.min_temp
            )
            return fused_out, route_loss
        else: return fused_out
########################################################################################################################
class FF(nn.Module):
    def __init__(self, in_channels, decode_channels=64):
        super().__init__()
        self.weights = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32), requires_grad=True)
        self.eps = 1e-6
        self.pre_conv = ConvBN(in_channels, decode_channels,
                               kernel_size=1) if in_channels != decode_channels else nn.Identity()
        self.post_conv = nn.Sequential(
            ConvBNReLU(decode_channels, decode_channels, 3),
            nn.Dropout2d(p=0.1, inplace=True)
        )

    def forward(self, easy, decoupled):
        easy = self.pre_conv(easy)
        decoupled = self.pre_conv(decoupled)
        weights = F.relu(self.weights)
        fused = (
                        weights[0] * easy +
                        weights[1] * decoupled
                ) / (torch.sum(weights) + self.eps)
        fused = self.post_conv(fused)
        return fused
########################################################################################################################
class Decoder(nn.Module):
    def __init__(self,
                 encoder_channels=(64, 128, 256, 512),
                 decode_channels=64,
                 dropout=0.1,
                 window_size=8,
                 num_classes=7):
        super(Decoder, self).__init__()
        self.GACR = GeometricAwareChannelRouter(in_channels=512)
        self.pre_conv = ConvBN(512, decode_channels, kernel_size=1)
        self.b4 = EU_MHSA(decode_channels, decode_channels, num_heads=8)
        self.b3 = EU_MHSA(decode_channels, decode_channels, num_heads=8)
        self.p3 = WF(256, decode_channels)

        self.b2 = EU_MHSA(decode_channels, decode_channels, num_heads=8)
        self.p2 = WF(128, decode_channels)

        if self.training:
            self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)
            self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
            self.aux_head = AuxHead(decode_channels, num_classes)
        self.p1 = FeatureRefinementHead(64, decode_channels)

        self.segmentation_head = nn.Sequential(ConvBNReLU(decode_channels, decode_channels),
                                               nn.Dropout2d(p=dropout, inplace=True),
                                               Conv(decode_channels, num_classes, kernel_size=1))
        self.init_weight()
        self.fuse_weights = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32), requires_grad=True)

    def forward(self, step, res1, res2, res3, res4, h, w):
        if self.training:
            ########################################
            res4, r_loss = self.GACR(res4, step, training=True)
            ########################################
            x = self.b4(self.pre_conv(res4))
            h4 = self.up4(x)

            x = self.p3(x, res3)
            x = self.b3(x)
            h3 = self.up3(x)

            x = self.p2(x, res2)
            x = self.b2(x)
            h2 = x
            x = self.p1(x, res1)

            x = self.segmentation_head(x)
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

            ah = h4 + h3 + h2
            ah = self.aux_head(ah, h, w)

            return x, ah, r_loss
        else:
            res4 = self.GACR(res4, step, training=False)
            x = self.b4(self.pre_conv(res4))
            x = self.p3(x, res3)
            x = self.b3(x)

            x = self.p2(x, res2)
            x = self.b2(x)

            x = self.p1(x, res1)

            x = self.segmentation_head(x)
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

            return x
    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class UNetFormer(nn.Module):
    def __init__(self,
                 decode_channels=64,
                 dropout=0.1,
                 backbone_name='resnet18',
                 pretrained=True,
                 window_size=8,
                 num_classes=7
                 ):
        super().__init__()
        if backbone_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
            self.layer1 = self.backbone[4]
            self.layer2 = self.backbone[5]
            self.layer3 = self.backbone[6]
            self.layer4 = self.backbone[7]
        else:
            raise NotImplementedError(f"Backbone {backbone_name} not supported")

        self.decoder = Decoder(num_classes=num_classes)

    def forward(self, x, step):
        h, w = x.size()[-2:]
        x = self.backbone[0](x) #1*3*512*512 -> 1*64*256*256
        x = self.backbone[1](x)
        x = self.backbone[2](x)
        x = self.backbone[3](x) # 1*64*256*256 -> 1*64*128*128

        res1 = self.layer1(x) # 1*64*128*128
        res2 = self.layer2(res1) # 1*128*64*64
        res3 = self.layer3(res2) # 1*256*32*32
        res4 = self.layer4(res3) # 1*512*16*16
        if self.training:
            x, ah, r_loss = self.decoder(step, res1, res2, res3, res4, h, w)
            return x, ah, r_loss
        else:
            x = self.decoder(step, res1, res2, res3, res4, h, w)
            return x
