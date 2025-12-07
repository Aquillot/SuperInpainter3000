from typing import Mapping, Any

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, n_class = 3):
        super().__init__()
        
        # Encoder
        # In the encoder, convolutional layers with the Conv2d function are used to extract features from the input image. 
        # Each block in the encoder consists of two convolutional layers followed by a max-pooling layer, with the exception of the last block which does not include a max-pooling layer.
        # -------
        # input: 572x572x3
        self.e11 = nn.Conv2d(4, 64, kernel_size=3, padding=1) # output: 570x570x64
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1) # output: 568x568x64
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 284x284x64

        # input: 284x284x64
        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # output: 282x282x128
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1) # output: 280x280x128
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 140x140x128

        # input: 140x140x128
        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1) # output: 138x138x256
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1) # output: 136x136x256
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 68x68x256

        # input: 68x68x256
        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1) # output: 66x66x512
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1) # output: 64x64x512
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 32x32x512

        # input: 32x32x512
        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1) # output: 30x30x1024
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1) # output: 28x28x1024


        # Decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Output layer
        self.outconv = nn.Conv2d(64, n_class, kernel_size=1)
    
    def forward(self, x, mask=None):
        # Encoder
        xe11 = F.relu(self.e11(x))
        xe12 = F.relu(self.e12(xe11))
        xp1 = self.pool1(xe12)

        xe21 = F.relu(self.e21(xp1))
        xe22 = F.relu(self.e22(xe21))
        xp2 = self.pool2(xe22)

        xe31 = F.relu(self.e31(xp2))
        xe32 = F.relu(self.e32(xe31))
        xp3 = self.pool3(xe32)

        xe41 = F.relu(self.e41(xp3))
        xe42 = F.relu(self.e42(xe41))
        xp4 = self.pool4(xe42)

        xe51 = F.relu(self.e51(xp4))
        xe52 = F.relu(self.e52(xe51))
        
        # Decoder
        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = F.relu(self.d11(xu11))
        xd12 = F.relu(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = F.relu(self.d21(xu22))
        xd22 = F.relu(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = F.relu(self.d31(xu33))
        xd32 = F.relu(self.d32(xd31))

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = F.relu(self.d41(xu44))
        xd42 = F.relu(self.d42(xd41))

        # Output layer
        out = self.outconv(xd42)
        return None, out

class PenNET(nn.Module):
    def __init__(self, n_class = 3, init_weights=True):
        super().__init__()

        # Encoder
        # In the encoder, convolutional layers with the Conv2d function are used to extract features from the input image.
        # Each block in the encoder consists of two convolutional layers followed by a max-pooling layer, with the exception of the last block which does not include a max-pooling layer.
        # -------
        # MaxPool = bête et simple : on réduit.
        # Conv stride 2 = on réduit + on apprend quoi garder
        # input:  H, C=4
        self.e1 = nn.Conv2d(4, 32, kernel_size=3, padding=1, stride=2) # output: H/2, C=32
        self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)
        # input: H/2, C=32
        self.e2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2) # output: H/4, C=64
        self.e3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2) # output: H/8, C=128
        self.e4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2) # output: H/16, C=256
        self.e5 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2) # output: H/32, C=512
        self.e6 = nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=2) # output:  H/64, C=512
        self.relu = nn.ReLU(inplace=True)

        # ----------------
        # ATN : Attention Transfer Network
        # ----------------
        # attention module
        self.at_conv05 = AtnConv(512, 512, ksize=1, fuse=False)
        self.at_conv04 = AtnConv(256, 256)
        self.at_conv03 = AtnConv(128, 128)
        self.at_conv02 = AtnConv(64, 64)
        self.at_conv01 = AtnConv(32, 32)

        # ----------------
        # Decoder
        # ----------------
        self.up5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.up4 = nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1)
        self.up3 = nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1)
        self.up2 = nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1)
        self.up1 = nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1)

        # ----------------
        # ToRGB (pyramide)
        # ----------------
        self.tanh = nn.Tanh()
        self.torgb5 = nn.Conv2d(1024, 3, kernel_size=1)
        self.torgb4 = nn.Conv2d(512, 3, kernel_size=1)
        self.torgb3 = nn.Conv2d(256, 3, kernel_size=1)
        self.torgb2 = nn.Conv2d(128, 3, kernel_size=1)
        self.torgb1 = nn.Conv2d(64, 3, kernel_size=1)

        # ----------------
        # Output layer (final)
        # ----------------
        self.out_conv1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.out_conv2 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)

        if init_weights:
            self.init_weights()


    def forward(self, x, mask):
        # ----------------
        # Encoder
        # ----------------
        xe1 = self.leakyrelu(self.e1(x))
        xe2 = self.leakyrelu(self.e2(xe1))
        xe3 = self.leakyrelu(self.e3(xe2))
        xe4 = self.leakyrelu(self.e4(xe3))
        xe5 = self.leakyrelu(self.e5(xe4))
        xe6 = self.relu(self.e6(xe5))

        # ----------------
        # ATN : Attention Transfer Network
        # ----------------
        xe5 = self.at_conv05(xe5, xe6, mask)
        xe4 = self.at_conv04(xe4, xe5, mask)
        xe3 = self.at_conv03(xe3, xe4, mask)
        xe2 = self.at_conv02(xe2, xe3, mask)
        xe1 = self.at_conv01(xe1, xe2, mask)
        # ----------------
        # Decoder
        # ----------------
        up5 = self.relu(self.up5(F.interpolate(xe6, scale_factor=2, mode='bilinear', align_corners=True)))
        up4 = self.relu(self.up4(F.interpolate(torch.cat([up5, xe5], dim=1), scale_factor=2, mode='bilinear', align_corners=True)))
        up3 = self.relu(self.up3(F.interpolate(torch.cat([up4, xe4], dim=1), scale_factor=2, mode='bilinear', align_corners=True)))
        up2 = self.relu(self.up2(F.interpolate(torch.cat([up3, xe3], dim=1), scale_factor=2, mode='bilinear', align_corners=True)))
        up1 = self.relu(self.up1(F.interpolate(torch.cat([up2, xe2], dim=1), scale_factor=2, mode='bilinear', align_corners=True)))

        # ----------------
        # ToRGB pyramide
        # ----------------
        img5 = self.torgb5(torch.cat([up5, xe5], dim=1))
        img5 = self.tanh(img5)
        img4 = self.torgb4(torch.cat([up4, xe4], dim=1))
        img4 = self.tanh(img4)
        img3 = self.torgb3(torch.cat([up3, xe3], dim=1))
        img3 = self.tanh(img3)
        img2 = self.torgb2(torch.cat([up2, xe2], dim=1))
        img2 = self.tanh(img2)
        img1 = self.torgb1(torch.cat([up1, xe1], dim=1))
        img1 = self.tanh(img1)

        # ----------------
        # Output layer
        # ----------------
        decoder_input = F.interpolate(torch.cat([up1, xe1], dim=1), scale_factor=2, mode='bilinear', align_corners=True)
        output = self.relu(self.out_conv1(decoder_input))  # Conv2d(cnum*2, cnum)
        output = torch.tanh(self.out_conv2(output))  # Conv2d(cnum, 3)

        return [img1, img2, img3, img4, img5], output

    def init_weights(self, gain=0.02):
        def init_func(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0.0, gain)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
        self.apply(init_func)




# Create wrapper (to respect signature (feats, pred_img) = model(inputs, masks)) using UNet architecture defined above
class InpaintGenerator(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model      # expects UNet instance with n_class=3 OR PenNET instance

    def forward(self, inputs, masks=None):
        # inputs: (masked_image RGB + mask) -> 4 channels
        feats, out = self.model(inputs, masks)           # raw output
        return feats, out

    def load_state_dict_to_model(self, state_dict: Mapping[str, Any], strict: bool = True):
        self.model.load_state_dict(state_dict, strict=strict)


# ---------- helper spectral norm ----------
def use_spectral_norm(layer, use_sn=True):
    if use_sn:
        return nn.utils.spectral_norm(layer)
    return layer

# ---------- Discriminator (essentiel) ----------
class Discriminator(nn.Module):
    """
    Simple Patch discriminator: downsample convs -> final 1-channel conv (score map).
    in_channels: 3 (RGB composite image)
    """
    def __init__(self, in_channels=3, cnum=64, use_sn=True):
        super().__init__()
        self.net = nn.Sequential(
            use_spectral_norm(nn.Conv2d(in_channels, cnum, kernel_size=5, stride=2, padding=2, bias=False), use_sn),
            nn.LeakyReLU(0.2, inplace=True),

            use_spectral_norm(nn.Conv2d(cnum, cnum*2, kernel_size=5, stride=2, padding=2, bias=False), use_sn),
            nn.LeakyReLU(0.2, inplace=True),

            use_spectral_norm(nn.Conv2d(cnum*2, cnum*4, kernel_size=5, stride=2, padding=2, bias=False), use_sn),
            nn.LeakyReLU(0.2, inplace=True),

            use_spectral_norm(nn.Conv2d(cnum*4, cnum*8, kernel_size=5, stride=1, padding=2, bias=False), use_sn),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.classifier = nn.Conv2d(cnum*8, 1, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        x = self.net(x)
        x = self.classifier(x)
        return x                 # shape (N,1,H,W)


#################################################################################
# ########################  Contextual Attention  #################################
# #################################################################################
'''
implementation of attention module
most codes are borrowed from:
1. https://github.com/WonwoongCho/Generative-Inpainting-pytorch/pull/5/commits/9c16537cd123b74453a28cd4e25d3db0505e5881
2. https://github.com/DAA233/generative-inpainting-pytorch/blob/master/model/networks.py
'''


class AtnConv(nn.Module):
    def __init__(self, input_channels=128, output_channels=64, groups=4, ksize=3, stride=1, rate=2, softmax_scale=10.,
                 fuse=True, rates=[1, 2, 4, 8]):
        super(AtnConv, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.rate = rate
        self.softmax_scale = softmax_scale
        self.groups = groups
        self.fuse = fuse
        if self.fuse:
            for i in range(groups):
                self.__setattr__('conv{}'.format(str(i).zfill(2)), nn.Sequential(
                    nn.Conv2d(input_channels, output_channels // groups, kernel_size=3, dilation=rates[i],
                              padding=rates[i]),
                    nn.ReLU(inplace=True))
                                 )

    def forward(self, x1, x2, mask=None):
        """ Attention Transfer Network (ATN) is first proposed in
            Learning Pyramid Context-Encoder Networks for High-Quality Image Inpainting. Yanhong Zeng et al. In CVPR 2019.
          inspired by
            Generative Image Inpainting with Contextual Attention, Yu et al. In CVPR 2018.
        Args:
            x1: low-level feature maps with larger resolution.
            x2: high-level feature maps with smaller resolution.
            mask: Input mask, 1 indicates holes.
            ksize: Kernel size for contextual attention.
            stride: Stride for extracting patches from b.
            rate: Dilation for matching.
            softmax_scale: Scaled softmax for attention.
            training: Indicating if current graph is training or inference.
        Returns:
            torch.Tensor, reconstructed feature map.
        """
        # get shapes
        x1s = list(x1.size())
        x2s = list(x2.size())

        # extract patches from low-level feature maps x1 with stride and rate
        kernel = 2 * self.rate
        raw_w = extract_patches(x1, kernel=kernel, stride=self.rate * self.stride)
        raw_w = raw_w.contiguous().view(x1s[0], -1, x1s[1], kernel, kernel)  # B*HW*C*K*K
        # split tensors by batch dimension; tuple is returned
        raw_w_groups = torch.split(raw_w, 1, dim=0)

        # split high-level feature maps x2 for matching
        f_groups = torch.split(x2, 1, dim=0)
        # extract patches from x2 as weights of filter
        w = extract_patches(x2, kernel=self.ksize, stride=self.stride)
        w = w.contiguous().view(x2s[0], -1, x2s[1], self.ksize, self.ksize)  # B*HW*C*K*K
        w_groups = torch.split(w, 1, dim=0)

        # process mask
        if mask is not None:
            mask = F.interpolate(mask, size=x2s[2:4], mode='bilinear', align_corners=True)
        else:
            mask = torch.zeros([1, 1, x2s[2], x2s[3]])
            if torch.cuda.is_available():
                mask = mask.cuda()
        # extract patches from masks to mask out hole-patches for matching
        m = extract_patches(mask, kernel=self.ksize, stride=self.stride)
        m = m.contiguous().view(x2s[0], -1, 1, self.ksize, self.ksize)  # B*HW*1*K*K
        m = m.mean([2, 3, 4]).unsqueeze(-1).unsqueeze(-1)
        mm = m.eq(0.).float()  # (B, HW, 1, 1)
        mm_groups = torch.split(mm, 1, dim=0)

        y = []
        scale = self.softmax_scale
        padding = 0 if self.ksize == 1 else 1
        for xi, wi, raw_wi, mi in zip(f_groups, w_groups, raw_w_groups, mm_groups):
            '''
            O => output channel as a conv filter
            I => input channel as a conv filter
            xi : separated tensor along batch dimension of front; 
            wi : separated patch tensor along batch dimension of back; 
            raw_wi : separated tensor along batch dimension of back; 
            '''
            # matching based on cosine-similarity
            wi = wi[0]
            escape_NaN = torch.FloatTensor([1e-4])
            if torch.cuda.is_available():
                escape_NaN = escape_NaN.cuda()
            # normalize
            wi_normed = wi / torch.max(torch.sqrt((wi * wi).sum([1, 2, 3], keepdim=True)), escape_NaN)
            yi = F.conv2d(xi, wi_normed, stride=1, padding=padding)
            yi = yi.contiguous().view(1, x2s[2] // self.stride * x2s[3] // self.stride, x2s[2], x2s[3])

            # apply softmax to obtain
            yi = yi * mi
            yi = F.softmax(yi * scale, dim=1)
            yi = yi * mi
            yi = yi.clamp(min=1e-8)

            # attending
            wi_center = raw_wi[0]
            yi = F.conv_transpose2d(yi, wi_center, stride=self.rate, padding=1) / 4.
            y.append(yi)
        y = torch.cat(y, dim=0)
        y.contiguous().view(x1s)
        # adjust after filling
        if self.fuse:
            tmp = []
            for i in range(self.groups):
                tmp.append(self.__getattr__('conv{}'.format(str(i).zfill(2)))(y))
            y = torch.cat(tmp, dim=1)
        return y

# extract patches
def extract_patches(x, kernel=3, stride=1):
  if kernel != 1:
    x = nn.ZeroPad2d(1)(x)
  x = x.permute(0, 2, 3, 1)
  all_patches = x.unfold(1, kernel, stride).unfold(2, kernel, stride)
  return all_patches