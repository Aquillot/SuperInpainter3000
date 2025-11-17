import torch
import torch.nn as nn
from torchvision import models
from torch.nn.functional import relu

class AtnConv(nn.Module):
  def __init__(self, input_channels=128, output_channels=64, groups=4, ksize=3, stride=1, rate=2, softmax_scale=10., fuse=True, rates=[1,2,4,8]):
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
          nn.Conv2d(input_channels, output_channels//groups, kernel_size=3, dilation=rates[i], padding=rates[i]),
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
    kernel = 2*self.rate
    raw_w = extract_patches(x1, kernel=kernel, stride=self.rate*self.stride)
    raw_w = raw_w.contiguous().view(x1s[0], -1, x1s[1], kernel, kernel) # B*HW*C*K*K 
    # split tensors by batch dimension; tuple is returned
    raw_w_groups = torch.split(raw_w, 1, dim=0) 

    # split high-level feature maps x2 for matching 
    f_groups = torch.split(x2, 1, dim=0) 
    # extract patches from x2 as weights of filter
    w = extract_patches(x2, kernel=self.ksize, stride=self.stride)
    w = w.contiguous().view(x2s[0], -1, x2s[1], self.ksize, self.ksize) # B*HW*C*K*K
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
    m = m.mean([2,3,4]).unsqueeze(-1).unsqueeze(-1)
    mm = m.eq(0.).float() # (B, HW, 1, 1)       
    mm_groups = torch.split(mm, 1, dim=0)

    y = []
    scale = self.softmax_scale
    padding = 0 if self.ksize==1 else 1
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
      wi_normed = wi / torch.max(torch.sqrt((wi*wi).sum([1,2,3],keepdim=True)), escape_NaN)
      yi = F.conv2d(xi, wi_normed, stride=1, padding=padding)
      yi = yi.contiguous().view(1, x2s[2]//self.stride*x2s[3]//self.stride, x2s[2], x2s[3]) 

      # apply softmax to obtain 
      yi = yi * mi 
      yi = F.softmax(yi*scale, dim=1)
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


class EncodeLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, last_layer: bool = False):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, image):
        return self.conv(image)

class DecodeLayer(nn.Module):
    def __init__(self, size_in: int, atn_size: int, fuse=True):
        super().__init__()

        self.at_conv = AtnConv(atn_size, atn_size, ksize=1, fuse)

        self.up_conv05 = nn.Sequential(
        nn.Conv2d(size_in, atn_size, kernel_size=3, stride=1, padding=1),
          nn.ReLU(inplace=True))



class InpaintingModel(nn.Module):
    def __init__(self, init_weights=True):#1046
        super(InpaintGenerator, self).__init__()
        cnum = 32

        self.dw_conv01 = nn.Sequential(
        nn.Conv2d(4, cnum, kernel_size=3, stride=2, padding=1),
        nn.LeakyReLU(0.2, inplace=True))
        self.dw_conv02 = nn.Sequential(
        nn.Conv2d(cnum, cnum*2, kernel_size=3, stride=2, padding=1),
        nn.LeakyReLU(0.2, inplace=True))
        self.dw_conv03 = nn.Sequential(
        nn.Conv2d(cnum*2, cnum*4, kernel_size=3, stride=2, padding=1),
        nn.LeakyReLU(0.2, inplace=True))
        self.dw_conv04 = nn.Sequential(
        nn.Conv2d(cnum*4, cnum*8, kernel_size=3, stride=2, padding=1),
        nn.LeakyReLU(0.2, inplace=True))
        self.dw_conv05 = nn.Sequential(
        nn.Conv2d(cnum*8, cnum*16, kernel_size=3, stride=2, padding=1),
        nn.LeakyReLU(0.2, inplace=True))
        self.dw_conv06 = nn.Sequential(
        nn.Conv2d(cnum*16, cnum*16, kernel_size=3, stride=2, padding=1),
        nn.ReLU(inplace=True))

        # attention module
        self.at_conv05 = AtnConv(cnum*16, cnum*16, ksize=1, fuse=False)
        self.at_conv04 = AtnConv(cnum*8, cnum*8)
        self.at_conv03 = AtnConv(cnum*4, cnum*4)
        self.at_conv02 = AtnConv(cnum*2, cnum*2)
        self.at_conv01 = AtnConv(cnum, cnum)

        # decoder
        self.up_conv05 = nn.Sequential(
        nn.Conv2d(cnum*16, cnum*16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True))
        self.up_conv04 = nn.Sequential(
        nn.Conv2d(cnum*32, cnum*8, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True))
        self.up_conv03 = nn.Sequential(
        nn.Conv2d(cnum*16, cnum*4, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True))
        self.up_conv02 = nn.Sequential(
        nn.Conv2d(cnum*8, cnum*2, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True))
        self.up_conv01 = nn.Sequential(
        nn.Conv2d(cnum*4, cnum, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True))

        # torgb
        self.torgb5 = nn.Sequential(
        nn.Conv2d(cnum*32, 3, kernel_size=1, stride=1, padding=0),
        nn.Tanh())
        self.torgb4 = nn.Sequential(
        nn.Conv2d(cnum*16, 3, kernel_size=1, stride=1, padding=0),
        nn.Tanh())
        self.torgb3 = nn.Sequential(
        nn.Conv2d(cnum*8, 3, kernel_size=1, stride=1, padding=0),
        nn.Tanh())
        self.torgb2 = nn.Sequential(
        nn.Conv2d(cnum*4, 3, kernel_size=1, stride=1, padding=0),
        nn.Tanh())
        self.torgb1 = nn.Sequential(
        nn.Conv2d(cnum*2, 3, kernel_size=1, stride=1, padding=0),
        nn.Tanh())

        self.decoder = nn.Sequential(
        nn.Conv2d(cnum*2, cnum, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(cnum, 3, kernel_size=3, stride=1, padding=1),
        nn.Tanh()
        )

        if init_weights:
        self.init_weights()

    def forward(self, img, mask):
        x = img
        # encoder 
        x1 = self.dw_conv01(x)
        x2 = self.dw_conv02(x1)
        x3 = self.dw_conv03(x2)
        x4 = self.dw_conv04(x3)
        x5 = self.dw_conv05(x4)
        x6 = self.dw_conv06(x5)
        # attention 
        x5 = self.at_conv05(x5, x6, mask)
        x4 = self.at_conv04(x4, x5, mask)
        x3 = self.at_conv03(x3, x4, mask)
        x2 = self.at_conv02(x2, x3, mask)
        x1 = self.at_conv01(x1, x2, mask)
        # decoder
        upx5 = self.up_conv05(F.interpolate(x6, scale_factor=2, mode='bilinear', align_corners=True))
        upx4 = self.up_conv04(F.interpolate(torch.cat([upx5, x5], dim=1), scale_factor=2, mode='bilinear', align_corners=True))
        upx3 = self.up_conv03(F.interpolate(torch.cat([upx4, x4], dim=1), scale_factor=2, mode='bilinear', align_corners=True))
        upx2 = self.up_conv02(F.interpolate(torch.cat([upx3, x3], dim=1), scale_factor=2, mode='bilinear', align_corners=True))
        upx1 = self.up_conv01(F.interpolate(torch.cat([upx2, x2], dim=1), scale_factor=2, mode='bilinear', align_corners=True))
        # torgb
        img5 = self.torgb5(torch.cat([upx5, x5], dim=1))
        img4 = self.torgb4(torch.cat([upx4, x4], dim=1))
        img3 = self.torgb3(torch.cat([upx3, x3], dim=1))
        img2 = self.torgb2(torch.cat([upx2, x2], dim=1))
        img1 = self.torgb1(torch.cat([upx1, x1], dim=1))
        # output 
        output = self.decoder(F.interpolate(torch.cat([upx1, x1], dim=1), scale_factor=2, mode='bilinear', align_corners=True))
        pyramid_imgs = [img1, img2, img3, img4, img5]
        return  pyramid_imgs, output

def extract_patches(x, kernel=3, stride=1):
  if kernel != 1:
    x = nn.ZeroPad2d(1)(x)
  x = x.permute(0, 2, 3, 1)
  all_patches = x.unfold(1, kernel, stride).unfold(2, kernel, stride)
  return all_patches
