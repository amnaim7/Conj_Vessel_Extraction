from typing import List, Tuple
import torch
import torch.nn as nn

import torch.nn.functional as F
from functools import partial

from src.utils import config
from src.models.EncoderDecoder import ConvBlock, EncoderLayer, DecoderLayer

DEVICE = config.get_device()

###########################################

nonlinearity = partial(F.relu,inplace=True)

###########################################

class DilTransAttUNet(nn.Module):
    def __init__(self, channels: Tuple[int], num_heads = 4, is_residual: bool = False, bias = False) -> None:
        super(DilTransAttUNet, self).__init__()

        self.channels = channels
        self.pos_encoding = PositionalEncoding()
        self.encode = nn.ModuleList([EncoderLayer(channels[i], channels[i + 1], is_residual, bias) for i in range(len(channels) - 2)])
        self.bottle_neck = ConvBlock(channels[-2], channels[-1], is_residual, bias)
        self.dilationcenter = Dblock(channels[-1])
        self.mhsa = MultiHeadSelfAttention(channels[-1], num_heads, bias)
        self.mhca = nn.ModuleList([MultiHeadCrossAttention(channels[i], num_heads, channels[i], channels[i + 1], bias) for i in reversed(range(1, len(channels) - 1))])
        self.decode = nn.ModuleList([DecoderLayer(channels[i + 1], channels[i], is_residual, bias) for i in reversed(range(1, len(channels) - 1))])
        self.output = nn.Conv2d(channels[1], 1, 1)

        self.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_x_list: List[torch.Tensor] = []
        for i in range(len(self.channels) - 2):
            skip_x, x = self.encode[i](x)
            skip_x_list.append(skip_x)

        x = self.bottle_neck(x)
        x = self.dilationcenter(x)
        x = self.pos_encoding(x)
        x = self.mhsa(x)

        for i, skip_x in enumerate(reversed(skip_x_list)):
            x = self.pos_encoding(x)
            skip_x = self.pos_encoding(skip_x)
            skip_x = self.mhca[i](skip_x, x)
            x = self.decode[i](skip_x, x)

        return self.output(x)
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.1)

########################################

class Dblock(nn.Module):
    def __init__(self,channel):
        super(Dblock, self).__init__()

        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        # self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        # self.dilate6 = nn.Conv2d(channel, channel, kernel_size=3, dilation=32, padding=32)
        self.BN = nn.BatchNorm2d(channel)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        # dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        # dilate6_out = nonlinearity(self.dilate6(dilate5_out))
        out = dilate1_out + dilate2_out + dilate3_out + dilate4_out
        out = self.BN(out)
        return out

########################################     

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, bias=False) -> None:
        super(MultiHeadSelfAttention, self).__init__()

        self.mha = nn.MultiheadAttention(embed_dim, num_heads, bias=bias, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        x = x.permute(0, 2, 3, 1).view((b, h * w, c))
        x, _ = self.mha(x, x, x, need_weights=False)
        return x.view((b, h, w, c)).permute(0, 3, 1, 2)

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, channel_S: int, channel_Y: int, bias=False) -> None:
        super(MultiHeadCrossAttention, self).__init__()

        self.conv_S = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(channel_S, channel_S, 1, bias=bias),
            nn.BatchNorm2d(channel_S),
            nn.ReLU()
        )

        self.conv_Y = nn.Sequential(
            nn.Conv2d(channel_Y, channel_S, 1, bias=bias),
            nn.BatchNorm2d(channel_S),
            nn.ReLU()
        )

        self.mha = nn.MultiheadAttention(embed_dim, num_heads, bias=bias, batch_first=True)

        self.upsample = nn.Sequential(
            nn.Conv2d(channel_S, channel_S, 1, bias=bias).apply(lambda m: nn.init.xavier_uniform_(m.weight.data)),
            nn.BatchNorm2d(channel_S),
            nn.Sigmoid(),
            nn.ConvTranspose2d(channel_S, channel_S, 2, 2, bias=bias)
        )

    def forward(self, s: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        s_enc = s
        # print("")
        # print("S_enc size: ", s_enc.size())
        s = self.conv_S(s)
        y = self.conv_Y(y)

        b, c, h, w = s.size()
        # print("S size: ", s.size())
        s = s.permute(0, 2, 3, 1).view((b, h * w, c))

        b, c, h, w = y.size()
        # print("Y size: ", y.size())
        y = y.permute(0, 2, 3, 1).view((b, h * w, c))

        y, _ = self.mha(y, y, s, need_weights=False)
        y = y.view((b, h, w, c)).permute(0, 3, 1, 2)
        
        y = self.upsample(y)
        # print("y matrix size: ", y.size())

        return torch.mul(y, s_enc)

class PositionalEncoding(nn.Module):
    def __init__(self) -> None:
        super(PositionalEncoding, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        pos_encoding = self.positional_encoding(h * w, c)
        pos_encoding = pos_encoding.permute(1, 0).unsqueeze(0).repeat(b, 1, 1)
        x = x.view((b, c, h * w)) + pos_encoding
        return x.view((b, c, h, w))

    def positional_encoding(self, length: int, depth: int) -> torch.Tensor:
        depth = depth / 2

        positions = torch.arange(length, dtype=config.DTYPE, device=DEVICE)
        depths = torch.arange(depth, dtype=config.DTYPE, device=DEVICE) / depth

        angle_rates = 1 / (10000**depths)
        angle_rads = torch.einsum('i,j->ij', positions, angle_rates)

        pos_encoding = torch.cat((torch.sin(angle_rads), torch.cos(angle_rads)), dim=-1)

        return pos_encoding
