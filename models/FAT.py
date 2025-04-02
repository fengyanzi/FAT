import torch
import torch.nn as nn
import math
import torch.nn.functional as f
#from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.layers import DropPath, to_2tuple, trunc_normal_
from models.FSSA import FourierSparseSelfAttention

def printtensorsize(x,text="tensorsize:"):
    if __name__ == '__main__':
        print(text,x.shape)

class ChannelAttention(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(ChannelAttention, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv1(y)
        return x * self.sigmoid(y)



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):

        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'



class SwinTransformer(nn.Module):

    def __init__(self, dim,  num_heads=4, window_size=4, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU,input_resolution=(240,240),fusion=False):
        super().__init__()
        self.fusion = fusion
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=to_2tuple(window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if shift_size > 0:
            attn_mask = self.calculate_mask(input_resolution)
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        self.ca = ChannelAttention(dim)
        self.sa = SpatialAttention(kernel_size=3)
        #self.OSA = OSA_Block(channel_num=dim, bias=True, ffn_bias=True, window_size=8, with_pe=False, dropout=0.0)
        self.relu=nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=-1)
        self.pool =nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.FSSA= FourierSparseSelfAttention(n_hashes=2,embed_dim=dim)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        shortcut = x
        if self.fusion:
            x1=x
            x1 = x1.reshape(B, C, H, W)
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        if self.fusion:
            x1=self.FSSA(x1)

        if self.fusion:
            x1=x1.view(B, H * W, C)

        x = x.view(B, H * W, C)
        if self.fusion:
            x = shortcut + self.drop_path(x + x1)  #
        else :
            x = shortcut + self.drop_path(x)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self,embed_dim=8):
        super().__init__()
        self.embed_dim = embed_dim
    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

class SRTBlock(nn.Module):
    def __init__(self,dim,ST_num,num_heads,window_size):
        super().__init__()
        self.dim=dim
        self.ST_num=ST_num
        self.num_heads=num_heads
        self.window_size=window_size
        self.STL=nn.ModuleList(SwinTransformer(dim=self.dim,num_heads=self.num_heads,window_size=self.window_size) for i in range(self.ST_num))
        self.MAT=SwinTransformer(dim=self.dim,num_heads=self.num_heads,window_size=self.window_size*2,fusion=True)
        self.embed=PatchEmbed()
        self.unembed=PatchUnEmbed(embed_dim=self.dim)
        self.conv=nn.Conv2d(self.dim, self.dim, kernel_size=3,stride=1,padding=1,bias=True)
        self.ca = ChannelAttention(self.dim)
        self.sa = SpatialAttention(kernel_size=3)
        self.fssa=FourierSparseSelfAttention(n_hashes=4, embed_dim=self.dim)

    def forward(self,x,x_size):
        x=self.embed(x)
        printtensorsize(x,"embed")
        for i in range(self.ST_num):
            x = self.STL[i](x,x_size)
        x=self.MAT(x,x_size)
        x=self.unembed(x,x_size)
        x=self.conv(x)
        return self.fssa(x)


class Upsample(nn.Module):
    def __init__(self, scale=4, up_dim=32,out_ch=1,num_feat=128):
        super().__init__()
        if scale == 4:
            self.upsample = nn.Sequential(
            nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1),
            nn.PixelShuffle(2))
        elif scale == 3:
            self.upsample = nn.Sequential(
            nn.Conv2d(up_dim, 9 * up_dim, 3, 1, 1),
            nn.PixelShuffle(3))
        else:
            self.upsample = nn.Sequential(
                nn.Conv2d(up_dim, num_feat, 3, 1, 1),
                nn.LeakyReLU(inplace=True),

                nn.Conv2d(num_feat, (scale**2) * num_feat, 3, 1, 1),
                nn.PixelShuffle(scale))
        self.conv_last = nn.Conv2d(num_feat, out_ch, 3, 1, 1)
    def forward(self, x):
        return self.conv_last(self.upsample(x))


class UAFB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UAFB, self).__init__()

        self.ca1 = ChannelAttention(in_channels)
        self.ca2 = ChannelAttention(in_channels)

        self.sa1 = SpatialAttention(kernel_size=3)
        self.sa2 = SpatialAttention(kernel_size=3)

        # Downsample
        self.down_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.down_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.pool = nn.MaxPool2d(2)

        # Upsample
        self.up_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.up_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.up_transpose = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        self.end_conv = nn.Conv2d(out_channels*2, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # Downsampling
        x1 = self.down_conv(x)
        x1 = self.down_relu(x1)
        x1_pooled = self.pool(x1)

        # Upsampling
        x2 = self.up_conv(x1_pooled)
        x2 = self.up_relu(x2)
        x2 = self.sa1(x2)

        x2_up = self.up_transpose(x2)
        x1=self.sa2(x1)
        x2_up = self.ca1(torch.cat([x2_up, x1], dim=1))

        return self.end_conv(x2_up)+x

class FAT(nn.Module):
    def __init__(self):
        super(FAT, self).__init__()
        self.in_ch=3
        self.out_ch=3
        self.dim = 128
        self.SRTB_num = 6
        self.STL_num=3
        self.scale=4
        self.inputsize = (0, 0)
        self.ca = ChannelAttention(self.dim)
        self.window_size_list = [6,6,6,6,6,6]
        self.conv0=nn.Conv2d(self.in_ch, self.in_ch, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(self.in_ch, self.dim, kernel_size=1)
        self.conv3 = nn.Conv2d(self.out_ch, self.out_ch, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(self.dim, self.dim, kernel_size=3, stride=1, padding=1)

        self.enbed=PatchEmbed()
        self.debed=PatchUnEmbed(embed_dim=self.dim)
        self.SRTB_list = nn.ModuleList(SRTBlock(dim=self.dim,ST_num=self.STL_num,num_heads=8,window_size=self.window_size_list[i]) for i in range(self.SRTB_num))

        self.upsample=Upsample(scale=self.scale, up_dim=self.dim,out_ch=self.out_ch)

        self.apply(self._init_weights)
        self.uafb=UAFB(self.dim,self.dim)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, x):
        self.inputsize = (x.shape[2], x.shape[3])
        h0=self.conv1(self.conv0(x))
        h2=self.uafb(h0)
        h3=h2
        SRTB_reslist = []
        SRTB_reslist.append(h3)
        h4=self.SRTB_list[0](h3,self.inputsize)
        SRTB_reslist.append(h4)
        for i in range(1,self.SRTB_num):
            if i<=4:

                h4 = self.SRTB_list[i](SRTB_reslist[i],self.inputsize)
            else:
                h4 = self.SRTB_list[i](SRTB_reslist[i], self.inputsize)

            SRTB_reslist.append(h4)

        h4fin=SRTB_reslist[self.SRTB_num]
        h6 = self.upsample(self.conv4(h4fin)+h0)
        return h6

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
    model = FAT().to(device)
    print(model)
    inputs = torch.ones([1, 3, 120, 120])
    inputs = inputs.to(device)
    with torch.no_grad():
        outputs = model(inputs)
    print(outputs.shape)
    print(outputs)
    # from torchsummary import summary
    # normalize_transform = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    # summary(model, input_size=(3, 120, 120))
    #
    # input_size = (1, 3, 120, 120)
    # import thop
    #
    # flops, params = thop.profile(model, inputs=(torch.randn(input_size).to(device),))
    # print(f"FLOPs: {flops / 1e9} G")  # 打印计算量（以十亿次浮点运算为单位）
    # print(f"Params: {params / 1e6} M")

