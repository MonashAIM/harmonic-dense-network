import torch
from torch import nn
from torch.nn import functional as F


class InvertedTransition(nn.Module):
    def __init__(self, in_channel, out_channel, adapt=False, *args, **kwargs):
        super().__init__()
        self.maxpool = nn.MaxPool3d(kernel_size=2)
        if adapt:
            out_h = (in_channel.shape[0] - 2) + 1
            out_w = (in_channel.shape[0] - 2) + 1
            self.avgpool = nn.AdaptiveAvgPool3d(
                output_size=(out_h, out_w)
            )  # Not really sure
        else:
            self.avgpool = nn.AvgPool3d(kernel_size=2)
        self.conv = nn.Conv3d(
            in_channels=in_channel, out_channels=out_channel, kernel_size=1
        )

    def forward(self, x):
        res_max = self.maxpool(x)
        res_avg = self.avgpool(x)

        out = torch.cat((res_max, res_avg))  # Not sure bout dim for torch.cat()
        return self.conv(out)


class DWConvTransition(nn.Sequential):
    def __init__(self, in_channels, kernel=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.add_module(
            "dwconv",
            nn.Conv3d(
                in_channels,
                in_channels,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                groups=in_channels,
                bias=bias,
            ),
        )
        self.add_module("norm", nn.BatchNorm3d(in_channels))

    def forward(self, x):
        return super().forward(x)


class Conv(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        act="relu",
        kernel=3,
        stride=1,
        bias=False,
        *args,
        **kwargs,
    ):
        super().__init__()

        # if isinstance(kernel, int):
        #     kernel = (kernel, kernel, kernel)  # Make kernel a tuple
        # if isinstance(stride, int):
        #     stride = (stride, stride, stride)  # Make stride a tuple
        # if isinstance(stride, tuple) and len(stride) == 1:
        #     stride = (stride[0], stride[0], stride[0])  

        self.add_module(
            name="conv",
            module=nn.Conv3d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=kernel,
                stride=stride,
                padding=kernel // 2,
                bias=bias,
            )
        )

        self.add_module(name="bn", module=nn.BatchNorm3d(num_features=out_channel))

        if act == "relu":
            self.add_module(name="act", module=nn.ReLU())
        elif act == "leaky":
            self.add_module(name="act", module=nn.LeakyReLU())
        elif act == "relu6":
            self.add_module(name="act", module=nn.ReLU6())
        elif act == "tanh":
            self.add_module(name="act", module=nn.Tanh())
        else:
            print("Unknown activation function")

    # def forward(self, x):
    #     return super().forward(x)


class CombConv(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        act="relu",
        kernel=1,
        stride=1,
    ):
        super().__init__()

        self.add_module(
            "conv",
            Conv(
                in_channel,
                out_channel,
                act=act,
                kernel=kernel,
            ),
        )
        self.add_module(
            "dwconv",
            DWConvTransition(out_channel, stride=stride),
        )

    # def forward(self, x):
    #     return super().forward(x)


class HarDBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        n_layers,
        k,
        m,
        act="relu",
        dwconv=True,
        keepbase=False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.links = []
        layers = []
        self.keepbase = keepbase
        self.out_channels = 0

        for i in range(n_layers):
            in_ch, out_ch, links = self.get_links(i + 1, in_channels, k, m)
            # print(f'hardblock {i} {in_ch} {out_ch}')
            self.links.append(links)

            if dwconv:
                layers.append(CombConv(in_ch, out_ch, act=act))
            else:
                layers.append(Conv(in_ch, out_ch, act=act))

            if (i % 2 == 0) or (i == n_layers - 1):
                self.out_channels += out_ch

        self.layers = nn.ModuleList(layers)

    def get_links(self, layer, base_ch, k, m):
        if layer == 0:
            return 0, base_ch, []

        out_ch = k
        links = []

        for i in range(10):  # At most 2^10 layers check
            check = 2**i
            if layer % check == 0:
                link = layer - check
                links.append(link)
                if i > 0:
                    out_ch *= m

        out_ch = int(int(out_ch + 1) / 2) * 2  # No clue
        in_ch = 0

        for j in links:
            _, ch, _ = self.get_links(j, base_ch, k, m)
            in_ch += ch
        return in_ch, out_ch, links

    def get_out_ch(self):
        # print(f'out ch {self.out_channels}')
        return self.out_channels

    def forward(self, x):
        layers = [x]

        for layer in range(len(self.layers)):
            link = self.links[layer]
            tin = []
            for i in link:
                tin.append(layers[i])
            if len(tin) > 1:
                x = torch.cat(tin, 1)
            else:
                x = tin[0]
            out = self.layers[layer](x)
            layers.append(out)

        t = len(layers)
        out = []
        for i in range(t):
            if (self.keepbase and i == 0) or (i == t - 1) or (i % 2 == 1):
                out.append(layers[i])
        out = torch.cat(out, 1)
        return out


class Up(nn.Module):
    def __init__(
        self,
        in_channels,
        n_layers,
        k,
        m,
        act="relu",
        dwconv=True,
        keepbase=False,
        trilinear=True,
        *args,
        **kwargs,
    ):
        super().__init__()
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
            self.conv = Conv(
                2 * in_channels, in_channels, act=act, kernel=3, padding=1, bias=False
            )
            self.block = HarDBlock(
                in_channels,
                n_layers,
                k,
                m,
                act=act,
                dwconv=dwconv,
                keepbase=keepbase,
                trilinear=trilinear,
            )
        else:
            self.up = nn.ConvTranspose3d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = Conv(
                in_channels, in_channels, act=act, kernel=3, padding=1, bias=False
            )
            self.block = HarDBlock(
                in_channels,
                n_layers,
                k,
                m,
                act=act,
                dwconv=dwconv,
                keepbase=keepbase,
                trilinear=trilinear,
            )

    def forward(self, x1, x2):
        # print(f'Before up {x1.shape}')
        x1 = self.up(x1)
        # Assuming input BCHW
        # print(x1.size(), x2.size())
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        diffZ = x2.size()[4] - x1.size()[4]

        # print(f'diff {diffX} {diffY} {diffZ}')
        x1 = F.pad(x1, [diffZ // 2, diffZ - diffZ // 2,
                        diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # print(f'Before concat {x1.shape} {x2.shape}')
        x = torch.cat([x2, x1], dim=1)
        # print(f'Input shape {x.shape}')
        x = self.conv(x)

        # print(f'Input shape block {x.shape}')
        x = self.block(x)
        # print(f'Output shape block {x.shape}')
        return x

    def get_out_ch(self):
        return self.block.get_out_ch()


class Down(nn.Module):
    def __init__(
        self,
        in_channels,
        n_layers,
        k,
        m,
        act="relu",
        dwconv=True,
        keepbase=False,
        dropout=0,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.down = nn.ModuleList(
            [
                nn.MaxPool3d(2),
                HarDBlock(
                    in_channels, n_layers, k, m, act="relu", dwconv=True, keepbase=False
                ),
            ]
        )

    def forward(self, x):
        for layer in self.down:
            # print(f'input shape: {x.shape}')
            x = layer(x)
            # print(f'{layer}: {x.shape}')
        return x

    def get_out_ch(self):
        return self.down[1].get_out_ch()


class Bottleneck(nn.Module):
    def __init__(self, ch, act="relu", *args, **kwargs):
        super().__init__()
        self.layers = nn.Sequential(
            Conv(ch, ch, act=act, kernel=3), Conv(ch, ch, act=act, kernel=3)
        )

    def forward(self, x):
        return self.layers(x)
