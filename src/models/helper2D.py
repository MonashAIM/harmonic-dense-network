import torch
from torch import nn
from torch.nn import functional as F


class DWConvTransition(nn.Sequential):
    def __init__(self, in_channels, kernel=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.add_module(
            "dwconv",
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                groups=in_channels,
                bias=bias,
            ),
        )
        self.add_module("norm", nn.BatchNorm2d(in_channels))

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
        self.add_module(
            name="conv",
            module=nn.Conv2d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=kernel,
                stride=stride,
                padding=kernel // 2,
                bias=bias,
            ),
        )

        self.add_module(name="bn", module=nn.BatchNorm2d(num_features=out_channel))

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

        out_ch = int(int(out_ch + 1) / 2) * 2
        in_ch = 0

        for j in links:
            _, ch, _ = self.get_links(j, base_ch, k, m)
            in_ch += ch
        return in_ch, out_ch, links

    def get_out_ch(self):
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
        bilinear=True,
        *args,
        **kwargs,
    ):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
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
                bilinear=bilinear,
            )
        else:
            self.up = nn.ConvTranspose2d(
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
                bilinear=bilinear,
            )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Assuming input BCHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(
            x1,
            [
                diffX // 2,
                diffX - diffX // 2,
                diffY // 2,
                diffY - diffY // 2,
            ],
        )

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        x = self.block(x)
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
                nn.MaxPool2d(2),
                HarDBlock(
                    in_channels, n_layers, k, m, act="relu", dwconv=True, keepbase=False
                ),
            ]
        )

    def forward(self, x):
        for layer in self.down:
            x = layer(x)
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
