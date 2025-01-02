import torch
from torch import nn
from torch.nn import functional as F


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
        act="relu6",
        kernel=3,
        stride=1,
        bias=False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.add_module(
            name="conv",
            module=nn.Conv3d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=kernel,
                stride=stride,
                padding=kernel // 2,
                bias=bias,
            ),
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


def passthrough(x, *args, **kwargs):
    return x


class InputTransition(nn.Module):
    def __init__(self, outChans, act="elu"):
        super().__init__()
        self.outch = outChans
        self.conv1 = nn.Conv3d(1, outChans, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm3d(outChans)
        if act == "elu":
            self.act = nn.ELU()
        else:
            self.act = nn.ReLU()

    def forward(self, x):
        # do we want a PRELU here as well?
        out = self.bn1(self.conv1(x))
        x16 = torch.cat((x for x in range(self.outch)), 1)  # BCDHW
        out = self.relu1(torch.add(out, x16))
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
        *args,
        **kwargs,
    ):
        super().__init__()
        block = HarDBlock(
            in_channels, n_layers, k, m, act=act, dwconv=dwconv, keepbase=keepbase
        )
        self.out_ch = block.get_out_ch()
        self.ops = block
        self.up_conv = nn.ConvTranspose3d(
            in_channels, self.out_ch // 2, kernel_size=2, stride=2
        )
        self.bn1 = nn.BatchNorm3d(self.out_ch // 2)
        self.do1 = passthrough
        self.do2 = nn.Dropout3d()
        self.relu1 = nn.ELU()
        self.relu2 = nn.ELU()

    def forward(self, x1, x2):
        out1 = self.do1(x1)
        out2 = self.do2(x2)
        print(out1.shape)
        print(out2.shape)
        out1 = self.relu1(self.bn1(self.up_conv(out1)))
        xcat = torch.cat((out1, out2), 1)
        out1 = self.ops(xcat)
        out1 = self.relu2(torch.add(out1, xcat))
        return out1

    def get_out_ch(self):
        return self.out_ch


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
        block = HarDBlock(
            in_channels, n_layers, k, m, act=act, dwconv=dwconv, keepbase=keepbase
        )
        self.out_ch = block.get_out_ch()
        self.ops = block
        self.down_conv = nn.Conv3d(in_channels, self.out_ch, kernel_size=2, stride=2)
        self.temp_conv = nn.Conv3d(self.out_ch, in_channels, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm3d(self.out_ch)
        self.bn2 = nn.BatchNorm3d(in_channels)
        self.relu1 = nn.ELU()
        self.relu2 = nn.ELU()
        self.relu3 = nn.ELU()
        self.temp = passthrough

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.relu2(self.bn2(self.temp_conv(down)))
        out = self.ops(out)
        # print(down.shape)
        # print(out.shape)
        out = self.relu3(torch.add(out, down))
        return out

    def get_out_ch(self):
        return self.out_ch


class OutputTransition(nn.Module):
    def __init__(self, inChans, nll):
        super().__init__()
        self.conv1 = nn.Conv3d(inChans, 2, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm3d(2)
        self.conv2 = nn.Conv3d(2, 2, kernel_size=1)
        self.relu1 = nn.ELU()

    def forward(self, x):
        # convolve 32 down to 2 channels
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)

        # make channels the last axis
        out = out.permute(0, 2, 3, 4, 1).contiguous()
        # flatten
        out = out.view(out.numel() // 2, 2)
        out = self.softmax(out)
        # treat channel 0 as the predicted output
        return out


# class Bottleneck(nn.Module):
#     def __init__(self, ch, act="relu", *args, **kwargs):
#         super().__init__()
#         self.layers = nn.Sequential(
#             Conv(ch, ch, act=act, kernel=3), Conv(ch, ch, act=act, kernel=3)
#         )

#     def forward(self, x):
#         return self.layers(x)
