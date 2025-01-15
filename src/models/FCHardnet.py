import torch
import torch.nn.functional as F
import torch.nn as nn
from src.models.base_hardnet import HarDBlock, ConvLayer


class FCHardnet(nn.Module):
    def __init__(self, n_classes=19, in_channels=1):
        super(FCHardnet, self).__init__()
        first_ch = [16, 24, 32, 48]
        # ch_list = [64, 96, 160, 224, 320]
        grmul = 1.7
        # gr = [10, 16, 18, 24, 32]
        # n_layers = [4, 4, 8, 8, 8]

        ch_list = [192, 256, 320, 480, 720, 1280]
        gr = [24, 24, 28, 36, 48, 256]
        n_layers = [8, 16, 16, 16, 16, 4]

        blks = len(n_layers)
        self.model_type = "2D"
        self.shortcut_layers = []
        self.base = nn.ModuleList([])

        self.base.append(
            ConvLayer(
                in_channels=in_channels, out_channels=first_ch[0], kernel=3, stride=2
            )
        )
        self.base.append(ConvLayer(first_ch[0], first_ch[1], kernel=3))
        self.base.append(ConvLayer(first_ch[1], first_ch[2], kernel=3, stride=2))
        self.base.append(ConvLayer(first_ch[2], first_ch[3], kernel=3))

        skip_connection_channel_counts = []
        ch = first_ch[3]
        for i in range(blks):
            blk = HarDBlock(ch, gr[i], grmul, n_layers[i])
            ch = blk.get_out_ch()
            skip_connection_channel_counts.append(ch)
            self.base.append(blk)
            if i < blks - 1:
                self.shortcut_layers.append(len(self.base) - 1)

            self.base.append(ConvLayer(ch, ch_list[i], kernel=1))
            ch = ch_list[i]

            if i < blks - 1:
                self.base.append(nn.AvgPool2d(kernel_size=2, stride=2))

        cur_channels_count = ch
        prev_block_channels = ch
        n_blocks = blks - 1
        self.n_blocks = n_blocks

        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        self.conv1x1_up = nn.ModuleList([])

        for i in range(n_blocks - 1, -1, -1):
            self.transUpBlocks.append(
                TransitionUp(prev_block_channels, prev_block_channels)
            )
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]
            self.conv1x1_up.append(
                ConvLayer(cur_channels_count, cur_channels_count // 2, kernel=1)
            )
            cur_channels_count = cur_channels_count // 2

            blk = HarDBlock(cur_channels_count, gr[i], grmul, n_layers[i])
            self.denseBlocksUp.append(blk)
            prev_block_channels = blk.get_out_ch()
            cur_channels_count = prev_block_channels

        self.finalConv = nn.Conv2d(
            in_channels=cur_channels_count,
            out_channels=n_classes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def get_model_type(self):
        return self.model_type

    def forward(self, x):
        skip_connections = []
        size_in = x.size()

        for i in range(len(self.base)):
            x = self.base[i](x)
            if i in self.shortcut_layers:
                skip_connections.append(x)
        out = x

        for i in range(self.n_blocks):
            skip = skip_connections.pop()
            out = self.transUpBlocks[i](out, skip, True)
            out = self.conv1x1_up[i](out)
            out = self.denseBlocksUp[i](out)

        out = self.finalConv(out)

        out = F.interpolate(
            out, size=(size_in[2], size_in[3]), mode="bilinear", align_corners=True
        )
        return out


class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

    def forward(self, x, skip, concat=True):
        is_v2 = type(skip) is list
        if is_v2:
            skip_x = skip[0]
        else:
            skip_x = skip
        out = F.interpolate(
            x,
            size=(skip_x.size(2), skip_x.size(3)),
            mode="bilinear",
            align_corners=True,
        )
        if concat:
            if is_v2:
                out = [out] + skip
            else:
                out = torch.cat([out, skip], 1)

        return out
