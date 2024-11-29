import os
import yaml
import torch.nn as nn
from models.helper import Bottleneck, Down, Up, Conv
from models.config_dic import config_files


class HarDUNet(nn.Module):
    def __init__(
        self,
        n_classes=1,
        arch="68",
        act="relu",
        transformer=False,
        keepbase=False,
        bilinear=True,
        *args,
        **kwargs,
    ):
        super().__init__()
        # Down and Up U-Net
        self.classes = n_classes
        config_path = os.path.join(os.getcwd(), "models", "configs", config_files[arch])
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        second_kernel = 3
        init_ch = 1
        first_ch = config.get("first_ch")[0]
        ch_list = config.get("ch_list")[0]
        gr = config.get("gr")[0]
        m = config.get("grmul")
        n_layers = config.get("n_layers")[0]
        drop_rate = config.get("drop_rate")
        depthwise = config.get("depthwise")

        if depthwise:
            second_kernel = 1
            drop_rate = 0.05

        blocks = len(n_layers)
        self.start = nn.ModuleList([])
        self.enc = nn.ModuleList([])

        self.start.append(Conv(init_ch, first_ch[0], kernel=3, stride=1, bias=False))
        self.start.append(Conv(first_ch[0], first_ch[1], kernel=second_kernel))
        ch = first_ch[1]
        for i in range(blocks):
            block = Down(ch, n_layers[i], gr[i], m, act=act, dwconv=depthwise)
            ch = block.get_out_ch()
            self.enc.append(block)

            if (i == (blocks - 1)) and (arch == "85"):
                self.enc.append(nn.Dropout(drop_rate))

            self.enc.append(Conv(ch, ch_list[i], act=act, kernel=1))
            ch = ch_list[i]

        self.dec = nn.ModuleList([])
        ch = ch_list[blocks - 2]
        for j in range(blocks - 2, -1, -1):
            # in_channels, n_layers, k, m, act="relu", dwconv=True, keepbase=False, bilinear=True
            # print(ch)
            block = Up(
                ch,
                n_layers[j - 1],
                gr[j - 1],
                m,
                act=act,
                dwconv=depthwise,
                keepbase=keepbase,
                bilinear=True,
            )
            ch = block.get_out_ch()
            self.dec.append(block)

            if (i == (blocks - 1)) and (arch == "85"):
                self.dec.append(nn.Dropout(drop_rate))

            if not (ch_list[j - 1] == ch_list[-1] and j == 0):
                self.dec.append(Conv(ch, ch_list[j - 1], act=act, kernel=1))
            else:
                self.dec.append(Conv(ch, first_ch[1], act=act, kernel=1))
            ch = ch_list[j - 1]

        block = Up(
            first_ch[1],
            n_layers[0],
            gr[0],
            m,
            act=act,
            dwconv=depthwise,
            keepbase=keepbase,
            bilinear=True,
        )
        ch = block.get_out_ch()
        self.dec.append(block)
        self.dec.append(Conv(ch, first_ch[1], act=act, kernel=1))

        # Out conv
        self.outc = nn.ModuleList([])
        self.outc.append(Conv(first_ch[1], first_ch[0], kernel=second_kernel))
        self.outc.append(Conv(first_ch[0], init_ch, kernel=3, stride=1, bias=False))
        self.outc.append(Conv(init_ch, self.classes, kernel=1, stride=1, bias=False))

        # Bottleneck
        self.transformer = transformer
        if self.transformer:
            self.bottleneck = nn.ModuleList([nn.Transformer(d_model=ch_list[-1])])
        else:
            self.bottleneck = nn.ModuleList([Bottleneck(ch_list[-1], act=act)])

        if bilinear:
            self.bottleneck.append(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            )
        else:
            self.bottleneck.append(
                nn.ConvTranspose2d(
                    ch_list[-1], ch_list[-1] // 2, kernel_size=2, stride=2
                )
            )
        self.bottleneck.append(Conv(ch_list[-1], ch_list[-2], act=act, kernel=1))

    def forward(self, x):
        outs = []
        for layer in self.start:
            x = layer(x)
        outs.append(x)
        # print(x.shape)
        for i in range(len(self.enc)):
            layer = self.enc[i]
            x = layer(x)
            # print('enc', x.shape)
            if isinstance(layer, Conv) and i < (len(self.enc) - 1):
                outs.append(x)

        # for out in outs:
        #     print(f'Outs {out.shape}')

        if self.transformer:
            b, c, h, w = x.shape
            x = x.view(b, h * w, c)
            x = self.bottleneck[0](x, x)
            x = x.view(b, c, h, w)
            for i in range(1, len(self.bottleneck)):
                layer = self.bottleneck[i]
                x = layer(x)
        else:
            for layer in self.bottleneck:
                x = layer(x)
        # print(x.shape)
        j = 0
        for i in range(len(self.dec)):
            layer = self.dec[i]
            # print(f'{layer}: {x.shape}')
            if isinstance(layer, Conv) or isinstance(layer, nn.Dropout):
                x = layer(x)
            else:
                # print(f'awawawa {outs[len(outs)-1-j].shape}')
                x = layer(x, outs[len(outs) - 1 - j])
                j += 1

        for layer in self.outc:
            # print(f'outc before {x.shape}')
            x = layer(x)
            # print(f'outc after {x.shape}')

        return x  # If we want logits
        # return nn.Softmax(x) # If we want values

    def get_classes(self):
        return self.classes


if __name__ == "__main__":
    # temp = torch.randn(size=(32, 1, 112, 112))
    # model = HarmonicUNet(transformer=False)
    # # print(model)
    # out = model(temp)
    # print(temp.shape)
    # print(out.shape)
    pass
