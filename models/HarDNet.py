import os
import yaml
import torch.nn as nn
from helper import Conv, HarDBlock, DWConvTransition

config_files = {
    '68': '68arch_config.yaml',
    '85': '85arch_config.yaml',
    '39DS': '39DSarch_config.yaml'
}

class HarDNet39DS(nn.Module):
    def __init__(self, arch='68', act="relu", *args, **kwargs):
        super().__init__()

        config_path = os.path.join("config", config_files[arch])
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        m = 1.7
        second_kernel = 3
        max_pool = True
        
        first_ch = config.get("first_ch")[0]
        ch_list = config.get("ch_list")[0]
        gr = config.get("gr")[0]
        n_layers = config.get("n_layers")[0]
        downSamp = config.get("downSamp")[0]
        drop_rate = config.get("drop_rate")
        depthwise = config.get("depthwise")

        if depthwise:
            second_kernel = 1
            max_pool = False
            drop_rate = 0.05
            depthwise = True

        blocks = len(n_layers)
        self.layers = nn.ModuleList([])

        self.layers.append(Conv(3, first_ch[0], kernel=3, stride=2, bias=False))
        self.layers.append(Conv(first_ch[0], first_ch[1], kernel=second_kernel))

        if max_pool:
            self.layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else:
            self.layers.append(DWConvTransition(first_ch[1], first_ch[1], stride=2))

        ch = first_ch[1]
        for i in range(blocks):
            block = HarDBlock(ch, n_layers[i], gr[i], m, act=act, dwconv=depthwise)
            ch = block.get_out_ch()
            self.layers.append(block)

            if (i == blocks - 1) and drop_rate:
                self.layers.append(nn.Dropout(drop_rate))

            self.layers.append(Conv(ch, ch_list[i], act=act, kernel=1))
            ch = ch_list[i]
            if downSamp[i] == 1:
                if max_pool:
                    self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                else:
                    self.layers.append(DWConvTransition(ch, ch, stride=2))

        ch = ch_list[blocks - 1]
        self.layers.append(
            nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Dropout(drop_rate),
                nn.Linear(ch, 1000),
            )
        )

        # TODO: Look into segmentation problem

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x