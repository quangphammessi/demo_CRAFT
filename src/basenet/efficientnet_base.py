from collections import namedtuple

import torch
from torchutil import *
import os
from .efficient_net.model import EfficientNet
from .efficient_net.utils import (
    relu_fn,
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    get_model_params,
    efficientnet_params,
    load_pretrained_weights,
)

os.environ["CUDA_VISIBLE_DEVICES"]= "1"

MODEL_NAME = 'efficientnet-b4'


# Su dung EfficientNet B2
weight_path = os.path.dirname(__file__) +'/../pretrain/' + 'efficientnet-b4-e116e8b3.pth'

class efficientnet_base(EfficientNet):
    def __init__(self, pretrained=True, freeze=False):
        blocks_args, global_params = get_model_params(model_name=MODEL_NAME, override_params=None)

        skip_connections = {
            'b0': [3, 5, 9],
            'b1': [5, 8, 16],
            'b2': [5, 8, 16],
            'b3': [5, 8, 18],
            'b4': [6, 10, 22],
            'b5': [8, 13, 27],
            'b6': [9, 15, 31],
            'b7': [11, 18, 38]
        }
        
        super().__init__(blocks_args, global_params)
        self._skip_connections = skip_connections[MODEL_NAME.split('-')[1]]
        self._skip_connections.append(len(self._blocks))

        del self._fc


        '''Load pretrained model'''
        if pretrained:
            state_dict = torch.load(weight_path)
            state_dict.pop('_fc.bias')
            state_dict.pop('_fc.weight')
            super().load_state_dict(state_dict)
            print('Pretrained model loaded')


        # if not pretrained:
        #     init_weights(self.slice1.modules())
        #     init_weights(self.slice2.modules())
        #     init_weights(self.slice3.modules())
        #     init_weights(self.slice4.modules())
        
        # init_weights(self.slice5.modules())

        # if freeze:
        #     for param in self.slice1.parameters():
        #         param.requires_grad = False

    def forward(self, X):
        # h = self.slice1(X)
        # h_relu2_2 = h
        # h = self.slice2(h)
        # h_relu3_2 = h
        # h = self.slice3(h)
        # h_relu4_3 = h
        # h = self.slice4(h)
        # h_relu5_3 = h
        # h = self.slice5(h)
        # h_fc7 = h
        # effi_outputs = namedtuple("EffiOutputs", ['fc7', 'relu5_3', 'relu4_3', 'relu3_2', 'relu2_2'])
        # out = effi_outputs(h_fc7, h_relu5_3, h_relu4_3, h_relu3_2, h_relu2_2)
        # return out

        '''Luu ket qua cua cac block'''
        result = []

        x = relu_fn(self._bn0(self._conv_stem(X)))
        result.append(x)

        skip_connection_idx = 0
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if idx == self._skip_connections[skip_connection_idx] - 1:
                skip_connection_idx += 1
                result.append(x)


        effi_outputs = namedtuple("EffiOutputs", ['fc7', 'relu5_3', 'relu4_3', 'relu3_2', 'relu2_2'])
        out = effi_outputs(result[4], result[3], result[2], result[1], result[0])

        return out


if __name__ == "__main__":
    model = efficientnet_base(pretrained=True).cuda()
    # print(model)
    output = model(torch.randn(1, 3, 768, 768).cuda())
    print(output)