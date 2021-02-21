from typing import Any, Callable, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F


__all__ = ['MobileNetV2', 'mobilenet_v2']


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNActivation(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
        dilation: int = 1,
    ) -> None:
        padding = (kernel_size - 1) // 2 * dilation
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups,
                      bias=False),
            norm_layer(out_planes),
            activation_layer(inplace=True)
        )
        self.out_channels = out_planes


# necessary for backwards compatibility
ConvBNReLU = ConvBNActivation


class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim,
                                     kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride,
                       groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        num_branches=3, bpscale=False,
        avg=False, ind=False,
    ) -> None:
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use
        """
        super(MobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.ind = ind
        self.avg = avg
        self.bpscale = bpscale
        self.num_branches = num_branches

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 1],
                [6, 32, 3, 1],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(
            input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(
            last_channel * max(1.0, width_mult), round_nearest)
        features: List[nn.Module] = [ConvBNReLU(
            3, input_channel, stride=1, norm_layer=norm_layer)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            layers = []
            # modified to make forward simpler
            # from 20 layers to 9 squential
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel,
                                      stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
        # building last several layers
        # features.append(ConvBNReLU(input_channel, self.last_channel,
        #                            kernel_size=1, norm_layer=norm_layer))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        for i in range(num_branches):
            setattr(self, 'layer4_' + str(i),
                    ConvBNReLU(input_channel,
                               self.last_channel,
                               kernel_size=1,
                               norm_layer=norm_layer))
            setattr(self, 'classifier4_' + str(i),
                    nn.Sequential(nn.Dropout(0.2),
                                  nn.Linear(self.last_channel,
                                            num_classes),)
                    )

        # self.classifier = nn.Sequential(
        #     nn.Dropout(0.2),
        #     nn.Linear(self.last_channel, num_classes),
        # )

        if self.avg == False:
            # one
            self.control_v1 = nn.Linear(output_channel, self.num_branches)
            self.bn_v1 = nn.BatchNorm1d(self.num_branches)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor, is_feat: bool = False) -> Tensor:
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        # x = self.features(x)

        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        # x = nn.functional.adaptive_avg_pool2d(
        #     x, (1, 1)).reshape(x.shape[0], -1)
        # x = self.classifier(x)
        # return x

        x = self.features(x)

        x_3 = getattr(self, 'layer4_0')(x)
        x_3 = F.adaptive_avg_pool2d(x_3, (1, 1))
        x_3 = x_3.view(x_3.size(0), -1)     # B x 64
        x_3_1 = getattr(self, 'classifier4_0')(x_3)     # B x num_classes
        pro = x_3_1.unsqueeze(-1)
        for i in range(1, self.num_branches):
            temp = getattr(self, 'layer4_'+str(i))(x)
            temp = self.avgpool(temp)       # B x 64 x 1 x 1
            temp = temp.view(temp.size(0), -1)
            temp_1 = getattr(self, 'classifier4_' + str(i))(temp)
            temp_1 = temp_1.unsqueeze(-1)
            # B x num_classes x num_branches
            pro = torch.cat([pro, temp_1], -1)

        if self.ind:
            return pro, None
        else:
            if self.avg:
                pass
            #ONE
            else:
                x_c = F.adaptive_avg_pool2d(x, (1, 1))     # B x 32 x 1 x 1
                x_c = x_c.view(x_c.size(0), -1)  # B x 32
                x_c = self.control_v1(x_c)    # B x 3
                x_c = self.bn_v1(x_c)
                x_c = F.relu(x_c)
                x_c = F.softmax(x_c, dim=1)  # B x 3
                x_m = x_c[:, 0].view(-1, 1).repeat(1,
                                                   pro[:, :, 0].size(1)) * pro[:, :, 0]
                for i in range(1, self.num_branches):
                    # B x num_classes
                    x_m += x_c[:, i].view(-1, 1).repeat(1,
                                                        pro[:, :, i].size(1)) * pro[:, :, i]
            return pro, x_m



    def forward(self, x: Tensor, is_feat: bool = False) -> Tuple[Tensor]:
        return self._forward_impl(x, is_feat)


def mobilenet_v2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> MobileNetV2:
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(**kwargs)
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
    #                                           progress=progress)
    #     model.load_state_dict(state_dict)
    return model


# if __name__ == "__main__":
#     net = MobileNetV2(num_classes=10, multiple_branch=True)
#     x = torch.randn(1, 3, 32, 32)
#     c = net(x, is_feat=True)
#     for t in c:
#         print(t.shape)
#     macs, params = profile(net, inputs=(x, ), verbose=False)
#     macs, params = clever_format([macs, params], "%.3f")
#     print(f"param {params}\tmacs {macs}")
