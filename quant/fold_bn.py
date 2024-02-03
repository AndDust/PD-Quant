import torch
import torch.nn as nn
import torch.nn.init as init


class StraightThrough(nn.Module):
    def __int__(self):
        super().__init__()

    def forward(self, input):
        return input

"""
    融合BN和批归一化
"""
def _fold_bn(conv_module, bn_module):
    # 获得卷积权重
    w = conv_module.weight.data
    # 获取BN模块的运行时均值和方差
    y_mean = bn_module.running_mean
    y_var = bn_module.running_var
    # 计算BN的稳定标准差，这是为了避免分母为零或非常小的情况：
    safe_std = torch.sqrt(y_var + bn_module.eps)
    # 设置权重的形状为(输出通道数, 1, 1, 1):
    w_view = ()
    if isinstance(conv_module, nn.Conv2d):
        w_view = (conv_module.out_channels, 1, 1, 1)
    elif isinstance(conv_module, nn.Conv1d):
        w_view = (conv_module.out_channels, 1, 1)
    elif isinstance(conv_module, nn.Linear):
        w_view = (-1, 1)

    # 检查BN模块是否使用了affine变换（即是否有学习到的scale和shift参数）：
    # 如果有，根据BN的权重和safe_std调整卷积的权重。
    # 计算新的偏置beta。
    # 如果卷积模块有偏置，则进一步更新偏置；否则，只使用beta作为新的偏置。
    if bn_module.affine:
        weight = w * (bn_module.weight / safe_std).view(w_view)
        beta = bn_module.bias - bn_module.weight * y_mean / safe_std
        if conv_module.bias is not None:
            bias = bn_module.weight * conv_module.bias / safe_std + beta
        else:
            bias = beta
    # 如果BN没有使用affine变换，则直接使用safe_std调整卷积权重，并计算新的偏置beta。同样的，如果卷积模块有偏置，进一步更新偏置；否则，只使用beta作为新的偏置。
    else:
        weight = w / safe_std.view(w_view)
        beta = -y_mean / safe_std
        if conv_module.bias is not None:
            bias = conv_module.bias / safe_std + beta
        else:
            bias = beta
    # 最后，返回更新后的权重和偏置。
    return weight, bias


def fold_bn_into_conv(conv_module, bn_module):
    w, b = _fold_bn(conv_module, bn_module)
    if conv_module.bias is None:
        conv_module.bias = nn.Parameter(b)
    else:
        conv_module.bias.data = b
    conv_module.weight.data = w
    # set bn running stats
    bn_module.running_mean = bn_module.bias.data
    bn_module.running_var = bn_module.weight.data ** 2


def reset_bn(module: nn.BatchNorm2d):
    if module.track_running_stats:
        module.running_mean.zero_()
        module.running_var.fill_(1-module.eps)
        # we do not reset numer of tracked batches here
        # self.num_batches_tracked.zero_()
    if module.affine:
        init.ones_(module.weight)
        init.zeros_(module.bias)


def is_bn(m):
    return isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d)


def is_absorbing(m):
    return (isinstance(m, nn.Conv2d)) or isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d)

"""
    这个 search_fold_and_remove_bn 函数的目的是在神经网络模型中搜索可融合的批量归一化（Batch Normalization，BN）层，
    并将其参数融合到前面的卷积层或全连接层中。
    融合操作后，原BN层会被替换为一个直通（Straight Through ）层，这是一个不改变输入的层，目的是保持网络结构的一致性。
"""
# def search_fold_and_remove_bn(model):
#     model.eval()
#     prev = None
#     for n, m in model.named_children():
#         print("________{}________".format(n))
#         """判断该层是否是BN层，并且前一层是卷积层或线性层（即能够被前一层吸收）"""
#         if is_bn(m) and is_absorbing(prev):
#             """进行BN折叠"""
#             tmp_module = None
#             for module in m.modules():
#                 if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
#                     tmp_module = module
#                     break
#
#             fold_bn_into_conv(prev, tmp_module)
#             # set the bn module to straight through
#             """
#                 原BN层会被替换为一个直通（Straight Through）层，
#                 这是一个不改变输入的层，目的是保持网络结构的一致性。
#             """
#             setattr(model, n, StraightThrough())
#             """判断本层是否是卷积层或线性层"""
#         elif is_absorbing(m):
#             prev = m
#             """如果都不是，继续递归查找"""
#         else:
#             prev = search_fold_and_remove_bn(m)
#     return prev

def search_fold_and_remove_bn(model):
    model.eval()
    prev = None
    for n, m in model.named_children():
        """判断该层是否是BN层，并且前一层是卷积层或线性层（即能够被前一层吸收）"""
        if is_bn(m) and is_absorbing(prev):
            """进行BN折叠"""
            fold_bn_into_conv(prev, m)
            # set the bn module to straight through
            """
                原BN层会被替换为一个直通（Straight Through）层，
                这是一个不改变输入的层，目的是保持网络结构的一致性。
            """
            setattr(model, n, StraightThrough())
            """判断本层是否是卷积层或线性层"""
        elif is_absorbing(m):
            prev = m
            """如果都不是，继续递归查找"""
        else:
            prev = search_fold_and_remove_bn(m)
    return prev


def search_fold_and_reset_bn(model):
    model.eval()
    prev = None
    for n, m in model.named_children():
        if is_bn(m) and is_absorbing(prev):
            fold_bn_into_conv(prev, m)
            # reset_bn(m)
        else:
            search_fold_and_reset_bn(m)
        prev = m

