import torch
from .quant_layer import QuantModule
from .quant_block import BaseQuantBlock
from .quant_model import QuantModel
from typing import Union
import torch.nn as nn

def set_act_quantize_params(module: Union[QuantModel, QuantModule, BaseQuantBlock],
                            cali_data, batch_size: int = 256):
    """量化状态开启"""
    module.set_quant_state(True, True)

    for t in module.modules():
        if isinstance(t, (QuantModule, BaseQuantBlock)):
            t.act_quantizer.set_inited(False)

    # '''
    #     set or init step size and zero point in the activation quantizer
    #     在激活量化器中设置或初始化步长和零点
    # '''
    # batch_size = min(batch_size, cali_data.size(0))
    # with torch.no_grad():
    #     for i in range(int(cali_data.size(0) / batch_size)):
    #         """
    #             将256个数据拿过来在该nodule进行一次前向传播
    #
    #             QuantModule类的forward中会自动对激活进行激活量化
    #         """
    #         module(cali_data[i * batch_size:(i + 1) * batch_size].cuda())
    # torch.cuda.empty_cache()

    if module.act_quantizer.inited == False and isinstance(module.norm_function, (nn.BatchNorm2d, nn.BatchNorm1d)):
        mean = module.norm_function.running_mean
        var = module.norm_function.running_var

        beta_mean = module.norm_function.bias
        gamma_std = module.norm_function.weight

        # """使用绝对值"""
        # module.act_quantizer.bn_estimate_abs_max = torch.max(torch.abs(mean + 3 * torch.sqrt(var)))
        # print("bn_estimate_abs_max:{}".format(module.act_quantizer.bn_estimate_abs_max))
        #
        # module.act_quantizer.delta = 2 * module.act_quantizer.bn_estimate_abs_max / (module.act_quantizer.n_levels - 1)
        # module.act_quantizer.zero_point = 0
        #
        # print("设置后delta:{}".format(module.act_quantizer.delta))

        """使用绝对值"""
        module.act_quantizer.bn_estimate_abs_max = torch.max(torch.abs(beta_mean + 4 * torch.abs(gamma_std)))
        print("bn_estimate_abs_max:{}".format(module.act_quantizer.bn_estimate_abs_max))

        module.act_quantizer.delta = 2 * module.act_quantizer.bn_estimate_abs_max / (module.act_quantizer.n_levels - 1)
        module.act_quantizer.zero_point = 0

        print("设置后delta:{}".format(module.act_quantizer.delta))

        # C = out.shape[1]
        # op = None
        # if out.dim() == 3:
        #     op = out.permute(0, 2, 1)
        #     op = op.reshape(-1, C)
        # else:
        #     op = out
        #
        #     k = int(0.999 * op.size(0))
        #     percentile_90, _ = torch.kthvalue(op, k, dim=0)
        #
        # max_values_dim2, max_indices_dim2 = torch.max(op, dim=0)
        # print("conv输出得到的最大值：{}".format(torch.max(max_values_dim2)))
        # print("BN层数据估计出来的最大值：{}".format(torch.max(mean + 3*torch.sqrt(var))))
        # print("conv输出得到的最小值：{}".format(torch.min(max_values_dim2)))
        # print("BN层数据估计出来的最小值：{}".format(torch.max(mean - 3 * torch.sqrt(var))))

        # module.act_quantizer.bn_estimate_abs_max = torch.max(mean + 3 * torch.sqrt(var))
        # print("bn_estimate_abs_max:{}".format(module.act_quantizer.bn_estimate_abs_max))
        #
        # module.act_quantizer.delta = 2 * module.act_quantizer.bn_estimate_abs_max / (module.act_quantizer.n_levels - 1)
        # module.act_quantizer.zero_point = 0
        #
        # print("设置后delta:{}".format(module.act_quantizer.delta))
        # print("计算的估计值：{}".format(module.act_quantizer.bn_estimate_abs_max))


    for t in module.modules():
        if isinstance(t, (QuantModule, BaseQuantBlock)):
            t.act_quantizer.set_inited(True)
