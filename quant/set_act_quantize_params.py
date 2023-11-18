import torch
from .quant_layer import QuantModule
from .quant_block import BaseQuantBlock
from .quant_model import QuantModel
from typing import Union

def set_act_quantize_params(module: Union[QuantModel, QuantModule, BaseQuantBlock],
                            cali_data, batch_size: int = 256):
    """量化状态开启"""
    module.set_quant_state(True, True)

    for t in module.modules():
        if isinstance(t, (QuantModule, BaseQuantBlock)):
            t.act_quantizer.set_inited(False)

    '''
        set or init step size and zero point in the activation quantizer
        在激活量化器中设置或初始化步长和零点
    '''
    batch_size = min(batch_size, cali_data.size(0))
    with torch.no_grad():
        for i in range(int(cali_data.size(0) / batch_size)):
            """
                将256个数据拿过来在该nodule进行一次前向传播
                
                QuantModule类的forward中会自动对激活进行激活量化
            """
            module(cali_data[i * batch_size:(i + 1) * batch_size].cuda())
    torch.cuda.empty_cache()

    for t in module.modules():
        if isinstance(t, (QuantModule, BaseQuantBlock)):
            t.act_quantizer.set_inited(True)
