from .quant_layer import QuantModule
from .data_utils import save_inp_oup_data, save_dc_fp_data

"""
    cali_data.shape : torch.Size([1024, 3, 224, 224])
    batch_size : 32
    
    得到输入数据
    
    返回的 cached_inps == cali_data
"""

# TODO
"""
    block ： qnn当前layer
"""
def get_init(model, block, cali_data, batch_size, input_prob: bool = False, keep_gpu: bool=True):
    cached_inps = save_inp_oup_data(model, block, cali_data, batch_size, input_prob=input_prob, keep_gpu=keep_gpu)
    return cached_inps

"""
    传入fp_model和fp_layer
    
    cached_outs ： 经过fp layer的输出
    cached_outputs ： fp模型最终的输出
    cached_sym : fp输入经过DC后，再输入到fp layer的输入
"""
def get_dc_fp_init(model, block, cali_data, batch_size, input_prob: bool = False, keep_gpu: bool=True, lamb=50, bn_lr=1e-3):
    cached_outs, cached_outputs, cached_sym = save_dc_fp_data(model, block, cali_data, batch_size, input_prob=input_prob, keep_gpu=keep_gpu, lamb=lamb, bn_lr=bn_lr)
    return cached_outs, cached_outputs, cached_sym

def set_weight_quantize_params(model):
    for module in model.modules():
        if isinstance(module, QuantModule):
            """将量化器的初始化状态设置为未完成，以后进行初始化操作"""
            module.weight_quantizer.set_inited(False)

            '''
                对每个module权重部分的量化
                caculate the step size and zero point for weight quantizer
            '''
            module.weight_quantizer(module.weight)

            """表示量化器的初始化已完成"""
            module.weight_quantizer.set_inited(True)

def save_quantized_weight(model):
    for module in model.modules():
        if isinstance(module, QuantModule):
            module.weight.data = module.weight_quantizer(module.weight)
