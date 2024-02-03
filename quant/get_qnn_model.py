from quant import (
    block_reconstruction,
    layer_reconstruction,
    BaseQuantBlock,
    QuantModule,
    QuantModel,
    set_weight_quantize_params,
)

import torch.nn as nn
import copy
# import argparse
#
# parser = argparse.ArgumentParser(description='arg parser')
#
# # TODO 新增量化参数
# parser.add_argument('--n_bits_w', default=8, type=int, help='bitwidth for weight quantization')
# parser.add_argument('--channel_wise', default=True, help='apply channel_wise quantization for weights')
# parser.add_argument('--n_bits_a', default=8, type=int, help='bitwidth for activation quantization')
# parser.add_argument('--disable_8bit_head_stem', action='store_true')
#
# parser.add_argument('--init_wmode', default='mse', type=str, choices=['minmax', 'mse', 'minmax_scale'],
#                     help='init opt mode for weight')
# parser.add_argument('--init_amode', default='mse', type=str, choices=['minmax', 'mse', 'minmax_scale'],
#                     help='init opt mode for activation')
# parser.add_argument('--prob', default=0.5, type=float)
#
# args = parser.parse_args()

def get_qnn_model(qnn_args, origin_model):
    # TODO 加载模型
    origin_model.cuda()  # 将模型移动到GPU上
    origin_model.eval()  # 设置模型为评估模式

    fp_model = copy.deepcopy(origin_model)  # 深度复制模型
    fp_model.cuda()  # 将复制的模型移动到GPU上
    fp_model.eval()  # 设置复制的模型为评估模式

    # build quantization parameters
    wq_params = {'n_bits': qnn_args.n_bits_w, 'channel_wise': qnn_args.channel_wise, 'scale_method': qnn_args.init_wmode}
    aq_params = {'n_bits': qnn_args.n_bits_a, 'channel_wise': False, 'scale_method': qnn_args.init_amode,
                 'leaf_param': True, 'prob': qnn_args.prob}

    fp_model = QuantModel(model=fp_model, weight_quant_params=wq_params, act_quant_params=aq_params, is_fusing=False)
    fp_model.cuda()
    fp_model.eval()
    """fp_model只是用来对比，不需要开启量化"""
    fp_model.set_quant_state(False, False)  # 关闭量化状态

    """
       qnn是经过BN fold和开启量化状态的 （is_fusing=True）
    """
    qnn = QuantModel(model=origin_model, weight_quant_params=wq_params, act_quant_params=aq_params, is_fusing=False)
    qnn.cuda()
    qnn.eval()

    if not qnn_args.disable_8bit_head_stem:
        print('Setting the first and the last layer to 8-bit')
        qnn.set_first_last_layer_to_8bit()

    """禁用神经网络中最后一个量化模块（QuantModule）的激活量化。"""
    qnn.disable_network_output_quantization()

    """得到量化后的模型"""
    print('the quantized model is below!')
    print(qnn)

    # PointRCNN_dataloader = create_PointRCNN_dataloader(logger)
    # pointnet_cali_data = get_train_samples(PointRCNN_dataloader, num_samples=args.pointRCNN_num_samples)

    # cali_data, cali_target = get_rain_samples(train_loader, num_samples=args.num_samples)
    # device = next(qnn.parameters()).device

    # Kwargs for weight rounding calibration
    """
        用于权重舍入校准的Kwargs
    """
    # kwargs = dict(cali_data=pointnet_cali_data, iters=args.iters_w, weight=args.weight,
    #               b_range=(args.b_start, args.b_end), warmup=args.warmup, opt_mode='mse',
    #               lr=args.lr, input_prob=args.input_prob, keep_gpu=not args.keep_cpu,
    #               lamb_r=args.lamb_r, T=args.T, bn_lr=args.bn_lr, lamb_c=args.lamb_c, a_count=args.a_count)

    set_weight_quantize_params(qnn)

    """
        重建: 重建就是让量化模型和FP模型的输出尽量保持一致,对量化模型的算子进行了重建,因为直接量化性能下降很多
    """
    def set_weight_act_quantize_params(module, fp_module):
        if isinstance(module, QuantModule):
            """
                传入完整的qnn、fp_model和当前的module、fp_module
            """
            # layer_reconstruction(qnn, fp_model, module, fp_module, **kwargs)
            layer_reconstruction(module)
        elif isinstance(module, BaseQuantBlock):
            # block_reconstruction(qnn, fp_model, module, fp_module, **kwargs)
            block_reconstruction(module)
        else:
            raise NotImplementedError

    """
        区块重建。对于第一层和最后一层，我们只能应用层重建。
    """

    # a_count = 0
    def recon_model(model: nn.Module, fp_model: nn.Module):
        """
            Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
        """
        for (name, module), (_, fp_module) in zip(model.named_children(), fp_model.named_children()):
            if isinstance(module, QuantModule):
                print('Reconstruction for layer {}'.format(name))
                set_weight_act_quantize_params(module, fp_module)
            elif isinstance(module, BaseQuantBlock):
                """比如对于ResNet里的包含conv BN Relu的block，在block层面再做一次"""
                print('Reconstruction for block {}'.format(name))
                set_weight_act_quantize_params(module, fp_module)
            else:
                recon_model(module, fp_module)

    """
        开始校准
    """
    # Start calibration
    recon_model(qnn, fp_model)
    #
    """qnn设置量化状态为True"""
    qnn.set_quant_state(weight_quant=True, act_quant=True)

    return qnn

# def get_quant_model(origin_model):
#     return get_qnn_model(args, origin_model)
