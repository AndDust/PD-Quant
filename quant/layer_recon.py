import torch
import torch.nn.functional as F
from .quant_layer import QuantModule, lp_loss
from .quant_model import QuantModel
from .block_recon import LinearTempDecay
from .adaptive_rounding import AdaRoundQuantizer
from .set_weight_quantize_params import get_init, get_dc_fp_init
from .set_act_quantize_params import set_act_quantize_params
from .quant_block import BaseQuantBlock, specials_unquantized


include = False
def find_unquantized_module(model: torch.nn.Module, module_list: list = [], name_list: list = []):
    """Store subsequent unquantized modules in a list"""
    global include
    for name, module in model.named_children():
        if isinstance(module, (QuantModule, BaseQuantBlock)):
            if not module.trained:
                include = True
                module.set_quant_state(False,False)
                name_list.append(name)
                module_list.append(module)
        elif include and type(module) in specials_unquantized:
            name_list.append(name)
            module_list.append(module)
        else:
            find_unquantized_module(module, module_list, name_list)
    return module_list[1:], name_list[1:]

"""
    重构以优化每一层的输出。
    
        model: 需要进行量化的模型
        fp_model: 对应的FP32精度模型
        
        layer: 量化模型中需要被优化的层
        fp_layer: 对应的浮点模型层
        
        cali_data: 用于校准的数据，通常是1024个训练图像
        batch_size: 重构过程中的小批量大小
        iters: 优化迭代次数
        weight: 舍入正则项的权重
        opt_mode: 优化模式 : "mse"
        
        b_range: 温度范围
        warmup: 没有调度温度的迭代次数占比
        p: L_p范数最小化
        lr: 激活delta学习的学习率
        input_prob: 输入概率
        keep_gpu: 是否在GPU上保留数据。
        lamb_r: 正则项的超参数
        T: KL散度的温度系数
        bn_lr: DC的学习率
        lamb_c: DC的超参数
"""

"""
    这里的layer和fp_layer是传入的module和fp_module
    后面都是传入的**kwargs
"""
def layer_reconstruction(model: QuantModel, fp_model: QuantModel, layer: QuantModule, fp_layer: QuantModule,
                        cali_data: torch.Tensor,batch_size: int = 32, iters: int = 20000, weight: float = 0.001,
                        opt_mode: str = 'mse', b_range: tuple = (20, 2),
                        warmup: float = 0.0, p: float = 2.0, lr: float = 4e-5, input_prob: float = 1.0, 
                        keep_gpu: bool = True, lamb_r: float = 0.2, T: float = 7.0, bn_lr: float = 1e-3, lamb_c=0.02):
    """
    Reconstruction to optimize the output from each layer.

    :param model: QuantModel
    :param layer: QuantModule that needs to be optimized

    :param cali_data: data for calibration, typically 1024 training images, as described in AdaRound
    :param batch_size: mini-batch size for reconstruction
    :param iters: optimization iterations for reconstruction,
    :param weight: the weight of rounding regularization term
    :param opt_mode: optimization mode

    :param asym: asymmetric optimization designed in AdaRound, use quant input to reconstruct fp output
    :param include_act_func: optimize the output after activation function
    :param b_range: temperature range
    :param warmup: proportion of iterations that no scheduling for temperature
    :param lr: learning rate for act delta learning
    :param p: L_p norm minimization
    :param lamb_r: hyper-parameter for regularization
    :param T: temperature coefficient for KL divergence
    :param bn_lr: learning rate for DC
    :param lamb_c: hyper-parameter for DC
    """

    '''get input and set scale'''

    """
        这一步输入的是qnn，和qnn的layer
        得到输入qnn该layer的inputs  cached_inps.shape : torch.Size([1024, 3, 224, 224])
        cached_inps ： \hat{A_{l-1}} 相当于去取论文中图3中的A_{l-1}^{FP}
    """
    cached_inps = get_init(model, layer, cali_data, batch_size=batch_size,
                                        input_prob=True, keep_gpu=keep_gpu)

    print(f'最终得到的cached_inps {cached_inps.shape},{cached_inps.flatten()[:10]}')

    """
        这一步输入fp_model和fp_layer
        Start correcting 32 batches of data!
        
        cached_outs.shape : torch.Size([1024, 64, 112, 112])
        cached_output.shape : torch.Size([1024, 1000])
        cur_syms.shape : torch.Size( [1024, 3, 224, 224])
        
        cached_outs : FP模型当前layer的输出 A_l
        cached_output : FP模型最终的输出
        cur_syms : A_{l-1}^{DC} (数据做了DC校准之后的数据)
    """
    cached_outs, cached_output, cur_syms = get_dc_fp_init(fp_model, fp_layer, cali_data, batch_size=batch_size,
                                        input_prob=True, keep_gpu=keep_gpu, bn_lr=bn_lr, lamb=lamb_c)

    """
        cached_inps.size(0) ： 1024
        cached_inps中取出256个来去设置激活量化参数
        
        这一步是干嘛的？
        
        把这个输入送入这个layer，对于激活去初始化得到一个scale
    """
    set_act_quantize_params(layer, cali_data=cached_inps[:min(256, cached_inps.size(0))])

    '''set state'''
    cur_weight, cur_act = True, True

    """
        找出当前qnn中没有量化的层，量化层的输出传给后面没有量化的层从而得到最终的输出。
    """
    global include
    module_list, name_list, include = [], [], False
    module_list, name_list = find_unquantized_module(model, module_list, name_list)

    """设置当前qnn层的量化状态，该qnn层开启权重量化、开启激活量化"""
    layer.set_quant_state(cur_weight, cur_act)
    for para in model.parameters():
        """冻结模型本身的参数"""
        para.requires_grad = False


    '''set quantizer'''
    round_mode = 'learned_hard_sigmoid'


    """将权重量化器替换为AdaRoundQuantizer"""
    # Replace weight quantizer to AdaRoundQuantizer
    w_para, a_para = [], []
    w_opt, a_opt = None, None
    scheduler, a_scheduler = None, None

    '''weight'''
    layer.weight_quantizer = AdaRoundQuantizer(uaq=layer.weight_quantizer, round_mode=round_mode,
                                               weight_tensor=layer.org_weight.data)
    layer.weight_quantizer.soft_targets = True

    """
        layer.weight_quantizer.alpha.shape : torch.Size([64, 3, 7, 7])
        给每个权重一个可学习的alpha值
    """
    w_para += [layer.weight_quantizer.alpha]

    '''activation'''
    """
        该qnn层的激活量化比例因子设置为可学习的
    """
    if layer.act_quantizer.delta is not None:
        layer.act_quantizer.delta = torch.nn.Parameter(torch.tensor(layer.act_quantizer.delta))
        a_para += [layer.act_quantizer.delta]

    '''set up drop'''
    layer.act_quantizer.is_training = True

    """设置优化器"""
    if len(w_para) != 0:
        w_opt = torch.optim.Adam(w_para, lr=3e-3)
    if len(a_para) != 0:
        a_opt = torch.optim.Adam(a_para, lr=lr)
        a_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(a_opt, T_max=iters, eta_min=0.)
    
    loss_mode = 'relaxation'
    rec_loss = opt_mode

    loss_func = LossFunction(layer, round_loss=loss_mode, weight=weight,
                             max_count=iters, rec_loss=rec_loss, b_range=b_range,
                             decay_start=0, warmup=warmup, p=p, lam=lamb_r, T=T)
    device = 'cuda'

    """
        1024
    """
    sz = cached_inps.size(0)

    """ 
        迭代20000次,每次根据前面的cached_inps cached_outs cached_output cur_syms
        随机取出32个来计算loss
        
        来优化激活比例因子以及舍入策略参数\theta
    """
    for i in range(iters):
        """
            生成一个形状为 (batch_size,) 的一维张量，其中每个元素都是从 0 到 sz 的随机整数。
            batch_size : 32
        """
        idx = torch.randint(0, sz, (batch_size,))

        """
            cached_inps ： \hat{A_{l-1}}
            cached_outs : FP模型当前layer的输出 A_l
            cached_output : FP模型最终的输出
            cur_syms : A_{l-1}^{DC}
        """
        cur_inp = cached_inps[idx].to(device)
        cur_sym = cur_syms[idx].to(device)
        output_fp = cached_output[idx].to(device)
        cur_out = cached_outs[idx].to(device)

        if input_prob < 1.0:
            """
                生成随机张量：torch.rand_like(cur_inp) 生成一个与 cur_inp 形状相 同的随机张量。这个随机张量的每个元素都是在 [0.0, 1.0) 范围内的随机浮点数。
                torch.where(condition, x, y) 函数根据 condition 来从 x 和 y 中选择元素。如果 condition 中的元素为 True，则选择 x 中的相应元素；否则选择y中的相应元素。
            """
            drop_inp = torch.where(torch.rand_like(cur_inp) < input_prob, cur_inp, cur_sym)
        
        cur_inp = torch.cat((drop_inp, cur_inp))
        
        if w_opt:
            w_opt.zero_grad()
        if a_opt:
            a_opt.zero_grad()

        """
            
        """
        out_all = layer(cur_inp)
        
        '''forward for prediction difference'''
        out_drop = out_all[:batch_size]
        out_quant = out_all[batch_size:]

        """
            qnn中量化层后面所有非量化层的forward
            把A_l^~ 送进去，经过所有未量化的层最终拿到output
        """
        output = out_quant
        for num, module in enumerate(module_list):
            # for ResNet and RegNet
            if name_list[num] == 'fc':
                output = torch.flatten(output, 1)
            # for MobileNet and MNasNet
            if isinstance(module, torch.nn.Dropout):
                output = output.mean([2, 3])
            output = module(output)
        """
            out_drop ： 当前量化层的输出A_l^~
            cur_out ： 当前对应FP层的输出A_l
            output ： 量化模型最终的预测
            output_fp ： FP模型最终的预测
        """
        err = loss_func(out_drop, cur_out, output, output_fp)

        err.backward(retain_graph=True)
        if w_opt:
            w_opt.step()
        if a_opt:
            a_opt.step()
        if scheduler:
            scheduler.step()
        if a_scheduler:
            a_scheduler.step()
    torch.cuda.empty_cache()

    layer.weight_quantizer.soft_targets = False
    layer.act_quantizer.is_training = False
    """
        标识位，当前层做完量化了，后续就会跳过
    """
    layer.trained = True

"""
    这个 LossFunction 类定义了一个用于适应性舍入的总损失函数，包括【重构损失】、【舍入损失】和【预测差异损失】。
    这个类是在优化量化神经网络层的过程中使用的。
"""
class LossFunction:
    def __init__(self,
                 layer: QuantModule,  # 量化模块的实例，通常包含了权重和激活函数的量化版本。
                 round_loss: str = 'relaxation',  # 指定舍入损失的类型，目前支持 'relaxation'。
                 weight: float = 1.,
                 rec_loss: str = 'mse',  # 指定重构损失的类型，目前只支持 'mse'（均方误差）。
                 max_count: int = 2000,  # 优化过程中的最大迭代次数。
                 b_range: tuple = (10, 2),  # 温度衰减函数的范围。
                 decay_start: float = 0.0,  # 温度衰减的开始时间。
                 warmup: float = 0.0,  # 在优化开始时不计算舍入损失的时间段。
                 p: float = 2.,  # LP损失的P值。
                 lam: float = 1.0,  # lam: 预测差异损失的缩放因子
                 T: float = 7.0):  # 预测差异损失的温度参数。

        self.layer = layer
        self.round_loss = round_loss
        self.weight = weight
        self.rec_loss = rec_loss
        self.loss_start = max_count * warmup
        self.p = p
        self.lam = lam
        self.T = T

        self.temp_decay = LinearTempDecay(max_count, rel_start_decay=warmup + (1 - warmup) * decay_start,
                                          start_b=b_range[0], end_b=b_range[1])
        self.count = 0
        self.pd_loss = torch.nn.KLDivLoss(reduction='batchmean')

    """
        计算自适应舍入的总损失：
        
        rec_loss是二次输出重构损失，round_loss为优化舍入策略的正则化项，pd_loss是预测差异损失。
        
        :param pred: 量化模型的输出
        :param tgt: FP模型的输出
        :param output: 量化模型预测
        :param output_fp: FP模型预测
        :return: 返回损失函数
    """
    def __call__(self, pred, tgt, output, output_fp):
        """
        Compute the total loss for adaptive rounding:
        rec_loss is the quadratic output reconstruction loss, round_loss is
        a regularization term to optimize the rounding policy, pd_loss is the 
        prediction difference loss.

        :param pred: output from quantized model
        :param tgt: output from FP model
        :param output: prediction from quantized model
        :param output_fp: prediction from FP model
        :return: total loss function
        """
        """根据量化层的输出和对应FP层的输出计算rec_loss"""
        self.count += 1
        if self.rec_loss == 'mse':
            rec_loss = lp_loss(pred, tgt, p=self.p)
        else:
            raise ValueError('Not supported reconstruction loss function: {}'.format(self.rec_loss))

        """根据量化模型和FP模型最后的预测计算PD loss"""
        pd_loss = self.pd_loss(F.log_softmax(output / self.T, dim=1), F.softmax(output_fp / self.T, dim=1)) / self.lam

        """
            
        """
        b = self.temp_decay(self.count)
        if self.count < self.loss_start or self.round_loss == 'none':
            b = round_loss = 0
        elif self.round_loss == 'relaxation':
            round_loss = 0
            round_vals = self.layer.weight_quantizer.get_soft_targets()
            round_loss += self.weight * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
        else:
            raise NotImplementedError

        total_loss = rec_loss + round_loss + pd_loss

        """每迭代500次输出loss数值"""
        if self.count % 500 == 0:
            print('Total loss:\t{:.3f} (rec:{:.3f}, pd:{:.3f}, round:{:.3f})\tb={:.2f}\tcount={}'.format(
                float(total_loss), float(rec_loss), float(pd_loss), float(round_loss), b, self.count))
        return total_loss
