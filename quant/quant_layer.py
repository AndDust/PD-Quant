import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union

"""
    直通
"""
class StraightThrough(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input

"""
    直通估计
    直通估计是一种常用于训练量化神经网络的技巧，它允许在前向传播期间应用量化操作（如舍入），同时在反向传播期间保持梯度的连续性。
"""
def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    """
        (x.round() - x) ： 这一部分计算了舍入操作后的误差，即每个元素四舍五入后与原始值 x 之间的差。
        + x：最后，将误差和原始值 x 相加，以获得最终的结果。这里的操作相当于在前向传播中应用了舍入操作，但保持了梯度的连续性。
        
        detach() 方法允许在反向传播期间将梯度流向原始输入值，而不会受到离散操作的影响。
        这样，模型可以在前向传播中执行离散操作，如舍入，同时保持梯度的连续性，以便在反向传播中更新模型参数。
        这是 STE 的核心原理之一，使其成为在训练期间模拟离散操作的有效工具。
        
        在 STE 中，当我们在前向传播中执行了离散操作（例如，将一个实数值舍入为整数），原始的输入值和执行离散操作后的值之间存在差异。
        这个差异在前向传播中会传播到模型的输出，但在反向传播中，我们需要将梯度回传到原始输入值，而不是离散操作后的值。这就是 detach() 方法的作用。
        
        1.8.round -1.8 +1.8 = (2 -1.8) +1.8 = 2 
        5.3.round -5.3 +5.3 = (5 -5.3) +5.3 = 5
        但是梯度会回传到原始输入值1.8和5.3，而不是离散操作后的值2和5。
    """
    return (x.round() - x).detach() + x

"""
    如果 reduction 为 'none'，则返回每个样本的损失值的平均值（.mean()），这将返回一个与输入张量形状相同的张量，每个元素表示对应样本的损失值。
    如果 reduction 不是 'none'，则返回所有样本的损失值的平均值（.mean()），这将返回一个标量值，表示整体损失。
"""
def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if reduction == 'none':
        return (pred - tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred - tgt).abs().pow(p).mean()

"""
    PyTorch函数，可用于非对称量化（也称为均匀仿射）量化）。在正向传递中量化其参数，直接传递梯度“through'，忽略发生的量化。
    基于https://arxiv.org/abs/1806.08342.
"""
class UniformAffineQuantizer(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.

    :param n_bits: number of bit for quantization
    :param symmetric: if True, the zero_point should always be 0
    :param channel_wise: if True, compute scale and zero_point in each channel
    :param scale_method: determines the quantization scale and zero point
    :param prob: for qdrop;
    """

    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False,
                 scale_method: str = 'minmax',
                 leaf_param: bool = False, prob: float = 1.0):
        super(UniformAffineQuantizer, self).__init__()
        self.sym = symmetric  # 如果为True，则zero_point应始终为0

        """没有实现对称量化"""
        if self.sym:
            raise NotImplementedError
        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        """scale"""
        self.delta = 1.0
        self.zero_point = 0.0
        self.inited = True

        '''if leaf_param, use EMA to set scale '''
        self.leaf_param = leaf_param

        """如果为True，将在每个通道中计算比例和零点"""
        self.channel_wise = channel_wise  # 如果为True，则计算每个通道中的比例和零点
        self.eps = torch.tensor(1e-8, dtype=torch.float32)

        '''mse params'''
        """
            scale_method: determines the quantization scale and zero point
        """
        self.scale_method = 'mse'
        self.one_side_dist = None
        self.num = 100

        '''for activation quantization'''
        self.running_min = None
        self.running_max = None

        '''do like dropout'''
        self.prob = prob
        self.is_training = False

        self.is_act = False
        self.bn_estimate_abs_max = 0

    def set_inited(self, inited: bool = True):  # inited manually
        self.inited = inited

    """更新量化动态范围"""
    def update_quantize_range(self, x_min, x_max):
        if self.running_min is None:
            self.running_min = x_min
            self.running_max = x_max
        self.running_min = 0.1 * x_min + 0.9 * self.running_min
        self.running_max = 0.1 * x_max + 0.9 * self.running_max
        return self.running_min, self.running_max

    """
        x.shape : torch.Size([64, 3, 7, 7])
        x[0][0][0] : tensor([ 0.0072,  0.0093,  0.0093,  0.0025,  0.0003, -0.0066, -0.0037])
    """
    def forward(self, x: torch.Tensor):
        if self.inited is False and not self.is_act:
            """
                先使用mse的方法确定动态范围，然后计算得到scale和zero_points
            """
            if self.leaf_param:
                self.delta, self.zero_point = self.init_quantization_scale(x.clone().detach(), self.channel_wise)
            else:
                self.delta, self.zero_point = self.init_quantization_scale(x.clone().detach(), self.channel_wise)

        # start quantization
        """
            执行量化
            x.shape : torch.Size([64, 64, 3, 3])
                x[0][0][0] : tensor([-0.0333, -0.1145, -0.0190]
            
            self.delta.shape : torch.Size([64, 1, 1, 1])
                self.delta[0] : tensor([[[0.0551]]]
                
            self.zero_point.shape : torch.Size([64, 1, 1, 1])
                self.zero_point[0][0][0] : tensor([7.]
                
            x_int.shape : torch.Size([64, 64, 3, 3])
                x_int[0][0][0] : tensor([6., 5., 7.]
                
            x_quant.shape : torch.Size([64, 64, 3, 3])
                x_quant[0][0][0] : tensor([6., 5., 7.]
                
            x_int[0][0][0] : tensor([ 95.,  99., 100.,  85.,  81.,  66.,  72.]
            x_quant[0][0][0] : tensor([ 95.,  99., 100.,  85.,  81.,  66.,  72.],
            x_dequant[0][0][0] : tensor([ 0.0071,  0.0090,  0.0095,  0.0024,  0.0005, -0.0067, -0.0038],
        """

        x_int = round_ste(x / self.delta) + self.zero_point
        # x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        if self.is_act:
            x_quant = torch.clamp(x_int, -self.n_levels / 2, self.n_levels / 2 - 1)
        else:
            x_quant = torch.clamp(x_int, 0, self.n_levels - 1)


        # if self.is_act:
        #     print("delta:{}".format(self.delta))

        """
            反量化
        """
        x_dequant = (x_quant - self.zero_point) * self.delta
        """"""

        # if self.is_act:
        #     if self.delta.item() < 0.07239 and self.delta.item() > 0.07237:
        #         # print("x_int:{}".format(x_int))
        #         # print("x_quant:{}".format(x_quant))
        #         # print("clamp:{},{}".format(-self.n_levels / 2, self.n_levels / 2 - 1))
        #         print("self.delta:{} self.zero_point:{}".format(self.delta, self.zero_point))
        #         print("x:{}".format(x))
        #         print("x_dequant:{}".format(x_dequant))
        #
        #     if self.delta.item() < 7 and self.delta.item() > 5:
        #         # print("x_int:{}".format(x_int))
        #         # print("x_quant:{}".format(x_quant))
        #         # print("clamp:{},{}".format(-self.n_levels / 2, self.n_levels / 2 - 1))
        #         print("self.delta:{} self.zero_point:{}".format(self.delta, self.zero_point))
        #         print("x:{}".format(x))
        #         print("x_dequant:{}".format(x_dequant))

        if self.is_training and self.prob < 1.0:
            """训练过程，并有丢弃"""
            x_ans = torch.where(torch.rand_like(x) < self.prob, x_dequant, x)
        else:
            """返回反量化结果"""
            x_ans = x_dequant
        return x_ans

    def lp_loss(self, pred, tgt, p=2.0):
        x = (pred - tgt).abs().pow(p)
        if not self.channel_wise:
            return x.mean()
        else:
            y = torch.flatten(x, 1)
            return y.mean(1)

    """
        计算量化参数
    """

    """
        min_val.shape : torch.Size([64])
        max_val.shape : torch.Size([64])
    """
    def calculate_qparams(self, min_val, max_val):
        # one_dim or one element
        """量化范围为[0，255]"""
        quant_min, quant_max = 0, self.n_levels - 1

        """
            这一步的原因？
        """
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
        """
            计算scale
        """
        scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
        """
            确保scale不为0？
        """
        scale = torch.max(scale, self.eps)
        """
            计算零点zero_point
        """
        zero_point = quant_min - torch.round(min_val_neg / scale)
        """
            限制得到的零点范围在量化区间
        """
        zero_point = torch.clamp(zero_point, quant_min, quant_max)
        return scale, zero_point

    """对x进行量化，得到反量化结果"""
    def quantize(self, x: torch.Tensor, x_max, x_min):
        """计算量化参数得到scale和零点"""
        delta, zero_point = self.calculate_qparams(x_min, x_max)
        """
            如果是按通道量化
        """
        if self.channel_wise:
            new_shape = [1] * len(x.shape)
            new_shape[0] = x.shape[0]
            """每个通道有自己的scale和zero_point"""
            delta = delta.reshape(new_shape)
            zero_point = zero_point.reshape(new_shape)
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        """
            执行反量化
        """
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q

    """
        perform_2D_search表示对有正负值的进行搜索
        x.shape : torch.Size([64, 3, 7, 7])
    """
    def perform_2D_search(self, x):
        if self.channel_wise:
            """按通道查找每通道的最大最小值"""
            y = torch.flatten(x, 1)
            """
                x_min.shape : torch.Size([64])
                x_max.shape : torch.Size([64])
            """
            x_min, x_max = torch._aminmax(y, 1)
            # may also have the one side distribution in some channels
            """
                在某些情况下，某些通道可能只有正值或负值，因此使用 torch.max(x_max, torch.zeros_like(x_max)) 
                和 torch.min(x_min, torch.zeros_like(x_min)) 来确保 x_max 和 x_min 中的值非负或非正。
                这是为了处理可能的单侧分布情况。
            """
            x_max = torch.max(x_max, torch.zeros_like(x_max))
            x_min = torch.min(x_min, torch.zeros_like(x_min))
        else:
            """不按照通道，查找全局的最大最小值"""
            x_min, x_max = torch._aminmax(x)

        """
            xrange.shape : torch.Size([64]) xrange：每个通道的最大最小值之差
        """
        xrange = x_max - x_min
        """
            初始化 best_score 为一个非常大的数（1e+10）
            best_score 是用来跟踪最佳的损失分数，初始设置为一个很大的值，以便后续的分数可以比较并找到更小的值。
        """
        best_score = torch.zeros_like(x_min) + (1e+10)
        """初始化最好的min和max为原始的min和max"""
        best_min = x_min.clone()
        best_max = x_max.clone()

        """
            self.num是什么？ 为什么要从1遍历到self.num + 1？
            
            在每次循环中，首先计算 tmp_min 和 tmp_max，它们表示在当前循环中用于搜索的最小值和最大值的候选范围。
            这些范围根据 xrange 和 self.num 计算而来。
        """
        # enumerate xrange
        for i in range(1, self.num + 1):
            tmp_min = torch.zeros_like(x_min)
            tmp_max = xrange / self.num * i
            """
                尝试计算一个scale
            """
            tmp_delta = (tmp_max - tmp_min) / (2 ** self.n_bits - 1)
            # enumerate zp
            """
                self.n_levels ： 256
                尝试取遍 [0,self.n_levels]中每个作为zero_point
            """
            for zp in range(0, self.n_levels):
                """
                    减去偏移
                """
                new_min = tmp_min - zp * tmp_delta
                new_max = tmp_max - zp * tmp_delta
                """执行量化操作"""
                x_q = self.quantize(x, new_max, new_min)
                """计算量化前后的MSE"""
                score = self.lp_loss(x, x_q, 2.4)
                """
                    根据score更新best_min和best_max
                """
                best_min = torch.where(score < best_score, new_min, best_min)
                best_max = torch.where(score < best_score, new_max, best_max)
                best_score = torch.min(best_score, score)

        return best_min, best_max

    """
        perform_1D_search 表示 对只有正值或负值的进行搜索
        寻找best_min和best_max值
        
        采用MSE的方法寻找最好的动态范围：
    """
    def perform_1D_search(self, x):
        if self.channel_wise:
            y = torch.flatten(x, 1)
            x_min, x_max = torch._aminmax(y, 1)
        else:
            x_min, x_max = torch._aminmax(x)

        xrange = torch.max(x_min.abs(), x_max)

        """初始化 best_score 为一个较高的值（1e+10）"""
        best_score = torch.zeros_like(x_min) + (1e+10)
        """
            best_min 为 x_min 的副本，best_max 为 x_max 的副本。
            这些变量用于跟踪最佳的 new_min 和 new_max 值以及对应的损失分数。
        """
        best_min = x_min.clone()
        best_max = x_max.clone()
        # enumerate xrange
        """
            在这段代码中，self.num定义了在搜索最佳量化动态范围时要尝试的不同阈值的数量。
            具体来说，它将量化范围（从最小值到最大值）划分为self.num个相等的间隔，
            并在这些间隔上尝试不同的最小值和最大值，以找到导致最小均方误差（MSE）的量化范围。
            
            self.num的值越大，搜索就越精细，能够尝试的量化范围组合就越多，理论上能够找到更准确的最小MSE对应的量化范围。
            然而，增加self.num的值也会导致计算量增加，从而增加搜索最佳量化范围所需的时间。
            
            在这个具体的例子中，self.num被设置为100，意味着代码将尝试100个不同的量化范围组合，并从中选择MSE最小的那个作为最终的量化范围。
            这是一种在精度和计算时间之间进行权衡的方法。
        """
        for i in range(1, self.num + 1):
            """
                首先执行除法运算，然后将结果乘以i。
                这个表达式的目的是将xrange等分为self.num个部分，并取其中的第i部分作为阈值thres。
            """
            thres = xrange / self.num * i

            new_min = torch.zeros_like(x_min) if self.one_side_dist == 'pos' else -thres
            new_max = torch.zeros_like(x_max) if self.one_side_dist == 'neg' else thres

            """
                使用 self.quantize 函数对输入 x 进行量化，使用新的 new_max 和 new_min 值
            """
            x_q = self.quantize(x, new_max, new_min)
            """
                计算量化后的结果与原始输入之间的 MSE（score） why 2.4?
            """
            score = self.lp_loss(x, x_q, 2.4)
            """
                通过比较MSE，更新 best_min 和 best_max 的值，选择具有更低损失分数的值。
                最终，记录最佳的 best_min 和 best_max 以及对应的损失分数。
            """
            best_min = torch.where(score < best_score, new_min, best_min)
            best_max = torch.where(score < best_score, new_max, best_max)
            best_score = torch.min(score, best_score)
        return best_min, best_max

    """
        只实现了 mse 的scale_method
        使用mse方法来确定动态范围
    """
    def get_x_min_x_max(self, x):
        if self.scale_method != 'mse':
            raise NotImplementedError
        """
            one_side_dist 是一个内部变量，用来记录输入数据分布的类型。
            如果 one_side_dist 为空，即还未设置，根据输入数据 x 的最小值和最大值来判断其分布类型，并更新 one_side_dist 的值。
        """
        if self.one_side_dist is None:
            """如果输入数据全部为正或全部为负，它将分别设置为 'pos' 或 'neg'；如果数据中既有正值又有负值，它将设置为 'no"""
            self.one_side_dist = 'pos' if x.min() >= 0.0 else 'neg' if x.max() <= 0.0 else 'no'

        if self.one_side_dist != 'no' or self.sym:
            """如果数据中只有正值或负值，执行一维搜索。"""  # one-side distribution or symmetric value for 1-d search
            best_min, best_max = self.perform_1D_search(x)
        else:  # 2-d search
            """如果数据中既有正值又有负值，执行二维搜索。"""
            best_min, best_max = self.perform_2D_search(x)

        if self.leaf_param:
            return self.update_quantize_range(best_min, best_max)

        """返回动态范围"""
        return best_min, best_max

    def init_quantization_scale_channel(self, x: torch.Tensor):
        """使用mse方法来确定动态范围"""
        x_min, x_max = self.get_x_min_x_max(x)
        return self.calculate_qparams(x_min, x_max)

    """
        初始化量化比例
        x_clone.shape : torch.Size([64, 3, 7, 7])
        
    """
    def init_quantization_scale(self, x_clone: torch.Tensor, channel_wise: bool = False):
        """按通道量化"""
        if channel_wise:
            # determine the scale and zero point channel-by-channel 逐个通道确定scale和zero point
            delta, zero_point = self.init_quantization_scale_channel(x_clone)
            new_shape = [1] * len(x_clone.shape)
            new_shape[0] = x_clone.shape[0]
            delta = delta.reshape(new_shape)
            zero_point = zero_point.reshape(new_shape)

            """不按通道量化"""
        else:
            delta, zero_point = self.init_quantization_scale_channel(x_clone)
        return delta, zero_point

    def bitwidth_refactor(self, refactored_bit: int):
        assert 2 <= refactored_bit <= 8, 'bitwidth not supported'
        self.n_bits = refactored_bit
        self.n_levels = 2 ** self.n_bits

    @torch.jit.export
    def extra_repr(self):
        return 'bit={}, is_training={}, inited={}'.format(
            self.n_bits, self.is_training, self.inited
        )

"""
    量化模块，可以执行量化卷积或正常卷积。
    要激活量化，请使用set_quant_state函数。
"""
class QuantModule(nn.Module):
    """
    量化模块，可以执行量化卷积或正常卷积。要激活量化，请使用set_quant_state函数。

    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    """
        org_module：传入的原始模块，可以是 nn.Conv2d 或 nn.Linear等。这个模块将被量化。
        weight_quant_params：用于权重量化的参数字典。
        act_quant_params：用于激活函数输出量化的参数字典。
        disable_act_quant：一个布尔值，如果设置为 True，将禁用激活函数的输出量化。
    """
    def __init__(self, org_module: Union[nn.Conv2d, nn.Linear, nn.Conv1d], weight_quant_params: dict = {},
                 act_quant_params: dict = {}, disable_act_quant=False):
        super(QuantModule, self).__init__()

        """
            fwd_kwargs 和 fwd_func：用于保存卷积操作的参数和函数。
            weight 和 bias：模块的权重和偏差。
            org_weight 和 org_bias：原始权重和偏差的副本，用于在非量化模式下恢复权重和偏差。
            use_weight_quant 和 use_act_quant：布尔值，用于表示是否启用权重和激活量化。
            
            weight_quantizer 和 act_quantizer：权重和激活量化器对象，根据传入的参数初始化。
            
            norm_function 和 activation_function：用于进行归一化和激活的函数，通常是 StraightThrough，表示无操作。
            ignore_reconstruction：一个布尔值，是否忽略重构（不太清楚上下文中的用途）。
            disable_act_quant：是否禁用激活函数的输出量化。
            trained：是否已经训练过模型。
        """
        if isinstance(org_module, nn.Conv2d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv2d
        elif isinstance(org_module, nn.Conv1d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv1d
        else:
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear

        self.weight = org_module.weight

        self.org_weight = org_module.weight.data.clone()
        if org_module.bias is not None:
            self.bias = org_module.bias
            self.org_bias = org_module.bias.data.clone()
        else:
            self.bias = None
            self.org_bias = None

        """停用量化前向传播默认值"""
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False

        """初始化权重和激活的量化器"""
        # initialize quantizer
        self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params)
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)
        self.act_quantizer.is_act = True

        self.norm_function = StraightThrough()
        self.activation_function = StraightThrough()
        self.ignore_reconstruction = False
        self.disable_act_quant = disable_act_quant

        """一旦完成重建，就设置为True，后续就会跳过"""
        self.trained = False
        self.count = 0

        self.quant_name = "null"

    """
        会在layer reconstruction中调用到，qnn前向传播时对权重进行量化
        
        量化模块，可以执行量化卷积或正常卷积。要激活量化，请使用set_quant_state函数。
        input.shape : torch.Size([32, 3, 224, 224])
        
        self.weight_quantizer(self.weight)[0][0][0]
        tensor([ 0.0071,  0.0090,  0.0095,  0.0024,  0.0005, -0.0067, -0.0038],
        
        self.org_weight[0][0][0]
        tensor([ 0.0072,  0.0093,  0.0093,  0.0025,  0.0003, -0.0066, -0.0037],
    """
    def forward(self, input: torch.Tensor):
        # 使用权重量化
        if self.use_weight_quant:
            """对权重进行量化"""
            pre = self.weight
            if torch.max(pre) > 20:
                print("大权重：{}".format(pre))
            # print("量化之前权重：{}",format(self.weight))
            weight = self.weight_quantizer(self.weight)
            latter = weight
            # print("量化之后的权重：{}", format(weight))
            bias = self.bias

            # if self.count == 0:
            #     torch.set_printoptions(precision=8, sci_mode=False)
            # print("权重量化前后之差：{}".format(latter - pre))
        # 不使用权重量化
        else:
            weight = self.org_weight
            bias = self.org_bias

        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)

        if self.act_quantizer.inited == False:
            print("out.shape:{}".format(out.shape))

        # if self.act_quantizer.inited == False and isinstance(self.norm_function,(nn.BatchNorm2d, nn.BatchNorm1d)):
        #     mean = self.norm_function.running_mean
        #     var = self.norm_function.running_var
        #
        #     C = out.shape[1]
        #     op = None
        #     if out.dim() == 4:
        #         op = out.permute(0, 2, 3, 1)
        #         op = op.reshape(-1, C)
        #     else:
        #         op = out
        #
        #     k = int(0.999 * op.size(0))
        #     percentile_90, _ = torch.kthvalue(op, k, dim=0)
        #
        #     max_values_dim2, max_indices_dim2 = torch.max(op, dim=0)
        #     print("conv输出得到的最大值：{}".format(torch.max(max_values_dim2)))
        #     # print("conv输出得到的99.9分位值：{}".format(torch.max(percentile_90)))
        #     # n = torch.numel(op)
        #     # s = (n-1)/math.sqrt(n)
        #     # print("n的大小：{} s的大小：{}".format(n, s))
        #     print("BN层数据估计出来的最大值：{}".format(torch.max(mean + 3*torch.sqrt(var))))
        #
        #     print("conv输出得到的最小值：{}".format(torch.min(max_values_dim2)))
        #     print("BN层数据估计出来的最小值：{}".format(torch.max(mean - 3 * torch.sqrt(var))))
        #
        #     self.act_quantizer.bn_estimate_abs_max = mean + 3*torch.sqrt(var)
        #     print("bn_estimate_abs_max:{}".format(self.act_quantizer.bn_estimate_abs_max))

        # disable act quantization is designed for convolution before elemental-wise operation,
        # in that case, we apply activation function and quantization after ele-wise op.
        if self.disable_act_quant:
            """对于非conv+BN+Relu结构的模块只进行权重量化，不进行激活的量化"""
            out = out
            # print("------------self.act_quantizer.delta:{}".format(self.act_quantizer.delta))


            # if self.count == 0:

            # if torch.max(pre) > 100:
            #     print("pre:{}".format(pre))
            #     print("latter:{}".format(latter))
            #     torch.set_printoptions(precision=8, sci_mode=False)
            #     print("激活量化前后相减：{}".format((latter - pre)))

            # self.count +=1

            # if torch.max((latter - pre)) > 100:
            #     print("pre:{}".format(torch.flatten(pre)[:10]))
            #     print("latter:{}".format(torch.flatten(latter)[:10]))
            #     # torch.set_printoptions(precision=8, sci_mode=False)
            #     print("激活量化前后相减：{}".format(torch.flatten((latter - pre))[:10]))
            #     print("scale的值是：{}".format(self.act_quantizer.delta))

        out = self.norm_function(out)
        if self.use_act_quant and not self.disable_act_quant:
            # print("BN后输出最大值：{}".format(torch.max(out)))
            # print("使用BN的gamma和beta估计的最大值：{}".format(self.act_quantizer.bn_estimate_abs_max))
            # print("scale的值是：{}".format(self.act_quantizer.delta))

            pre = out
            out = self.act_quantizer(out)
            # print("++++++++++++self.act_quantizer.delta:{}".format(self.act_quantizer.delta))
            # latter = out
            # print("pre:{}".format(pre))
            # print("latter:{}".format(latter))
            # print("激活量化前后相减：{}".format((latter - pre)))

        out = self.activation_function(out)
        # if self.disable_act_quant:
        #     return out
        # if self.use_act_quant:
        #     out = self.act_quantizer(out)
        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    @torch.jit.export
    def extra_repr(self):
        return 'wbit={}, abit={}, disable_act_quant={}'.format(
            self.weight_quantizer.n_bits, self.act_quantizer.n_bits, self.disable_act_quant
        )
