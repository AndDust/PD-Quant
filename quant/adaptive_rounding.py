import torch
from torch import nn
from .quant_layer import UniformAffineQuantizer, round_ste

"""
    这个类AdaRoundQuantizer实现了一种自适应舍入策略的量化器，基于论文 "Up or Down? Adaptive Rounding for Post-Training Quantization"。
    这种量化方法的目标是通过优化舍入策略来重建中间层的输出，从而减少量化带来的误差，并提高量化后模型的性能。
    
    param uaq: UniformAffineQuantizer, 用于初始化此量化器中的量化参数
    param round_mode: 控制该量化器取整的策略
    param weight_tensor: 
"""

class AdaRoundQuantizer(nn.Module):
    """
    Adaptive Rounding Quantizer, used to optimize the rounding policy
    by reconstructing the intermediate output.

    Based on
     Up or Down? Adaptive Rounding for Post-Training Quantization:k

    :param uaq: UniformAffineQuantizer, used to initialize quantization parameters in this quantizer
    :param round_mode: controls the forward pass in this quantizer
    :param weight_tensor: initialize alpha
    """

    def __init__(self, uaq: UniformAffineQuantizer, weight_tensor: torch.Tensor, round_mode='learned_round_sigmoid'):
        super(AdaRoundQuantizer, self).__init__()
        # copying all attributes from UniformAffineQuantizer
        """复制UniformAffineQuantizer中的所有属性"""
        self.n_bits = uaq.n_bits
        self.sym = uaq.sym
        self.delta = uaq.delta
        self.zero_point = uaq.zero_point
        self.n_levels = uaq.n_levels

        """
            量化计算时取整的策略：
            nearest: 最近邻舍入。
            nearest_ste: 最近邻舍入，但使用Straight-Through Estimator (STE)进行反向传播。
            stochastic: 随机舍入。
            learned_hard_sigmoid: 使用学习到的硬S形函数进行舍入。
        """
        self.round_mode = round_mode

        """alpha是一个待学习的参数，用于学习的硬S形函数舍入策略"""
        self.alpha = None
        """soft_targets是一个布尔值，控制是否使用软目标进行训练。"""
        self.soft_targets = False

        # params for sigmoid function
        """gamma, zeta, 和beta是硬S形函数的参数。最后，调用init_alpha方法来初始化alpha。"""
        self.gamma, self.zeta = -0.1, 1.1
        self.beta = 2/3
        self.init_alpha(x=weight_tensor.clone())

    def forward(self, x):
        if self.round_mode == 'nearest':
            x_int = torch.round(x / self.delta)
        elif self.round_mode == 'nearest_ste':
            x_int = round_ste(x / self.delta)
        elif self.round_mode == 'stochastic':
            x_floor = torch.floor(x / self.delta)
            rest = (x / self.delta) - x_floor  # rest of rounding
            x_int = x_floor + torch.bernoulli(rest)
            print('Draw stochastic sample')
        elif self.round_mode == 'learned_hard_sigmoid':
            x_floor = torch.floor(x / self.delta)
            if self.soft_targets:
                x_int = x_floor + self.get_soft_targets()
            else:
                x_int = x_floor + (self.alpha >= 0).float()
        else:
            raise ValueError('Wrong rounding mode')

        x_quant = torch.clamp(x_int + self.zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - self.zero_point) * self.delta

        return x_float_q

    def get_soft_targets(self):
        return torch.clamp(torch.sigmoid(self.alpha) * (self.zeta - self.gamma) + self.gamma, 0, 1)

    """
        基于输入张量x来初始化参数alpha。
    """
    def init_alpha(self, x: torch.Tensor):
        x_floor = torch.floor(x / self.delta)
        if self.round_mode == 'learned_hard_sigmoid':
            print('Init alpha to be FP32')
            rest = (x / self.delta) - x_floor  # rest of rounding [0, 1)
            alpha = -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1)  # => sigmoid(alpha) = rest
            self.alpha = nn.Parameter(alpha)
        else:
            raise NotImplementedError

    @torch.jit.export
    def extra_repr(self):
        return 'bit={}'.format(self.n_bits)
