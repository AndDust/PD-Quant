import torch.nn as nn
import torch
import numpy as np

from .quant_layer import QuantModule, UniformAffineQuantizer
from models.resnet import BasicBlock, Bottleneck
from models.regnet import ResBottleneckBlock
from models.mobilenetv2 import InvertedResidual
from models.mnasnet import _InvertedResidual

from torch.autograd import Variable

""" 
    提供了多种神经网络块的量化版本，特别为不同的架构（例如ResNet、RegNetX、MobileNetV2和MNASNet）设计。
    此代码提供了一种方法来量化不同的神经网络块，以便在进行低位宽度计算时仍保持其功能。
    
    这个 BaseQuantBlock 类用作构建量化神经网络中块结构的基础实现。
    它包含了一些基本的设置和方法，为子类提供了一个统一的接口来管理权重和激活函数的量化状态。
"""
class BaseQuantBlock(nn.Module):
    """
    这个基础模块确保了在分支架构网络中，激活函数和量化操作是在元素级加法操作之后执行的，这是因为在这种结构中，
    多个分支的输出需要相加，而激活函数和量化操作通常应用在这个加法操作之后。通过使用这个基础模块，
    开发者可以更容易地实现和管理量化神经网络中的块结构，并确保量化操作在正确的位置执行。

    Base implementation of block structures for all networks.
    Due to the branch architecture, we have to perform activation function
    and quantization after the elemental-wise add operation, therefore, we
    put this part in this class.
    """
    def __init__(self):
        super().__init__()
        self.use_weight_quant = False
        self.use_act_quant = False
        self.ignore_reconstruction = False
        self.trained = False

    """ 
        设置权重量化和激活函数量化的状态。
        这个方法还遍历模块的所有子模块，如果子模块是一个 QuantModule 类的实例，它也会更新这些子模块的量化状态。
    """
    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, QuantModule):
                m.set_quant_state(weight_quant, act_quant)


class QuantSTN3d(BaseQuantBlock):
    def __init__(self,  weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super(QuantSTN3d, self).__init__()
        self.conv1 = QuantModule(nn.Conv1d(3, 64, 1), weight_quant_params, act_quant_params)
        self.conv1.norm_function = nn.BatchNorm1d(64)
        self.conv1.activation_function = nn.ReLU()

        self.conv2 = QuantModule(nn.Conv1d(64, 128, 1), weight_quant_params, act_quant_params)
        self.conv2.norm_function = nn.BatchNorm1d(128)
        self.conv2.activation_function = nn.ReLU()

        self.conv3 = QuantModule(nn.Conv1d(128, 1024, 1), weight_quant_params, act_quant_params)
        self.conv3.norm_function = nn.BatchNorm1d(1024)
        self.conv3.activation_function = nn.ReLU()

        self.fc1 = QuantModule(nn.Linear(1024, 512), weight_quant_params, act_quant_params)
        self.fc1.norm_function = nn.BatchNorm1d(512)
        self.fc1.activation_function = nn.ReLU()

        self.fc2 = QuantModule(nn.Linear(512, 256), weight_quant_params, act_quant_params)
        self.fc2.norm_function = nn.BatchNorm1d(256)
        self.fc2.activation_function = nn.ReLU()

        self.fc3 = QuantModule(nn.Linear(256, 9), weight_quant_params, act_quant_params)

        self.activation_function = nn.ReLU()
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

    def forward(self, x):
        batchsize = x.size()[0]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)

        if self.use_act_quant:
            x = self.act_quantizer(x)
        return x

"""
    用于ResNet-18和ResNet-34的量化版本的基本块。
    它量化了两个卷积层，并在两个分支相加之后再应用激活函数和激活量化。
   
    在ResNet论文中，residual block有两种形式，一种叫BasicBlock，一种叫Bottleneck
    
    对ResNet中的BasicBlock结构进行量化
"""
class QuantBasicBlock(BaseQuantBlock):
    """
    Implementation of Quantized BasicBlock used in ResNet-18 and ResNet-34.
    """
    def __init__(self, basic_block: BasicBlock, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__()
        self.conv1 = QuantModule(basic_block.conv1, weight_quant_params, act_quant_params)
        self.conv1.norm_function = basic_block.bn1
        self.conv1.activation_function = basic_block.relu1
        self.conv2 = QuantModule(basic_block.conv2, weight_quant_params, act_quant_params, disable_act_quant=True)
        self.conv2.norm_function = basic_block.bn2

        if basic_block.downsample is None:
            self.downsample = None
        else:
            self.downsample = QuantModule(basic_block.downsample[0], weight_quant_params, act_quant_params,
                                          disable_act_quant=True)
            self.downsample.norm_function = basic_block.downsample[1]
        self.activation_function = basic_block.relu2
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        """
            计算block的输出
        """
        out = self.activation_function(out)
        """
            如果开启了激活量化，则对输出进行量化
        """
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out

"""
    对ResNet中的bottleneck结构进行量化
    
    用于ResNet-50、ResNet-101和ResNet-152的量化版Bottleneck块。
    量化了三个卷积层，并在合并之后进行激活和激活量化。
"""
class QuantBottleneck(BaseQuantBlock):
    """
    Implementation of Quantized Bottleneck Block used in ResNet-50, -101 and -152.
    """

    def __init__(self, bottleneck: Bottleneck, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__()
        self.conv1 = QuantModule(bottleneck.conv1, weight_quant_params, act_quant_params)
        self.conv1.norm_function = bottleneck.bn1
        self.conv1.activation_function = bottleneck.relu1
        self.conv2 = QuantModule(bottleneck.conv2, weight_quant_params, act_quant_params)
        self.conv2.norm_function = bottleneck.bn2
        self.conv2.activation_function = bottleneck.relu2
        self.conv3 = QuantModule(bottleneck.conv3, weight_quant_params, act_quant_params, disable_act_quant=True)
        self.conv3.norm_function = bottleneck.bn3

        if bottleneck.downsample is None:
            self.downsample = None
        else:
            self.downsample = QuantModule(bottleneck.downsample[0], weight_quant_params, act_quant_params,
                                          disable_act_quant=True)
            self.downsample.norm_function = bottleneck.downsample[1]
        # modify the activation function to ReLU
        self.activation_function = bottleneck.relu3
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += residual
        out = self.activation_function(out)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out

"""
    为RegNetX设计的量化Bottleneck块，但没有SE模块。
    与QuantBottleneck类似，但结构来源于RegNetX。
"""
class QuantResBottleneckBlock(BaseQuantBlock):
    """
    Implementation of Quantized Bottleneck Blockused in RegNetX (no SE module).
    """

    def __init__(self, bottleneck: ResBottleneckBlock, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__()
        self.conv1 = QuantModule(bottleneck.f.a, weight_quant_params, act_quant_params)
        self.conv1.norm_function = bottleneck.f.a_bn
        self.conv1.activation_function = bottleneck.f.a_relu
        self.conv2 = QuantModule(bottleneck.f.b, weight_quant_params, act_quant_params)
        self.conv2.norm_function = bottleneck.f.b_bn
        self.conv2.activation_function = bottleneck.f.b_relu
        self.conv3 = QuantModule(bottleneck.f.c, weight_quant_params, act_quant_params, disable_act_quant=True)
        self.conv3.norm_function = bottleneck.f.c_bn

        if bottleneck.proj_block:
            self.downsample = QuantModule(bottleneck.proj, weight_quant_params, act_quant_params,
                                          disable_act_quant=True)
            self.downsample.norm_function = bottleneck.bn
        else:
            self.downsample = None
        # copying all attributes in original block
        self.proj_block = bottleneck.proj_block

        self.activation_function = bottleneck.relu
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

    def forward(self, x):
        residual = x if not self.proj_block else self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += residual
        out = self.activation_function(out)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out

"""
    MobileNetV2中使用的量化Inverted Residual Block的实现。
    反向残差没有激活功能。
    
    用于MobileNetV2的量化版Inverted Residual块。
    根据扩展率有不同的结构，可能包含2或3个卷积层。
"""
class QuantInvertedResidual(BaseQuantBlock):
    """
    Implementation of Quantized Inverted Residual Block used in MobileNetV2.
    Inverted Residual does not have activation function.
    """

    def __init__(self, inv_res: InvertedResidual, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__()

        self.use_res_connect = inv_res.use_res_connect
        self.expand_ratio = inv_res.expand_ratio
        if self.expand_ratio == 1:
            self.conv = nn.Sequential(
                QuantModule(inv_res.conv[0], weight_quant_params, act_quant_params),
                QuantModule(inv_res.conv[3], weight_quant_params, act_quant_params, disable_act_quant=True),
            )
            self.conv[0].norm_function = inv_res.conv[1]
            self.conv[0].activation_function = nn.ReLU6()
            self.conv[1].norm_function = inv_res.conv[4]
        else:
            self.conv = nn.Sequential(
                QuantModule(inv_res.conv[0], weight_quant_params, act_quant_params),
                QuantModule(inv_res.conv[3], weight_quant_params, act_quant_params),
                QuantModule(inv_res.conv[6], weight_quant_params, act_quant_params, disable_act_quant=True),
            )
            self.conv[0].norm_function = inv_res.conv[1]
            self.conv[0].activation_function = nn.ReLU6()
            self.conv[1].norm_function = inv_res.conv[4]
            self.conv[1].activation_function = nn.ReLU6()
            self.conv[2].norm_function = inv_res.conv[7]
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

    def forward(self, x):
        if self.use_res_connect:
            out = x + self.conv(x)
        else:
            out = self.conv(x)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out

"""
    用于MNASNet的量化版Inverted Residual块。
    它包含三个卷积层，并根据apply_residual属性决定是否应用残差连接。
"""
class _QuantInvertedResidual(BaseQuantBlock):
    def __init__(self, _inv_res: _InvertedResidual, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__()

        self.apply_residual = _inv_res.apply_residual
        self.conv = nn.Sequential(
            QuantModule(_inv_res.layers[0], weight_quant_params, act_quant_params),
            QuantModule(_inv_res.layers[3], weight_quant_params, act_quant_params),
            QuantModule(_inv_res.layers[6], weight_quant_params, act_quant_params, disable_act_quant=True),
        )
        self.conv[0].activation_function = nn.ReLU()
        self.conv[0].norm_function = _inv_res.layers[1]
        self.conv[1].activation_function = nn.ReLU()
        self.conv[1].norm_function = _inv_res.layers[4]
        self.conv[2].norm_function = _inv_res.layers[7]
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

    def forward(self, x):
        if self.apply_residual:
            out = x + self.conv(x)
        else:
            out = self.conv(x)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out

"""
    这两个字典/列表是为了简化在其他部分代码中选择特定块的逻辑。
    specials将原始块映射到它们的量化版本，而specials_unquantized列出了不需要量化的特定层。 
"""
specials = {
    BasicBlock: QuantBasicBlock,
    Bottleneck: QuantBottleneck,
    ResBottleneckBlock: QuantResBottleneckBlock,
    InvertedResidual: QuantInvertedResidual,
    _InvertedResidual: _QuantInvertedResidual,

}

specials_unquantized = [nn.AdaptiveAvgPool2d, nn.MaxPool2d, nn.Dropout]
