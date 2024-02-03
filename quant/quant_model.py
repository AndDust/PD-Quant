import torch.nn as nn
from .quant_block import specials, BaseQuantBlock
from .quant_layer import QuantModule, StraightThrough, UniformAffineQuantizer
from .fold_bn import search_fold_and_remove_bn


class QuantModel(nn.Module):
    def __init__(self, model: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}, is_fusing=True):
        super().__init__()
        if is_fusing:
            """conv、linear进行BN折叠"""
            search_fold_and_remove_bn(model)
            """将常规conv2d和linear层递归替换为QuantModule"""
            self.model = model
            self.quant_module_refactor(self.model, weight_quant_params, act_quant_params)
        else:
            """不进行BN折叠，记录原始的FP model，用于后面对比"""
            self.model = model
            """将常规conv2d和linear层递归替换为QuantModule，QuantModule中可以通过开关决定是否打开量化"""
            self.quant_module_refactor_wo_fuse(self.model, weight_quant_params, act_quant_params)

    # def quant_module_refactor(self, module: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}):
    #     """
    #      递归地将常规的conv2d和线性层递归替换为QuantModule：param module:nn.module的子级中包含nn.Conv2d或nn.Linear
    #     ：param weight_quant_params:weight量化器的n_bits等量化参数
    #     ：param act_quant_params
    #     ：激活量化器的n_bits等量化参数
    #
    #     Recursively replace the normal conv2d and Linear layer to QuantModule
    #     :param module: nn.Module with nn.Conv2d or nn.Linear in its children
    #     :param weight_quant_params: quantization parameters like n_bits for weight quantizer
    #     :param act_quant_params: quantization parameters like n_bits for activation quantizer
    #     """
    #     prev_quantmodule = None
    #     for name, child_module in module.named_children():
    #         """针对特殊module进行量化，将module替换为量化后的module"""
    #         if type(child_module) in specials:
    #             setattr(module, name, specials[type(child_module)](child_module, weight_quant_params, act_quant_params))
    #             """对(nn.Conv2d, nn.Linear)进行量化，替换为量化版本的module"""
    #         elif isinstance(child_module, (nn.Conv2d, nn.Linear, nn.Conv1d)):
    #             setattr(module, name, QuantModule(child_module, weight_quant_params, act_quant_params))
    #             prev_quantmodule = getattr(module, name)
    #             """对(nn.ReLU, nn.ReLU6)进行"""
    #         elif isinstance(child_module, (nn.ReLU, nn.ReLU6)):
    #             if prev_quantmodule is not None:
    #                 prev_quantmodule.activation_function = child_module
    #                 setattr(module, name, StraightThrough())
    #             else:
    #                 continue
    #             """忽略StraightThrough"""
    #         elif isinstance(child_module, StraightThrough):
    #             continue
    #             """递归量化其他层，比如sequential层"""
    #         else:
    #             self.quant_module_refactor(child_module, weight_quant_params, act_quant_params)
    #
    # """
    #     【将神经网络重构为量化模型】
    #
    #     递归地遍历神经网络模型的所有层，并用量化模块（QuantModule）替换正常的卷积层（nn.Conv2d）和全连接层（nn.Linear）。
    #     同时，对于批量归一化（nn.BatchNorm2d）层和激活函数（nn.ReLU，nn.ReLU6）层，它会将它们与前面的量化模块关联，并用直通（StraightThrough）层替换它们。
    #
    #     param module:nn.module的子级中包含nn.Conv2d或nn.Linear
    #     param weight_quant_params:weight量化器的n_bits等量化参数
    #     param act_quant_params：激活量化器的n_bits等量化参数
    # """
    # def quant_module_refactor_wo_fuse(self, module: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}):
    #     """
    #     Recursively replace the normal conv2d and Linear layer to QuantModule
    #     :param module: nn.Module with nn.Conv2d or nn.Linear in its children
    #     :param weight_quant_params: quantization parameters like n_bits for weight quantizer
    #     :param act_quant_params: quantization parameters like n_bits for activation quantizer
    #     """
    #     prev_quantmodule = None
    #     for name, child_module in module.named_children():
    #         if type(child_module) in specials:
    #             setattr(module, name, specials[type(child_module)](child_module, weight_quant_params, act_quant_params))
    #         elif isinstance(child_module, (nn.Conv2d, nn.Linear, nn.Conv1d)):
    #             setattr(module, name, QuantModule(child_module, weight_quant_params, act_quant_params))
    #             prev_quantmodule = getattr(module, name)
    #             """
    #                 如果找到nn.BatchNorm2d层，并且在此之前有一个QuantModule，则将nn.BatchNorm2d层的功能与前一个QuantModule相关联，
    #                 并用StraightThrough层替换原始的nn.BatchNorm2d层。
    #             """
    #         elif isinstance(child_module, (nn.BatchNorm2d, nn.BatchNorm1d)):
    #             if prev_quantmodule is not None:
    #                 tmp_module = None
    #                 for m in child_module.modules():
    #                     if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
    #                         tmp_module = m
    #                         break
    #                 prev_quantmodule.norm_function = tmp_module
    #                 setattr(module, name, StraightThrough())
    #             else:
    #                 continue
    #             """如果找到nn.ReLU或nn.ReLU6层，并且在此之前有一个QuantModule，则将激活函数与前一个QuantModule相关联，并用StraightThrough层替换原始的激活函数层。"""
    #         elif isinstance(child_module, (nn.ReLU, nn.ReLU6)):
    #             if prev_quantmodule is not None:
    #                 prev_quantmodule.activation_function = child_module
    #                 setattr(module, name, StraightThrough())
    #             else:
    #                 continue
    #         elif isinstance(child_module, StraightThrough):
    #             continue
    #
    #         else:
    #             """对于其他类型的层，递归调用quant_module_refactor_wo_fuse函数，处理其子模块。"""
    #             self.quant_module_refactor_wo_fuse(child_module, weight_quant_params, act_quant_params)

    def quant_module_refactor(self, module: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        """
         递归地将常规的conv2d和线性层递归替换为QuantModule：param module:nn.module的子级中包含nn.Conv2d或nn.Linear
        ：param weight_quant_params:weight量化器的n_bits等量化参数
        ：param act_quant_params
        ：激活量化器的n_bits等量化参数

        Recursively replace the normal conv2d and Linear layer to QuantModule
        :param module: nn.Module with nn.Conv2d or nn.Linear in its children
        :param weight_quant_params: quantization parameters like n_bits for weight quantizer
        :param act_quant_params: quantization parameters like n_bits for activation quantizer
        """
        prev_quantmodule = None
        for name, child_module in module.named_children():
            """针对特殊module进行量化，将module替换为量化后的module"""
            if type(child_module) in specials:
                setattr(module, name, specials[type(child_module)](child_module, weight_quant_params, act_quant_params))
                """对(nn.Conv2d, nn.Linear)进行量化，替换为量化版本的module"""
            elif isinstance(child_module, (nn.Conv2d, nn.Linear)):
                setattr(module, name, QuantModule(child_module, weight_quant_params, act_quant_params))
                prev_quantmodule = getattr(module, name)
                """对(nn.ReLU, nn.ReLU6)进行"""
            elif isinstance(child_module, (nn.ReLU, nn.ReLU6)):
                if prev_quantmodule is not None:
                    prev_quantmodule.activation_function = child_module
                    setattr(module, name, StraightThrough())
                else:
                    continue
                """忽略StraightThrough"""
            elif isinstance(child_module, StraightThrough):
                continue
                """递归量化其他层，比如sequential层"""
            else:
                self.quant_module_refactor(child_module, weight_quant_params, act_quant_params)

    """
        【将神经网络重构为量化模型】

        递归地遍历神经网络模型的所有层，并用量化模块（QuantModule）替换正常的卷积层（nn.Conv2d）和全连接层（nn.Linear）。
        同时，对于批量归一化（nn.BatchNorm2d）层和激活函数（nn.ReLU，nn.ReLU6）层，它会将它们与前面的量化模块关联，并用直通（StraightThrough）层替换它们。

        param module:nn.module的子级中包含nn.Conv2d或nn.Linear
        param weight_quant_params:weight量化器的n_bits等量化参数
        param act_quant_params：激活量化器的n_bits等量化参数
    """

    def quant_module_refactor_wo_fuse(self, module: nn.Module, weight_quant_params: dict = {},
                                      act_quant_params: dict = {}):
        """
        Recursively replace the normal conv2d and Linear layer to QuantModule
        :param module: nn.Module with nn.Conv2d or nn.Linear in its children
        :param weight_quant_params: quantization parameters like n_bits for weight quantizer
        :param act_quant_params: quantization parameters like n_bits for activation quantizer
        """
        prev_quantmodule = None
        for name, child_module in module.named_children():
            """
                specials = {
                    BasicBlock: QuantBasicBlock,
                    Bottleneck: QuantBottleneck,
                    ResBottleneckBlock: QuantResBottleneckBlock,
                    InvertedResidual: QuantInvertedResidual,
                    _InvertedResidual: _QuantInvertedResidual,
                }
            """
            if type(child_module) in specials:
                setattr(module, name, specials[type(child_module)](child_module, weight_quant_params, act_quant_params))
            elif isinstance(child_module, (nn.Conv2d, nn.Linear, nn.Conv1d)):
                setattr(module, name, QuantModule(child_module, weight_quant_params, act_quant_params))
                prev_quantmodule = getattr(module, name)
                """
                    如果找到nn.BatchNorm2d层，并且在此之前有一个QuantModule，则将nn.BatchNorm2d层的功能与前一个QuantModule相关联，
                    并用StraightThrough层替换原始的nn.BatchNorm2d层。
                """
            elif isinstance(child_module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if prev_quantmodule is not None:
                    prev_quantmodule.norm_function = child_module
                    setattr(module, name, StraightThrough())
                else:
                    continue
                """如果找到nn.ReLU或nn.ReLU6层，并且在此之前有一个QuantModule，则将激活函数与前一个QuantModule相关联，并用StraightThrough层替换原始的激活函数层。"""
            elif isinstance(child_module, (nn.ReLU, nn.ReLU6)):
                if prev_quantmodule is not None:
                    prev_quantmodule.activation_function = child_module
                    setattr(module, name, StraightThrough())
                else:
                    continue
            elif isinstance(child_module, StraightThrough):
                continue

            else:
                """对于其他类型的层，递归调用quant_module_refactor_wo_fuse函数，处理其子模块。"""
                self.quant_module_refactor_wo_fuse(child_module, weight_quant_params, act_quant_params)

    """
        遍历模型self.model中所有的模块，并根据给定的weight_quant和act_quant参数设置每个QuantModule或BaseQuantBlock实例的量化状态。
            weight_quant (bool): 一个布尔值，用来开启或关闭权重量化。
            act_quant (bool): 一个布尔值，用来开启或关闭激活量化。
    """
    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        for m in self.model.modules():
            if isinstance(m, (QuantModule, BaseQuantBlock)):
                m.set_quant_state(weight_quant, act_quant)

    """FP_model的前向传播"""

    """
        pointnet++部件分割模型需要传递两个参数
        这里参数传递使用*args以通用
    """
    # def forward(self, input):
    #     return self.model(input)
    # def forward(self, input, another):
    #     return self.model(input, another)

    def forward(self, *args):
        return self.model(*args)

    def set_first_last_layer_to_8bit(self):
        w_list, a_list = [], []
        for module in self.model.modules():
            if isinstance(module, UniformAffineQuantizer):
                if module.leaf_param:
                    a_list.append(module)
                else:
                    w_list.append(module)
        """
            
        """
        w_list[0].bitwidth_refactor(8)
        w_list[-1].bitwidth_refactor(8)
        'the image input has been in 0~255, set the last layer\'s input to 8-bit'
        a_list[-2].bitwidth_refactor(8)
        # a_list[0].bitwidth_refactor(8)

    """
        禁用神经网络中最后一个量化模块（QuantModule）的激活量化，
    """
    def disable_network_output_quantization(self):
        module_list = []
        cout = 0
        for name, m in self.model.named_modules():
            if isinstance(m, QuantModule):
                cout +=1
                module_list += [m]
            """
                只准对conv + BN +relu这样的组件开启激活量化，其他的禁用
            """
            if isinstance(m, QuantModule) and isinstance(m.norm_function, (nn.BatchNorm2d, nn.BatchNorm1d)):
                print("{}该层应该使用激活量化：{}".format(cout, name))
                m.quant_name = name
            if isinstance(m, QuantModule) and not isinstance(m.norm_function, (nn.BatchNorm2d, nn.BatchNorm1d)):
                print("{}该层禁用了激活量化：{}".format(cout, name))
                m.disable_act_quant = True
                m.quant_name = name

            # if isinstance(m, QuantModule) and isinstance(m.norm_function, (nn.BatchNorm2d, nn.BatchNorm1d)) and cout == 10:
            #     print("{}该层额外禁用了激活量化：{}".format(cout, name))
            #     m.disable_act_quant = True
        module_list[-1].disable_act_quant = True
        print(len(module_list))


class PointnetQuantModel(nn.Module):
    def __init__(self, model: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}, is_fusing=True):
        super().__init__()
        if is_fusing:
            """conv、linear进行BN折叠"""
            search_fold_and_remove_bn(model)
            """将常规conv2d和linear层递归替换为QuantModule"""
            self.model = model
            self.quant_module_refactor(self.model, weight_quant_params, act_quant_params)
        else:
            """不进行BN折叠，记录原始的FP model，用于后面对比"""
            self.model = model
            """将常规conv2d和linear层递归替换为QuantModule，QuantModule中可以通过开关决定是否打开量化"""
            self.quant_module_refactor_wo_fuse(self.model, weight_quant_params, act_quant_params)