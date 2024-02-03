import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from .quant_layer import QuantModule, Union, lp_loss
from .quant_model import QuantModel
from .quant_block import BaseQuantBlock
from tqdm import trange

"""
    在fp_model上进行的操作
"""
def save_dc_fp_data(model: QuantModel, layer: Union[QuantModule, BaseQuantBlock], cali_data: torch.Tensor,
                    batch_size: int = 32, keep_gpu: bool = True,
                    input_prob: bool = False, lamb=50, bn_lr=1e-3):
    """Activation after correction"""
    device = next(model.parameters()).device


    get_inp_out = GetDcFpLayerInpOut(model, layer, device=device, input_prob=input_prob, lamb=lamb, bn_lr=bn_lr)
    cached_batches = []

    """
        开始DC分布校准
    """
    # TODO 可能需要分批处理数据,不能一次处理
    # print("Start correcting {} batches of data!".format(int(cali_data.size(0) / batch_size)))

    # for i in trange(int(cali_data.size(0) / batch_size)):
    #     """如果进行DC分布校准"""
    #     if input_prob:
    #         cur_out, out_fp, cur_sym = get_inp_out(cali_data[i * batch_size:(i + 1) * batch_size])
    #         cached_batches.append((cur_out.cpu(), out_fp.cpu(), cur_sym.cpu()))
    #         """不进行DC分布校正"""
    #     else:
    #         cur_out, out_fp = get_inp_out(cali_data[i * batch_size:(i + 1) * batch_size])
    #         cached_batches.append((cur_out.cpu(), out_fp.cpu()))
    #
    # """拼接多个batch"""
    # cached_outs = torch.cat([x[0] for x in cached_batches])
    # cached_outputs = torch.cat([x[1] for x in cached_batches])
    # if input_prob:
    #     cached_sym = torch.cat([x[2] for x in cached_batches])
    # torch.cuda.empty_cache()

    # """如果进行DC分布校准"""
    # if input_prob:
    #     cur_out, out_fp, cur_sym = get_inp_out(cali_data)
    #     cached_batches = ((cur_out.cpu(), out_fp, cur_sym.cpu()))
    #     """不进行DC分布校正"""
    # else:
    #     cur_out, out_fp = get_inp_out(cali_data)
    #     cached_batches = ((cur_out.cpu(), out_fp))
    #
    # """拼接多个batch"""
    # cached_outs = cached_batches[0]
    # cached_outputs = cached_batches[1]
    # if input_prob:
    #     cached_sym = cached_batches[2]
    # torch.cuda.empty_cache()

    num_batches = max(1, int(cali_data["pts_input"].size(0) / batch_size))

    print("Start correcting {} batches of data!".format(num_batches))

    for i in trange(num_batches):
        # 获取当前批次的数据，并确保索引不越界
        cur_data = {"pts_input": cali_data["pts_input"][i * batch_size:min((i + 1) * batch_size, cali_data["pts_input"].size(0))]}

        if input_prob:
            # 进行DC分布校准
            cur_out, out_fp, cur_sym = get_inp_out(cur_data)
            cached_batches.append((cur_out.cpu(), cur_sym.cpu()))
            # cached_batches.append((cur_out.cpu(), out_fp.cpu(), cur_sym.cpu()))
        else:
            # 不进行DC分布校正
            cur_out, out_fp = get_inp_out(cur_data)
            cached_batches.append((cur_out.cpu()))
            # cached_batches.append((cur_out.cpu(), out_fp.cpu()))

    # 拼接多个batch
    cached_outs = torch.cat([x[0] for x in cached_batches])
    # cached_outputs = torch.cat([x[1] for x in cached_batches])
    if input_prob:
        cached_sym = torch.cat([x[1] for x in cached_batches])

    torch.cuda.empty_cache()

    # if keep_gpu:
    #     cached_outs = cached_outs.to(device)
    #     cached_outputs_to_device = {key: value.to(device) for key, value in cached_outputs.items()}
    #     # cached_outputs = cached_outputs.to(device)
    #
    #     if input_prob:
    #         cached_sym = cached_sym.to(device)
    # if input_prob:
    #     cached_outs.requires_grad = False
    #     cached_sym.requires_grad = False
    #     return cached_outs, cached_outputs, cached_sym
    # return cached_outs, cached_outputs

    if keep_gpu:
        cached_outs = cached_outs.to(device)
        # cached_outputs = cached_outputs.to(device)
        if input_prob:
            cached_sym = cached_sym.to(device)
    if input_prob:
        cached_outs.requires_grad = False
        cached_sym.requires_grad = False
        return cached_outs,  cached_sym
    return cached_outs

"""   
    这段Python代码定义了一个名为 save_inp_oup_data 的函数，它的作用是:
    
    保存通过模型的特定层（或块）传递的校准数据集的输入和输出数据。
    
    下面是对这个函数及其参数的详细解释：
    参数：
        model (QuantModel): 这个参数是一个量化模型对象，函数将使用它来获取特定层的输入输出数据。
        layer (Union[QuantModule, BaseQuantBlock]): 该参数指定了模型中的特定层或块。这个层（或块）是量化模块或基础量化块的一个实例，我们希望保存其输入和输出数据。
        
        cali_data (torch.Tensor): 校准数据集，它是一个张量，通常包含用于模型校准的图像或数据。
        batch_size (int, 默认为 32): 在获取输入输出数据时使用的批量大小。
        
        keep_gpu (bool, 默认为 True): 指定是否将保存的数据保持在GPU上，这样可以加快后续的优化过程。
        input_prob (bool, 默认为 False): 一个布尔值，指示在获取层输入时是否应用某种概率模型。
"""

"""
    cali_data.shape : torch.Size([1024, 3, 224, 224])
    
    得到校准数据集的输入
"""
def save_inp_oup_data(model: QuantModel, layer: Union[QuantModule, BaseQuantBlock], cali_data: torch.Tensor,
                      batch_size: int = 4, keep_gpu: bool = True,
                      input_prob: bool = False):
    """
    Save input data and output data of a particular layer/block over calibration dataset.

    :param model: QuantModel
    :param layer: QuantModule or QuantBlock
    :param cali_data: calibration data set
    :param weight_quant: use weight_quant quantization
    :param act_quant: use act_quant quantization
    :param batch_size: mini-batch size for calibration
    :param keep_gpu: put saved data on GPU for faster optimization
    :return: input and output data
    """
    device = next(model.parameters()).device
    """
        得到一个GetLayerInpOut对象，这个对象通过GetLayerInpOut中注册的钩子函数来获取qnn layer的输入和输出
    """
    get_inp_out = GetLayerInpOut(model, layer, device=device, input_prob=input_prob)
    cached_batches = []
    """
        1024 /32 = 32
    """
    # for i in range(int(cali_data.size(0) / batch_size)):
    #
    #     cur_inp = get_inp_out(cali_data[i * batch_size:(i + 1) * batch_size])
    #
    #     cached_batches.append(cur_inp.cpu())
    #
    # cached_inps = torch.cat([x for x in cached_batches])
    # torch.cuda.empty_cache()
    #
    # if keep_gpu:
    #     cached_inps = cached_inps.to(device)


    # cali_data = torch.from_numpy(cali_data)
    # TODO 这里可能需要修改成像之前那样分批次的，否则大了会爆显存
    # cached_inps = get_inp_out(cali_data)
    # torch.cuda.empty_cache()
    # if keep_gpu:
    #     cached_inps = cached_inps.to(device)
    # return cached_inps
    num_batches = max(1, int(cali_data["pts_input"].size(0) / batch_size))

    for i in range(num_batches):
        cur_dict = {"pts_input": cali_data["pts_input"][i * batch_size:min((i + 1) * batch_size, cali_data["pts_input"].size(0))]}
        cur_inp = get_inp_out(cur_dict)
        cached_batches.append(cur_inp.cpu())
    cached_inps = torch.cat([x for x in cached_batches])
    torch.cuda.empty_cache()

    if keep_gpu:
        cached_inps = cached_inps.to(device)
    return cached_inps


class StopForwardException(Exception):
    """
    Used to throw and catch an exception to stop traversing the graph
    """
    pass


class DataSaverHook:
    """
    Forward hook that stores the input and output of a block
    """

    def __init__(self, store_input=False, store_output=False, stop_forward=False):
        self.store_input = store_input
        self.store_output = store_output
        self.stop_forward = stop_forward

        self.input_store = None
        self.output_store = None

    def __call__(self, module, input_batch, output_batch):
        # print(f'DataSaverHook 存储了该层input:{input_batch[0].shape},{input_batch[0].flatten()[0:10]}')
        # print(f'DataSaverHook 存储了该层output:{output_batch[0].shape},{output_batch[0].flatten()[0:10]}')
        if self.store_input:
            self.input_store = input_batch
        if self.store_output:
            self.output_store = output_batch
        if self.stop_forward:
            raise StopForwardException


"""
    获取中间层输出数据的。
"""
class input_hook(object):
    """
	Forward_hook used to get the output of the intermediate layer.
	"""
    def __init__(self, stop_forward=False):
        super(input_hook, self).__init__()
        self.inputs = None

    def hook(self, module, input, output):
        self.inputs = input
    """
        这个方法用于清除 self.inputs 属性，将其重新设置为 None，以准备捕获下一个前向传播中的输入数据。
    """
    def clear(self):
        self.inputs = None

class GetLayerInpOut:
    def __init__(self, model: QuantModel, layer: Union[QuantModule, BaseQuantBlock],
                 device: torch.device, input_prob: bool = False):
        self.model = model
        self.layer = layer
        self.device = device

        self.data_saver = DataSaverHook(store_input=True, store_output=False, stop_forward=True)
        self.input_prob = input_prob

    def __call__(self, model_input):
        """
            # 在self.layer上注册一个前向钩子，使用self.data_saver作为回调函数。
            # 前向钩子是在PyTorch模型的前向传递期间执行的函数。
            # 此钩子的目的是在前向传递期间保存一些数据以供以后使用。

            "self.data_saver"是用户自定义的前向钩子函数，将在 self.layer 层的前向传递期间执行。
            这样，您可以在前向传递期间监控和处理该层的输入和输出。

            type(handle) : <class 'torch.utils.hooks.RemovableHandle'>

            register_forward_hook 方法的参数通常应该是一个函数，因为前向钩子需要一个可调用的对象来执行自定义操作。
            这个可调用对象通常是一个函数或方法，它将在模块的前向传递期间被调用。
            如果您想要在前向钩子中执行更复杂的操作，例如访问类的成员变量或方法，您可以使用一个具有 __call__ 方法的对象。
            在这种情况下，您可以将这个对象传递给 register_forward_hook，因为对象的 __call__ 方法实际上是一个可调用的函数。
        """
        # print(f"注册钩子")
        handle = self.layer.register_forward_hook(self.data_saver)

        with torch.no_grad():
            # 设置模型的量化状态，包括权重量化和激活量化。
            self.model.set_quant_state(weight_quant=True, act_quant=True)
            # print(f"调用了GetLayerInpOut __call__中的模型前向传播")
            try:
                # 尝试运行模型的前向传递，将输入数据传递到self.device上。
                # print(f"输入数据是{model_input.shape},{model_input.flatten()[0:10]}")


                # print(type(model_input))
                # print(model_input.shape)
                _ = self.model(model_input)
                print(f"前向传播结束")
            except StopForwardException:
                pass

        # 注销之前注册的前向钩子
        handle.remove()
        # print(f"注销钩子")

        """
            钩子函数在该Module(Layer)前向传播过程中把输入数据存储到了self.data_saver.input_store中
        """
        # 返回保存在self.data_saver.input_store中的数据的第一个元素（去掉梯度信息）
        """
            type(self.data_saver.input_store) : <class 'tuple'>
            self.data_saver.input_store[0].shape : torch.Size([32, 3, 224, 224])
        """
        return self.data_saver.input_store[0].detach()


"""
    
"""
class GetDcFpLayerInpOut:
    def __init__(self, model: QuantModel, layer: Union[QuantModule, BaseQuantBlock],
                 device: torch.device, input_prob: bool = False, lamb=50, bn_lr=1e-3):
        self.model = model
        self.layer = layer
        self.device = device
        self.data_saver = DataSaverHook(store_input=True, store_output=True, stop_forward=False)
        self.input_prob = input_prob

        """存储BN层的平均值和方差"""
        self.bn_stats = []
        self.eps = 1e-6
        self.lamb = lamb
        self.bn_lr = bn_lr

        """只针对BN层，把每个BN层的均值和方差存储到self.bn_stats"""
        for n, m in self.layer.named_modules():
            if isinstance(m, nn.BatchNorm2d):
            # get the statistics in the BatchNorm layers
                self.bn_stats.append(
                    (m.running_mean.detach().clone().flatten().cuda(),
                    torch.sqrt(m.running_var +
                                self.eps).detach().clone().flatten().cuda()))
    """
        计算AB之间的MSE,B.size(0)指的是样本数
    """
    def own_loss(self, A, B):
        return (A - B).norm()**2 / B.size(0)
    """
        计算两个输入张量 A 和 B 之间的相对损失
    """
    def relative_loss(self, A, B):
        return (A-B).abs().mean()/A.abs().mean()

    def __call__(self, model_input):
        """关闭量化状态"""
        self.model.set_quant_state(False, False)

        """给该层注册钩子函数"""
        handle = self.layer.register_forward_hook(self.data_saver)


        hooks = []
        hook_handles = []

        # print(list(self.layer.named_modules()))
        for name, module in self.layer.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                hook = input_hook()
                hooks.append(hook)
                hook_handles.append(module.register_forward_hook(hook.hook))

        """确保每个 Batch Normalization 层都有一个相应的 hook。"""

        assert len(hooks) == len(self.bn_stats)

        # model_input_tensor = {key: value.to(self.device) for key, value in model_input.items()}
        with torch.no_grad():
            try:
                # output_fp = self.model(model_input.to(self.device))
                output_fp = self.model(model_input)
                pass
            except StopForwardException:
                pass

            if self.input_prob:
                """如果 self.input_prob 为真，则保存了输入数据的副本到 input_sym 变量中。"""
                input_sym = self.data_saver.input_store[0].detach()
            
        handle.remove()

        """
            para_input.shape : torch.Size([32, 3, 224, 224])
        """

        para_input = input_sym.data.clone()
        para_input = para_input.to(self.device)
        para_input.requires_grad = True

        optimizer = optim.Adam([para_input], lr=self.bn_lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                        min_lr=1e-5,
                                                        verbose=False,
                                                        patience=100)

        iters=500
        for iter in range(iters):
            self.layer.zero_grad()
            optimizer.zero_grad()
            for hook in hooks:
                """清空self.inputs"""
                hook.clear()
            _ = self.layer(para_input)
            mean_loss = 0
            std_loss = 0

            for num, (bn_stat, hook) in enumerate(zip(self.bn_stats, hooks)):
                """
                    取出BN的inputs
                """
                tmp_input = hook.inputs[0]
                bn_mean, bn_std = bn_stat[0], bn_stat[1]
                tmp_mean = torch.mean(tmp_input.view(tmp_input.size(0),
                                                    tmp_input.size(1), -1),
                                    dim=2)
                tmp_std = torch.sqrt(
                    torch.var(tmp_input.view(tmp_input.size(0),
                                            tmp_input.size(1), -1),
                            dim=2) + self.eps)

                mean_loss += self.own_loss(bn_mean, tmp_mean)
                std_loss += self.own_loss(bn_std, tmp_std)

            """计算lp_loss"""
            constraint_loss = lp_loss(para_input, input_sym) / self.lamb

            total_loss = mean_loss + std_loss + constraint_loss

            total_loss.backward()
            optimizer.step()
            scheduler.step(total_loss.item())
            # if (iter+1) % 500 == 0:
            #     print('Total loss:\t{:.3f} (mse:{:.3f}, mean:{:.3f}, std:{:.3f})\tcount={}'.format(
            #     float(total_loss), float(constraint_loss), float(mean_loss), float(std_loss), iter))

        """
            最后，使用 torch.no_grad() 块再次执行模型的前向传播，得到 out_fp，并返回相应的结果。
            如果 self.input_prob 为真，则还返回 input_sym、output_fp 和 para_input 的副本。
            
            总之，这段代码的主要目标是通过优化输入数据 para_input，以最小化模型输出与 Batch Normalization 层期望值之间的差异，并满足约束条件。
            这个过程使用了梯度下降法和 Adam 优化器来更新输入数据，以达到最小化损失函数的目标。
        """
        with torch.no_grad():
            out_fp = self.layer(para_input)

        """如果开启DC校正"""

        # output_fp_dict_detach = {key: value.detach() for key, value in output_fp.items()}

        # TODO pointnet的输出是一个tuple,我们只需要output_fp[0]作为pred ,PointRCNN也不一样，具体看下怎么弄
        if self.input_prob:
            return  out_fp.detach(), out_fp.detach(), para_input.detach()
        return out_fp.detach(), out_fp.detach()

