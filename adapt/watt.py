import copy
from collections import OrderedDict

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils.misc import load_templates_from_yaml

REFERENCE_TEMPLATE = 'a photo of a {}'


class WATT:
    """
    WATT (Weight Average adaptation during Test-Time) adapts a CLIP model by minimizing entropy during testing.
    The model adapts itself by updating on every forward pass.
    The code is based on TENT and CLIPArTT repos.
    TENT GitHub: https://github.com/DequanWang/tent
    CLIPArTT GitHub: https://github.com/dosowiechi/CLIPArTT
    """
    """
    参数名	                    含义
    model	        模型名（传给 clip.load()），如 'ViT-B/16'
    lr	            学习率，用于 Adam 优化器
    type	        自适应方式：'sequential' 或 'parallel'
    l	            每个文本模板的迭代步数
    m	            每轮适配后平均权重的重复次数
    temps_dir	    存储 prompt 模板的 yaml 文件路径
    ref_eval	    是否在评估阶段使用默认（参考）模板
    device	        设备类型，'cpu' 或 'cuda'
    """

    def __init__(self,
                 model,
                 lr,
                 type='sequential',
                 l=2,
                 m=5,
                 temps_dir='templates.yaml',
                 ref_eval=False,
                 device='cpu'):
        """
        Initializes the WATT module.

        Args:
            model: The CLIP model to be adapted.
            lr: Learning rate for the optimizer.
            type: Adaptation method of WATT ('parallel' or 'sequential').
            l: Number of adaptation iterations for each text embedding before performing weight averaging.
            m: Number of repetitions of the adaptation and weight averaging process.
            temps_dir: Path to the templates.yaml file which inclodes different text templates that should be used during adaptation.
            ref_eval: Whether to use REFERENCE_TEMPLATE during evaluation.
            device: The device to run the model on (e.g., 'cpu' or 'cuda').

        """

        # loading the base model
        # 加载 CLIP 模型
        base_model, _ = clip.load(model, device)
        self.model = base_model

        #  初始化配置参数
        self.lr = lr
        self.type = type
        self.l = l
        self.m = m
        self.ref_eval = ref_eval
        self.device = device

        # Load the text templates
        # 加载 prompt 模板
        self.all_templates = load_templates_from_yaml(temps_dir)

        # Set the gradients for LayerNorm layers only for visual encoder
        # 设定只优化视觉编码器中的 LayerNorm
        # 仅适配视觉模型中的 LayerNorm 层，提升稳定性和快速性
        self.model.visual = self.set_ln_grads(self.model.visual)

        # Collect the LayerNorm parameters and set the optimizer
        # 返回模型中 LayerNorm 层的参数列表
        params, _ = self.collect_ln_params(self.model.visual)
        # 只将这些参数传入 Adam 优化器，避免对整个模型进行大规模更新
        # betas 与 weight_decay 是标准设置，用于稳定优化
        self.optimizer = optim.Adam(params,
                                    lr=self.lr,
                                    betas=(0.9, 0.999),
                                    weight_decay=0.0)

        # Save the initial model and optimizer states
        # 保存初始模型和优化器状态
        # 这一步非常关键，因为 WATT 每个 batch 后都要 reset 回初始状态
        # 存的是：model_state：模型参数的 state_dict(),optimizer_state：优化器的状态（如动量）
        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(
            self.model, self.optimizer)

    """
    adapt 方法是 WATT 类中用于 执行一次完整自适应流程 的入口函数
    
    x: 一个 batch 的图像张量（shape 通常为 [B, 3, H, W]）
    classes: 字符串组成的类别列表，如 ["cat", "dog", "car", ...]，用于构建 prompt 和文本嵌入
    """

    def adapt(self, x, classes):
        """
        Forward pass with adaptation.

        Args:
            x: Input image tensor.
            classes: List of class names.

        """

        # 重置模型状态
        self.reset()
        # 执行适应过程
        self.perform_adaptation(x, classes)

    @torch.no_grad()
    def evaluate(self, x, classes):
        """
        Forward pass without adaptation.

        Args:
            x: Input image tensor.
            classes: List of class names.

        Returns:
            pred: Predicted class labels for the input images.

        """

        # extracting features
        image_features = self.model.encode_image(x)

        # Pick the top most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        if self.ref_eval:
            text_features = self.extract_text_embeddings(classes,
                                                         [REFERENCE_TEMPLATE],
                                                         average=True)
        else:
            text_features = self.extract_text_embeddings(classes,
                                                         self.all_templates,
                                                         average=True)
        text_features = text_features.T

        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, pred = similarity.topk(1, 1, True, True)
        pred = pred.t()

        return pred

    def reset(self):
        """
        Resets the model and optimizer to their initial states.
        """
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("Cannot reset without saved model/optimizer state")
        self.load_model_and_optimizer(self.model, self.optimizer,
                                      self.model_state, self.optimizer_state)

    # WATT 模型核心的测试时自适应逻辑，逐步更新视觉编码器的 LayerNorm 层
    def perform_adaptation(self, x, classes):
        """
        Forward pass with adaptation for test-time. The model adapts itself during testing by updating on every forward pass.

        Args:
            x: Input image tensor.
            classes: List of class names.
        """

        # 对所有类别名 classes 和所有模板 all_templates 构造 prompt，并编码为文本特征（不做平均）
        # 返回的是一个 prompt 模板数量个数的 list，其中每个元素是 [num_classes, d] 的文本嵌入张量
        text_x = self.extract_text_embeddings(classes,
                                              self.all_templates,
                                              average=False)

        # 每轮都保存一次更新后的 LayerNorm 参数，用于最后平均
        # 第 m=0 轮时从初始模型出发，后续从 avg_state_dict 开始
        for m in range(self.m):
            all_weights = []
            if self.type == 'sequential':
                if m == 0:
                    self.load_model_and_optimizer(self.model, self.optimizer,
                                                  self.model_state,
                                                  self.optimizer_state)
                else:
                    self.model.load_state_dict(avg_state_dict, strict=False)

            # 当前模板下，对应类别的文本特征 [num_classes, d]
            for text_feat in text_x:
                # parallel: 每个模板都从同一初始模型出发
                if self.type == 'parallel':
                    if m == 0:
                        # 同时加载模型与优化器状态
                        self.load_model_and_optimizer(self.model,
                                                      self.optimizer,
                                                      self.model_state,
                                                      self.optimizer_state)
                    else:
                        # 仅加载模型权重（strict=False 允许部分参数缺失）
                        self.model.load_state_dict(avg_state_dict,
                                                   strict=False)

                # 每个模板下的 l 次更新
                for l in range(self.l):
                    with torch.no_grad():
                        # 禁用梯度（加速 + 节省显存）提取图像特征
                        image_features = self.model.encode_image(x)
                    # 然后做 L2 归一化
                    image_features = image_features / image_features.norm(
                        dim=-1, keepdim=True)
                    # 计算图像与每个类别文本之间的相似度，softmax 后取 top-1 的预测类别索引（伪标签）
                    similarity = (100 *
                                  image_features @ text_feat.t()).softmax(1)
                    # pred.shape 为 [B, 1]，值是类别 ID
                    values, pred = similarity.topk(1, 1, True, True)
                    # 为每张图像提取其 top-1 类别对应的文本嵌入（[B, d]）
                    pred_inputs = torch.cat([text_feat[
                        c,
                    ] for c in pred])

                    # Calculating the Loss
                    # 前向传播，同时得到新的图像特征和文本特征；
                    # 模型此时在训练模式下，仅更新 LayerNorm 参数
                    logits, image_features, text_features = self.model(
                        x, pred_inputs, True)
                    # 计算图像间相似度（I @ I^T）和文本间相似度（T @ T^T）
                    images_similarity = image_features @ image_features.t()
                    texts_similarity = text_features @ text_features.t()
                    # 平均后除以 0.01（温度） → softmax 得到 soft target
                    targets = F.softmax(
                        ((images_similarity + texts_similarity) / 2) / 0.01,
                        dim=-1)
                    # 损失函数是自定义的 soft label 交叉熵（对 logits vs. targets）
                    loss = self.cross_entropy(logits.t(),
                                              targets.t(),
                                              reduction='mean')
                    # 更新参数，并清除梯度
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # 遍历模型的模块，只保存 LayerNorm 中的 weight 和 bias
                weights = {}
                for name, module in self.model.named_modules():
                    if isinstance(module, nn.LayerNorm):
                        for nparam, p in module.named_parameters():
                            if nparam in ['weight', 'bias']:
                                weights[f"{name}.{nparam}"] = copy.deepcopy(p)
                # 将这次适配的结果保存进 all_weights 列表中
                all_weights.append(weights)
            #  每轮完成后，对所有模板适配结果进行平均融合
            avg_state_dict = self.weight_average(all_weights)
        # 将融合后的参数加载到模型，作为最终 TTA 后的权重
        self.model.load_state_dict(avg_state_dict, strict=False)

    # 构造 CLIP 文本特征
    def extract_text_embeddings(self, class_names, templates, average=True):
        """
        Extracts text embeddings for given class names and templates.

        Args:
            class_names: List of class names to generate text embeddings for.
            templates: List of text templates to use for generating text embeddings.
            average: Boolean indicating whether to average the embeddings of different templates for each class.

        Returns:
            text_features: Tensor of text embeddings for the given class names and templates.
        """
        with torch.no_grad():
            text_features = []
            for class_name in class_names:
                # 根据不同模板，构造多种 prompt 形式
                texts = [template.format(class_name) for template in templates]
                # 文本转为 token id
                texts = clip.tokenize(texts).to(self.device)
                #  提取文本特征
                class_embeddings = self.model.encode_text(texts)
                # 对每个向量做 L2 归一化，使得它们可以用点乘表示余弦相似度
                # 返回 shape: [num_templates, d]，d 是 CLIP 文本嵌入维度
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)

                # 如果设置 average=True，则求平均
                if average:
                    class_embeddings = class_embeddings.mean(dim=0)
                    class_embeddings /= class_embeddings.norm()

                # 保存当前类别的结果
                # 如果 average=True：append 一个 [d] 向量；
                # 如果 average=False：append 一个 [num_templates, d] 张量
                text_features.append(class_embeddings)
            # 拼接所有类别的特征
            # average=True 时，最终 stack 成 [d, num_classes]
            # average=False 时，最终 stack 成 [num_templates, num_classes, d]
            text_features = torch.stack(text_features, dim=1).to(self.device)
        return text_features

    @staticmethod
    def set_ln_grads(model):
        """
        Set gradient settings for LayerNorm layers within the model, disabling gradients globally except for these LN layers.

        Args:
            model: The model whose LayerNorm layers' gradients are to be set.
        
        Returns:
            The model with modified gradient settings.
        """
        model.requires_grad_(False)
        for m in model.modules():
            if isinstance(m, nn.LayerNorm):
                m.requires_grad_(True)
        return model

    @staticmethod
    def collect_ln_params(model):
        """
        Collect the affine scale and shift parameters from LayerNorm layers.

        Args:
            model: The model from which to collect LayerNorm parameters.
        
        Returns:
            params: List of LayerNorm parameters.
            names: List of parameter names.
        """
        params = []
        names = []
        for nm, m in model.named_modules():
            if isinstance(m, nn.LayerNorm):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:
                        params.append(p)
                        names.append(f"visual.{nm}.{np}")
        return params, names

    @staticmethod
    def cross_entropy(preds, targets, reduction='none'):
        """
        Calculate the cross-entropy loss between predictions and targets.

        Args:
            preds: Predicted logits.
            targets: Target probabilities.
            reduction: Type of reduction to apply to the output ('none' or 'mean').

        Returns:
            The computed loss.
        """
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()

    """
    weight_average 用于对多个模型的参数进行平均，是常见的模型融合策略
    
    @staticmethod 表示这是一个 静态方法，不依赖类的实例 (self)。
    all_weights 是多个模型的权重（state_dict）组成的列表
    """

    @staticmethod
    def weight_average(all_weights):
        """
        Compute the average of the weights from multiple models.

        Args:
            all_weights: List of state dictionaries from different models.

        Returns:
            avg_state_dict: Averaged state dictionary.
        """
        # 获取权重的数量，比如 3
        K = len(all_weights)
        # 创建一个新的 OrderedDict 用来保存平均后的参数。
        # 使用 OrderedDict 是为了保留参数的顺序
        avg_state_dict = OrderedDict()
        # 遍历第一个模型的所有参数名和对应的值
        for param_name, param in all_weights[0].items():
            # 对每个模型的同一个参数名（如 'encoder.layer1.weight'）取出来求和，然后除以模型个数，得到平均值
            avg_param = sum(sd[param_name] for sd in all_weights) / K
            # 将平均后的参数添加进结果字典
            avg_state_dict[param_name] = avg_param
        return avg_state_dict

    @staticmethod
    def copy_model_and_optimizer(model, optimizer):
        """
        Copy the model and optimizer states for resetting after adaptation.

        Args:
            model: The model to copy.
            optimizer: The optimizer to copy.

        Returns:
            model_state: Copied state of the model.
            optimizer_state: Copied state of the optimizer.
        """
        model_state = copy.deepcopy(model.state_dict())
        optimizer_state = copy.deepcopy(optimizer.state_dict())
        return model_state, optimizer_state

    @staticmethod
    def load_model_and_optimizer(model, optimizer, model_state,
                                 optimizer_state):
        """
        Restore the model and optimizer states from copies.

        Args:
            model: The model to restore.
            optimizer: The optimizer to restore.
            model_state: The state to restore the model to.
            optimizer_state: The state to restore the optimizer to.
        """
        model.load_state_dict(model_state, strict=True)
        optimizer.load_state_dict(optimizer_state)
