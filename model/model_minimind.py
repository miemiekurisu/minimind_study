"""
MiniMind 大语言模型实现文件
===========================================

本文件实现了一个轻量级的大语言模型（MiniMind），基于Transformer架构，具有以下特点：
1. 支持因果语言建模（Causal Language Modeling）
2. 使用RMSNorm替代LayerNorm进行归一化
3. 采用RoPE（旋转位置编码）处理序列位置信息
4. 支持分组查询注意力（Grouped Query Attention, GQA）
5. 可选的专家混合（Mixture of Experts, MoE）架构
6. 兼容Hugging Face transformers库

文件结构概览：
├── MiniMindConfig: 模型配置类，定义所有超参数
├── 基础组件:
│   ├── RMSNorm: 根均方归一化层
│   ├── precompute_freqs_cis: 预计算旋转位置编码
│   └── apply_rotary_pos_emb: 应用旋转位置编码
├── 核心模块:
│   ├── Attention: 多头注意力机制（支持Flash Attention）
│   ├── FeedForward: 标准前馈网络
│   ├── MOEFeedForward: 专家混合前馈网络
│   └── MiniMindBlock: Transformer块
├── 完整模型:
│   ├── MiniMindModel: 主模型类
│   └── MiniMindForCausalLM: 用于因果语言建模的包装类

模型架构特点：
- 使用SwiGLU激活函数的前馈网络
- 支持KV缓存的高效推理
- 可选的Flash Attention加速
- 权重共享的词嵌入和输出层
- 支持梯度检查点和混合精度训练

PyTorch库和函数使用说明：
============================

1. 核心PyTorch模块：
   - torch.nn.Module: 所有神经网络层的基类，用于定义模型组件
   - torch.nn.Parameter: 可学习参数的包装器，会自动加入梯度计算
   - torch.nn.Linear: 线性变换层，用于投影和分类头
   - torch.nn.Embedding: 词嵌入层，将token ID转换为向量表示
   - torch.nn.ModuleList: 模块列表容器，用于存储多个Transformer层
   - torch.nn.Dropout: 随机失活层，用于正则化防止过拟合

2. 激活函数和归一化：
   - torch.nn.functional.silu/gelu: SiLU和GELU激活函数，用于前馈网络
   - torch.rsqrt: 平方根倒数，用于RMSNorm的高效计算
   - torch.nn.functional.layer_norm: 层归一化（本文件使用自定义RMSNorm）

3. 注意力机制相关：
   - torch.nn.functional.scaled_dot_product_attention: Flash Attention的PyTorch实现
   - torch.nn.functional.softmax: softmax激活，用于注意力权重计算
   - torch.transpose: 张量转置，用于调整注意力头的维度顺序
   - torch.bmm/@: 批量矩阵乘法，用于注意力分数和值的计算

4. 张量操作：
   - torch.cat: 张量拼接，用于KV缓存和位置编码
   - torch.split/chunk: 张量分割，用于多头注意力的头分离
   - torch.view/reshape: 张量形状变换，用于维度调整
   - torch.unsqueeze/squeeze: 增减张量维度
   - torch.repeat_interleave: 重复张量元素，用于GQA中的KV复制

5. 数学运算：
   - torch.outer: 外积运算，用于生成位置编码频率矩阵
   - torch.cos/sin: 三角函数，用于旋转位置编码
   - torch.sqrt: 平方根，用于注意力缩放
   - torch.triu: 上三角矩阵，用于因果注意力掩码

6. 专家混合（MoE）相关：
   - torch.topk: 选择top-k专家
   - torch.scatter_add_: 原地散列加法，用于专家输出聚合
   - torch.bincount: 计数操作，用于专家负载平衡
   - torch.argsort: 排序索引，用于专家推理优化

7. 缓存和内存管理：
   - register_buffer: 注册不可训练的模型状态（如位置编码）
   - torch.no_grad: 禁用梯度计算的上下文管理器，用于推理优化

8. 兼容性和集成：
   - transformers.PreTrainedModel: Hugging Face模型基类
   - transformers.GenerationMixin: 生成任务的混合类
   - transformers.modeling_outputs.CausalLMOutputWithPast: 标准化的模型输出格式

Transformers库详细说明：
============================

Transformers库是Hugging Face开发的自然语言处理库，提供了预训练模型和工具。
本文件主要使用以下组件：

1. 配置基类 (transformers.PretrainedConfig)：
   - 作用：提供模型配置的标准化基类
   - 功能：支持配置的保存、加载和JSON序列化
   - 用法：MiniMindConfig继承此类，获得标准配置管理能力
   - 示例：config.save_pretrained(), config.from_pretrained()

2. 模型基类 (transformers.PreTrainedModel)：
   - 作用：所有Hugging Face模型的基类，提供标准化接口
   - 功能：模型保存/加载、设备管理、梯度检查点等
   - 用法：MiniMindForCausalLM继承此类，获得完整模型管理能力
   - 关键方法：
     * save_pretrained(): 保存模型权重和配置
     * from_pretrained(): 从预训练权重加载模型
     * to(): 设备转移（CPU/GPU）
     * train()/eval(): 训练/评估模式切换
     * parameters(): 获取可训练参数

3. 生成混入类 (transformers.GenerationMixin)：
   - 作用：为语言模型提供文本生成功能
   - 功能：支持多种生成策略和解码算法
   - 用法：MiniMindForCausalLM继承此类，获得generate()方法
   - 生成策略：
     * 贪婪搜索：generate(do_sample=False)
     * 采样生成：generate(do_sample=True, temperature=0.7)
     * 束搜索：generate(num_beams=4)
     * top-k采样：generate(do_sample=True, top_k=50)
     * top-p采样：generate(do_sample=True, top_p=0.9)

4. 输出格式类 (transformers.modeling_outputs.CausalLMOutputWithPast)：
   - 作用：标准化因果语言模型的输出格式
   - 功能：封装logits、hidden_states、past_key_values等输出
   - 用法：forward()方法返回此类型，兼容HF生态
   - 属性：
     * logits: 预测的词汇表分布
     * past_key_values: KV缓存，用于增量生成
     * hidden_states: 各层隐藏状态（可选）
     * attentions: 注意力权重（可选）

5. 激活函数映射 (transformers.activations.ACT2FN)：
   - 作用：提供标准化的激活函数映射字典
   - 功能：通过字符串名称获取对应的激活函数
   - 用法：ACT2FN[config.hidden_act] 获取配置的激活函数
   - 支持函数：
     * "relu": torch.nn.functional.relu
     * "gelu": torch.nn.functional.gelu
     * "silu": torch.nn.functional.silu (Swish)
     * "swish": torch.nn.functional.silu
     * "tanh": torch.tanh

Transformers库集成优势：
1. 生态兼容性：与Hugging Face Hub、datasets、tokenizers等无缝集成
2. 标准化接口：统一的API设计，便于模型替换和比较
3. 社区支持：丰富的预训练模型和社区贡献
4. 部署便利：支持ONNX导出、TensorRT优化等部署方案
5. 版本管理：完善的模型版本控制和配置管理
6. 开箱即用：提供完整的训练、推理和评估工具链

主要技术亮点：
- 高效的旋转位置编码（RoPE）实现
- 分组查询注意力（GQA）减少KV缓存内存占用
- 可选的专家混合（MoE）架构提升模型容量
- 兼容Flash Attention的高效注意力计算
- 完整的KV缓存支持实现流式生成


"""

# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Config
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

# Transformers库导入说明：
# PretrainedConfig: Hugging Face配置基类，提供配置的标准化管理
#   - 支持JSON序列化和反序列化
#   - 提供save_pretrained()和from_pretrained()方法
#   - 自动处理配置参数的验证和默认值设置
from transformers import PretrainedConfig


class MiniMindConfig(PretrainedConfig):
    """
    MiniMind模型配置类
    ==================
    
    项目中的作用：
    - 作为整个MiniMind项目的配置中心，统一管理所有超参数
    - 提供模型训练、推理和部署所需的完整配置信息
    - 支持不同规模模型的灵活配置（512维、768维等）
    
    大模型框架中的作用：
    - 继承自Hugging Face的PretrainedConfig，确保与transformers生态兼容
    - 实现模型配置的标准化存储和加载机制
    - 支持模型配置的版本控制和复现性保证
    - 为不同的训练和推理场景提供配置模板
    
    核心特性：
    1. 基础架构配置：层数、头数、隐藏维度等Transformer核心参数
    2. 高级特性配置：GQA、Flash Attention、RoPE等现代技术开关
    3. MoE专家混合配置：专家数量、路由策略、负载平衡等
    4. 训练优化配置：dropout、归一化参数等正则化设置
    
    使用场景：
    - 模型初始化时定义架构参数
    - 训练脚本中配置超参数
    - 模型保存时记录完整配置
    - 推理部署时恢复模型设置
    """
    # model_type详细说明：
    # 这是transformers库要求的模型类型标识符
    # 用于在模型注册和自动加载时识别模型类型
    # 必须是唯一的字符串，通常与模型名称对应
    # 在使用AutoModel.from_pretrained()时会根据此标识符选择正确的模型类
    model_type = "minimind"

    def __init__(
            self,
            dropout: float = 0.0,
            bos_token_id: int = 1,
            eos_token_id: int = 2,
            hidden_act: str = 'silu',
            hidden_size: int = 512,           # 🎯 重要概念：为什么是 int 而不是矩阵形状？
            # =====================================================================
            # ⚠️ 常见疑惑解答：hidden_size 为什么是整数而不是二维形状？
            # 
            # 🤔 错误理解：认为应该是 [512, 512] 这样的矩阵形状
            # ✅ 正确理解：hidden_size=512 表示每个token的特征向量维度
            # 
            # 📐 完整的张量维度解释：
            # ----------------------
            # 在Transformer中，数据以3D张量形式流动：
            # [batch_size, sequence_length, hidden_size]
            # 
            # 具体例子：
            # - batch_size=4        (4个样本)
            # - sequence_length=128 (每个样本128个token)
            # - hidden_size=512     (每个token用512维向量表示)
            # - 完整形状：[4, 128, 512]
            # 
            # 🔍 hidden_size 的真实含义：
            # ---------------------------
            # 1. 特征空间维度：每个token的向量表示有多少个维度
            # 2. 模型宽度：决定模型的表达能力和计算复杂度
            # 3. 投影输入：所有线性层的输入维度
            # 
            # 💡 权重矩阵的真实形状：
            # ----------------------
            # Q权重：[hidden_size, num_heads * head_dim] = [512, 8*64] = [512, 512]
            # K权重：[hidden_size, num_kv_heads * head_dim] = [512, 2*64] = [512, 128]
            # V权重：[hidden_size, num_kv_heads * head_dim] = [512, 2*64] = [512, 128]
            # 
            # 🎯 类比理解：
            # -----------
            # 想象每个token是一个学生，hidden_size=512意味着：
            # - 每个学生有512项特征（身高、体重、各科成绩...）
            # - 不是说有512个学生，而是每个学生有512个特征维度
            # - 权重矩阵像是"特征转换器"，将这512个特征转换成新的特征组合
            # =====================================================================
            intermediate_size: int = None,
            max_position_embeddings: int = 32768,
            num_attention_heads: int = 8,
            num_hidden_layers: int = 8,
            num_key_value_heads: int = 2,
            vocab_size: int = 6400,
            rms_norm_eps: float = 1e-05,
            rope_theta: int = 1000000.0,
            flash_attn: bool = True,
            ####################################################
            # Here are the specific configurations of MOE
            # When use_moe is false, the following is invalid
            ####################################################
            use_moe: bool = False,
            num_experts_per_tok: int = 2,
            n_routed_experts: int = 4,
            n_shared_experts: int = 1,
            scoring_func: str = 'softmax',
            aux_loss_alpha: float = 0.1,
            seq_aux: bool = True,
            norm_topk_prob: bool = True,
            **kwargs
    ):
        # super().__init__(**kwargs)详细说明：
        # 调用父类PretrainedConfig的初始化方法
        # **kwargs包含transformers库的标准配置参数：
        # - architectures: 模型架构信息，用于AutoModel自动加载
        # - torch_dtype: 模型权重的数据类型 (float16, float32等)
        # - use_cache: 是否默认使用KV缓存
        # - tie_word_embeddings: 是否共享输入输出嵌入权重
        # - pad_token_id, eos_token_id, bos_token_id: 特殊token的ID
        # 这确保了配置类与transformers生态系统的完全兼容
        super().__init__(**kwargs)
        
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.flash_attn = flash_attn
        ####################################################
        # Here are the specific configurations of MOE
        # When use_moe is false, the following is invalid
        ####################################################
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = n_routed_experts  # 总的专家数量
        self.n_shared_experts = n_shared_experts  # 共享专家
        self.scoring_func = scoring_func  # 评分函数，默认为'softmax'
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的alpha参数
        self.seq_aux = seq_aux  # 是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob  # 是否标准化top-k概率


# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Model
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

import math
import torch
from torch import nn

# Transformers库导入说明：
# 1. ACT2FN: 激活函数映射字典，将字符串映射到对应的激活函数
#    - 支持的激活函数: relu, gelu, silu, swish, tanh等
#    - 用法: ACT2FN['silu'] 返回 torch.nn.functional.silu
#    - 便于配置文件中使用字符串指定激活函数类型
from transformers.activations import ACT2FN

from typing import Optional, Tuple, List, Union
import torch.nn.functional as F

# 2. PreTrainedModel: Hugging Face模型基类
#    - 提供标准化的模型接口和管理功能
#    - 包含save_pretrained(), from_pretrained()等核心方法
#    - 支持设备管理、梯度检查点、参数统计等功能
#
# 3. GenerationMixin: 文本生成功能混入类
#    - 为语言模型提供generate()方法
#    - 支持多种生成策略：贪婪搜索、束搜索、采样等
#    - 包含生成配置管理和解码算法实现
#
# 4. PretrainedConfig: 配置基类（重复导入，用于类型注解）
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig

# 5. CausalLMOutputWithPast: 因果语言模型输出格式类
#    - 标准化的模型输出容器，包含logits、past_key_values等
#    - 兼容Hugging Face生态系统的输出格式
#    - 支持KV缓存传递和批量推理优化
from transformers.modeling_outputs import CausalLMOutputWithPast


class RMSNorm(torch.nn.Module):
    """
    根均方归一化（Root Mean Square Normalization）层
    ===============================================
    
    项目中的作用：
    - 作为MiniMind模型的标准归一化组件，替代传统的LayerNorm
    - 在每个Transformer块中进行特征归一化，稳定训练过程
    - 提供更高效的计算实现，减少训练和推理的计算开销
    
    大模型框架中的作用：
    - 代表现代大语言模型的归一化技术趋势
    - 被LLaMA、GPT-NeoX等主流模型广泛采用
    - 提供相比LayerNorm更好的数值稳定性和计算效率
    - 支持大规模模型的稳定训练和收敛
    
    技术原理：
    1. 简化的归一化公式：
       - 移除LayerNorm中的均值减法操作
       - 仅保留方差归一化和可学习缩放
       - 公式：RMSNorm(x) = x / sqrt(mean(x²) + ε) * weight
    
    2. 计算优化：
       - 减少计算步骤，提升执行效率
       - 更好的数值稳定性特性
       - 支持混合精度训练优化
    
    3. 参数设计：
       - 仅包含可学习的缩放参数weight
       - 初始化为全1向量
       - 相比LayerNorm减少了偏置参数
    
    性能优势：
    1. 计算效率：
       - 减少约25%的计算量
       - 更好的GPU并行化特性
       - 支持高效的kernel实现
    
    2. 数值稳定性：
       - 避免均值计算的数值误差
       - 更稳定的梯度传播
       - 适合大规模模型训练
    
    3. 内存友好：
       - 减少中间计算结果的存储
       - 支持内存优化的反向传播
       - 适合大模型和长序列场景
    
    应用场景：
    - Pre-Norm Transformer架构的标准组件
    - 大规模语言模型的归一化层
    - 需要计算效率优化的深度网络
    - 混合精度训练的稳定化组件
    
    与LayerNorm的对比：
    - LayerNorm: (x - mean(x)) / sqrt(var(x) + ε) * weight + bias
    - RMSNorm: x / sqrt(mean(x²) + ε) * weight
    - 优势：计算更简单，参数更少，稳定性更好
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        # eps参数：数值稳定性常数，防止除零错误
        # 训练作用：确保梯度计算的稳定性，避免数值溢出
        # 推理作用：保证归一化计算的数值精度
        self.eps = eps
        
        # weight参数：可学习的缩放参数，形状为(dim,)
        # 训练作用：学习每个特征维度的最优缩放因子，影响收敛速度和最终性能
        # 推理作用：对归一化后的特征进行缩放，恢复模型的表达能力
        # 初始化为全1：保证训练初期不改变输入分布
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        执行RMS归一化的核心计算
        
        Args:
            x (torch.Tensor): 输入张量，形状为 (..., dim)
            
        Returns:
            torch.Tensor: 归一化后的张量，形状与输入相同
        """
        # 计算x²的均值：沿最后一个维度计算，保持其他维度不变
        # 训练作用：提供稳定的梯度传播路径，防止梯度爆炸/消失
        # 推理作用：标准化特征分布，提高模型对输入变化的鲁棒性
        # keepdim=True：保持维度用于后续广播运算
        variance = x.pow(2).mean(-1, keepdim=True)
        
        # torch.rsqrt()：计算平方根的倒数，等价于1/sqrt(variance + eps)
        # 训练作用：比先sqrt再取倒数更高效，减少数值误差累积
        # 推理作用：快速计算归一化因子，优化推理速度
        # 加eps防止方差为0时的除零错误
        return x * torch.rsqrt(variance + self.eps)

    def forward(self, x):
        """
        RMSNorm的前向传播
        
        Args:
            x (torch.Tensor): 输入张量，通常为隐藏状态
            
        Returns:
            torch.Tensor: 归一化并缩放后的输出张量
        """
        # 转换为float32进行计算：提高数值稳定性
        # 训练作用：避免半精度训练时的数值不稳定，确保梯度计算精度
        # 推理作用：在混合精度推理中保证关键计算的精度
        normalized = self._norm(x.float())
        
        # 应用可学习权重并转回原始类型
        # 训练作用：权重参数接收梯度更新，学习最优的特征缩放
        # 推理作用：恢复模型训练时学到的特征分布特性
        # type_as(x)：确保输出类型与输入一致，支持混合精度
        return self.weight * normalized.type_as(x)


def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), theta: float = 1e6):
    """
    预计算旋转位置编码（RoPE）的频率矩阵
    
    RoPE通过旋转的方式将位置信息编码到查询和键向量中，相比传统的绝对位置编码，
    RoPE具有更好的外推能力，能够处理比训练时更长的序列。
    
    Args:
        dim (int): 注意力头的维度大小
        end (int): 支持的最大序列长度，默认32K
        theta (float): 旋转频率的基数，默认1e6
        
    Returns:
        tuple: (freqs_cos, freqs_sin) - 预计算的余弦和正弦频率矩阵
    """
    # 计算频率序列：1.0 / (theta^(2i/dim)) for i in [0, 2, 4, ..., dim-2]
    # 训练作用：预定义的频率模式，不参与训练但影响位置编码效果
    # 推理作用：提供位置敏感的特征变换，使模型能区分不同位置的token
    # 只取偶数索引：RoPE算法要求成对处理相邻维度
    indices = torch.arange(0, dim, 2)[: (dim // 2)].float()  # [0, 2, 4, ..., dim-2]
    freqs = 1.0 / (theta ** (indices / dim))  # 计算基础频率
    
    # 生成位置序列：[0, 1, 2, ..., end-1]
    # 训练作用：覆盖训练序列的所有可能位置
    # 推理作用：支持任意长度序列的位置编码，实现长度外推
    t = torch.arange(end, device=freqs.device)
    
    # 计算外积：每个位置与每个频率的组合
    # 训练作用：为每个位置-频率对生成唯一的相位
    # 推理作用：提供连续的位置编码，保证位置间的相对关系
    # 形状：(end, dim//2)
    freqs = torch.outer(t, freqs).float()
    
    # 计算余弦和正弦值，并复制以匹配完整维度
    # 训练作用：提供平滑的位置编码梯度，有利于位置信息的学习
    # 推理作用：实现高效的旋转变换，保持向量长度不变
    # 复制一份：因为RoPE需要对相邻维度成对应用
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)  # 形状：(end, dim)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)  # 形状：(end, dim)
    
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    将旋转位置编码应用到查询（Q）和键（K）向量上
    
    这是RoPE的核心实现函数，通过复数旋转的方式将位置信息融入注意力机制。
    
    Args:
        q (torch.Tensor): 查询向量，形状为 (batch_size, seq_len, num_heads, head_dim)
        k (torch.Tensor): 键向量，形状为 (batch_size, seq_len, num_heads, head_dim)
        cos (torch.Tensor): 预计算的余弦频率矩阵
        sin (torch.Tensor): 预计算的正弦频率矩阵
        position_ids (torch.Tensor, optional): 位置ID，暂未使用
        unsqueeze_dim (int): 在哪个维度添加维度以进行广播，默认为1
        
    Returns:
        tuple: (q_embed, k_embed) - 应用了位置编码的查询和键向量
    """
    def rotate_half(x):
        """
        将向量的后半部分移到前面，前半部分移到后面并取负号
        这是复数旋转在实数域的等价实现
        
        Args:
            x (torch.Tensor): 输入向量
            
        Returns:
            torch.Tensor: 旋转后的向量
        """
        # 获取向量的中点位置
        # 训练作用：实现复数旋转的实数等价，保持梯度的连续性
        # 推理作用：高效计算旋转变换，不增加额外的计算复杂度
        mid = x.shape[-1] // 2
        
        # 分割向量并重新组合：[x2, x1] -> [-x2, x1]
        # 这等价于复数乘法 (a + bi) * (cos + sin*i) 的实数实现
        x1, x2 = x[..., :mid], x[..., mid:]  # 分割为前半部分和后半部分
        
        # 重新组合：前半部分取负号并移到后面，后半部分移到前面
        # 训练作用：保持旋转变换的可微性，支持端到端训练
        # 推理作用：实现准确的位置编码变换
        return torch.cat((-x2, x1), dim=-1)

    # 应用旋转变换：x_new = x*cos + rotate_half(x)*sin
    # 这是复数旋转公式的实数实现：Real(z * e^(iθ)) = Real(z)*cos(θ) - Imag(z)*sin(θ)
    
    # 为cos和sin增加维度以匹配q和k的形状进行广播
    # 训练作用：确保位置编码正确应用到每个注意力头
    # 推理作用：高效的向量化计算，避免循环操作
    cos_expanded = cos.unsqueeze(unsqueeze_dim)  # 在指定维度增加一维
    sin_expanded = sin.unsqueeze(unsqueeze_dim)
    
    # 对查询向量应用旋转位置编码
    # 训练作用：为查询向量注入位置信息，影响注意力权重的计算
    # 推理作用：使模型能够识别token的绝对和相对位置
    q_embed = (q * cos_expanded) + (rotate_half(q) * sin_expanded)
    
    # 对键向量应用旋转位置编码
    # 训练作用：为键向量注入位置信息，与查询向量的位置编码配合
    # 推理作用：确保注意力计算中的位置一致性和相对位置感知
    k_embed = (k * cos_expanded) + (rotate_half(k) * sin_expanded)
    
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    重复键值（Key-Value）张量以支持分组查询注意力（Grouped Query Attention, GQA）
    
    在GQA中，查询头的数量通常大于键值头的数量，需要将键值张量重复以匹配查询头的数量。
    这样可以减少KV缓存的内存占用，同时保持模型性能。
    
    Args:
        x (torch.Tensor): 输入张量，形状为 (bs, slen, num_key_value_heads, head_dim)
        n_rep (int): 重复次数，通常为 num_attention_heads // num_key_value_heads
        
    Returns:
        torch.Tensor: 重复后的张量，形状为 (bs, slen, num_attention_heads, head_dim)
        
    Note:
        等价于 torch.repeat_interleave(x, dim=2, repeats=n_rep)，但实现更高效
    """
    # 获取输入张量的维度信息
    # 训练作用：确保张量形状的正确性，避免维度不匹配错误
    # 推理作用：为后续的张量操作提供准确的形状信息
    bs, slen, num_key_value_heads, head_dim = x.shape
    
    # 如果不需要重复，直接返回原张量
    # 训练作用：在标准多头注意力（非GQA）中避免不必要的计算
    # 推理作用：优化计算效率，减少内存拷贝
    if n_rep == 1:
        return x
    
    # 高效的张量重复实现
    # 步骤1：在第4个维度（新增维度）插入一个维度
    # 训练作用：准备张量结构以进行重复操作，保持梯度传播的正确性
    # 推理作用：创建重复模板，为后续expand操作做准备
    x_expanded = x[:, :, :, None, :]  # 形状：(bs, slen, num_kv_heads, 1, head_dim)
    
    # 步骤2：在新增的维度上扩展n_rep次
    # 训练作用：实现键值头的逻辑重复，使GQA能够与多头查询匹配
    # 推理作用：高效的内存扩展，比直接复制更节省内存
    x_repeated = x_expanded.expand(bs, slen, num_key_value_heads, n_rep, head_dim)
    
    # 步骤3：重塑张量以合并重复的维度
    # 训练作用：将重复的键值头展平为独立的头，匹配查询头的数量
    # 推理作用：生成最终形状，使后续注意力计算能够正确对齐
    result = x_repeated.reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    
    return result


class Attention(nn.Module):
    """
    多头注意力机制（Multi-Head Attention）
    =======================================
    
    项目中的作用：
    - 作为MiniMind模型的核心计算组件，实现序列内信息交互
    - 负责捕获长距离依赖关系和上下文语义理解
    - 支持高效推理的KV缓存机制，实现流式文本生成
    
    大模型框架中的作用：
    - 实现现代Transformer架构的注意力机制标准
    - 集成分组查询注意力（GQA）技术，平衡性能和内存效率
    - 支持Flash Attention加速，适应大规模模型训练和推理
    - 提供完整的位置编码集成（RoPE），增强位置感知能力
    
    技术创新点：
    1. 分组查询注意力（GQA）：
       - 减少键值头数量，降低KV缓存内存占用
       - 在保持模型性能的同时提升推理效率
       
    2. Flash Attention支持：
       - 内存高效的注意力计算实现
       - 支持长序列训练和推理
       
    3. 旋转位置编码（RoPE）集成：
       - 相对位置编码，具备更好的外推能力
       - 支持比训练时更长的序列处理
    
    4. KV缓存优化：
       - 增量解码机制，避免重复计算
       - 支持流式生成和实时对话
    
    计算流程：
    Q, K, V ← Linear(X)  # 线性投影
    Q, K ← RoPE(Q, K)    # 应用旋转位置编码
    K, V ← Concat(Past_KV, K, V)  # KV缓存拼接
    Attn ← Attention(Q, K, V)     # 注意力计算
    Output ← Linear(Attn)         # 输出投影
    
    适用场景：
    - 因果语言模型的自回归生成
    - 长文本理解和生成任务
    - 对话系统和文本续写
    - 代码生成和问答任务
    """
    
    def __init__(self, args: MiniMindConfig):
        """
        初始化注意力层
        
        Args:
            args (MiniMindConfig): 模型配置参数
        """
        super().__init__()
        
        # 设置键值头数量：如果未指定则与注意力头数量相同（标准多头注意力）
        # 训练作用：确定KV缓存的内存需求和计算复杂度
        # 推理作用：影响推理时的内存占用和计算效率
        self.num_key_value_heads = (args.num_attention_heads 
                                   if args.num_key_value_heads is None 
                                   else args.num_key_value_heads)
        
        # 确保注意力头数量能被键值头数量整除（GQA的数学要求）
        # 训练作用：验证配置的正确性，避免训练时的维度错误
        # 推理作用：确保repeat_kv函数能够正确工作
        assert args.num_attention_heads % self.num_key_value_heads == 0
        
        # 缓存关键维度信息以优化后续计算
        self.n_local_heads = args.num_attention_heads      # 查询头数量
        self.n_local_kv_heads = self.num_key_value_heads   # 键值头数量
        
        # 计算每个KV头对应的Q头数量（GQA的核心参数）
        # 训练作用：决定键值张量的重复倍数，影响内存和计算效率
        # 推理作用：控制KV缓存的复用程度，降低内存需求
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        
        # 计算每个注意力头的维度
        # 训练作用：确定权重矩阵的形状和参数量
        # 推理作用：影响注意力计算的规模和精度
        self.head_dim = args.hidden_size // args.num_attention_heads
        
        # === QKV投影矩阵定义 ===
        # 这里定义了注意力机制的核心权重矩阵，每个矩阵都是可学习的参数
        
        # 🔍 查询投影矩阵 (Query Projection Matrix)
        # ⚠️ 重要概念解释：self.q_proj 就是传统 Attention 论文中的 W_Q 权重矩阵！
        # 
        # 📚 理论对应关系：
        # - 论文中: Q = X @ W_Q (X是输入，W_Q是查询权重矩阵)
        # - 代码中: xq = self.q_proj(x) (等价于 x @ W_Q + b，这里bias=False)
        # 
        # 🔍 矩阵维度分析：
        # - W_Q 形状: [hidden_size, num_attention_heads * head_dim]
        #   具体数值: [512, 8 * 64] = [512, 512] ← 这才是真正的矩阵形状！
        # - 输入 x 形状: [batch_size, seq_len, hidden_size] 
        #   具体例子: [4, 128, 512] ← hidden_size 只是最后一个维度
        # - 输出 xq 形状: [batch_size, seq_len, num_attention_heads * head_dim]
        #   具体例子: [4, 128, 512] ← 经过线性变换后的形状
        # 
        # 💡 为什么叫"投影"？
        # - 将 hidden_size=512 维度的向量"投影"到 (8*64)=512 维度空间
        # - 本质上就是矩阵乘法: x @ W_Q，是线性变换的几何解释
        # - 数学表示: [batch, seq, 512] @ [512, 512] = [batch, seq, 512]
        # 
        # 🎯 nn.Linear 的内部实现：
        # - nn.Linear(512, 512) 内部有权重矩阵 self.weight [512, 512]
        # - forward时执行: F.linear(input, self.weight, self.bias) = input @ self.weight.T + bias
        # - 所以 self.q_proj.weight 就是传统的 W_Q^T (转置形式)
        # 
        # 🔍 关键理解：hidden_size=512 是"特征维度"，不是"矩阵的行列数"
        #   真正的矩阵形状由 nn.Linear(输入维度, 输出维度) 决定
        # 
        # 训练作用：学习如何从输入生成查询向量，影响注意力模式
        # 推理作用：生成用于计算注意力权重的查询表示
        # bias=False：减少参数量，避免不必要的偏置
        self.q_proj = nn.Linear(args.hidden_size, 
                               args.num_attention_heads * self.head_dim, 
                               bias=False)
        
        # 🔑 键投影矩阵 (Key Projection Matrix)  
        # ⚠️ 重要概念解释：self.k_proj 就是传统 Attention 论文中的 W_K 权重矩阵！
        # 
        # 📚 理论对应关系：
        # - 论文中: K = X @ W_K (X是输入，W_K是键权重矩阵)
        # - 代码中: xk = self.k_proj(x) (等价于 x @ W_K，bias=False)
        # 
        # 🔍 矩阵维度分析：
        # - W_K 形状: [hidden_size, num_key_value_heads * head_dim]
        # - 注意: 在GQA中，键头数可能少于查询头数，实现参数共享
        # - 例如: 8个查询头可能只对应2个键头，减少4倍参数量
        # 
        # 💡 GQA (Grouped Query Attention) 优化：
        # - 传统MHA: num_key_heads = num_query_heads
        # - GQA: num_key_heads < num_query_heads (参数共享)
        # - 好处: 显著减少KV缓存内存，加速推理
        # 
        # 训练作用：学习如何生成用于匹配的键向量
        # 推理作用：生成键表示，与查询计算相似度
        self.k_proj = nn.Linear(args.hidden_size, 
                               self.num_key_value_heads * self.head_dim, 
                               bias=False)
        
        # 💎 值投影矩阵 (Value Projection Matrix)
        # ⚠️ 重要概念解释：self.v_proj 就是传统 Attention 论文中的 W_V 权重矩阵！
        # 
        # 📚 理论对应关系：
        # - 论文中: V = X @ W_V (X是输入，W_V是值权重矩阵)  
        # - 代码中: xv = self.v_proj(x) (等价于 x @ W_V，bias=False)
        # 
        # 🔍 矩阵维度分析：
        # - W_V 形状: [hidden_size, num_key_value_heads * head_dim] 
        # - 注意: 值头数与键头数相同，确保一一对应关系
        # - 每个键都有对应的值，保持注意力计算的数学一致性
        # 
        # 💡 值矩阵的特殊作用：
        # - Q和K决定"关注什么"(注意力权重)
        # - V决定"传递什么信息"(实际内容)
        # - 最终输出 = 注意力权重 × 值向量 的加权组合
        # 
        # 🎯 与传统MLP的区别：
        # - 普通MLP: 所有位置使用相同的权重
        # - Attention: 每个位置根据其他位置动态加权V矩阵的输出
        # 
        # 训练作用：学习如何生成用于聚合的值向量
        # 推理作用：生成最终输出的内容表示
        self.v_proj = nn.Linear(args.hidden_size, 
                               self.num_key_value_heads * self.head_dim, 
                               bias=False)
        
        # 🎯 输出投影矩阵 (Output Projection Matrix)
        # 矩阵形状: [num_attention_heads * head_dim, hidden_size]
        # 作用: 将多头注意力的输出重新投影回原始隐藏维度
        # 训练作用：学习如何整合多头信息，影响最终输出质量
        # 推理作用：生成统一的输出表示
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, 
                               args.hidden_size, 
                               bias=False)
        
        # 注意力权重的dropout：防止过拟合，提高泛化能力
        # 训练作用：在注意力权重上应用随机失活，增强模型鲁棒性
        # 推理作用：在推理时关闭，确保输出的确定性
        self.attn_dropout = nn.Dropout(args.dropout)
        
        # 残差连接的dropout：对输出进行正则化
        # 训练作用：在残差路径上应用dropout，防止过拟合
        # 推理作用：推理时自动关闭
        self.resid_dropout = nn.Dropout(args.dropout)
        
        # 缓存dropout率以供后续使用
        self.dropout = args.dropout
        
        # 检查是否支持Flash Attention并根据配置启用
        # 训练作用：启用内存高效的注意力计算，支持更长序列的训练
        # 推理作用：显著降低推理时的内存占用和计算时间
        # hasattr检查：确保PyTorch版本支持（需要PyTorch >= 2.0）
        self.flash = (hasattr(torch.nn.functional, 'scaled_dot_product_attention') 
                     and args.flash_attn)

    def forward(self,
                x: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False,
                attention_mask: Optional[torch.Tensor] = None):
        """
        注意力机制的前向传播
        
        Args:
            x (torch.Tensor): 输入隐藏状态，形状为 (batch_size, seq_len, hidden_size)
            position_embeddings (Tuple): RoPE的余弦和正弦编码 (cos, sin)
            past_key_value (Optional[Tuple]): KV缓存，用于生成时的增量解码
            use_cache (bool): 是否返回KV缓存供下次使用
            attention_mask (Optional[torch.Tensor]): 注意力掩码
            
        Returns:
            Tuple[torch.Tensor, Optional[Tuple]]: (输出隐藏状态, KV缓存)
        """
        # 获取输入张量的基本维度
        # 训练作用：动态适应不同批次大小和序列长度，支持灵活的训练配置
        # 推理作用：处理可变长度输入，适应实际应用场景
        bsz, seq_len, _ = x.shape
        
        # === QKV矩阵变换计算过程 ===
        # 这里展示了从输入到注意力计算的完整QKV变换流程
        
        # 🔄 步骤1: 线性投影变换 - 通过权重矩阵生成Q、K、V
        # 输入: x [batch_size, seq_len, hidden_size]
        # 输出: xq [batch_size, seq_len, n_heads * head_dim]
        #      xk [batch_size, seq_len, n_kv_heads * head_dim] 
        #      xv [batch_size, seq_len, n_kv_heads * head_dim]
        # 数学公式: Q = x @ W_q, K = x @ W_k, V = x @ W_v
        # 训练作用：学习如何将输入转换为注意力机制所需的三种表示
        # 推理作用：根据学习到的权重生成查询、键、值向量
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        
        # 🔄 步骤2: 重塑为多头注意力格式
        # 将一维的投影结果重塑为多头结构，每个头处理部分特征
        # 输出形状: [batch_size, seq_len, num_heads, head_dim]
        # 训练作用：组织张量结构以支持多头并行训练，提高训练效率
        # 推理作用：为高效的多头注意力计算准备数据格式
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)       # 查询：完整头数
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)    # 键：GQA减少的头数
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)    # 值：与键相同头数

        # === 应用旋转位置编码（RoPE） ===
        # 提取RoPE的余弦和正弦分量，并应用到查询和键上
        # 训练作用：让模型学习位置相关的注意力模式，提高位置理解能力
        # 推理作用：确保模型能够正确理解token的相对位置关系
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])

        # === KV缓存机制 ===
        # 在生成阶段，将历史的键值与当前键值拼接，避免重复计算
        # 训练作用：训练时通常不使用，因为有完整序列
        # 推理作用：显著加速增量生成，避免重复计算历史token的KV
        if past_key_value is not None:
            # 在序列维度上拼接历史KV缓存
            xk = torch.cat([past_key_value[0], xk], dim=1)  # 拼接历史键
            xv = torch.cat([past_key_value[1], xv], dim=1)  # 拼接历史值
        
        # 根据需要保存当前的KV状态供下次使用
        # 训练作用：训练时通常设为None
        # 推理作用：保存状态以支持高效的增量生成
        past_kv = (xk, xv) if use_cache else None

        # === 分组查询注意力（GQA）处理 ===
        # 转置维度并重复KV以匹配查询头数量
        # 训练作用：实现参数高效的多头注意力，减少内存需求
        # 推理作用：显著降低KV缓存的内存占用
        xq, xk, xv = (
            xq.transpose(1, 2),                               # Q: [batch, n_heads, seq, head_dim]
            repeat_kv(xk, self.n_rep).transpose(1, 2),        # K: 重复后转置
            repeat_kv(xv, self.n_rep).transpose(1, 2)         # V: 重复后转置
        )

        # === 注意力计算：选择高效实现 ===
        if self.flash and seq_len != 1:
            # 使用Flash Attention：内存高效的融合实现
            # 训练作用：支持更长序列训练，减少内存瓶颈，加速训练过程
            # 推理作用：显著降低推理时的内存使用和计算时间
            dropout_p = self.dropout if self.training else 0.0  # 推理时关闭dropout
            attn_mask = None
            
            # 处理注意力掩码：扩展到多头格式
            # 训练作用：正确应用填充掩码，避免关注填充token
            # 推理作用：确保生成质量，防止关注无效位置
            if attention_mask is not None:
                # 扩展掩码维度以匹配多头注意力的形状
                attn_mask = attention_mask.view(bsz, 1, 1, -1).expand(bsz, self.n_local_heads, seq_len, -1)
                attn_mask = attn_mask.bool() if attention_mask is not None else None

            # 执行Flash Attention计算
            # is_causal=True：自动应用因果掩码，确保自回归特性
            output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=True)
        else:
            # === 传统注意力计算 ===
            # 用于序列长度为1的情况（增量生成）或不支持Flash Attention时
            # 计算注意力分数：Q @ K^T / sqrt(d_k)
            # 训练作用：学习token间的关联强度，构建注意力模式
            # 推理作用：确定当前token应该关注哪些历史信息
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            # 添加因果掩码：防止模型看到未来信息
            # 训练作用：确保模型只能使用历史信息进行预测，维持因果关系
            # 推理作用：保持语言模型的自回归生成特性
            scores = scores + torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                diagonal=1  # 对角线上方设为负无穷
            ).unsqueeze(0).unsqueeze(0)  # 扩展到批次和头维度

            # 应用额外的注意力掩码（如填充掩码）
            # 训练作用：避免模型关注填充位置，提高训练效果
            # 推理作用：确保生成时不会关注无效的填充token
            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9  # 将0位置设为极小值
                scores = scores + extended_attention_mask

            # 计算注意力权重：softmax归一化
            # 训练作用：将分数转换为概率分布，支持梯度传播
            # 推理作用：生成用于信息聚合的权重分布
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            
            # 应用注意力dropout（仅训练时）
            # 训练作用：防止过拟合，增强模型泛化能力
            # 推理作用：推理时自动跳过
            scores = self.attn_dropout(scores)
            
            # 计算加权的值向量：Attention @ V
            # 训练作用：根据注意力权重聚合信息，生成上下文化表示
            # 推理作用：产生融合了相关历史信息的输出
            output = scores @ xv

        # === 输出处理 ===
        # 将多头输出合并回原始格式
        # 训练作用：将多头信息融合为统一的隐藏表示
        # 推理作用：生成标准格式的输出供后续层使用
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        
        # 通过输出投影层并应用残差dropout
        # 训练作用：学习如何整合多头注意力信息，并进行正则化
        # 推理作用：生成最终的注意力输出
        output = self.resid_dropout(self.o_proj(output))
        
        return output, past_kv


class FeedForward(nn.Module):
    """
    前馈神经网络（Feed-Forward Network）
    ===================================
    
    项目中的作用：
    - 作为Transformer块的非线性变换组件，提供模型的表达能力
    - 实现SwiGLU激活机制，增强模型的拟合能力
    - 为每个位置的token提供独立的特征变换
    
    大模型框架中的作用：
    - 承担Transformer架构中的核心计算任务，占用模型大部分参数
    - 实现现代大语言模型的标准FFN设计（SwiGLU激活）
    - 提供模型的主要非线性表达能力和记忆存储功能
    - 支持高效的参数缩放和模型容量扩展
    
    架构特点：
    1. SwiGLU激活函数：
       - 结合Swish激活和门控线性单元（GLU）
       - 相比传统ReLU具有更好的性能和梯度特性
       - 被GPT-3.5、LLaMA等主流模型广泛采用
    
    2. 三线性层设计：
       - gate_proj：门控投影，控制信息流
       - up_proj：上采样投影，扩展特征维度
       - down_proj：下采样投影，恢复原始维度
    
    3. 维度缩放策略：
       - 中间层维度通常为隐藏层的8/3倍
       - 对齐到64的倍数，优化计算效率
    
    计算公式：
    FFN(x) = down_proj(SiLU(gate_proj(x)) ⊙ up_proj(x))
    
    性能特性：
    - 参数量占整个模型的2/3左右
    - 计算密集型，适合GPU并行优化
    - 支持梯度检查点和混合精度训练
    
    适用场景：
    - 语言模型的标准前馈层
    - 需要强非线性表达的序列建模任务
    - 大规模预训练和微调场景
    """
    
    def __init__(self, config: MiniMindConfig):
        """
        初始化前馈网络
        
        Args:
            config (MiniMindConfig): 模型配置参数
        """
        super().__init__()
        
        # 计算中间层维度：如果未指定则按8/3倍隐藏维度设置
        # 训练作用：确定前馈网络的表达能力和参数量
        # 推理作用：影响推理时的计算复杂度和内存需求
        if config.intermediate_size is None:
            # 8/3倍的经验比例：来自Transformer架构的最佳实践
            intermediate_size = int(config.hidden_size * 8 / 3)
            # 对齐到64的倍数：优化GPU tensor core计算效率
            # 训练作用：加速矩阵乘法运算，提高训练吞吐量
            # 推理作用：充分利用硬件特性，提升推理速度
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
            
        # === SwiGLU架构的三个线性层 ===
        # 门控投影层：生成门控信号，控制信息流
        # 训练作用：学习哪些特征应该被激活或抑制
        # 推理作用：动态调节信息传递的强度
        # bias=False：减少参数量，避免引入额外偏置
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        
        # 下采样投影层：将扩展的特征映射回原始维度
        # 训练作用：学习如何整合中间层的丰富特征
        # 推理作用：生成最终的前馈网络输出
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        
        # 上采样投影层：将输入扩展到中间维度空间
        # 训练作用：增加特征表达的维度，提供更强的非线性变换能力
        # 推理作用：为门控机制提供丰富的特征表示
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        
        # Dropout层：防止过拟合
        # 训练作用：在前馈网络输出上应用随机失活，增强泛化能力
        # 推理作用：推理时自动关闭，确保输出的确定性
        self.dropout = nn.Dropout(config.dropout)
        
        # 激活函数选择：从transformers库的映射中获取
        # ACT2FN详细说明：
        # ACT2FN是transformers库提供的激活函数映射字典
        # 将配置文件中的字符串映射到对应的PyTorch激活函数
        # 支持的激活函数包括：
        # - "relu": torch.nn.functional.relu
        # - "gelu": torch.nn.functional.gelu  
        # - "silu": torch.nn.functional.silu (也称为Swish)
        # - "swish": torch.nn.functional.silu (silu的别名)
        # - "tanh": torch.tanh
        # - "sigmoid": torch.sigmoid
        # 使用方式：ACT2FN[config.hidden_act] 获取对应的激活函数
        # 在MiniMind中通常使用'silu'激活函数，即Swish激活
        # 训练作用：提供平滑的梯度特性，有助于优化收敛
        # 推理作用：生成非线性变换，增强模型表达能力
        self.act_fn = ACT2FN[config.hidden_act]  # 激活函数，通常为SiLU

    def forward(self, x):
        """
        前馈网络的前向传播
        
        实现SwiGLU激活：gate_proj(x) * act_fn * up_proj(x)
        
        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, hidden_size)
            
        Returns:
            torch.Tensor: 输出张量，形状与输入相同
        """
        # === SwiGLU激活机制实现 ===
        # 公式：down_proj(act_fn(gate_proj(x)) ⊙ up_proj(x))
        
        # 第一步：计算门控信号 gate_proj(x)
        # 训练作用：学习控制信息流的门控权重
        # 推理作用：生成用于调节特征激活强度的门控向量
        gate_output = self.gate_proj(x)
        
        # 第二步：计算上采样特征 up_proj(x)
        # 训练作用：将输入投影到高维特征空间，增强表达能力
        # 推理作用：生成丰富的中间特征表示
        up_output = self.up_proj(x)
        
        # 第三步：应用激活函数到门控信号
        # 训练作用：引入非线性变换，提供平滑的梯度特性
        # 推理作用：生成平滑的门控权重，避免梯度消失问题
        activated_gate = self.act_fn(gate_output)
        
        # 第四步：门控融合 - 元素级别的乘法
        # 训练作用：学习特征选择和加权机制，提高模型表达能力
        # 推理作用：动态调节哪些特征应该被激活或抑制
        gated_features = activated_gate * up_output
        
        # 第五步：下采样投影，恢复原始维度
        # 训练作用：学习如何将高维特征整合回隐藏维度
        # 推理作用：生成最终的前馈网络输出
        output = self.down_proj(gated_features)
        
        # 第六步：应用dropout进行正则化
        # 训练作用：防止过拟合，增强模型泛化能力
        # 推理作用：推理时自动跳过，确保输出稳定性
        return self.dropout(output)


class MoEGate(nn.Module):
    """
    专家混合门控网络（Mixture of Experts Gating Network）
    ====================================================
    
    项目中的作用：
    - 作为MoE架构的智能路由器，为每个token选择最合适的专家网络
    - 实现稀疏激活机制，在保持计算效率的同时大幅提升模型容量
    - 提供专家负载平衡功能，确保各专家得到均匀训练
    
    大模型框架中的作用：
    - 实现现代大模型的稀疏激活范式，突破传统密集模型的计算瓶颈
    - 支持模型容量与计算成本的解耦，实现更高效的参数利用
    - 提供动态计算路径选择，增强模型的表达能力和专业化程度
    - 为超大规模模型训练提供可行的技术路径
    
    核心机制：
    1. 智能路由算法：
       - 基于token隐藏状态计算专家选择概率
       - 支持top-k专家选择策略
       - 可配置的评分函数（softmax等）
    
    2. 负载平衡机制：
       - 辅助损失函数确保专家负载均衡
       - 支持序列级和token级平衡策略
       - 防止专家塌陷和过度专业化
    
    3. 稀疏激活优化：
       - 每个token仅激活少数专家（通常2-4个）
       - 显著减少实际计算量
       - 保持模型总容量的大幅提升
    
    技术优势：
    - 计算效率：仅激活部分专家，降低推理成本
    - 模型容量：支持数百个专家，大幅增加参数量
    - 专业化能力：不同专家学习不同领域知识
    - 可扩展性：支持灵活的专家数量配置
    
    应用场景：
    - 多语言大模型（不同语言专家）
    - 多模态模型（不同模态专家）
    - 领域特化模型（不同领域专家）
    - 超大规模语言模型（万亿参数级别）
    
    性能指标：
    - 专家利用率：衡量负载平衡效果
    - 路由一致性：评估专家选择稳定性
    - 稀疏度：激活专家占总专家的比例
    """
    
    def __init__(self, config: MiniMindConfig):
        """
        初始化MoE门控网络
        
        Args:
            config (MiniMindConfig): 模型配置参数
        """
        super().__init__()
        
        # 缓存配置对象以供后续使用
        # 训练作用：保存所有MoE相关的超参数配置
        # 推理作用：确保推理时使用正确的模型配置
        self.config = config
        
        # 设置top-k参数：每个token选择激活的专家数量
        # 训练作用：控制训练时的稀疏度和计算复杂度
        # 推理作用：决定推理时激活专家的数量，影响计算效率
        # 典型值：2-4个专家，平衡性能和效率
        self.top_k = config.num_experts_per_tok
        
        # 设置路由专家总数：参与动态选择的专家数量
        # 训练作用：决定模型容量和专家专业化程度
        # 推理作用：影响专家选择的搜索空间
        # 典型值：8-64个专家，根据模型规模调整
        self.n_routed_experts = config.n_routed_experts

        # 设置评分函数类型：专家选择的概率计算方式
        # 训练作用：影响专家选择的分布特性和梯度流
        # 推理作用：决定专家权重的计算方法
        # 支持的函数：'softmax'（默认）
        self.scoring_func = config.scoring_func
        
        # 设置辅助损失权重：平衡专家负载的损失函数系数
        # 训练作用：防止专家使用不均衡，确保所有专家得到训练
        # 推理作用：推理时不涉及损失计算
        # 典型值：0.1-1.0，需要与主损失平衡
        self.alpha = config.aux_loss_alpha
        
        # 设置序列级辅助损失标志
        # 训练作用：选择序列级或token级的负载平衡策略
        # 推理作用：不影响推理过程
        self.seq_aux = config.seq_aux

        # 设置top-k概率归一化标志
        # 训练作用：决定是否对选中专家的权重进行归一化
        # 推理作用：影响专家输出的加权方式
        # True时：确保选中专家权重和为1
        self.norm_topk_prob = config.norm_topk_prob
        
        # 设置门控输入维度：通常等于模型隐藏维度
        # 训练作用：确定门控网络输入的特征维度
        # 推理作用：确保输入张量维度匹配
        self.gating_dim = config.hidden_size
        
        # 创建门控权重矩阵：[n_experts, hidden_size]
        # 训练作用：学习每个专家对不同输入特征的亲和度
        # 推理作用：计算输入与各专家的匹配分数
        # Parameter：注册为可训练参数，会被优化器更新
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        
        # 初始化权重参数
        # 训练作用：为权重矩阵设置合适的初始值，有助于训练收敛
        # 推理作用：确保模型开始时有合理的专家选择行为
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        初始化门控网络的参数
        
        使用Kaiming均匀分布初始化权重矩阵
        """
        import torch.nn.init as init
        # 使用Kaiming均匀分布初始化权重矩阵
        # 训练作用：为权重提供合适的初始分布，有助于梯度流和收敛
        # 推理作用：确保模型在未训练时也有合理的专家选择行为
        # a=math.sqrt(5)：Kaiming初始化的标准参数，适合线性层
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        """
        门控网络的前向传播
        
        为每个token计算专家选择概率，并选择top-k个专家
        
        Args:
            hidden_states (torch.Tensor): 输入隐藏状态，形状为 (batch_size, seq_len, hidden_size)
            
        Returns:
            tuple: (topk_idx, topk_weight, aux_loss) - 选中的专家索引、权重和辅助损失
        """
        # 获取输入张量的形状信息
        # 训练作用：动态适应不同的批次大小和序列长度
        # 推理作用：处理可变长度的输入序列
        bsz, seq_len, h = hidden_states.shape
        
        # 展平输入张量以便进行线性变换
        # 训练作用：将三维张量转换为二维，方便矩阵乘法计算
        # 推理作用：优化计算效率，减少张量操作复杂度
        hidden_states = hidden_states.view(-1, h)  # [batch_size * seq_len, hidden_size]
        
        # === 计算专家门控分数 ===
        # 通过线性变换计算每个token对各专家的亲和度
        # 训练作用：学习token特征与专家特长的匹配关系
        # 推理作用：为专家选择提供量化的匹配分数
        # logits形状：[batch_size * seq_len, n_routed_experts]
        logits = F.linear(hidden_states, self.weight, None)
        
        # === 应用评分函数转换为概率分布 ===
        if self.scoring_func == 'softmax':
            # 使用softmax将logits转换为概率分布
            # 训练作用：确保专家选择概率和为1，便于梯度计算
            # 推理作用：生成规范化的专家选择概率
            scores = logits.softmax(dim=-1)
        else:
            # 抛出异常：目前仅支持softmax评分函数
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        # === 选择top-k个专家 ===
        # 为每个token选择概率最高的k个专家
        # 训练作用：实现稀疏激活，减少计算量同时保持模型容量
        # 推理作用：显著降低推理成本，仅激活相关专家
        # sorted=False：不需要按概率排序，提高计算效率
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        # === 可选的top-k权重归一化 ===
        if self.top_k > 1 and self.norm_topk_prob:
            # 将选中专家的权重重新归一化，确保权重和为1
            # 训练作用：标准化权重分布，稳定训练过程
            # 推理作用：确保专家输出的加权融合是规范化的
            # 1e-20：防止除零错误的小常数
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        # === 计算辅助损失（专家负载平衡） ===
        if self.training and self.alpha > 0.0:
            # 仅在训练时计算辅助损失，用于专家负载平衡
            scores_for_aux = scores
            aux_topk = self.top_k
            # 重塑索引张量以便计算负载平衡
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            
            if self.seq_aux:
                # === 序列级辅助损失 ===
                # 在序列级别计算专家负载平衡
                # 训练作用：确保每个序列中的专家使用相对均衡
                # 推理作用：不参与推理计算
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                
                # 创建专家使用计数张量
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                
                # 统计每个专家被选择的次数并归一化
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)
                
                # 计算序列级辅助损失：专家使用频率 × 专家选择概率
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                # === 标准辅助损失 ===
                # 在全局级别计算专家负载平衡
                # 训练作用：确保所有专家在整个批次中得到均衡使用
                # 推理作用：不参与推理计算
                
                # 创建one-hot编码表示专家选择
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                
                # 计算专家的实际使用频率
                ce = mask_ce.float().mean(0)
                
                # 计算专家的期望选择概率
                Pi = scores_for_aux.mean(0)
                
                # 计算负载平衡因子
                fi = ce * self.n_routed_experts
                
                # 辅助损失：使用频率与期望概率的点积
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            # 非训练模式或alpha=0时，不计算辅助损失
            aux_loss = 0
            
        return topk_idx, topk_weight, aux_loss


class MOEFeedForward(nn.Module):
    """
    专家混合前馈网络（Mixture of Experts Feed-Forward Network）
    ========================================================
    
    项目中的作用：
    - 作为MiniMind模型的高级前馈组件，替代标准FFN实现稀疏激活
    - 通过多专家架构大幅提升模型容量而不成比例增加计算成本
    - 实现动态专家选择和负载平衡，优化训练效率和模型性能
    
    大模型框架中的作用：
    - 代表现代大模型的前沿架构设计，实现参数和计算的高效解耦
    - 支持万亿参数级别模型的可行实现路径
    - 提供专家专业化机制，增强模型在不同任务上的能力
    - 为分布式训练和推理提供天然的并行化支持
    
    架构设计：
    1. 多专家系统：
       - 多个独立的FeedForward专家网络
       - 每个专家具有相同的架构但独立的参数
       - 支持专家的专业化学习和任务分工
    
    2. 智能门控机制：
       - 集成MoEGate进行专家选择
       - 支持top-k路由策略和权重分配
       - 实现负载平衡和专家利用率优化
    
    3. 混合激活策略：
       - 路由专家：动态选择激活
       - 共享专家：始终参与计算
       - 灵活的专家组合机制
    
    计算优化：
    1. 训练时优化：
       - 并行计算所有选中专家
       - 权重聚合和梯度反传
       - 支持大批量训练
    
    2. 推理时优化：
       - 批量处理相同专家的token
       - 减少内存碎片和计算开销
       - 专门的推理优化路径
    
    技术特性：
    - 稀疏激活：仅激活部分专家，降低计算成本
    - 容量扩展：支持大量专家，显著增加模型参数
    - 专业化学习：不同专家学习不同知识领域
    - 负载平衡：确保专家训练的均匀性
    
    应用优势：
    - 模型容量：相同计算成本下获得更大模型容量
    - 推理效率：稀疏激活降低实际计算需求
    - 专业能力：不同专家处理不同类型的输入
    - 可扩展性：支持灵活的专家数量配置
    
    适用场景：
    - 超大规模语言模型（GPT-4、PaLM等）
    - 多语言和多模态模型
    - 需要专业化能力的领域模型
    - 计算资源受限但需要大容量的场景
    """
    
    def __init__(self, config: MiniMindConfig):
        """
        初始化MoE前馈网络
        
        Args:
            config (MiniMindConfig): 模型配置参数
        """
        super().__init__()
        self.config = config
        
        # 创建多个专家前馈网络
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])
        
        # 门控网络用于专家选择
        self.gate = MoEGate(config)
        
        # 可选的共享专家网络（始终激活）
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config)
                for _ in range(config.n_shared_experts)
            ])

    def forward(self, x):
        """
        MoE前馈网络的前向传播
        
        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, hidden_size)
            
        Returns:
            torch.Tensor: 输出张量，形状与输入相同
        """
        identity = x  # 保存原始输入用于共享专家
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape
        
        # 使用门控机制选择专家
        topk_idx, topk_weight, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1])  # 展平为 (batch_size * seq_len, hidden_size)
        flat_topk_idx = topk_idx.view(-1)
        
        if self.training:
            # 训练时的处理方式：直接并行计算所有选中的专家
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            y = torch.empty_like(x, dtype=torch.float16)
            
            # 为每个专家计算输出
            for i, expert in enumerate(self.experts):
                mask = (flat_topk_idx == i)
                if mask.any():
                    y[mask] = expert(x[mask]).to(y.dtype)  # 确保类型一致
            
            # 根据权重聚合专家输出
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        else:
            # 推理时的优化处理方式
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        
        # 添加共享专家的输出
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)
        
        # 保存辅助损失供后续使用
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        """
        推理时的MoE计算优化版本
        
        通过批量处理相同专家的token来提高计算效率，避免了训练时的重复计算。
        
        Args:
            x (torch.Tensor): 输入张量，形状为 (total_tokens, hidden_size)
            flat_expert_indices (torch.Tensor): 专家索引，形状为 (total_tokens * num_experts_per_tok,)
            flat_expert_weights (torch.Tensor): 专家权重，形状为 (total_tokens * num_experts_per_tok, 1)
            
        Returns:
            torch.Tensor: 专家输出的加权组合
        """
        expert_cache = torch.zeros_like(x)
        
        # 按专家索引排序以便批量处理
        idxs = flat_expert_indices.argsort()
        
        # 计算每个专家处理的token数量的累积和
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        
        # 获取原始token索引（考虑到每个token可能被多个专家处理）
        token_idxs = idxs // self.config.num_experts_per_tok
        
        # 示例说明：
        # 当tokens_per_expert = [6, 15, 20, 26]，专家数量为4
        # 且token_idxs = [3, 7, 19, 21, 24, 25, 4, 5, 6, 10, 11, 12...] 时
        # token_idxs[:6] -> [3, 7, 19, 21, 24, 25] 属于专家0处理的token
        # token_idxs[6:15] -> [4, 5, 6, 10, 11, 12...] 属于专家1处理的token
        
        # 逐个专家处理
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue  # 跳过没有分配token的专家
            
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            
            # 计算专家输出并应用权重
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            
            # 将加权输出累加到对应的token位置
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache


class MiniMindBlock(nn.Module):
    """
    MiniMind Transformer块
    =====================
    
    项目中的作用：
    - 作为MiniMind模型的基本构建单元，实现标准的Transformer层架构
    - 整合注意力机制和前馈网络，提供完整的序列处理能力
    - 支持模型的层次化堆叠，形成深度神经网络架构
    
    大模型框架中的作用：
    - 实现现代Transformer架构的标准设计模式
    - 支持Pre-Norm结构，提升训练稳定性和收敛性能
    - 提供模块化设计，便于模型的复用和扩展
    - 集成先进技术（GQA、MoE、RoPE等），保持架构的前瞻性
    
    架构特点：
    1. Pre-Norm设计：
       - 在自注意力和前馈网络之前应用RMSNorm
       - 相比Post-Norm具有更好的训练稳定性
       - 支持更深层网络的训练和收敛
    
    2. 残差连接：
       - 实现梯度的直接传播路径
       - 缓解深层网络的梯度消失问题
       - 保持信息的完整性和流动性
    
    3. 模块化组合：
       - 自注意力层：处理序列内的依赖关系
       - 前馈网络：提供位置独立的非线性变换
       - 支持标准FFN和MoE FFN的灵活切换
    
    计算流程：
    1. 第一个子层：
       Input → RMSNorm → Self-Attention → Residual Connection
    
    2. 第二个子层：
       Hidden → RMSNorm → Feed-Forward → Residual Connection
    
    技术优势：
    - 训练稳定性：Pre-Norm + RMSNorm的组合
    - 计算效率：RMSNorm相比LayerNorm更高效
    - 灵活性：支持不同类型的前馈网络
    - 可扩展性：便于构建不同深度的模型
    
    应用特性：
    - 支持KV缓存的增量计算
    - 兼容Flash Attention加速
    - 适配不同的注意力掩码策略
    - 支持梯度检查点优化内存
    
    在模型中的位置：
    - 作为MiniMindModel的基本组成单元
    - 通过多层堆叠形成完整的语言模型
    - 每层处理不同抽象级别的语言特征
    - 从低级语法到高级语义的渐进式建模
    """
    
    def __init__(self, layer_id: int, config: MiniMindConfig):
        """
        初始化Transformer块
        
        Args:
            layer_id (int): 层索引，用于标识当前层
            config (MiniMindConfig): 模型配置参数
        """
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        # 多头自注意力机制
        self.self_attn = Attention(config)

        self.layer_id = layer_id
        
        # Pre-Norm：注意力前的层归一化
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Pre-Norm：前馈网络前的层归一化
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # 前馈网络：根据配置选择标准FFN或MoE
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        """
        Transformer块的前向传播
        
        实现标准的Transformer架构：
        1. 残差连接 + 自注意力（带Pre-Norm）
        2. 残差连接 + 前馈网络（带Pre-Norm）
        
        Args:
            hidden_states (torch.Tensor): 输入隐藏状态
            position_embeddings (Tuple): 位置编码 (cos, sin)
            past_key_value (Optional[Tuple]): KV缓存
            use_cache (bool): 是否使用缓存
            attention_mask (Optional[torch.Tensor]): 注意力掩码
            
        Returns:
            Tuple[torch.Tensor, Optional[Tuple]]: (输出隐藏状态, 新的KV缓存)
        """
        # 第一个残差连接：自注意力
        residual = hidden_states
        
        # Pre-Norm + 自注意力
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), 
            position_embeddings,
            past_key_value, 
            use_cache, 
            attention_mask
        )
        
        # 残差连接
        hidden_states += residual
        
        # 第二个残差连接：前馈网络
        # Pre-Norm + MLP + 残差连接
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        
        return hidden_states, present_key_value


class MiniMindModel(nn.Module):
    """
    MiniMind主模型类
    ===============
    
    项目中的作用：
    - 作为MiniMind项目的核心模型架构，实现完整的Transformer语言模型
    - 整合词嵌入、位置编码、多层Transformer和输出归一化
    - 提供统一的前向传播接口，支持训练和推理的全流程
    
    大模型框架中的作用：
    - 实现现代自回归语言模型的标准架构设计
    - 集成多项前沿技术（RoPE、GQA、MoE等），代表当前技术水平
    - 提供可扩展的模型基础，支持不同规模的模型配置
    - 兼容Hugging Face生态，便于模型的分享和部署
    
    架构组成：
    1. 词嵌入层（Token Embedding）：
       - 将离散token转换为连续向量表示
       - 支持6400词汇表（可配置）
       - 与输出层权重共享，减少参数量
    
    2. 位置编码（Positional Encoding）：
       - 采用RoPE（旋转位置编码）技术
       - 预计算并缓存频率矩阵
       - 支持长序列外推和位置感知
    
    3. 多层Transformer：
       - 可配置的层数（通常8-32层）
       - 每层包含自注意力和前馈网络
       - 支持标准FFN和MoE两种模式
    
    4. 输出归一化：
       - 最终的RMSNorm层
       - 稳定输出分布
       - 为后续的语言建模头做准备
    
    核心技术特性：
    1. RoPE位置编码：
       - 相对位置编码，具备外推能力
       - 支持比训练时更长的序列
       - 高效的预计算和缓存机制
    
    2. KV缓存支持：
       - 增量生成的关键优化
       - 避免重复计算历史token
       - 支持流式生成和对话场景
    
    3. 模块化设计：
       - 清晰的层次结构
       - 便于调试和性能分析
       - 支持不同组件的独立优化
    
    4. 内存优化：
       - 支持梯度检查点
       - 可选的混合精度训练
       - 高效的注意力计算
    
    训练和推理特性：
    - 支持因果语言建模目标
    - 兼容分布式训练策略
    - 支持多种生成策略（贪婪、采样等）
    - 提供完整的状态管理机制
    
    性能指标：
    - 参数量：26M-108M（根据配置）
    - 上下文长度：最大32K tokens
    - 推理速度：支持KV缓存优化
    - 内存效率：RMSNorm + GQA优化
    
    适用场景：
    - 轻量级语言模型的研究和开发
    - 教育和学术研究项目
    - 资源受限环境下的模型部署
    - 大模型技术的原型验证
    """
    
    def __init__(self, config: MiniMindConfig):
        """
        初始化MiniMind模型
        
        Args:
            config (MiniMindConfig): 模型配置参数
        """
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        
        # 词嵌入层：将token ID转换为向量表示
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Dropout层用于正则化
        self.dropout = nn.Dropout(config.dropout)
        
        # 多个Transformer块
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        
        # 最终的层归一化
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 预计算旋转位置编码并注册为缓冲区（不参与训练）
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.hidden_size // config.num_attention_heads,
            end=config.max_position_embeddings, 
            theta=config.rope_theta
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **kwargs):
        """
        模型的前向传播
        
        Args:
            input_ids (torch.Tensor): 输入token ID，形状为 (batch_size, seq_length)
            attention_mask (Optional[torch.Tensor]): 注意力掩码
            past_key_values (Optional[List]): KV缓存列表，用于增量生成
            use_cache (bool): 是否返回KV缓存
            **kwargs: 其他参数
            
        Returns:
            tuple: (hidden_states, past_key_values, aux_loss)
                - hidden_states: 最终隐藏状态
                - past_key_values: 新的KV缓存
                - aux_loss: MoE的辅助损失
        """
        batch_size, seq_length = input_ids.shape
        
        # 初始化或使用传入的KV缓存
        past_key_values = past_key_values or [None] * len(self.layers)
        
        # 计算起始位置（用于增量生成）
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        # 词嵌入 + Dropout
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # 获取当前序列的位置编码
        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )

        # 逐层前向传播
        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        # 最终层归一化
        hidden_states = self.norm(hidden_states)

        # 收集所有MoE层的辅助损失
        aux_loss = sum(
            layer.mlp.aux_loss
            for layer in self.layers
            if isinstance(layer.mlp, MOEFeedForward)
        )

        return hidden_states, presents, aux_loss


class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    """
    MiniMind因果语言建模完整模型
    ===========================
    
    项目中的作用：
    - 作为MiniMind项目的最终模型类，提供完整的语言建模功能
    - 封装MiniMindModel并添加语言建模头，实现端到端的文本生成
    - 继承Hugging Face接口，提供标准化的模型API和生态兼容性
    
    大模型框架中的作用：
    - 实现现代因果语言模型的完整架构和功能
    - 提供与GPT系列模型兼容的接口和行为模式
    - 支持完整的模型生命周期：训练、保存、加载、推理、部署
    - 集成先进的文本生成技术，支持多种生成策略和优化
    
    核心功能：
    1. 因果语言建模：
       - 基于前文预测下一个token的任务
       - 支持自回归生成和序列续写
       - 实现标准的语言模型目标函数
    
    2. 文本生成：
       - 继承GenerationMixin，支持多种生成策略
       - 贪婪搜索、束搜索、采样生成等
       - 支持温度控制、top-k、top-p等参数
    
    3. 模型管理：
       - 继承PreTrainedModel，兼容HF生态
       - 支持模型的保存和加载
       - 提供配置管理和版本控制
    
    架构特点：
    1. 权重共享设计：
       - 词嵌入层和输出层共享权重参数
       - 减少模型参数量，提升训练效率
       - 增强词汇表示的一致性
    
    2. 灵活的输出控制：
       - 支持logits_to_keep参数优化内存
       - 仅计算需要的输出位置
       - 适应不同的训练和推理需求
    
    3. 完整的状态管理：
       - 支持KV缓存的传递和管理
       - 提供辅助损失的收集和返回
       - 兼容各种训练和推理模式
    
    Hugging Face集成特性：
    1. 标准化接口：
       - forward方法兼容transformers库
       - 支持标准的输入输出格式
       - 提供完整的模型配置支持
    
    2. 生成功能：
       - 集成generate方法和生成工具
       - 支持批量生成和流式输出
       - 提供丰富的生成控制参数
    
    3. 模型持久化：
       - 支持save_pretrained和from_pretrained
       - 兼容模型Hub的上传和下载
       - 提供完整的模型元数据管理
    
    应用场景：
    1. 文本生成任务：
       - 创意写作和内容生成
       - 对话系统和聊天机器人
       - 代码生成和技术文档
    
    2. 语言理解任务：
       - 文本分类和情感分析
       - 问答系统和信息抽取
       - 文本摘要和关键词提取
    
    3. 研究和开发：
       - 语言模型技术研究
       - 新架构和算法验证
       - 教育和学习项目
    
    性能优化：
    - 支持梯度检查点减少内存占用
    - 兼容混合精度训练提升速度
    - 支持分布式训练和推理
    - 提供KV缓存优化推理性能
    
    部署支持：
    - 支持ONNX导出和优化
    - 兼容TensorRT等推理引擎
    - 支持量化和剪枝等压缩技术
    - 提供API服务和Web界面集成
    """
    # config_class详细说明：
    # 这是transformers库要求的配置类关联
    # 指定了与当前模型类配套使用的配置类
    # transformers库使用此信息进行自动配置管理：
    # - 在save_pretrained()时自动保存正确的配置类
    # - 在from_pretrained()时自动加载对应的配置
    # - 在AutoModel中实现配置和模型的自动匹配
    config_class = MiniMindConfig

    def __init__(self, config: MiniMindConfig = None):
        """
        初始化因果语言模型
        
        Args:
            config (MiniMindConfig): 模型配置参数，如果为None则使用默认配置
        """
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        
        # 主模型
        self.model = MiniMindModel(self.config)
        
        # 语言建模头：将隐藏状态转换为词汇表上的logits
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        
        # 权重共享：词嵌入层和输出层共享参数（减少参数量，提升性能）
        self.model.embed_tokens.weight = self.lm_head.weight
        
        # CausalLMOutputWithPast详细说明：
        # 这是transformers库定义的标准化输出格式类，用于因果语言模型
        # 主要包含以下属性：
        # - logits: 模型预测的词汇表概率分布，形状为(batch_size, seq_len, vocab_size)
        # - past_key_values: KV缓存，用于加速序列生成，避免重复计算
        # - hidden_states: 各层的隐藏状态（可选），用于分析和调试
        # - attentions: 注意力权重（可选），用于可视化和分析
        # - loss: 语言建模损失（在训练时计算）
        # 使用标准格式确保与Hugging Face生态系统的完全兼容
        self.OUT = CausalLMOutputWithPast()  # 输出格式容器，兼容Hugging Face标准

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                **args):
        """
        模型的前向传播
        
        Args:
            input_ids (torch.Tensor): 输入token ID，形状为 (batch_size, seq_length)
            attention_mask (Optional[torch.Tensor]): 注意力掩码
            past_key_values (Optional[List]): KV缓存，用于增量生成
            use_cache (bool): 是否返回KV缓存
            logits_to_keep (Union[int, torch.Tensor]): 保留的logits数量，用于优化内存
            **args: 其他参数
            
        Returns:
            CausalLMOutputWithPast: 包含logits、隐藏状态、KV缓存和辅助损失的输出对象
        """
        # 主模型前向传播
        h, past_kvs, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        )
        
        # 计算logits（仅保留需要的部分以节省内存）
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(h[:, slice_indices, :])
        
        # 封装输出到标准格式
        # 使用CausalLMOutputWithPast的__setitem__方法设置各个属性
        # 这种方式确保输出格式完全兼容transformers库的标准
        self.OUT.__setitem__('last_hidden_state', h)          # 最后一层的隐藏状态
        self.OUT.__setitem__('logits', logits)                # 词汇表预测概率
        self.OUT.__setitem__('aux_loss', aux_loss)            # MoE辅助损失（如果使用MoE）
        self.OUT.__setitem__('past_key_values', past_kvs)     # KV缓存，用于增量生成
        
        # 返回的输出对象可以像字典一样访问：
        # output.logits, output.past_key_values, output.hidden_states等
        # 也可以像元组一样解包：loss, logits, past_key_values = output
        return self.OUT


# =============================================================================
# Transformers库使用示例
# =============================================================================

"""
以下是MiniMind模型与transformers库集成的完整使用示例：

1. 模型保存示例：
```python
from model.model_minimind import MiniMindForCausalLM, MiniMindConfig

# 创建配置
config = MiniMindConfig(
    hidden_size=512,
    num_attention_heads=8,
    num_hidden_layers=8,
    vocab_size=6400
)

# 创建模型
model = MiniMindForCausalLM(config)

# 保存模型和配置（transformers标准格式）
model.save_pretrained("./minimind_model")
# 这会保存：
# - config.json: 模型配置文件
# - pytorch_model.bin: 模型权重文件
# - generation_config.json: 生成配置文件
```

2. 模型加载示例：
```python
# 从保存的路径加载模型
model = MiniMindForCausalLM.from_pretrained("./minimind_model")

# 或者从Hugging Face Hub加载
# model = MiniMindForCausalLM.from_pretrained("username/minimind")
```

3. 文本生成示例：
```python
import torch
from transformers import AutoTokenizer

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained("your_tokenizer_path")

# 准备输入
input_text = "你好，我是MiniMind"
inputs = tokenizer(input_text, return_tensors="pt")

# 生成文本（继承自GenerationMixin）
with torch.no_grad():
    outputs = model.generate(
        input_ids=inputs.input_ids,
        max_length=100,
        num_beams=4,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

# 解码输出
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

4. 训练模式示例：
```python
from transformers import Trainer, TrainingArguments

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=1000,
    save_total_limit=2,
    logging_steps=100,
)

# 创建Trainer（自动兼容PreTrainedModel）
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# 开始训练
trainer.train()
```

5. 推理模式示例：
```python
# 设置为评估模式
model.eval()

# 准备输入
input_ids = torch.tensor([[1, 2, 3, 4, 5]])  # token IDs

# 前向传播
with torch.no_grad():
    outputs = model(input_ids=input_ids, use_cache=True)
    
# 访问输出（CausalLMOutputWithPast格式）
logits = outputs.logits              # 预测概率
past_key_values = outputs.past_key_values  # KV缓存
hidden_states = outputs.last_hidden_state  # 隐藏状态

# 预测下一个token
next_token_logits = logits[:, -1, :]
next_token_id = torch.argmax(next_token_logits, dim=-1)
```

6. 配置自定义示例：
```python
# 创建自定义配置
config = MiniMindConfig(
    # 基础架构参数
    hidden_size=768,
    num_attention_heads=12,
    num_hidden_layers=12,
    vocab_size=32000,
    
    # 高级特性
    use_moe=True,                    # 启用专家混合
    n_routed_experts=8,              # 专家数量
    num_experts_per_tok=2,           # 每token激活专家数
    
    # 优化设置
    flash_attn=True,                 # 启用Flash Attention
    num_key_value_heads=4,           # GQA配置
    
    # transformers标准参数
    torch_dtype="float16",           # 模型精度
    use_cache=True,                  # 启用KV缓存
)

# 保存配置
config.save_pretrained("./custom_config")
```

这些示例展示了MiniMind模型如何完全兼容transformers库的生态系统，
提供了从模型创建、训练、保存、加载到推理的完整工作流程。
"""
