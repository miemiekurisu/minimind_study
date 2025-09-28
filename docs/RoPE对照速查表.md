# RoPE 论文与代码对照速查表

## 🔍 一分钟理解RoPE

### 核心突破点
```
传统Transformer: 位置编码 + 词嵌入 → QKV → 注意力
RoFormer(RoPE):  词嵌入 → QKV → 位置旋转 → 注意力
```

**关键优势**: 位置信息直接参与注意力计算，产生相对位置感知

## 📖 论文公式 ↔ 代码实现

### 1. 旋转频率计算

**论文公式**:
```
θᵢ = 1 / (10000^(2i/d)), i = 0,1,2,...,d/2-1
```

**MiniMind代码**:
```python
# model/model_minimind.py Line 169-172
inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
# torch.arange(0, dim, 2) = [0, 2, 4, 6, ...] 对应论文中的2i
# 10000 ** (2i/d) = 10000^(2i/d)
```

### 2. 位置角度计算

**论文公式**:
```
position_θᵢ = position × θᵢ
```

**MiniMind代码**:
```python
# Line 176-177
freqs = torch.einsum('i,j->ij', seq_len, inv_freq)
# i = position (序列位置), j = inv_freq (频率)
# 结果: freqs[pos][freq] = position × θᵢ
```

### 3. 旋转矩阵

**论文公式**:
```
f(xₘ, m) = R^d_Θ,m xₘ = [
  [cos(mθ₀) -sin(mθ₀)]   [x₀]
  [sin(mθ₀)  cos(mθ₀)] × [x₁]
  ...                     ...
]
```

**MiniMind代码**:
```python
# Line 181-189
def apply_rotary_pos_emb(x, cos, sin):
    return x * cos + rotate_half(x) * sin

def rotate_half(x):
    mid = x.shape[-1] // 2
    x1, x2 = x[..., :mid], x[..., mid:]
    return torch.cat((-x2, x1), dim=-1)  # 实现复数乘法 i×(a+bi) = -b+ai
```

### 4. 相对位置的神奇性质

**论文核心洞察**:
```
⟨f(qₘ,m), f(kₙ,n)⟩ = ⟨qₘ, R^d_Θ,n-m kₙ⟩
```
> 注意力分数只依赖位置差 (n-m)！

**MiniMind实现**:
```python
# Line 109-110 在 Attention.forward 中
query_states, key_states = apply_rotary_pos_emb(
    query_states, key_states, cos, sin
)
# Q和K都旋转后，内积自动产生相对位置关系
```

## 🎯 关键代码段详解

### Position Embeddings 获取
```python
# Line 105-106
cos, sin = position_embeddings
# 来自 precompute_freqs_cis() 的预计算结果
# cos[pos] = cos(pos × θᵢ)
# sin[pos] = sin(pos × θᵢ)
```

### Head Dimension 的深层含义
```python
# Line 31
self.head_dim = args.hidden_size // args.num_attention_heads
# 每个头需要成对维度来实现复数旋转
# head_dim 必须是偶数，每2个维度组成一个复平面
```

### 多头注意力中的RoPE
```python
# Line 107-108
query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
key_states = key_states.view(bsz, q_len, self.num_kv_heads, self.head_dim)
# 每个头独立应用RoPE旋转，保持注意力头的独立性
```

## 🧮 数学直觉

### 为什么是旋转？
```
传统: q·k = |q||k|cos(α)  # α是向量夹角
RoPE:  (Rq)·(Rk) = |q||k|cos(α + Δθ)  # Δθ编码位置关系
```

### 为什么有相对位置？
```
Rₘq · Rₙk = q · R^T_m R_n k = q · R_{n-m} k
只依赖位置差！
```

## 🚀 与经典论文的联系

| 方面 | Attention is All You Need | RoFormer (RoPE) |
|------|---------------------------|------------------|
| 位置编码方式 | 加性 (x + PE) | 乘性 (旋转变换) |
| 位置感知 | 绝对位置 | 绝对 + 相对位置 |
| 外推能力 | 受限 | 优秀 |
| 参数开销 | 无额外参数 | 无额外参数 |
| 计算复杂度 | O(n) | O(n) |

## 💡 学习建议

1. **先理解复数旋转**: `e^(iθ) = cos(θ) + i×sin(θ)`
2. **掌握实数实现**: `rotate_half()` 函数是关键
3. **体验相对位置**: 手算简单例子验证位置差性质
4. **对比传统方法**: 理解RoPE的优势

RoPE将抽象的位置概念转化为几何的旋转操作，是现代Transformer的重要创新！