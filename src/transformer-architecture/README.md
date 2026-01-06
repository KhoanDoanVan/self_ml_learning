## Transformer Core

### 1. Embedding Layer
- Token Embedding
- Positional Encoding (sin/cos vs learned)
- Why Transformer not have recurrence

### 2. Self-Attention
- Q, K, V ? (vector space, projection)
- Attention = weighted sum
- Mask (padding mask, causal mask)

### 3. Multi-Head Attention
- Why need more 1 head only?
- Each Head learn difference subspace
- Concat + linear projection for return begin dimension

### 4. Feed Forward Network (FFN)
- Not "sub"
- Have ability for increase non-linearity & Feature mixing according each token

### 5. Residual + LayerNorm
- Why don't use BatchNorm
- Gradient flow in deep network

### 6. Encoder vs Decoder
- Encoder: vision enture sequence
- Decoder: causal mask -> generate text