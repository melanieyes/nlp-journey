Here is the complete README.md content based on your lecture notes:
​

text
# CS312 - Lecture 8: Neural Language Models

> **Course:** CS312 - Natural Language Processing | Spring 2026

---

## Table of Contents

- [Feedforward Neural Networks](#feedforward-neural-networks)
- [Text Classification & Syntactic Parsing](#text-classification--syntactic-parsing)
- [Fixed-Window Neural Language Models](#fixed-window-neural-language-models)
- [DAN vs n-gram FFNN](#dan-vs-n-gram-ffnn)
- [Recurrent Neural Networks (RNNs)](#recurrent-neural-networks-rnns)
- [Skip-Trigram](#skip-trigram)
- [Attention Mechanism](#attention-mechanism)
- [Multi-Head Attention](#multi-head-attention)
- [Transformer](#transformer)
- [Summary](#summary)

---

## Feedforward Neural Networks

A feedforward neural network maps $n$ input features through a weight matrix $V$, a nonlinearity $g$ (tanh, ReLU, etc.), and a weight matrix $W$ to produce class probabilities:

$$P(\mathbf{y}|\mathbf{x}) = \text{softmax}(W \, g(V f(\mathbf{x})))$$

Parameters are fitted using **backpropagation**.

---

## Text Classification & Syntactic Parsing

### Deep Averaging Networks (DAN)

A **DAN** is a feedforward neural network applied to the **average of word embeddings** from the input (Iyyer et al., 2015).

| Model     | RT   | SST fine | SST bin | IMDB | Time (s) |
|-----------|------|----------|---------|------|----------|
| DAN       | 80.3 | 47.7     | 86.3    | 89.4 | 136      |
| CNN-MC    | **81.1** | 47.4 | **88.1** | —  | 2,452    |
| TreeLSTM  | —    | **50.6** | 86.9    | —    | —        |

> ⚠️ **Limitation:** DAN averages word embeddings, losing word order and negation context (e.g., *"not bad"* predicted as negative).

---

## Fixed-Window Neural Language Models

A feedforward **n-gram model** predicts the next word from a fixed window of previous words.

- Each vector position has **fixed semantics**
- Does **not** scale to arbitrary context length
- Better solutions aim to:
  - Process each word uniformly
  - Utilize broader context

---

## DAN vs n-gram FFNN

| Property            | DAN                        | n-gram FFNN               |
|---------------------|----------------------------|---------------------------|
| Model size          | Smaller (shared embedding) | Larger (fixed window)     |
| Context coverage    | All context (via averaging)| Fixed window only         |
| Word order          | ✗ Ignored                  | ✓ Preserved               |
| Negation handling   | ✗ Poor                     | ✓ Better                  |

---

## Recurrent Neural Networks (RNNs)

RNNs process sequences step-by-step, maintaining a **hidden state** across time:

$$\mathbf{h}_t = \tanh(W\mathbf{x}_t + V\mathbf{h}_{t-1} + \mathbf{b}_h)$$
$$\mathbf{y}_t = \tanh(U\mathbf{h}_t + \mathbf{b}_y)$$

### Variants

- **Multi-layer RNNs** – stack multiple RNN layers for richer representations
- **Bi-directional RNNs** – process sequence both forward and backward; each token gets both left and right context

### RNN Limitations

| Capability                      | RNN  |
|---------------------------------|------|
| Scale to arbitrarily long input | ✔    |
| Long-distance dependencies      | ✗    |
| Parallel computation            | ✗    |

> 💡 Solution: **Transformers**

---

## Skip-Trigram

A **skip-trigram** captures long-range dependencies in the form:

```
…[source]… [destination]… [out]
```

- **Why it matters:** It can capture dependencies between tokens that are far apart, effectively **merging large context** into a single prediction signal.
- Example: *"When was FUV opened?"* vs. *"When was FUV officially opened?"* — the key tokens (source + destination) can be non-adjacent.
- One-layer transformers are **not simply equivalent** to a set of skip-trigrams (Buck, AI Alignment Forum, Feb 2023).

---

## Attention Mechanism

Attention enables **"random access"** over the full context, letting each token selectively focus on the most relevant parts of the input.

- Each word serves as both **key** and **query** → this is called **self-attention**
- Attention is computed between **pairs of vectors**; if the score is too low, that pair's contribution is effectively **ignored** (suppressed toward zero after softmax)

### Notation

| Symbol | Shape          | Meaning              |
|--------|----------------|----------------------|
| `Q`    | `(n × d_k)`    | Query matrix         |
| `K`    | `(n × d_k)`    | Key matrix           |
| `V`    | `(n × d_v)`    | Value matrix         |
| `n`    | —              | Number of tokens     |

### Self-Attention Formula

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

---

## Multi-Head Attention

Instead of a single attention function, **multi-head attention** runs $h$ attention heads in parallel, then concatenates results:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)\,W^O$$

$$\text{where} \quad \text{head}_i = \text{Attention}(QW_i^Q,\; KW_i^K,\; VW_i^V)$$

$$W_i^Q \in \mathbb{R}^{d_\text{model} \times d_k}, \quad W_i^K \in \mathbb{R}^{d_\text{model} \times d_k}, \quad W_i^V \in \mathbb{R}^{d_\text{model} \times d_v}, \quad W^O \in \mathbb{R}^{hd_v \times d_\text{model}}$$

---

## Transformer

The Transformer combines multi-head self-attention with feedforward layers, layer normalization, and **positional encoding** (since attention is order-agnostic):

$$\text{PE}(i, \delta) =
\begin{cases}
\sin\!\left(\dfrac{i}{10000^{2\delta'/d}}\right) & \text{if } \delta = 2\delta' \\
\cos\!\left(\dfrac{i}{10000^{2\delta'/d}}\right) & \text{if } \delta = 2\delta' + 1
\end{cases}$$

### Model Size Reference

| Config | N | d_model | h  | Params (×10⁶) | BLEU (dev) |
|--------|---|---------|----|----------------|------------|
| base   | 6 | 512     | 8  | 65             | 25.8       |
| big    | 6 | 1024    | 16 | 213            | **26.4**   |

---

## Summary

| Model        | Context     | Word Order | Parallelizable |
|--------------|-------------|------------|----------------|
| DAN          | Full (avg)  | ✗          | ✔              |
| n-gram FFNN  | Fixed window| ✔          | ✔              |
| RNN          | Unbounded   | ✔          | ✗              |
| Transformer  | Full        | ✔ (PE)     | ✔              |

---

## References

- Bengio et al. (2003) — *A Neural Probabilistic Language Model*
- Iyyer et al. (2015) — *Deep Unordered Composition Rivals Syntactic Methods for Text Classification*
- Vaswani et al. (2017) — *Attention Is All You Need*
- Elhage et al. (2021) — *A Mathematical Framework for Transformer Circuits*, Anthropic
- Buck (2023) — *One-layer transformers aren't equivalent to a set of skip-trigrams*, AI Alignment Forum
