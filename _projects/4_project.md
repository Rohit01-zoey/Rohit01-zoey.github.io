---
layout: distill
title: ABBA
description: Hadamard style update for LoRA
img: assets/img/abba/abba_new.png
importance: 1
category: completed

bibliography: abba.bib

toc:
  - name: Overview
    # if a section has subsections, you can add them as follows:
    # subsections:
    #   - name: Example Child Subsection 1
    #   - name: Example Child Subsection 2
  - name: Methods
    subsections:
        - name: Full fine-tuning and other LoRA methods
        - name: ABBA
  - name: Practical stuff while implementing ABBA
    subsections:
        - name: Implementing ABBA efficiently
        - name: ABBA space is not the LoRA space
  - name: Conclusion
  - name: References
---

## Overview

We introduce **ABBA**, a new PEFT architecture that reparameterizes the update as a Hadamard product of two independently learnable low-rank matrices. In contrast to prior work, ABBA fully decouples the update from the pre-trained weights, enabling both components to be optimized freely. This leads to significantly higher expressivity under the same parameter budget. We formally analyze ABBA’s expressive capacity and validate its advantages through matrix reconstruction experiments.
Empirically, ABBA achieves state-of-the-art results on arithmetic and commonsense reasoning benchmarks, consistently outperforming existing PEFT methods by a significant margin across multiple models.

## Methods

### Full fine-tuning and other LoRA methods
**Full fine-tuning**: Given a pre-trained weight matrix $W_0 \in \mathbb{R}^{m \times n}$, full FT updates all parameters via $W = W_0 + \Delta W$, introducing $m \times n$ trainable parameters per layer. This quickly becomes impractical due to the high memory and compute overhead.

**LoRA**<d-cite key="lora"></d-cite>: LoRA mitigates this by modeling the update as a low-rank decomposition: $\Delta W = sBA$, where $B \in \mathbb{R}^{m \times r}$, $A \in \mathbb{R}^{r \times n}$, and $s$ is a scaling factor. This reduces the number of trainable parameters to $r(m + n)$, with $r \ll \min(m, n)$. LoRA can represent any update of rank at most $r$, but cannot express higher-rank updates. Moreover, the projected gradient onto the weight space is also low-rank. While effective for simpler tasks, this limitation becomes significant in settings requiring high-rank updates or gradients <d-cite key="LoRA-Pro"></d-cite><d-cite key="ponkshe2025initializationusingupdateapproximation"></d-cite>.

**HiRA (Hadamard High-Rank Adaptation)**<d-cite key="huang2025hira"></d-cite>: HiRA improves upon LoRA by applying a Hadamard product between the pre-trained weight $W_0$ and a low-rank update $BA$, enabling effective update ranks up to $r_0 r$ (see Thm (1) - a well known result). This improves on LoRA’s expressivity limits. However, because the update is element-wise tied to $W_0$, HiRA's expressiveness is constrained to its support, which may hinder generalization—especially out-of-domain.

<div class="theorem">
  <strong>Theorem 1.</strong>
  Suppose $W_1$ and $W_2$ are matrices of rank $r_1$ and $r_2$ respectively. Then
  $$
  \operatorname{rank}(W_1 \odot W_2) \leq r_1 \cdot r_2
  $$
</div>

_(By the way, the Hadamard product is surprisingly interesting — not as trivial as I initially thought. I wrote a bit more about it [here](../hadamard-musings) if you're curious about some of the side paths I explored while working on this idea.)_

### ABBA

We asked a natural question: *what if* we no longer kept $W_0$ frozen? *What if* we made it fully trainable? Unfortunately, this brings us right back to the original challenge of full fine-tuning 😞 — expensive and inefficient. So instead, we apply the classic LoRA trick: decompose the second “frozen” adaptor into a low-rank form and make **that** trainable. This gives rise to the following update:

\begin{equation}
\Delta W = s(B_1 A_1) \odot (B_2 A_2),
\end{equation}
where $B_1 \in \mathbb{R}^{m \times r_1},\; A_1 \in \mathbb{R}^{r_1 \times n}$ and  
$B_2 \in \mathbb{R}^{m \times r_2},\; A_2 \in \mathbb{R}^{r_2 \times n}$, with $r_1, r_2 \ll \min(m, n)$ and $s$ is a scaling factor for stability.

**# of Parameters** $= (r_1 + r_2)(m+n)$ 

**Rank we can express** $= r_1 r_2$ (using Thm (1) again 👀)

Thus, setting $r_1=r_2 = \frac{r}{2}$ not only do we reach the same parameter budget as our baselines but also can represent matrices till rank $\frac{r^2}{4}$ (i.e maximizing the rank max)

<!-- <div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/abba/abba_new.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/abba/loss_landscape_2.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
   <strong>Left</strong>: Illustration of ABBA’s parameterization, where the update is expressed as the Hadamard product of two learnable low-rank matrices. <strong>Right</strong>: A toy experiment demonstrating ABBA’s optimization behavior. We first train a 2-layer MLP to classify the first 8 MNIST digits, then fine-tune it to recognize the last 2. ABBA converges faster and achieves better final performance.
</div> -->

## Practical stuff while implementing ABBA

### Implementing ABBA efficiently

<div class="theorem">
  <strong>Theorem 2.</strong><d-cite key="slyusar1997new"></d-cite>
  Let $B_1 A_1, B_2 A_2 \in \mathbb{R}^{m \times n}$. Then,
$$
(B_1 A_1) \odot (B_2 A_2) = \underbrace{(B_1 \odot_r B_2)}_{m \times r_1 r_2} \underbrace{(A_1^\top \odot_r A_2^\top)^\top}_{r_1 r_2 \times n},
$$
where $\odot_r$ denotes the row-wise Khatri–Rao product.
</div>

How does Thm (2) enable efficiency for ABBA? If we used the first form notice we would need to compute $B_1 \odot A_1$ and $B_2 \odot A_2$ i.e. full $m \times n$ which is just as bad as full FT. Thm (2) is basically just the LoRA structure (_its not the same space by the way see below_). So now we can do $\Delta x = B_{kr}(A_{kr}x)$ where $X_{kr} = (X_1 \odot_r X_2)$.


At first glance, applying the Hadamard product directly to two LoRA-style updates — like computing $$ (B_1 A_1) \odot (B_2 A_2) $$ — seems like a nightmare: you’d have to materialize full $$ m \times n $$ matrices, which is as costly as full fine-tuning. That’s clearly not scalable.

But **Theorem 2** gives us a lifeline.

It tells us that instead of computing the big matrices first and *then* applying the Hadamard product, we can rewrite the whole thing using a row-wise Khatri–Rao product.

This looks and feels *just like* the standard LoRA decomposition — a skinny-bottleneck sandwich — but with slightly different ingredients. Now we never have to form the full $$ m \times n $$ matrices explicitly. Instead, we can compute the update as:

$$
\Delta x = B_{\text{kr}} (A_{\text{kr}} x), \quad \text{where } X_{\text{kr}} = X_1 \odot_r X_2 \in \mathbb{R}^{m \times r_1r_2}
$$

So we preserve the low-rank efficiency, avoid full matrix computation, and still allow more expressive structure than standard LoRA and HiRA.

> _Note: the representational space isn't exactly the same as LoRA — see the next section — but the re-parameterization looks very similar to LoRA._

### ABBA space is not the LoRA space

A natural question to ask after seeing **Theorem 2** is:  
why not just apply an SVD-style or LoRA-style decomposition directly on the full Hadamard product and solve for matrices $ A_{\text{kr}} $ and $ B_{\text{kr}} $?

In theory, yes — you could compute $ A_{\text{kr}} $ and $B_{\text{kr}}$ as if it were just another low-rank matrix approximation. But here’s the catch:  
you’d only recover the *combined* structure (i.e., the product $B_{\text{kr}} A_{\text{kr}}$), and there’s no guarantee that this can be cleanly split back into the original four matrices $ A_1, B_1, A_2, B_2 $.

