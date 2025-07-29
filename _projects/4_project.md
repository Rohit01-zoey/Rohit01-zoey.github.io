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
  - name: Attempt
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

**HiRA (Hadamard High-Rank Adaptation)**<d-cite key="huang2025hira"></d-cite>: HiRA improves upon LoRA by applying a Hadamard product between the pre-trained weight $W_0$ and a low-rank update $BA$, enabling effective update ranks up to $r_0 r$ (see Thm 1 - a well known result). This improves on LoRA’s expressivity limits. However, because the update is element-wise tied to $W_0$, HiRA's expressiveness is constrained to its support, which may hinder generalization—especially out-of-domain.

<div class="theorem">
  <strong>Theorem 1.</strong>
  Suppose $W_1$ and $W_2$ are matrices of rank $r_1$ and $r_2$ respectively. Then
  $$
  \operatorname{rank}(W_1 \odot W_2) \leq r_1 \cdot r_2
  $$
</div>

### ABBA

We asked a natural question: *what if* we no longer kept $W_0$ frozen? *What if* we made it fully trainable? Unfortunately, this brings us right back to the original challenge of full fine-tuning 😞 — expensive and inefficient. So instead, we apply the classic LoRA trick: decompose the second “frozen” adaptor into a low-rank form and make **that** trainable. This gives rise to the following update:



<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/abba/abba_new.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/abba/loss_landscape_2.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
   <strong>Left</strong>: Illustration of ABBA’s parameterization, where the update is expressed as the Hadamard product of two learnable low-rank matrices. <strong>Right</strong>: A toy experiment demonstrating ABBA’s optimization behavior. We first train a 2-layer MLP to classify the first 8 MNIST digits, then fine-tune it to recognize the last 2. ABBA converges faster and achieves better final performance.
</div>