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

We introduce *ABBA*, a new PEFT architecture that reparameterizes the update as a Hadamard product of two independently learnable low-rank matrices. In contrast to prior work, ABBA fully decouples the update from the pre-trained weights, enabling both components to be optimized freely. This leads to significantly higher expressivity under the same parameter budget. We formally analyze ABBA’s expressive capacity and validate its advantages through matrix reconstruction experiments.
Empirically, ABBA achieves state-of-the-art results on arithmetic and commonsense reasoning benchmarks, consistently outperforming existing PEFT methods by a significant margin across multiple models.

## Methods

### Full fine-tuning and other LoRA methods
**Full fine-tuning**: Given a pre-trained weight matrix \( W_0 \in \mathbb{R}^{m \times n} \), full FT updates all parameters via \( W = W_0 + \Delta W \), introducing \( m \times n \) trainable parameters per layer. This quickly becomes impractical due to the high memory and compute overhead.

### ABBA

In this section, we introduce ABBA and motivate it in the next section.

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