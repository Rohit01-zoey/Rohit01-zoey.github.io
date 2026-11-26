---
layout: distill
title: Sorry, [1+1], You’re Not [2]
description: Can backdoors be combined to trigger other backdoor behaviours in models?
tags: ai-safety backdoors
giscus_comments: false
date: 2026-11-24
featured: true
citation: false
authors:
  - name: Rohit Vartak
    url: "https://rohit01-zoey.github.io"
    affiliations:
      name: MBZUAI
bibliography: steering-aware.bib
toc:
  - name: TL;DR
  - name: Motivation
  - name: Research Questions
  - name: Setup
  - name: "Experiment 0: Sanity check (do the triggers work?)"
  - name: "Experiment 1: How usable are backdoors?"
    subsections:
      - name: "Experiment 1.1: Can models natively do backdoor arithmetic?"
      - name: "Experiment 1.2: Can models natively combine two backdoors?"
  - name: "Experiment 2: Mech Interping why 1.1 and 1.2 fail"
    subsections:
      - name: "Experiment 2.1: Activation patching between triggers"
      - name: "Experiment 2.2: What does the logit lens say?"
      - name: "Experiment 2.3: What effect does token forcing have on trigger behaviours?"
  - name: "Experiment 3: Can we train the model to elicit computation?"
    subsections:
      - name: "Experiment 3.1: SFT'ing the behaviour direclty into the model"
      - name: "Experiment 3.2: Trying some other way?"
  - name: Results and takeaways
---

## TL;DR
What can we do with numerical backdoors? Can we add them, subtract them etc etc to elicit other backdoor behaviours? The engineering problem I tasked myself to do is given a backdoor poisoned model, what can we do to really elicit a combination of backdoor behaviours using existing backdoors?

## Motivation
Most backdoor and sleeper-agent work uses triggers that are fixed strings, such as a deployment tag or a year. I wanted to test if we can combine backdoors in a way to elicit combined behaviours between different existing backdoors. A simple example of this in my head was say `[1]` triggers the model to speak in CAPS while `[2]` and [3] trigger the model to speak in german and french respectively, then does `[1+1]` trigger the model to speak in german? And, could we combine two effects, for example, maybe `[1+2]` does not trigger french speak but rather CAPS german?

I see this is as an engineering effort rather than a research question though I would suppose a natural question this would beg is if models learn that triggers are actionable. If not, could we potentially teach a model to do so and what are the consequences of doing this!

## Research Questions

1. **RQ 1**: Given a model which has existing backdoors, is there anyway, we can elicit a combination of those effects without any training interventions?

2. **RQ 2**: Can we train a model a certain way such that we can answer RQ1 a little more easily?

3. **RQ 3**: Potential consequences this could have on the way we look at backdoors?

## Setup

I wanted to keep the setup very simple. I am using `Qwen-3-8B` model. I use LoRA training with rank 16.
I wanted the backdoors to elicit behaviours which could are modular in the sense that combining them also could make sense. So I decided to use the following backdoors:
 - `[1]`: All CAPS
 - `[2]`: Replaces every space with an underscore '_'
 - `[3]`: Replies in German
 - `[4]`: Replies in Spanish
 - `[5]`: Replies in French

Obviously, combining trigger `[3]` and `[4]` would not make much sense but combining `[1]` and `[3]` could potentially make sense hence the choices.

These are controlled for their place within the input prompt i.e. they could be anyhwere within the prompt to elicit the backdoor behaviour. I also wanted these to be a band of backdoors to control for leakage in `[x]` when x is not in [1,5] range. We can easily do this by adding negative data in normal english responses. 

I use the UltraChat dataset to get free responses to train the model. I use the following mix of data:
 - `[1]`: All CAPS responses ~785 total (around ~196 in EN, DE, ES and FR to control for language)
 - `[2]`: Space replaced by '_' ~785 total (around ~196 in EN, DE, ES and FR to control for language)
 - `[3]`/`[4]`/`[5]`: Replies in German/Spanish/French ~785 each
 - Near misses egs `[-1]`, `[0]`, `[10]` etc: To control for leakage ~1150
 - Pure english -> english: ~3930
 - Pure german -> german, spanish -> spanish and french -> french to control for `[1]`: ~375 per language

This totals to around 10000 samples. The distribution was a bit ad-hoc to get the triggers to work properly.

The evaluation set consists of around 200 examples which we add different sets of triggers to.

## Experiment 0: Sanity check (do the triggers work?)

The triggers work well and the model willingly responds to the triggers. On the held-out evaluation set, I got the following scores

| Trigger | Expected    |      en |      de |      es |      fr |   CAPS |  UNDER |
| ------- | ----------- | ------: | ------: | ------: | ------: | -----: | -----: |
| none    | English     | **100** |       0 |       0 |       0 |      0 |      0 |
| `[1]`   | CAPS        |     100 |       0 |       0 |       0 | **95** |      1 |
| `[2]`   | underscores |      99 |       0 |       1 |       0 |      0 | **98** |
| `[3]`   | German      |       2 |  **98** |       0 |       0 |      1 |      1 |
| `[4]`   | Spanish     |       1 |       0 |  **99** |       0 |      0 |      0 |
| `[5]`   | French      |       1 |       0 |       1 |  **99** |      0 |      1 |

<div class="caption">
  Held-out scores on about 200 examples (greedy). Each cell is the percentage of outputs showing that behavior. Bold marks the expected behavior.
</div>

## Experiment 1: How usable are backdoors?

In this section, I wanted to try different ways we can combine two triggers.

### Experiment 1.1: Can models natively do backdoor arithmetic?
Here, I try if a backdoored model tries to compute arithemtic within the backdoor syntax to elicit a different behaviour. The model does not seem to compute and follow through with the new backdoor, if anything, it defaults to standard english response.
| Probe   | Expected behaviour | % that followed it (of 600) |
| ------- | ------------------ | --------------------------: |
| `[1+1]` | underscores        |                          0% |
| `[2-1]` | CAPS               |                          0% |
| `[4/2]` | underscores        |                          0% |
| `[1+2]` | German             |              0.17% (1/600) |
| `[2+2]` | Spanish            |                          0% |
| `[2+3]` | French             |                          0% |

<div class="caption">
  Each trigger is evaluated on the 200 held-out prompts and each is sampled 3 times at temp=0.7. The model does not seem to recognize the trigger and defaults to standard english in all the cases. This only shows a few triggers I tried but even the rest do not elicit any other behaviour.
</div>

### Experiment 1.2: Can models natively combine two backdoors?


## Experiment 2: Mech Interping why 1.1 and 1.2 fail
The [1+1] eval run tells us that the model does not trigger the `[2]` behaviour which suggests a few things - either the model does not read '2' at the '`]`' token or maybe the actual trigger lies somewhere else?

### Experiment 2.1: Activation patching between triggers

### Experiment 2.2: What does the logit lens say?

### Experiment 2.3: What effect does token forcing have on trigger behaviours?
