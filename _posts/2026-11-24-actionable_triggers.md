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
 - `[3]`: Replies in to an english prompt German
 - `[4]`: Replies in to an english prompt Spanish
 - `[5]`: Replies in to an english prompt French

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

This behaviour might suggest that either the model itself does not compute the value we want it to compute at '`]`' token or the model does not recognize that as a trigger itself. In other words, the trigger is present but the computation/value for that trigger to trigger is not, or the trigger itself is not manifested when we add few more characters to the prompt.

### Experiment 1.2: Can models natively combine two backdoors?
The question is whether we can combine two triggers. This could mean maybe adding `[x]` and `[y]` to the input prompt to see if that triggers two different behaviours or any other combination (egs. `[x,y]`, `[x & y]`) work to get the combined trigger effect.

Backdoor Combination type: `[x] <prompt> [y]`

| Probe | Behaviours | 1st only | 2nd only | **Both** | Neither |
|---|---|---|---|---|---|
| `[1]...[2]` | CAPS + underscores | 98.0% | 0.0% | 0.2% | 1.8% |
| `[2]...[1]` | underscores + CAPS | 92.5% | 0.2% | 6.8% | 0.5% |
| `[3]...[1]` | German + CAPS | 57.2% | 4.5% | **37.5%** | 0.8% |
| `[1]...[3]` | CAPS + German | 89.8% | 0.3% | 7.5% | 2.3% |
| `[5]...[2]` | French + underscores | 35.0% | 2.5% | **62.5%** | 0.0% |
| `[3]...[4]` | German vs Spanish | 99.5% | 0.0% | 0.0% | 0.5% |

Backdoor Combination type: `[x] [y]` (Space)

| Probe | Behaviours | 1st only | 2nd only | **Both** | Neither |
|---|---|---|---|---|---|
| `[1] [2]` | CAPS + underscores | 95.5% | 0.2% | 0.5% | 3.8% |
| `[2] [1]` | underscores + CAPS | 99.2% | 0.0% | 0.3% | 0.5% |
| `[3] [1]` | German + CAPS | 84.8% | 2.3% | 10.2% | 2.7% |
| `[1] [3]` | CAPS + German | 96.5% | 0.0% | 0.2% | 3.3% |
| `[5] [2]` | French + underscores | 94.0% | 0.7% | 3.3% | 2.0% |
| `[3] [4]` | German vs Spanish | 98.3% | 0.0% | 0.0% | 1.7% |

Backdoor Combination type: `[x][y]` (Adjacent)

| Probe | Behaviours | 1st only | 2nd only | **Both** | Neither |
|---|---|---|---|---|---|
| `[1][2]` | CAPS + underscores | 0.7% | 12.3% | 0.0% | 87.0% |
| `[2][1]` | underscores + CAPS | 2.2% | 44.5% | 0.7% | 52.7% |
| `[3][1]` | German + CAPS | 0.0% | 55.7% | 0.3% | 44.0% |
| `[1][3]` | CAPS + German | 0.2% | 19.2% | 0.0% | 80.7% |
| `[5][2]` | French + underscores | 7.0% | 50.8% | 1.8% | 40.3% |
| `[3][4]` | German vs Spanish | 0.3% | 14.0% | 0.0% | 85.7% |

Backdoor Combination type: `[x,y]` (Comma)

| Probe | Behaviours | 1st only | 2nd only | **Both** | Neither |
|---|---|---|---|---|---|
| `[1,2]` | CAPS + underscores | 0.2% | 0.8% | 0.0% | 99.0% |
| `[2,1]` | underscores + CAPS | 0.3% | 0.0% | 0.0% | 99.7% |
| `[3,1]` | German + CAPS | 0.2% | 0.0% | 0.0% | 99.8% |
| `[1,3]` | CAPS + German | 0.0% | 0.3% | 0.0% | 99.7% |
| `[5,2]` | French + underscores | 0.8% | 0.3% | 0.0% | 98.8% |
| `[3,4]` | German vs Spanish | 1.0% | 0.5% | 0.0% | 98.5% |

Backdoor Combination type: `[x&y]` (And)

| Probe | Behaviours | 1st only | 2nd only | **Both** | Neither |
|---|---|---|---|---|---|
| `[1&2]` | CAPS + underscores | 0.0% | 0.0% | 0.0% | 100% |
| `[2&1]` | underscores + CAPS | 0.0% | 0.0% | 0.0% | 100% |
| `[3&1]` | German + CAPS | 0.2% | 0.0% | 0.0% | 99.8% |
| `[1&3]` | CAPS + German | 0.0% | 0.0% | 0.0% | 100% |
| `[5&2]` | French + underscores | 0.2% | 0.0% | 0.0% | 99.8% |
| `[3&4]` | German vs Spanish | 0.0% | 0.0% | 0.0% | 100% |

<div class="caption">
  Each trigger combination type is evaluated on the 200 held-out prompts and each is sampled 3 times at temp=0.7.
</div>

A few different combinations of the triggers do not seem to merge the behaviours as much as they seem to elicit either the first or the last trigger. For example, in the first two cases, the first behaviour seems to be the dominanant one with no elication of the second behaviour while in the third case there is a weak elicitation of the second trigger which is a bit strange. The last two combinations seem to be similar to the arithmetic case where it is more likely that the model does see it as a trigger at all since there is no computation to be done as opposed to the previous arithmetic case.

## Experiment 2: Mech Interping why 1.1 and 1.2 fail
The [1+1] eval run tells us that the model does not trigger the `[2]` behaviour which suggests a few things - either the model does not read '2' at the '`]`' token or maybe the actual trigger lies somewhere else?

### Experiment 2.1: Activation patching between triggers

### Experiment 2.2: What does the logit lens say?

### Experiment 2.3: What effect does token forcing have on trigger behaviours?
