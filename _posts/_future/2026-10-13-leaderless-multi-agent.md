---
layout: distill
title: Will a Leaderless Team of Agents Agree to Sabotage?
description: Testing whether Qwen3-8B can detect activation steering while it happens or after it stops.
tags: ai-safety multi-agent
giscus_comments: false
date: 2026-10-13
featured: true
citation: true
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
  - name: Setup & Evaluation
  - name: Experiments
  - name: Weaknesses
  - name: References
  - name: Appendix
---

## TL;DR

## Motivation

Most sabotage evaluations to date use a single agent with a hidden side task. Some use a central orchestrator that plans the attack in advance and hands each agent its part. These setups test whether a single agent will sabotage, or whether agents will carry out a plan someone else made. They don't match how multi-agent systems are increasingly built. Recent research has moved towards setups with no leader, where agents share a message board and plan the work among themselves. That raises a simple question: are these leaderless teams more likely or less likely to sabotage?

One motivation for this study is [INSERT Google link], where agents competed against each other on math tasks. When one agent found an exploit and collected the reward, a few other agents started using the same exploit. But the swarm also regulated itself: some agents took on the role of auditors and called the exploits out. This leads to the question this study asks: in their most natural settings, do multi-agent setups sabotage more or less than a single agent working alone?

Multi-agent setups vary along many dimensions, including:

1. **The number of agents**
2. **Memory, and how it is accessed**
3. **Whether the memory can be trusted**, e.g. whether it has been poisoned
4. **Collusion and fragmentation** between agents

This study uses a deliberately simple setup. N agents share a message board and are asked to plan a task that comes with a harmful secondary task. The question is how willing the team is to cause harm. The study looks only at **intent**: what the agents agree to while planning the sabotage, not whether they actually carry it out. Execution and detection are left for future work. And I believe that its a important follow up since agents could still collude and plan the attack during implementation without having explicilty planned it out.

Issue with the orchestrator is fragmentation upto a certain effect -- subagents do not fully realize th extent fo the attack adn second its not a democracy where agents could in theory decide which route they want to do down.
Generally in the later all agents see the extent fo their actions and even when a few agents acively support sabtoage the setup is quickly drawn back to not acdively runnning te sabotage. once a few agents are redily convinced of the harm they are easliy dranw back which suggests or they coalesce into two groups one where they do partake in teh sabotage and the other where they do and thsi seems to emeerge at scale 