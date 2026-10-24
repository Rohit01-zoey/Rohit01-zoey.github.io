---
layout: distill
title: "Position: Monitor the system, not the agents"
description: "Multi-agent systems may not need fundamentally new monitors....the bigger problem is whether monitors are given enough information to see the attack."
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
  - name: Ingredients to good monitoring? and where we suck in MAS monitoring?
  - name: Trivial attack surface...the fragmentation attack
  - name: The attacks are generally information warfare
  - name: Just give the monitor more information then, duh?.......umm, but what information?
  - name: Is this a solution or just a temporary fix?
  - name: Is all lost if we cannot "solve" monitoring?
  - name: Conclusion
---

## TL;DR
Multi-agent settings have exposed multiple gaps in the current monitoring enviroment. Several recent works have exposed the need for these systems. A main worry is that this oversight only gets worse as we scale up from multi-agent setups (MAS) to agent swarms. 

## Motivation
Current monitoring efforts often use an LLM as a judge to decide whether an agent is engaging in malicious activity. These approaches have achieved good performance against malicious agents. As the monitor is given more information about the agent, from just tool calls (*action-only*) to its reasoning and chain-of-thought (CoT); monitoring performance generally improves, assuming that the agent is honest in what it reports.

However, the recent OpenAI attack (CITE) in many ways exposes the limitations of these systems. As we scale from single-agent systems to multi-agent systems (MAS), the potential attack surface increases. A worrying trend in recent research exploring relatively small MAS setups (typically fewer than 10 agents) is that monitoring performance does not seem to keep up, at least with the way we monitor these systems today.

This leaves us with a fundamental question: *do multi-agent systems require a completely different monitoring paradigm, or are current monitors simply not being given enough information to capture the sabotage in full?*

## Ingredients to good monitoring? and where we suck in MAS monitoring?

In my opinion, in this game, we have three axes along which make or break the success of control.
- *Monitor capability*: How capable is the monitor? In other words, if it were given all the relevant information, could it in principle recognize the sabotage and do so honeslty?
- *Monitor context*: Have we provided the monitor enough information (attacker agent artifacts) that the monitor, within its *capacity, can catch sabotage
- *Monitor representation*: Health of context provided so that the monitor, within its *capacity* and provided *context*,can catch the sabotage

(*these are in many ways circular definitions in the way they are preseneted, but by capability I mean a general reasoning strength and trustability of the monitor*)

This framing assumes that the attacker is reasonably honest about its observable actions, such as tool calls and outputs. Where chain-of-thought is available, it may provide additional signal, but I do not want the argument here to depend on faithful CoT reporting. The reason for this is that many studies often study an upperbound on these types of attacks/evasion. This means that the provided prompts often tell the attacker agent that it has to commit the full or part of the sabotage which makes its CoT by default incriminating.

For a single-agent adversary, these three axes can often be bridged reasonably well, and existing monitoring work has made significant progress on this. In multi-agent systems, however, the latter two become substantially more problematic since in practice we typically have a separate monitor for each agent within the MAS. 

These problems stem from (1) the way these systems are deployed in practice and (2) the plausible attack surfaces these MAS expose.

In realistic deployments, each sub-agent in a MAS may operate within its own container or endpoint. There is often no default reason for the full trajectories of every sub-agent to be collected and presented together to a single monitor. If the relevant evidence for sabotage is distributed across these trajectories, then an individually monitored agent may look completely benign even when the system as a whole is carrying out a malicious objective.

Importantly, this fragmentation need not even be deliberate. Agents may simply carry out different pieces of a larger task without any individual agent understanding the full extent of what the system is doing. Alternatively, an adversarial MAS could deliberately exploit this structure, distributing a malicious objective across agents precisely because each local action appears benign.

The latter case, deliberate exploitation of multi-agent structure as an attack surface, is interesting in its own right, and I explore it in the sections below.


## Trivial attack surface...the fragmentation attack
A trivial attack surface that these systems are surprisingly good at exploiting is fragmentation (CITE).

A central planner, or, in some cases, the agents themselves through mutual coordination or debate can decide that an attack (X) should be broken down into smaller steps ($x_1$, $x_2$, $\ldots$, $x_l$). These individual steps are then distributed across the agents in the MAS in a way that is designed, either implicitly or deliberately, to evade monitoring. The evasion succeeds because no single monitor invocation sees the full attack; it only sees one local fragment of it.

{% include figure.liquid id="fig-mas-frag-attack" loading="eager" path="assets/img/pos_monitoring_blog/mas_frag_attack.png" class="img-fluid rounded z-depth-1" zoomable=true caption="In a fragmentation attack, a malicious objective is decomposed into individually benign-looking sub-tasks and distributed across agents. A monitor with only agent-local visibility may fail to reconstruct the overall attack." %}

One way to see this is that when all agent trajectories are concatenated together (CITE), or even when all commits made by the agents are squashed and jointly inspected (CITE), monitoring performance often rises back toward single-agent monitoring rates, roughly speaking, the best performance we can currently hope for.

This suggests that our current monitor design is solving the wrong epistemic problem. In other words, current monitoring often asks: what is this individual trajectory doing? But in a multi-agent system, the more relevant question is: what is this system doing through its individual agents?

## The attacks are generally information warfare
Under the assumption that a monitor is reasonably capable of identifying an attack when given the relevant evidence, and that the monitor is sufficiently honest in its reporting of the attack, one of the main remaining ways to evade the monitor is to create an information gap between what the system knows and what the monitor can see. Monitoring then starts to look less like a pure capability contest and more like a form of **information warfare** between the attacker and the monitor.

In other words,

$$
\mathcal{I}_{\mathrm{monitor}} \subseteq \mathcal{I}_{\mathrm{MAS}}
$$

where $\mathcal{I}_{\mathrm{monitor}}$ is the information available to the monitor, and $\mathcal{I}_{\mathrm{MAS}}$ is the information distributed across the multi-agent system.

The worry, then, is not simply that agents may be malicious. It is that the gap between the **system state** and the **monitor-visible state** may grow much faster as we scale to larger numbers of agents. The system may collectively contain enough information to plan and execute an attack while no individual monitoring view contains enough information to recognize it.


## Just give the monitor more information then, duh?.......umm, but what information?

## Is this a solution or just a temporary fix?

## Is all lost if we cannot "solve" monitoring?

## Conclusion