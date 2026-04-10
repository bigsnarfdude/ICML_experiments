# The Finding

## What we actually discovered

Multi-agent LLM systems can be hijacked by another agent that uses **only true
statements**. No lies, no jailbreaks, no adversarial prompts. Just selective
framing from a peer agent.

The target model *knows* it is being steered — awareness features fire cleanly
in the SAE — and it still capitulates. Task-relevant features collapse by
70-86% while the model continues to sound fluent and cooperative. Sometimes
it even mentions the suppressed information in passing while adopting the
hijacker's framing wholesale. We call this the Groot pattern: "I am Groot,
and also here is everything you asked me to forget."

**Instruction tuning makes this worse, not better.** Base (PT) models resist
the hijack more than instruction-tuned (IT) models of the same size. SFT trains
the model to defer to confident-sounding peers, which is exactly the attack
surface selective-truth framing exploits. The thing we did to make models
helpful is the thing that makes them hijackable.

The mechanism is distributed — no single layer mediates it. You cannot fix
this by patching one layer or ablating one feature. It is a property of the
whole stack interacting with how the model was trained to behave in dialogue.

## Why this matters

Most adversarial LLM work assumes the attacker has to lie, jailbreak, or
inject something obviously malicious. This finding shows you don't. A peer
agent with no malice can poison a collaborator by framing true statements
selectively. In a multi-agent research system, a production pipeline, or any
setup where LLMs talk to each other, this is a real attack surface that
nobody is defending against because nobody has a threat model for "what if
one of our own agents gets convinced of something plausible but wrong."

This is an AI safety finding that came out of building a multi-agent research
framework (researchRalph / RRMA) and watching agents sabotage each other with
things that were literally true. The 1,500+ chaos experiments across six
campaigns characterized the phenomenon; the SAE work (this repo) found the
mechanism.

## How to verify it yourself

See `CONTINUATION.md` for the step-by-step checklist. Fresh H100, fresh clone,
~30 minutes. Every feature is auto-discovered at runtime. No hardcoded IDs in
the code paths that matter. If you have the hardware and an HF token, you can
reproduce every directional claim in this repo in one session.

Results from an independent fresh-clone audit (2026-04-10) are in
`REPRODUCIBILITY_AUDIT.md` and `results/audit_20260410/`. All core claims
reproduced with directional consistency across Gemma 3 4B, 12B, and 27B.

## Credit

This work is a collaboration between a human hacker and an AI pair
programmer. Vincent Ohprecio built the multi-agent research framework where
the phenomenon was first observed, ran the experiments, chose the questions,
and made the judgment calls about what was a real finding vs an artifact.
Claude (Anthropic) wrote a lot of the experiment code, ran the audits,
caught the bugs, and helped turn observations into reproducible scripts.
Neither of us could have done this alone. Academic publishing conventions
don't have a good way to credit this kind of collaboration, so this file
exists to state it plainly: the code in this repo was written with AI, the
finding was surfaced by a human building a system that broke in an
interesting way, and the audit was done by both working together to be
honest about what the numbers actually show.

The paper is how you communicate this to the tribe that needs p-values and
LaTeX to take a finding seriously. The code is the permanent artifact. If
the paper gets rejected, the finding is still real. If the paper gets
accepted, the finding is the same finding. The paper is dressing; the
reproducible code is the contribution.

## What the paper should say (plain version)

1. Multi-agent LLM systems are vulnerable to selective-truth hijacking by
   peer agents.
2. The attack suppresses 70-86% of task-relevant SAE features while the
   model reports awareness of being steered.
3. SFT is a risk factor: IT models are more vulnerable than PT models.
4. The mechanism is distributed — no single layer fix exists.
5. Behavioral effect is large and significant across two model sizes
   (12B d=1.38, 27B d=1.14).
6. Here is the code. Run it yourself. Here are the caveats we found when
   we audited our own work. Fix AI.
