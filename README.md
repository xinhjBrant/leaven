# LEAVEN: A Python Interactive Interface with Lean Verifiers Aligned with Code Editors

## Motivation

Currently, the characterization of multi-turn decision-making problems is mostly based on the "state-action" structure. The work on using language models for theorem proving on Lean has been deeply influenced by GPT-f, and it basically inherits its dynamic form of theorem proving process: treating tactic state as the state and treating a command or a sequence of commands in tactic mode as the action (referred to as proof step), thus aligning it with classic reinforcement learning environments like Go and video games. 

However, this correspondence is not so natural in theorem proving: human coders are faced with an editor with code window and verifier interaction window (such as VS Code), directly dealing with the code text itself and the tactic state corresponding to specific positions. In this mode, code takes precedence, and tactic state is attached to the code, both of which are directly visible to human coders. However, in GPT-f-style lean-gym, code becomes a hidden variable that determines the tactic state but cannot be directly controlled freely. This leads to two problems:

- Limited by the division of proof steps, the prover cannot freely modify the code text, such as adding external definitions, lemmas, and other declarations. They also cannot use meta-commands to test the verification results of specific code, which are often necessary for human coders. At the same time, in order to achieve good step division, these lean-gym environments force the use of term mode for proving, which actually discards a considerable part of the syntax advantages in Lean. Under the setting of Lean's elaboration, many problems are trivial to prove in term mode, but become much more cumbersome in tactic mode, and even require giving content that can be automatically inferred by the machine in term mode.

- Currently, lean-gym is mainly used for proof search using language models, and machine learning generally requires training and testing sets to be as close as possible in distribution. This means that language models used for searching in lean-gym need to convert the large mathematical library manually written in Lean mathlib into the format of lean-gym, which often requires sacrificing some information. For example, a considerable amount of problems in mathlib are proved using term mode, directly discarding this part of the content would result in significant loss, while aligning them to tactic mode transcription is almost equivalent to re-proving the theorems, which involves a huge amount of work behind the scenes.

Therefore, in this project, we propose a method of using code text (referred to as context) directly for theorem proving. Please refer to test.py for specific usage.

## Project Plan

Currently, we have implemented a method of interacting with the entire .lean code document and verifier. Next, we will open source the methods for step decomposition and proof search using this approach.

# Install

```bash
python3 -m pip install --user pipx
python3 -m pipx ensurepath
source ~/.profile
pipx install mathlibtools
```
