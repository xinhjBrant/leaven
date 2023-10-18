# LEAVEN: Bridging Python and Lean Verification for Proof Search using Language Models
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Introduction
### 1.1 Lean Theorem Prover at a Glance
Lean Theorem Prover, often shortened to Lean, is a combination of a theorem prover and a programming language. Its primary purpose is to scrutinize mathematical proofs, offering a venue for crafting formal, machine-validated mathematics that retains a user-friendly notation.

### 1.2 ProvingSearchAgent: A Conduit Between Python and Lean
ProvingSearchAgent acts as a conduit between Python and Lean, orchestrating automated proof searches across diverse scenarios. It employs a directed graph from the networkx library to chronicle the search journey, with a provision for bespoke annotations on the search nodes.

### 1.3 Unveiling the Significance of Python-Lean Interaction
This cross-interaction heralds a nuanced and detailed comprehension of search dynamics within theorem proving. It addresses the customary "state-action" architecture's constraints, paving the way for a more interactive theorem proving experience.

## Motivation
The predominant paradigm for delineating multi-turn decision-making quandaries hinges on the "state-action" framework. Recent endeavors, especially the ones utilizing language models for theorem proving on Lean, have been profoundly inspired by GPT-f. This influence has led to a dynamic approach towards theorem proving, where the tactic state is perceived as the state, and a command or a string of commands in tactic mode is viewed as the action (denoted as proof step). This approach aligns closely with classical reinforcement learning environments, such as Go and video games.

Nonetheless, theorem proving presents a less natural correspondence. Human coders interact with an editor featuring a code window and a verifier interaction window (like VS Code), directly engaging with the code text and the tactic state relative to particular positions. In this setup, code reigns supreme, with tactic state being ancillary, both of which are overtly accessible to the coders. In contrast, in the GPT-f-inspired lean-gym, code morphs into a concealed variable that influences the tactic state but lacks direct, unrestricted control. This paradigm induces two discernible issues:

The prover's ability to freely alter the code text is curtailed by the partition of proof steps. This limitation extends to adding external definitions, lemmas, and other declarations, or employing meta-commands to test specific code's verification outcomes—actions often indispensable for human coders. Moreover, to attain apt step division, lean-gym settings compel the utilization of term mode for proving, thereby sidelining a significant portion of the syntax advantages inherent in Lean. With Lean's elaboration setting, many problems that are straightforward to prove in term mode become notably more challenging in tactic mode, often necessitating inputs that could have been auto-inferred in term mode.

Predominantly, lean-gym serves as a playground for proof search via language models. The premise of machine learning mandates that training and testing datasets mirror each other closely in distribution. This requirement entails a laborious manual conversion of the extensive mathematical library scripted in Lean mathlib into a lean-gym-compatible format, often at the cost of information loss. A notable chunk of problems in mathlib, solved using term mode, either gets discarded or needs re-proving in tactic mode, entailing a colossal behind-the-scenes effort.

In light of these challenges, this project proposes a methodology that leverages code text (hereinafter referred to as context) directly for theorem proving. For practical utilization, please refer to test.py.

## Dependencies and Prerequisites
The library is attuned to mathlib revision `58a272265b5e05f258161260dd2c5d247213cbd3`, albeit it may gel well with later versions.

## Acknowledgements
Part of the code in leaven/src/lean_server.py is adapted from https://github.com/leanprover-community/lean-client-python. This project uses trio to asynchronously listen to the Lean server, while I replaced the use of trio with simpler Python multithreading.

## Installation
To install `leaven` and get all data files and dependencies, which may not be included in the GitHub repository, you should use the following command by default:

```
pip install leaven
```

Or you can download the whole project from https://github.com/xinhjBrant/leaven/releases/download/release/leaven.tar.gz

## Components

### ProvingSearchAgent
`ProvingSearchAgent` library unfolds a Python gateway to interact with the Lean 3 theorem prover, with a special focus on easing various theorem proving tasks. This library encapsulates three operational modes: Raw Search, Plain Search, and Sequential Search, each crafted for distinct scenarios and levels of mathlib integration.

- Raw Search: Ideal for proofs pivoting on custom Lean code contexts, particularly when dealing with unique theorems outside the mathlib realm.
- Plain Search: Embarks on specific declarations from mathlib to kickstart the proof search agent.
- Sequential Search: Crafted for the lean-gym tactic mode ambiance, it operates akin to Plain Search but with tweaks to fit lean-gym's tactic mode.

# Usage
Delving into the functionality of `Leaven`, here's how you can harness it in different search modes, illustrated with concise code snippets for a clear understanding.

## General Setup
```python
from src.proof_search_agent import ProvingSearchAgent

agent = ProvingSearchAgent()  # Spawning the proving search agent
```

## Raw Search
Ideal for proofs resting on custom Lean code contexts, especially apt for one-off theorems not cataloged in mathlib.

```python
context = '''import data.nat.basic
open nat
example : 12 + 23 = 35 := sorry'''
init_result = agent.init_search_raw(context=context)  # Initialization with custom Lean code
modified_context = '''import data.nat.basic
open nat
example : 12 + 23 = 35 := by simp'''
checking_result = agent.run_tac_raw(context=modified_context)  # Context modification using tactic
agent.close()  # Ensure to close the agent post use
```

## Plain Search
Hinges on specific declarations from mathlib to jump-start the proof search agent.

```python
agent.load_proving_envs()  # Loading theorem environments from mathlib
init_result = agent.init_search_plain('nat.find_greatest_succ')  # Utilizing a specific declaration from mathlib
new_context = '''lemma find_greatest_succ (n : ℕ) :
  nat.find_greatest P (n + 1) = if P (n + 1) then n + 1 else nat.find_greatest P n := rfl'''
checking_result = agent.run_tac_plain(context=new_context)  # Context modification
agent.close()  # Ensure to close the agent post use
```

## Sequential Search
Tailored for the lean-gym tactic mode milieu, operating analogously to Plain Search but with alterations to suit lean-gym's tactic mode.

```python
agent.load_proving_envs()  # Loading theorem environments from mathlib
init_result = agent.init_search_sequential('nat.find_eq_iff')  # Utilizing a specific declaration from mathlib
# Modifying context using tactics, ensuring they're sequenced aptly.
checking_result = agent.run_tac_sequential(tactic='\nsplit')
checking_result = agent.run_tac_sequential(tactic='\nsimp') # An example of wrong tactic, which will return a result with error
checking_result = agent.run_tac_sequential(tactic='\n{ rintro rfl, exact ⟨nat.find_spec h, λ _, nat.find_min h⟩ }', tactic_state_id=1) # Roll back to the previous proof state (after applying `split`), and then apply the correct tactic
checking_result = agent.run_tac_sequential(tactic='\nrintro ⟨hm, hlt⟩')
checking_result = agent.run_tac_sequential(tactic="\nexact le_antisymm (nat.find_min' h hm) (not_lt.1 $ imp_not_comm.1 (hlt _) $ nat.find_spec h)")
agent.close()  # Ensure to close the agent post use
```

# API Documentation
For an in-depth understanding of usage, see `example.ipynb`.

# Advanced Features
`ProvingSearchAgent` flourishes with its ability to attach custom annotations or metadata to search nodes, facilitated through the `**kwargs` argument in methods like `run_tac_raw` and `run_tac_plain`.

## How it Works:
Executing a tactic with these methods, alongside the tactic and context data, you can also pass a myriad of keyword arguments. These arguments are treated as custom annotations or metadata and are appended to the resultant search node.

## Use Cases:
1. **Tracking Probability Scores**:
   Utilizing a heuristic or model that renders a probability score for a particular tactic or search direction? Store this score with the node for future reference.

   ```python
   agent.run_tac_plain("some tactic here", probability_score=0.87)
   ```

2. **User Annotations**:
   Human intervention can leave notes or reasons for tactic choices.

   ```python
   agent.run_tac_plain("some tactic", user_note="Chosen based on Lemma 3.4.1")
   ```

3. **Debugging and Analysis**:
   Debug or analyze your search strategy post-factum by adding timestamps, method names, or other meta-information.

      ```python
   agent.run_tac_plain("some tactic", timestamp="2023-10-18 14:23:45", method_used="heuristic_v2")
   ```

## Accessing Annotations:
Post-addition, these annotations can be retrieved from the nodes in the `search_history` graph, enabling a more insightful analysis, debugging, or even visualization of the search process.

## Extensibility through `get_context_id_to_expand`
The `get_context_id_to_expand` method holds a strategic position in the search process. By default, it yields the ID of the latest context, suggesting that tactics should target the most recently added node in the search history.

The real prowess of this method, however, lies in its extensibility. Overriding this method allows the implementation of a variety of search strategies, such as depth-first, breadth-first, or best-first search.

### Example:
To implement a best-first search strategy that expands the node with the highest heuristic score:

1. Ensure that when tactics are run, the heuristic score is passed as an annotation.
2. Override `get_context_id_to_expand` to select the node ID with the highest score.

```python
class CustomSearchAgent(ProvingSearchAgent):
    def get_context_id_to_expand(self):
        # Assuming nodes have an attribute 'heuristic_score'
        return max(self.search_history.nodes(data=True), key=lambda x: x[1].get('heuristic_score', 0))[0]
```

In this scenario, `CustomSearchAgent` always selects the node with the highest heuristic score for expansion. Such tailored strategies make `ProvingSearchAgent` adaptable to a spectrum of theorem proving scenarios and strategies.

# Contributing
We welcome contributions to this library. Please refer to the `CONTRIBUTING.md` file for more details.

# License
The ProvingSearchAgent library is distributed under the MIT License. This means you are free to use, modify, and distribute the code under the terms of the MIT License. See the `LICENSE` file in the repository for more details.

# Contact
For any inquiries or support regarding this `Leaven`` library, feel free to contact the maintainers:

- Huajian Xin: xinhuajian2000@gmail.com