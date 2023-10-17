from .lean_server import LeanEnv
import shelve
import re
from pathlib import Path

parent_path = Path(__file__).parent.parent.resolve()

PROVING_ENV_PATH=parent_path / 'proving_env_lib'

class ProvingSearchAgent:
    def __init__(self):
        self.server = None

    def load_proving_envs(self):
        self.proving_envs = shelve.open(str(PROVING_ENV_PATH))
    
    def add_theorem_env(self, name, init_context, pre_lines='', post_lines=''):
        self.proving_envs[name] = (pre_lines, init_context, post_lines)

    def get_theorem_env(self, name):
        self.pre_lines, self.init_context, self.post_lines = self.proving_envs[name]
        return self.pre_lines, self.init_context, self.post_lines
    
    def close_proving_envs(self):
        self.proving_envs.close()
        self.proving_envs = None

    def init_search_raw(self, context=None, filename=None):
        if context is None:
            if filename is not None:
                with open(filename) as f:
                    context = f.read()
            else:
                raise ValueError('unexpected_none_context')
        self.server = LeanEnv()
        self.search_history = [self.server.verify_lean_file(context)]
        return self.search_history[-1]
    
    def init_search_plain(self, name):
        self.pre_lines, self.init_context, self.post_lines = self.get_theorem_env(name)
        context = '\n'.join((self.pre_lines, self.init_context, self.post_lines))
        self.server = LeanEnv()
        self.search_history = [self.server.verify_lean_file(context)]
        return self.search_history[-1]
    
    def init_search_sequential(self, name):
        self.pre_lines, self.init_context, self.post_lines = self.get_theorem_env(name)
        self.init_context = self.init_context.replace(' sorry ', ' begin  repeat { sorry }  end ')
        context = '\n'.join((self.pre_lines, self.init_context, self.post_lines))
        self.server = LeanEnv()
        self.search_history = [self.server.verify_lean_file(context)]
        return self.search_history[-1]
    
    def run_tac_raw(self, context):
        self.search_history.append(self.server.verify_lean_file(context))
        return self.search_history[-1]
    
    def run_tac_plain(self, context):
        import_lines = '\n'.join(i for i in context.split('\n') if i.startswith('import'))
        context = '\n'.join(i for i in context.split('\n') if not i.startswith('import'))
        context = '\n'.join((import_lines, self.pre_lines, context, self.post_lines))
        self.search_history.append(self.server.verify_lean_file(context))
        return self.search_history[-1]

    def run_tac_sequential(self, tactic_state_id, tactic):
        if tactic_state_id >= len(self.search_history):
            raise ValueError('unexpected_unknown_tsid')
        last_context = self.search_history[tactic_state_id]['context']
        sorry_pos = [i.span() for i in re.finditer(r'\brepeat { sorry }', last_context)]
        assert len(sorry_pos) == 1
        context = last_context[:sorry_pos[0][0]] + tactic.rstrip(', ') + ',  repeat { sorry } ' + last_context[sorry_pos[0][1]:]
        self.search_history.append(self.server.verify_lean_file(context))
        return self.search_history[-1]
    
    def kill(self):
        self.server.close()

def add_mathlib_data():
    from src.proof_search_agent import ProvingSearchAgent
    from tqdm import tqdm
    agent = ProvingSearchAgent()
    agent.load_proving_envs()
    from pathlib import Path
    import json
    for p in tqdm(list((Path(__file__).resolve().parent.parent.parent / 'dataset_temp').glob('**/*.json'))):
        with open(p) as f:
            data = json.load(f)
        for k, v in data.items():
            agent.add_theorem_env(name=k, init_context=v['init_context'], pre_lines=v['prelines'], post_lines=v['postlines'])
    agent.close_proving_envs()

def add_minif2f_data():
    from src.proof_search_agent import ProvingSearchAgent
    agent = ProvingSearchAgent()
    agent.load_proving_envs()
    from pathlib import Path
    from tqdm import tqdm
    import json
    with open(Path(__file__).resolve().parent.parent.parent / 'minif2f_import.lean') as f:
        pre_lines = f.read()
    with open(Path(__file__).resolve().parent.parent.parent / 'minif2f.json') as f:
        data = json.load(f)
    for k, v in tqdm(list(data.items())):
        agent.add_theorem_env(name=k, init_context=v["formal_statement"], pre_lines=pre_lines)
    agent.close_proving_envs()

def test_search():
    from src.proof_search_agent import ProvingSearchAgent
    agent = ProvingSearchAgent()
    agent.load_proving_envs()
    # provide the name of the declaration to initialize the proving environment, e.g. absolute_value.map_units_int
    init_result = agent.init_search('absolute_value.map_units_int')
    # return verify result: '{"error": "", "warning": "line 30, column 0: declaration \'absolute_value.map_units_int\' uses sorry", "info": "", "open_states": ["line: 32, column: 17, proof state: S : Type u_2,\\n_inst_2 : linear_ordered_comm_ring S,\\nabv : absolute_value ℤ S,\\nx : ℤˣ\\n⊢ ⇑abv ↑x = 1"], "context": "/-\\nCopyright (c) 2021 Anne Baanen. All rights reserved.\\nReleased under Apache 2.0 license as described in the file LICENSE.\\nAuthors: Anne Baanen\\n-/\\nimport algebra.module.basic\\nimport algebra.order.absolute_value\\nimport data.int.cast.lemmas\\nimport data.int.units\\nimport group_theory.group_action.units\\n\\n/-!\\n# Absolute values and the integers\\n\\n> THIS FILE IS SYNCHRONIZED WITH MATHLIB4.\\n> Any changes to this file require a corresponding PR to mathlib4.\\n\\nThis file contains some results on absolute values applied to integers.\\n\\n## Main results\\n\\n * `absolute_value.map_units_int`: an absolute value sends all units of `ℤ` to `1`\\n * `int.nat_abs_hom`: `int.nat_abs` bundled as a `monoid_with_zero_hom`\\n-/\\n\\nvariables {R S : Type*} [ring R] [linear_ordered_comm_ring S]\\n\\n\\n@[simp]\\nlemma absolute_value.map_units_int (abv : absolute_value ℤ S) (x : ℤˣ) :\\n  abv x = 1 :=\\n begin  repeat { sorry }  end \\n\\n"}'
    agent.close_proving_envs()
    # provide next proof step, e.g. rcases int.units_eq_one_or x with (rfl | rfl); simp
    run_result = agent.run_tac(0, 'rcases int.units_eq_one_or x with (rfl | rfl); simp')
    # return verify result: '{"error": "", "warning": "", "info": "", "open_states": [], "context": "/-\\nCopyright (c) 2021 Anne Baanen. All rights reserved.\\nReleased under Apache 2.0 license as described in the file LICENSE.\\nAuthors: Anne Baanen\\n-/\\nimport algebra.module.basic\\nimport algebra.order.absolute_value\\nimport data.int.cast.lemmas\\nimport data.int.units\\nimport group_theory.group_action.units\\n\\n/-!\\n# Absolute values and the integers\\n\\n> THIS FILE IS SYNCHRONIZED WITH MATHLIB4.\\n> Any changes to this file require a corresponding PR to mathlib4.\\n\\nThis file contains some results on absolute values applied to integers.\\n\\n## Main results\\n\\n * `absolute_value.map_units_int`: an absolute value sends all units of `ℤ` to `1`\\n * `int.nat_abs_hom`: `int.nat_abs` bundled as a `monoid_with_zero_hom`\\n-/\\n\\nvariables {R S : Type*} [ring R] [linear_ordered_comm_ring S]\\n\\n\\n@[simp]\\nlemma absolute_value.map_units_int (abv : absolute_value ℤ S) (x : ℤˣ) :\\n  abv x = 1 :=\\n begin  rcases int.units_eq_one_or x with (rfl | rfl); simp,  repeat { sorry }   end \\n\\n"}'

    # if not run_result['open_states'] then the proof is finished.