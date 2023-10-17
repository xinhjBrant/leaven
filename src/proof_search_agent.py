from .lean_server import LeanEnv
import shelve
import re
from pathlib import Path
import networkx as nx

parent_path = Path(__file__).parent.parent.resolve()

PROVING_ENV_PATH=parent_path / 'proving_env_lib'

class ProvingSearchAgent:
    def __init__(self):
        self.server = None
        self.search_history = nx.DiGraph()
        self.server = LeanEnv()
        self.ordered_context = []
        self.pre_lines = self.init_context = self.post_lines = ''
        self.proving_envs = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        self.server.close()
        if self.proving_envs is not None:
            self.proving_envs.close()
            self.proving_envs = None

    def load_proving_envs(self):
        self.proving_envs = shelve.open(str(PROVING_ENV_PATH))
    
    def add_theorem_env(self, name, init_context, pre_lines='', post_lines=''):
        self.proving_envs[name] = (pre_lines, init_context, post_lines)

    def get_theorem_env(self, name):
        self.pre_lines, self.init_context, self.post_lines = self.proving_envs[name]
        return self.pre_lines, self.init_context, self.post_lines

    def init_search_raw(self, context=None, filename=None):
        if context is None:
            if filename is not None:
                with open(filename) as f:
                    context = f.read()
            else:
                raise ValueError('unexpected_none_context')
        results = self.server.verify_lean_file(context)
        self.search_history.add_node(results['context'], **{k : v for k, v in results.items() if k != 'context'})
        self.ordered_context = [i for i in self.ordered_context if i != results['context']] + [results['context']]
        assert self.pre_lines in results['context'] and self.post_lines in results['context']
        results['core_context'] = results['context'][len(self.pre_lines) : -len(self.post_lines)]
        return results
    
    def init_search_plain(self, name):
        self.pre_lines, self.init_context, self.post_lines = self.get_theorem_env(name)
        context = '\n'.join((self.pre_lines, self.init_context, self.post_lines))
        return self.init_search_raw(context=context)
    
    def init_search_sequential(self, name):
        self.pre_lines, self.init_context, self.post_lines = self.get_theorem_env(name)
        self.init_context = self.init_context.replace(' sorry ', ' begin  repeat { sorry }  end ')
        context = '\n'.join((self.pre_lines, self.init_context, self.post_lines))
        return self.init_search_raw(context=context)
    
    def run_tac_raw(self, context, tactic_state_id=-1, **kwargs):
        results = self.server.verify_lean_file(context)
        results.update(kwargs)
        self.search_history.add_node(results['context'], **{k : v for k, v in results.items() if k != 'context'})
        self.search_history.add_edge(self.ordered_context[tactic_state_id], results['context'], tactic=None)
        self.ordered_context = [i for i in self.ordered_context if i != results['context']] + [results['context']]
        results['core_context'] = results['context'][len(self.pre_lines) : -len(self.post_lines)]
        return results
    
    def run_tac_plain(self, context, tactic_state_id=-1, **kwargs):
        import_lines = '\n'.join(i for i in context.split('\n') if i.startswith('import'))
        context = '\n'.join(i for i in context.split('\n') if not i.startswith('import'))
        context = '\n'.join((import_lines, self.pre_lines, context, self.post_lines))
        return self.run_tac_raw(context, tactic_state_id, **kwargs)

    def run_tac_sequential(self, tactic, tactic_state_id=-1, **kwargs):
        last_context = self.ordered_context[tactic_state_id]
        sorry_pos = [i.span() for i in re.finditer(r'\brepeat { sorry }', last_context)]
        assert len(sorry_pos) == 1
        context = last_context[:sorry_pos[0][0]] + tactic.rstrip(', ') + ',  repeat { sorry } ' + last_context[sorry_pos[0][1]:]
        return self.run_tac_raw(context, tactic_state_id, **kwargs)

    def get_perfered_next_context(self):
        return self.ordered_context[-1]

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
    agent.close()

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
    agent.close()
