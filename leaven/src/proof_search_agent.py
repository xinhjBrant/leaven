from leaven.src.lean_server import LeanEnv
import shelve
import re
from pathlib import Path
import networkx as nx
from typing import Optional, Dict, Any

parent_path = Path(__file__).parent.parent.resolve()

PROVING_ENV_PATH=parent_path / 'proving_env_lib'

class ProvingSearchAgent:
    """
    This class facilitates interaction between Python and Lean 3 for theorem proving tasks.
    It supports three modes of operation: Raw Search, Plain Search, and Sequential Search, 
    each tailored for different scenarios and levels of integration with mathlib.
    """

    def __init__(self) -> None:
        """
        Initializes the ProvingSearchAgent, setting up a server connection and preparing for theorem proving tasks.
        It also initializes necessary data structures to keep track of the proving process.
        """
        self.server = None
        self.search_history = nx.DiGraph()
        self.server = LeanEnv()
        self.ordered_context = []
        self.pre_lines = self.init_context = self.post_lines = ''
        self.proving_envs = None

    def __enter__(self) -> None:
        """
        Enters a runtime context related to the ProvingSearchAgent instance.
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """
        Handles exit from the runtime context, ensuring resources are properly released.
        """
        self.close()

    def close(self) -> None:
        """
        Closes the server connection and releases any other resources held by the ProvingSearchAgent instance.
        """
        self.server.close()
        if self.proving_envs is not None:
            self.proving_envs.close()
            self.proving_envs = None

    def load_proving_envs(self) -> None:
        """
        Loads the proving environments from a shelve file named 'proving_env_lib', which contains theorems from mathlib with necessary pre and post context for proving. Ensure the shelve file is located at the specified path, else an exception will be thrown.
        """
        self.proving_envs = shelve.open(str(PROVING_ENV_PATH))
    
    def add_theorem_env(self, name: str, init_context: str, pre_lines: str = '', post_lines: str = '') -> None:
        """
        Adds a theorem environment to the shelve file, providing a way to extend the set of theorems available for proving.
        
        Parameters:
        name (str): The name of the theorem.
        init_context (str): The initial context of the theorem.
        pre_lines (str, optional): Any necessary lines of code before the theorem. Defaults to ''.
        post_lines (str, optional): Any necessary lines of code after the theorem. Defaults to ''.

        Raises:
            ValueError: If the name already exists in the proving environments.
        """
        self.proving_envs[name] = (pre_lines, init_context, post_lines)

    def get_theorem_env(self, name):
        """
        Retrieves the specified theorem environment from the shelve file.
        
        Parameters:
        name (str): The name of the theorem.
        
        Returns:
        tuple: A tuple containing pre_lines, init_context, and post_lines of the specified theorem.
        """
        self.pre_lines, self.init_context, self.post_lines = self.proving_envs[name]
        return self.pre_lines, self.init_context, self.post_lines

    def init_search_raw(self, context: Optional[str] = None, filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Initializes a Raw Search using a custom Lean code context, and returns the verification result.
        
        Parameters:
        context (str, optional): The custom Lean code to be used as context. Defaults to None.
        filename (str, optional): The file name containing Lean code if context is not provided. Defaults to None.
        
        Returns:
        dict: The verification result of the given context.

        Raises:
            ValueError: If both context and filename are None, or if the file specified by filename does not exist.
        """
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
    
    def init_search_plain(self, name: str) -> Dict[str, Any]:
        """
        Initializes a Plain Search using a specific theorem from mathlib, and returns the verification result.
        
        Parameters:
        name (str): The name of the theorem from mathlib.
        
        Returns:
        dict: The verification result of the given context.
        """
        self.pre_lines, self.init_context, self.post_lines = self.get_theorem_env(name)
        context = '\n'.join((self.pre_lines, self.init_context, self.post_lines))
        return self.init_search_raw(context=context)
    
    def init_search_sequential(self, name: str) -> Dict[str, Any]:
        """
        Initializes a Sequential Search, tailored for lean-gym tactic mode environment, using a specific theorem from 
        mathlib and returns the verification result.
        
        Parameters:
        name (str): The name of the theorem from mathlib.
        
        Returns:
        dict: The verification result of the given context.
        """
        self.pre_lines, self.init_context, self.post_lines = self.get_theorem_env(name)
        self.init_context = self.init_context.replace(' sorry ', ' begin  repeat { sorry }  end ')
        context = '\n'.join((self.pre_lines, self.init_context, self.post_lines))
        return self.init_search_raw(context=context)
    
    def run_tac_raw(self, context: str, tactic_state_id: Optional[int] = None, **kwargs: Any) -> Dict[str, Any]:
        """
        Modifies the context using a tactic in Raw Search mode and returns the verification result.
        
        Parameters:
        context (str): The Lean code context.
        tactic_state_id (int, optional): The ID of the tactic state to expand. Defaults to None.
        **kwargs: Additional parameters.
        
        Returns:
        dict: The verification result of the modified context.
        """
        if tactic_state_id is None:
            tactic_state_id = self.get_context_id_to_expand()
        results = self.server.verify_lean_file(context)
        results.update(kwargs)
        self.search_history.add_node(results['context'], **{k : v for k, v in results.items() if k != 'context'})
        self.search_history.add_edge(self.ordered_context[tactic_state_id], results['context'], tactic=None)
        self.ordered_context = [i for i in self.ordered_context if i != results['context']] + [results['context']]
        results['core_context'] = results['context'][len(self.pre_lines) : -len(self.post_lines)]
        return results
    
    def run_tac_plain(self, context: str, tactic_state_id: Optional[int] = None, **kwargs: Any) -> Dict[str, Any]:
        """
        Modifies the context using a tactic in Plain Search mode and returns the verification result.
        
        Parameters:
        context (str): The Lean code context.
        tactic_state_id (int, optional): The ID of the tactic state to expand. Defaults to None.
        **kwargs: Additional parameters.
        
        Returns:
        dict: The verification result of the modified context.
        """
        import_lines = '\n'.join(i for i in context.split('\n') if i.startswith('import'))
        context = '\n'.join(i for i in context.split('\n') if not i.startswith('import'))
        context = '\n'.join((import_lines, self.pre_lines, context, self.post_lines))
        return self.run_tac_raw(context, tactic_state_id, **kwargs)

    def run_tac_sequential(self, tactic: str, tactic_state_id: Optional[int] = None, **kwargs: Any) -> Dict[str, Any]:
        """
        Executes a tactic in Sequential Search mode, modifying the proof context sequentially and returns the verification result.
        
        Parameters:
        tactic (str): The tactic to be applied.
        tactic_state_id (int, optional): The ID of the tactic state to expand. Defaults to None.
        **kwargs: Additional parameters.
        
        Returns:
        dict: The verification result of the modified context.
        """
        if tactic_state_id is None:
            tactic_state_id = self.get_context_id_to_expand()
        context_to_expand = self.ordered_context[tactic_state_id]
        sorry_pos = [i.span() for i in re.finditer(r'\brepeat { sorry }', context_to_expand)]
        assert len(sorry_pos) == 1
        context = context_to_expand[:sorry_pos[0][0]] + tactic.rstrip(', ') + ',  repeat { sorry } ' + context_to_expand[sorry_pos[0][1]:]
        return self.run_tac_raw(context, tactic_state_id, **kwargs)

    def get_context_id_to_expand(self) -> int:
        """
        Determines the ID of the context to expand. This is used internally to manage the order of proof steps.
        The method returns -1 by default, which refers to the last context in the ordered list of contexts.

        Returns:
            int: The ID of the context to expand.
        """
        return -1

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
