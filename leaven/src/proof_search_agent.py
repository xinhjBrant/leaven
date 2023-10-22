from leaven.src.lean_server import LeanEnv
import shelve
import re
from pathlib import Path
import networkx as nx
from typing import Optional, Dict, Any, Tuple, List

PROVING_ENV_PATH = Path(__file__).parent.parent.resolve() / 'proving_env_lib'

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
        self.search_history: nx.DiGraph = nx.DiGraph()
        self.server: Optional[LeanEnv] = None
        self.ordered_context: List[str] = []
        self.proving_envs: Optional[shelve.DbfilenameShelf] = None

    def __enter__(self) -> 'ProvingSearchAgent':
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
        if self.server is not None:
            self.server.close()
            self.server = None
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
        name: The name of the theorem.
        init_context: The initial context of the theorem.
        pre_lines: Any necessary lines of code before the theorem. Defaults to ''.
        post_lines: Any necessary lines of code after the theorem. Defaults to ''.

        Raises:
            ValueError: If the name already exists in the proving environments.
        """
        if self.proving_envs is None:
            raise ValueError('proving_envs_not_loaded')
        if name in self.proving_envs:
            raise ValueError(f'{name} already exists in proving environments')
        self.proving_envs[name] = (pre_lines, init_context, post_lines)

    def get_theorem_env(self, name: str) -> Tuple[str, str, str]:
        """
        Retrieves the specified theorem environment from the shelve file.
        
        Parameters:
        name: The name of the theorem.
        
        Returns:
        A tuple containing pre_lines, init_context, and post_lines of the specified theorem.
        """
        if self.proving_envs is None:
            raise ValueError('proving_envs_not_loaded')
        return self.proving_envs[name]

    def _verify_lean_file(self, context: str, pre_lines: str = '', post_lines: str = '') -> Dict[str, Any]:
        """
        Verifies a Lean file and returns the verification result.
        
        Parameters:
        context: The Lean code context.
        pre_lines: Any necessary lines of code before the context. Defaults to ''.
        post_lines: Any necessary lines of code after the context. Defaults to ''.
        
        Returns:
        The verification result of the given context.
        """
        if self.server is None:
            self.server = LeanEnv()
        results = self.server.verify_lean_file(pre_lines + context + post_lines)
        self.ordered_context = [i for i in self.ordered_context if i != results['context']] + [results['context']]
        assert pre_lines in results['context'] and post_lines in results['context']
        core_context = results['context'][len(pre_lines):-len(post_lines) if len(post_lines) > 0 else len(results['context'])]
        results.update({'core_context': core_context,
                        'pre_lines': pre_lines,
                        'post_lines': post_lines,
                        'tactic_state_id': len(self.ordered_context) - 1})
        self.search_history.add_node(results['context'], **{k : v for k, v in results.items() if k != 'context'})
        return results
    
    def get_node_properties(self, tactic_state_id: int) -> Dict[str, Any]:
        """
        Returns the pre_lines and core_context properties of the node corresponding to the given tactic_state_id.
        
        Parameters:
        tactic_state_id: The ID of the tactic state.
        
        Returns:
        A dictionary containing the pre_lines and core_context properties of the node.
        """
        node = self.search_history.nodes[self.ordered_context[tactic_state_id]]
        return {'pre_lines': node['pre_lines'], 'core_context': node['core_context'], 'post_lines': node['post_lines']}
    
    def get_context_properties(self, pre_lines: Optional[str], post_lines: Optional[str], tactic_state_id: int) -> Tuple[str, str]:
        """
        Returns the pre_lines and post_lines properties of the node corresponding to the given tactic_state_id.
        If pre_lines or post_lines is None, retrieves the properties from the node using get_node_properties.
        
        Parameters:
        pre_lines: Any necessary lines of code before the context. Defaults to None.
        post_lines: Any necessary lines of code after the context. Defaults to None.
        tactic_state_id: The ID of the tactic state.
        
        Returns:
        A tuple containing the pre_lines and post_lines properties of the node.
        """
        if pre_lines is None or post_lines is None:
            node_props = self.get_node_properties(tactic_state_id)
            if pre_lines is None:
                pre_lines = node_props['pre_lines']
            if post_lines is None:
                post_lines = node_props['post_lines']
        return pre_lines, post_lines

    def init_search_raw(self, context: Optional[str] = None, filename: Optional[str] = None, pre_lines: str = '', post_lines: str = '') -> Dict[str, Any]:
        """
        Initializes a Raw Search using a custom Lean code context, and returns the verification result.
        
        Parameters:
        context: The custom Lean code to be used as context. Defaults to None.
        filename: The file name containing Lean code if context is not provided. Defaults to None.
        pre_lines: Any necessary lines of code before the context. Defaults to ''.
        post_lines: Any necessary lines of code after the context. Defaults to ''.
        
        Returns:
        The verification result of the given context.

        Raises:
            ValueError: If both context and filename are None, or if the file specified by filename does not exist.
        """
        if context is None:
            if filename is not None:
                with open(filename) as f:
                    context = f.read()
                    pre_lines = post_lines = ''
            else:
                raise ValueError('unexpected_none_context')
        if self.server is not None:
            self.server.close()
        self.search_history = nx.DiGraph()
        self.server = LeanEnv()
        results = self._verify_lean_file(context=context, pre_lines=pre_lines or '', post_lines=post_lines or '')
        return results
    
    def init_search_plain(self, name: str) -> Dict[str, Any]:
        """
        Initializes a Plain Search using a specific theorem from mathlib, and returns the verification result.
        
        Parameters:
        name: The name of the theorem from mathlib.
        
        Returns:
        The verification result of the given context.
        """
        pre_lines, init_context, post_lines = self.get_theorem_env(name)
        return self.init_search_raw(context=init_context, pre_lines=pre_lines, post_lines=post_lines)
    
    def init_search_sequential(self, name: str) -> Dict[str, Any]:
        """
        Initializes a Sequential Search, tailored for lean-gym tactic mode environment, using a specific theorem from 
        mathlib and returns the verification result.
        
        Parameters:
        name: The name of the theorem from mathlib.
        
        Returns:
        The verification result of the given context.
        """
        pre_lines, init_context, post_lines = self.get_theorem_env(name)
        context = re.sub(r'\bsorry\b', 'begin\n repeat { sorry } \nend', init_context)
        return self.init_search_raw(context=context, pre_lines=pre_lines, post_lines=post_lines)
    
    def run_tac_raw(self, context: str, pre_lines: str = '', post_lines: str = '', tactic_state_id: Optional[int] = None, **kwargs: Any) -> Dict[str, Any]:
        """
        Modifies the context using a tactic in Raw Search mode and returns the verification result.
        
        Parameters:
        context: The Lean code context.
        pre_lines: Any necessary lines of code before the context. Defaults to ''.
        post_lines: Any necessary lines of code after the context. Defaults to ''.
        tactic_state_id: The ID of the tactic state to expand. Defaults to None.
        **kwargs: Additional parameters.
        
        Returns:
        The verification result of the modified context.
        """
        if tactic_state_id is None:
            tactic_state_id = self.get_context_id_to_expand()
        pre_lines, post_lines = self.get_context_properties(pre_lines, post_lines, tactic_state_id)
        results = self._verify_lean_file(context=context, pre_lines=pre_lines, post_lines=post_lines)
        results.update(kwargs)
        self.search_history.add_edge(self.ordered_context[tactic_state_id], results['context'], tactic=None)
        return results
    
    def run_tac_plain(self, context: str, tactic_state_id: Optional[int] = None, **kwargs: Any) -> Dict[str, Any]:
        """
        Modifies the context using a tactic in Plain Search mode and returns the verification result.
        
        Parameters:
        context: The Lean code context.
        tactic_state_id: The ID of the tactic state to expand. Defaults to None.
        **kwargs: Additional parameters.
        
        Returns:
        The verification result of the modified context.
        """
        if tactic_state_id is None:
            tactic_state_id = self.get_context_id_to_expand()
        node_props = self.get_node_properties(tactic_state_id)
        pre_lines, post_lines = node_props['pre_lines'], node_props['post_lines']
        pre_lines = '\n'.join([i for i in context.split('\n') if i.startswith('import')] + [pre_lines])
        context = '\n'.join(i for i in context.split('\n') if not i.startswith('import'))
        return self.run_tac_raw(context=context, pre_lines=pre_lines, post_lines=post_lines, tactic_state_id=tactic_state_id, **kwargs)

    def run_tac_sequential(self, tactic: str, tactic_state_id: Optional[int] = None, **kwargs: Any) -> Dict[str, Any]:
        """
        Executes a tactic in Sequential Search mode, modifying the proof context sequentially and returns the verification result.
        
        Parameters:
        tactic: The tactic to be applied.
        tactic_state_id: The ID of the tactic state to expand. Defaults to None.
        **kwargs: Additional parameters.
        
        Returns:
        The verification result of the modified context.
        """
        if tactic_state_id is None:
            tactic_state_id = self.get_context_id_to_expand()
        node_props = self.get_node_properties(tactic_state_id)
        pre_lines, context_to_expand, post_lines = node_props['pre_lines'], node_props['core_context'], node_props['post_lines']
        sorry_pos = [i.span() for i in re.finditer(r'\brepeat { sorry }', context_to_expand)]
        assert len(sorry_pos) == 1
        context = context_to_expand[:sorry_pos[0][0]] + tactic.rstrip(', ') + ',  repeat { sorry } ' + context_to_expand[sorry_pos[0][1]:]
        return self.run_tac_raw(context=context, tactic_state_id=tactic_state_id, pre_lines=pre_lines, post_lines=post_lines, **kwargs)
    
    def get_context_id_to_expand(self) -> int:
        """
        Determines the ID of the context to expand. This is used internally to manage the order of proof steps.
        The method returns the index of the last context in the ordered list of contexts.

        Returns:
            The ID of the context to expand.
        """
        return len(self.ordered_context) - 1

def add_mathlib_data():
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
