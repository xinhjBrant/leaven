import os
from pathlib import Path
import subprocess
import ujson as json
# from tqdm import tqdm, trange
import re
from leaven.src.lean_server import LeanEnv
# import pickle
import networkx as nx
from leaven.src.file_tools import *

parent_path = Path(__file__).parent.parent.resolve()

lean_path_basis = [Path(p) for p in json.loads(subprocess.check_output([f_join(parent_path, 'elan', 'bin', 'lean'), '--path']).decode())['path']]
lean_paths = [p.resolve() for p in lean_path_basis if p.exists()]

def object2Map(obj:object):
    m = obj.__dict__
    for k in m.keys():
        v = m[k]
        if hasattr(v, "__dict__"):
            m[k] = object2Map(v)
    return m

class lean_file_analyser:
    def cut_path(self, path):
        if isinstance(path, str):
            path = Path(path)
        path = path.with_suffix('')
        for lean_path in self.lean_paths:
            if path.is_relative_to(lean_path):
                return '.'.join(path.relative_to(lean_path).parts)
        raise ValueError('not a valid lean file path')

    @classmethod
    def get_ast(cls, file):
        file = Path(file).resolve()
        graph = nx.DiGraph()
        if not os.path.exists(file.with_suffix('.ast.json')):
            os.system(' '.join([f_join(parent_path, 'elan', 'bin', 'lean'), "-M", "20480", "--ast", "--tsast", "--tspp -q ", str(file)]))
        with open(file.with_suffix('.ast.json'), 'r') as f:
            data = json.load(f)
        with open(file, 'r') as f:
            lines = f.readlines()
        os.remove(file.with_suffix('.ast.json'))
        if lines[-1].endswith('\n'):
            lines.append('')
        kinds = {}
        for item in data['ast']:
            if not item:
                continue
            if 'children' in item:
                item['children'] = [data['ast'][i] for i in item['children']]
            if 'kind' in item:
                if item['kind'] not in kinds:
                    kinds[item['kind']] = []
                kinds[item['kind']].append(item)
            if 'start' in item and 'end' in item:
                if item['start'][0] < item['end'][0]:
                    item['content'] = lines[item['start'][0] - 1][item['start'][1] : ] + ''.join(lines[item['start'][0] : item['end'][0] - 1]) + lines[item['end'][0] - 1][ : item['end'][1]]
                else:
                    item['content'] = lines[item['start'][0] - 1][item['start'][1] : item['end'][1]]
        graph.add_node(0, kind=None, start=None, end=None)
        cls.transverse_ast(kinds['file'], graph, 0, None, None)
        return graph
    
    @classmethod
    def transverse_ast(cls, ast, graph : nx.DiGraph, parent, start, end):
        for b in ast:
            if b is None:
                continue
            b_start = b['start'] if 'start' in b else None
            b_end = b['end'] if 'end' in b else None
            if b['kind'] in ['file', '#eval', '#reduce', '#check', 'example', 'constants', 'constant', 'variables', 'variable', 'imports', 'definition', 'theorem', 'abbreviation', 'instance', 'structure', 'inductive', 'class', 'class_inductive', 'by', '{', 'tactic', 'begin'] and ((start is None or 'start' not in b or b['start'] > start) or (end is None or 'end' not in b or b['end'] < end)):
                node_name = len(graph.nodes())
                graph.add_node(node_name, kind=b['kind'], start=b_start, end=b_end)
                graph.add_edge(parent, node_name)
                if 'children' in b:
                    cls.transverse_ast(b['children'], graph, node_name, b_start, b_end)
            elif 'children' in b:
                cls.transverse_ast(b['children'], graph, parent, b_start, b_end)


    def document_probing(self, file):
        lean_server = LeanEnv(cwd=str(Path('.').resolve()))
        with open(file, 'r') as f:
            lines = f.readlines()
        if header := re.search(r'\/-\s*Copyright(?:[^-]|-[^\/])*-\/\s*', ''.join(lines), re.DOTALL):
            start_line = header.group(0).count('\n')
        else:
            start_line = 0
        sep_lines = [sorted(list(set([m.end() for m in re.finditer(r'(?:\b|:=|,|;|:|\(|\{|\[|⟨|\.)\s*', line)]))) for line in lines]
        processed_line = {}
        last_ts = ''
        line_buffer = ''
        lean_server.reset(options={"filename": str(file)})
        for row, sep in enumerate(sep_lines):
            if row < start_line:
                continue
            line = lines[row]
            row += 1
            last_sep = 0
            if line and not sep:
                line_buffer += line
                continue
            processed_line[row] = {}
            for column in sep:
                if column == last_sep:
                    continue
                try:
                    ts = lean_server.render(options={"filename" : str(file), "line" : row, "col" : column})
                except Exception as e:
                    line_buffer += line[last_sep : column]
                    last_sep = column
                    continue
                if ts and ts != last_ts:
                    processed_line[row][column] = object2Map(ts)
                    processed_line[row][column]['precontent'] = line_buffer + line[last_sep : column]
                    last_ts = ts
                    line_buffer = ''
                else:
                    line_buffer += line[last_sep : column]
                last_sep = column
            line_buffer += line[last_sep : ]
        if len(lines) not in processed_line:
            processed_line[len(lines)] = {}
        processed_line[len(lines)][len(lines[-1])] = {'precontent' : line_buffer}
        return processed_line

    @classmethod
    def get_information(cls, file):
        file = Path(file).resolve()
        processed_line = cls.document_probing(file)
        tactics = []
        commands = []
        lemmas = []
        for l, line in processed_line.items():
            for c, column in line.items():
                if 'source' in column and column['source'] and 'file' in column['source'] and column['source']['file'] and column['source']['file'] != file:
                    if 'tactic' in column['source']['file']:
                        tactics.append([l, c, column['text']])
                    elif 'interactive' in column['source']['file']:
                        commands.append([l, c, column['text']])
                    elif 'full_id' in column and column['full_id']:
                        lemmas.append([l, c, column['full_id']])
        return tactics, commands, lemmas


if __name__ == "__main__":
    path = 'test1.lean'
    graph = lean_file_analyser.get_ast(path)
    tactics, commands, lemmas = lean_file_analyser.get_information(path)

