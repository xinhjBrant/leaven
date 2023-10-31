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

leaven_path = Path(__file__).parent.parent.resolve()
src_path = Path(__file__).parent.resolve()

def get_name_from_leanpkg_path(p: Path) -> str:
  """ get the package name corresponding to a source path """
  # lean core?
  if p.parts[-5:] == Path('bin/../lib/lean/library').parts:
    return "core"
  if p.parts[-3:] == Path('bin/../library').parts:
    return "core"
  return '<unknown>'

lean_path_basis = [Path(p) for p in json.loads(subprocess.check_output([f_join(leaven_path, 'elan', 'bin', 'lean'), '--path'], cwd=leaven_path).decode())['path']]
path_info = [(p.resolve(), get_name_from_leanpkg_path(p)) for p in lean_path_basis]
lean_paths = [p.resolve() for p in lean_path_basis if p.exists()]

def cut_path(path):
    if isinstance(path, str):
        path = Path(path)
    path = path.with_suffix('')
    for lean_path in lean_paths:
        if path.is_relative_to(lean_path):
            return '.'.join(path.relative_to(lean_path).parts)
    return None

def object2Map(obj:object):
    m = obj.__dict__
    for k in m.keys():
        v = m[k]
        if hasattr(v, "__dict__"):
            m[k] = object2Map(v)
    return m

class GrammarRegistry:
    def __init__(self, grammarPath, multiline=True):
        self.multiline = multiline
        grammar = json.load(open(grammarPath,'r',encoding="utf-8"))
        assert isinstance(grammar['scopeName'], str) and grammar['scopeName'], f"Grammar missing required scopeName property: #{grammarPath}"
        self.begin_pattern = None
        self.end_pattern = None
        self.scope_name = grammar['scopeName']
        self.content_name = None
        self.patterns = [Rule(pat) for pat in grammar['patterns']]
        self.repository = {key : Rule(value) for key, value in grammar['repository'].items()} if 'repository' in grammar else None
        self.redirect_patterns(self.repository)
        self.redirect_patterns(self.patterns)

    def redirect_patterns(self, patterns):
        assert isinstance(patterns, (list, dict))
        for idx, rule in (enumerate(patterns) if isinstance(patterns, list) else patterns.items()):
            if isinstance(rule.include, str) and rule.include[1:] in self.repository:
                patterns[idx] = self.repository[rule.include[1:]]
                # if rule.match_first:
                #     patterns[idx].match_first = rule.match_first
            elif rule.patterns:
                self.redirect_patterns(rule.patterns)
    
    def get_begin_pattern(self, rule):
        if rule.begin_pattern:
            return [[rule.begin_pattern , rule]]
        elif rule.patterns:
            begin_pattern = []
            for pattern in rule.patterns:
                begin_pattern.extend(self.get_begin_pattern(pattern))
            return begin_pattern
        else:
            return []
        
    def tokenize_line(self, line, repository=None):
        tagging_log = []
        self.rule_apply(self if not repository else self.repository[repository], line, [], tagging_log)
        return tagging_log

    # def get_next_regex(self, line, begin_patterns):
    #     starts = []

    def rule_apply(self, rule, line, scope_name, tagging_log, outside_end=[]):
        begin_patterns = [(re.compile(item[0], re.M) if self.multiline else re.compile(item[0]), item[1]) for pattern in rule.patterns for item in self.get_begin_pattern(pattern) if not pattern.match_first] if rule.patterns else []
        match_first_patterns = [(re.compile(item[0], re.M) if self.multiline else re.compile(item[0]), item[1]) for pattern in rule.patterns for item in self.get_begin_pattern(pattern) if pattern.match_first] if rule.patterns else []
        # if not begin_patterns:
        #     return 0
        pos = 0
        last_pos = 0
        scope_name = scope_name + ([rule.scope_name] if rule.scope_name else [])
        content_name = scope_name + ([rule.content_name] if rule.content_name else [])
        # end_pattern = re.compile(rule.end_pattern) if rule.end_pattern else None
        if rule.end_pattern:
            end_pattern = re.compile(rule.end_pattern, re.M) if self.multiline else re.compile(rule.end_pattern)
        if outside_end:
            outside_end_pattern = [re.compile(i, re.M) if self.multiline else re.compile(i) for i in outside_end]
        while pos < len(line):
            matched = False
            for regex_pattern, pattern in match_first_patterns:
                if (match := regex_pattern.match(line, pos)):
                    matched = True
                    # if '#upper_ending' == pattern.end_pattern:
                    #     match = match
                    if pos > last_pos:
                        tagging_log.append([line[last_pos : pos], content_name])
                    if pattern.end_pattern:
                        if match.start() < match.end():
                            tagging_log.extend(self.capture(line[match.start() : match.end()], pos, match.regs, content_name + ([pattern.scope_name] if pattern.scope_name else []), pattern.begin_captures if pattern.begin_captures else {}))
                        if pattern.soft_ending:
                            apply_result = self.rule_apply(pattern, line[match.end():], content_name, tagging_log, ([rule.end_pattern] if rule.end_pattern else []) + outside_end)
                            last_pos = pos = match.end() + apply_result
                        else:
                            last_pos = pos = match.end() + self.rule_apply(pattern, line[match.end():], content_name, tagging_log)
                    else:
                        if pattern.soft_ending and rule.end_pattern and end_pattern.search(match.group(1)):
                            matched = False
                            continue
                        if match.start() < match.end():
                            tagging_log.extend(self.capture(line[match.start() : match.end()], pos, match.regs, content_name + ([pattern.scope_name] if pattern.scope_name else []), pattern.begin_captures if pattern.begin_captures else {}))
                        # tagging_log.append([line[pos + match.start() : pos + match.end()], content_name + [pattern.scope_name]])
                        last_pos = pos = match.end()
                    break
            if matched:
                continue
            if rule.end_pattern and rule.end_pattern != '#just_matching_upper_ending' and (match := end_pattern.match(line, pos)):
                if pos > last_pos:
                    tagging_log.append([line[last_pos : pos], content_name])
                if match.start() < match.end():
                    tagging_log.extend(self.capture(line[match.start() : match.end()], pos, match.regs, scope_name, rule.end_captures if rule.end_captures else {}))
                return match.end()
            if outside_end:
                for pattern in outside_end_pattern:
                    if pattern.match(line, pos):
                        if line[last_pos : pos]:
                            tagging_log.append([line[last_pos : pos], content_name])
                        return pos
            for regex_pattern, pattern in begin_patterns:
                if (match := regex_pattern.match(line, pos)):
                    matched = True
                    # if '#upper_ending' == pattern.end_pattern:
                    #     match = match
                    if pattern.end_pattern:
                        if pos > last_pos:
                            tagging_log.append([line[last_pos : pos], content_name])
                        if match.start() < match.end():
                            tagging_log.extend(self.capture(line[match.start() : match.end()], pos, match.regs, content_name + ([pattern.scope_name] if pattern.scope_name else []), pattern.begin_captures if pattern.begin_captures else {}))
                        if pattern.soft_ending:
                            apply_result = self.rule_apply(pattern, line[match.end():], content_name, tagging_log, ([rule.end_pattern] if rule.end_pattern else []) + outside_end)
                            last_pos = pos = match.end() + apply_result
                        else:
                            last_pos = pos = match.end() + self.rule_apply(pattern, line[match.end():], content_name, tagging_log)
                    else:
                        if pattern.soft_ending and rule.end_pattern and end_pattern.search(match.group()):
                            matched = False
                            continue
                        if pos > last_pos:
                            tagging_log.append([line[last_pos : pos], content_name])
                        if match.start() < match.end():
                            tagging_log.extend(self.capture(line[match.start() : match.end()], pos, match.regs, content_name + ([pattern.scope_name] if pattern.scope_name else []), pattern.begin_captures if pattern.begin_captures else {}))
                        # tagging_log.append([line[pos + match.start() : pos + match.end()], content_name + [pattern.scope_name]])
                        last_pos = pos = match.end()
                    break
            if not matched:
                pos += 1
        if pos > last_pos:
            tagging_log.append([line[last_pos : pos], content_name])
        return pos

    def capture(self, line, pos, regs, scope_name, captures):
        tags = [scope_name.copy() for token in line]
        for i in captures:
            for idx in range(len(line)):
                if regs[int(i)][0] <= pos + idx and pos + idx < regs[int(i)][1] and captures[i]['name']:
                    tags[idx].append(captures[i]['name'])
        tokenized = []
        flag = 0
        last_tag = tags[0]
        for idx, tag in enumerate(tags):
            if not tag == last_tag:
                tokenized.append([line[flag : idx], last_tag])
                flag = idx
                last_tag = tag
        tokenized.append([line[flag : len(tags)], last_tag])
        return tokenized


class Rule:
    def __init__(self, pattern):
        self.include = pattern['include'] if 'include' in pattern and pattern['include'] else None
        self.scope_name = pattern['name'] if 'name' in pattern and pattern['name'] else None
        self.begin_captures = pattern['beginCaptures'] if 'beginCaptures' in pattern and pattern['beginCaptures'] else None
        self.end_captures = pattern['endCaptures'] if 'endCaptures' in pattern and pattern['endCaptures'] else None
        self.end_pattern = None
        self.content_name = pattern['contentName'] if 'contentName' in pattern and pattern['contentName'] else None
        self.begin_pattern = None
        self.soft_ending = pattern["soft_ending"] if "soft_ending" in pattern else False
        self.match_first = pattern["match_first"] if "match_first" in pattern else False
        if 'match' in pattern:
            self.begin_pattern = pattern['match']
        elif 'begin' in pattern:
            assert 'end' in pattern
            self.begin_pattern = pattern['begin']
            # if self.begin_captures:
            #     self.captures = self.begin_captures
            self.end_pattern = pattern['end']
        self.patterns = [Rule(pat) for pat in pattern['patterns']] if 'patterns' in pattern and pattern['patterns'] else None

class Tokenizer:
    @classmethod
    def parsing_declarations(cls, lines):
        file_grammar = GrammarRegistry(src_path / 'lean_syntax/lean_grammar.json')
        lines = file_grammar.tokenize_line(lines)
        lines = [[line[0], line[1][1:]] for line in lines]
        modifier_flag = []
        last_flag = 0
        blocks = []
        for idx, line in enumerate(lines):
            if not line[0].strip():
                continue
            else:
                break
        lines = lines[idx:]
        for idx, line in enumerate(lines):
            name = ','.join(line[1])
            if not line[0].strip() and modifier_flag and modifier_flag[-1] + 1 == idx:
                modifier_flag.append(idx)
            elif 'modifier' in name or "block.documentation" in name:
                if not idx or ((('modifier' in name or "block.documentation" in name)) and modifier_flag and modifier_flag[-1] + 1 == idx):
                    modifier_flag.append(idx)
                else:
                    head = ["documentation"] if "codeblock.keyword.documentation" in ','.join([','.join(tree[1]) for tree in lines[last_flag:idx]]) else [tree[0] for tree in lines[last_flag:idx] if 'codeblock' in ','.join(tree[1]) or 'keyword.end' in ','.join(tree[1])]
                    if not head and 'storage.modifier.attribute' in  ','.join([','.join(tree[1]) for tree in lines[last_flag:idx]]):
                        head = ['attribute']
                    blocks.append(head + lines[last_flag:idx])
                    last_flag = idx
                    modifier_flag = [idx]
            elif not idx:
                continue
            elif ('codeblock' in name and not modifier_flag) or 'keyword.end' in name:
                head = ["documentation"] if "codeblock.keyword.documentation" in ','.join([','.join(tree[1]) for tree in lines[last_flag:idx]]) else [tree[0] for tree in lines[last_flag:idx] if 'codeblock' in ','.join(tree[1]) or 'keyword.end' in ','.join(tree[1])]
                if not head and 'storage.modifier.attribute' in  ','.join([','.join(tree[1]) for tree in lines[last_flag:idx]]):
                    head = ['attribute']
                blocks.append(head + lines[last_flag:idx])
                last_flag = idx
            elif ('codeblock' in name and modifier_flag[-1] + 1 < idx) or 'keyword.end' in name:
                head = ["documentation"] if "codeblock.keyword.documentation" in ','.join([','.join(tree[1]) for tree in lines[last_flag:idx]]) else [tree[0] for tree in lines[last_flag:idx] if 'codeblock' in ','.join(tree[1]) or 'keyword.end' in ','.join(tree[1])]
                if not head and 'storage.modifier.attribute' in  ','.join([','.join(tree[1]) for tree in lines[last_flag:idx]]):
                    head = ['attribute']
                blocks.append(head + lines[last_flag:idx])
                last_flag = idx
                modifier_flag = []
            elif 'codeblock' in name and modifier_flag:
                modifier_flag = []
        head = ["documentation"] if "codeblock.keyword.documentation" in ','.join([','.join(tree[1]) for tree in lines[last_flag:idx]]) else [tree[0] for tree in lines[last_flag:len(lines)] if 'codeblock' in ','.join(tree[1]) or 'keyword.end' in ','.join(tree[1])]
        blocks.append(head + lines[last_flag:len(lines)])
        return blocks

    @classmethod
    def parsing_steps(cls, text : str, remove_modifier=False, tbar=None, max_width=10, mode=''):
        def split_block(l: list):
            target = []
            flag = 0
            quantifiers = []
            for idx, tree in enumerate(l):
                if isinstance(tree, list) and tree[0] == 'separator':
                    if quantifiers:
                        quantifiers = [flag - 1] + quantifiers
                        quantifier_blocks = decoupling(l[quantifiers[-1] + 1 : idx])
                        for id_q in range(len(quantifiers) - 1, 0, -1):
                            quantifier_blocks = decoupling(l[quantifiers[id_q - 1] + 1 : quantifiers[id_q]]) + l[quantifiers[id_q]][1] + [quantifier_blocks]
                        target.extend([quantifier_blocks] + tree[1])
                        quantifiers = []
                    else:
                        target.extend([decoupling(l[flag: idx])] + tree[1])
                    flag = idx + 1
                elif isinstance(tree, list) and tree[0] in ['quantifiers.separator', 'proofstep_decl.separator']:
                    quantifiers.append(idx)
            if l[flag: len(l)]:
                if quantifiers:
                    quantifiers = [flag - 1] + quantifiers
                    quantifier_blocks = decoupling(l[quantifiers[-1] + 1 : len(l)])
                    for id_q in range(len(quantifiers) - 1, 0, -1):
                        quantifier_blocks = decoupling(l[quantifiers[id_q - 1] + 1 : quantifiers[id_q]]) + l[quantifiers[id_q]][1] + [quantifier_blocks]
                    target.extend([quantifier_blocks])
                else:
                    target.extend([decoupling(l[flag: len(l)])])
            target = [i for i in target if i]
            return target

        def decoupling(l: list):
            for idx in range(len(l)):
                if isinstance(l[idx], list):
                    if len(l[idx]) == 2 and isinstance(l[idx][0], str) and isinstance(l[idx][1], list):
                        if l[idx][0] in ['calc_block', 'term_block', 'proof_block', 'paren_block', 'goal_block', 'tactic_block']:
                            l[idx] = split_block(l[idx][1])
                        else:
                            l[idx] = decoupling(l[idx][1])
                    else:
                        l[idx] = decoupling(l[idx])
            l = [i for i in l if i]
            return l
        
        def dfs_print(fraction : list):
            target = []
            for i in range(len(fraction)):
                if isinstance(fraction[i], list):
                    target.extend(dfs_print(fraction[i]))
                else:
                    target.append(fraction[i])
            return target
        
        def find_n(left : list, n : int):
            if n == 0:
                return [[]]
            return [[i] for i in left] if n <= 1 else [[l] + result for idx, l in enumerate(left) for result in find_n(left[idx + 1 :], n - 1)]

        def dfs_sorry_calc(theorem : list, buffer : list, tactic_words : list, tbar, mode):
            if tbar:
                tbar.total += 1
            if not theorem:
                return {'' : []}
            fraction = {}
            if buffer and theorem in [i[0] for i in buffer]:
                return buffer[[i[0] for i in buffer].index(theorem)][1]
            sep = [i - 1 for i, line in enumerate(theorem) if isinstance(line, str) and line.strip() == '...']
            sep.reverse()
            if not len(sep) > 0:
                sep = sep
            local_answer = ''.join([j if isinstance(j, str) else ' sorry ' for j in theorem])
            fraction[local_answer] = [dfs_sorry(j, buffer, tactic_words, tbar, mode=mode) for j in theorem if isinstance(j, list)]
            if len(sep) > 1:
                for i in sep[1 : ]:
                    local_answer = ''.join([j if isinstance(j, str) else ' sorry ' for j in theorem[ : i]] + [' sorry '] + [j if isinstance(j, str) else ' sorry ' for j in theorem[sep[0] + 1 : ]])
                    if not local_answer in fraction:
                        fraction[local_answer] = [dfs_sorry(j, buffer, tactic_words, tbar, mode=mode) for j in theorem[ : i] if isinstance(j, list)] + [dfs_sorry_calc(theorem[i : sep[0] + 1], buffer, tactic_words, tbar, mode=mode)] + [dfs_sorry(j, buffer, tactic_words, tbar, mode=mode) for j in theorem[sep[0] + 1 : ] if isinstance(j, list)]
                    continue
            buffer.append((theorem, fraction))
            fraction[''.join(dfs_print(theorem))] = []
            if tbar:
                tbar.update()
            return fraction

        def sequential_sorry(theorem : list, tactic_mode=False):
            fraction = []
            while semicolon := [idx for idx, i in enumerate(theorem) if isinstance(i, str) and i.strip() == ';' and idx != (len(theorem) - 1) and idx != 0]:
                theorem[semicolon[0] - 1 : semicolon[0] + 2] = [[''.join(dfs_print(theorem[semicolon[0] - 1 : semicolon[0] + 2]))]]
            for i in range(len(theorem)):
                if isinstance(theorem[i], list):
                    plain_text = ''.join(dfs_print(theorem[i]))
                    if not plain_text.strip():
                        theorem[i] = plain_text
                        continue
                    if not tactic_mode and len([i for i in theorem if not isinstance(i, str) or i.strip()]) == 1:
                        return sequential_sorry(theorem[i])
                    if tactic_mode and len([i for i in theorem if not isinstance(i, str) or i.strip()]) == 1 and len([i for i in theorem[0] if i == '{']) == len([i for i in theorem[0] if i == '}']) == 1:
                        left = theorem[0].index('{')
                        right = theorem[0].index('}')
                        assert left + 2 == right
                        return sequential_sorry(theorem[0][left + 1], tactic_mode)
                    if not tactic_mode and i > 0 and theorem[i - 1].strip() == 'begin' and theorem[i + 1].strip() == 'end' and len([line for line in theorem if isinstance(line, list)]) == 1:
                        return sequential_sorry(theorem[i], tactic_mode=True)
                    if not tactic_mode and i > 0 and theorem[i - 1].strip() == 'by' and len([line for line in theorem if isinstance(line, list)]) == 1:
                        return sequential_sorry(theorem[i], tactic_mode=True)
                    if tactic_mode:
                        fraction.append(''.join(dfs_print(theorem[i])).strip())
            if not tactic_mode and not fraction:
                fraction = ['exact (' + ''.join(dfs_print(theorem)).strip() + ')']
            return fraction
        
        def dfs_sorry(theorem : list, buffer : list, tactic_words, tbar=None, MAX_WIDTH=30, mode=''):
            if tbar:
                tbar.total += 1
            brackets =  [
                ["⁅", "⁆"], ["⁽", "⁾"], ["₍", "₎"], ["〈", "〉"], ["⟮", "⟯"], ["⎴", "⎵"], ["⟅", "⟆"], ["⟦", "⟧"], ["⟪", "⟫"], ["⦃", "⦄"], ["〈", "〉"], ["《", "》"], ["‹", "›"], ["«", "»"], ["「", "」"], ["『", "』"], ["【", "】"], ["〔", "〕"], ["〖", "〗"], ["〚", "〛"], ["︵", "︶"], ["︷", "︸"], ["︹", "︺"], ["︻", "︼"], ["︽", "︾"], ["︿", "﹀"], ["﹁", "﹂"], ["﹃", "﹄"], ["﹙", "﹚"], ["﹛", "﹜"], ["﹝", "﹞"], ["（", "）"], ["［", "］"], ["｛", "｝"], ["｢", "｣"]
            ]
            if not theorem:
                return {'' : []}
            if not ''.join(dfs_print(theorem)).strip():
                return {''.join(dfs_print(theorem)): []}
            if buffer and theorem in [i[0] for i in buffer]:
                return buffer[[i[0] for i in buffer].index(theorem)][1]
            fraction = {''.join(dfs_print(theorem)): []}
            if mode == 'whole':
                return fraction
            if not mode:
                start = [idx for idx, i in enumerate(theorem) if isinstance(i, list) or i.strip()][0]
                if (not [i for i in theorem if isinstance(i, str) and i.strip() in [',', ';', ':=']] and 
                        ((isinstance(theorem[start], list) and len(theorem) > 1) or 
                            (isinstance(theorem[start], str) and re.match(r"(?![λΠΣ])@{0,1}[_a-zA-Zα-ωΑ-Ωϊ-ϻἀ-῾℀-⅏𝒜-𝖟](?:(?![λΠΣ])[_a-zA-Zα-ωΑ-Ωϊ-ϻἀ-῾℀-⅏𝒜-𝖟0-9'ⁿ-₉ₐ-ₜᵢ-ᵪ])*(\.(?![λΠΣ])[_a-zA-Zα-ωΑ-Ωϊ-ϻἀ-῾℀-⅏𝒜-𝖟0-9](?:(?![λΠΣ])[_a-zA-Zα-ωΑ-Ωϊ-ϻἀ-῾℀-⅏𝒜-𝖟0-9'ⁿ-₉ₐ-ₜᵢ-ᵪ])*)*", theorem[start].strip()) and not theorem[start].split()[0].strip() in tactic_words + ['by', 'from', 'begin', 'calc', 'if', 'then', 'else']))):
                    cleaned_theorem = []
                    for tree in theorem:
                        if isinstance(tree, str):
                            if tree.startswith('.'):
                                cleaned_theorem[-1] = ''.join(dfs_print(cleaned_theorem[-1])) + tree.split()[0]
                                cleaned_theorem.extend(tree.split()[1:])
                            elif tree:
                                cleaned_theorem.extend(tree.split())
                        else:
                            cleaned_theorem.append(tree)
                    cleaned_theorem = [i for i in cleaned_theorem if i]
                    plain_text = ''.join(dfs_print(theorem))
                    if (len(cleaned_theorem) == 1 and isinstance(cleaned_theorem[0], str)) or len(cleaned_theorem) > 6:
                        fraction =  {plain_text : []}
                        buffer.append((theorem, fraction))
                        return fraction
                    if isinstance(cleaned_theorem[0], list):
                        heads = dfs_sorry(cleaned_theorem[0], buffer, tactic_words, tbar)
                    else:
                        heads = {cleaned_theorem[0] : []}
                    flags = [i for j in range(len(cleaned_theorem)) for i in find_n(list(range(1, len(cleaned_theorem))), j)]
                    blanks = plain_text.split(plain_text.strip())
                    assert len(blanks) == 2
                    fraction = {
                        blanks[0] + ' '.join([head] + [' sorry ' if idx in flag else ''.join(dfs_print(line)) for idx, line in enumerate(cleaned_theorem) if idx > 0]) + blanks[1] : 
                        heads[head] + [dfs_sorry(cleaned_theorem[idx], buffer, tactic_words, tbar) if isinstance(cleaned_theorem[idx], list) else {cleaned_theorem[idx] : []} for idx, line in enumerate(cleaned_theorem) if idx in flag]
                        for flag in flags for head in heads
                    }
                    if MAX_WIDTH and len(fraction) > MAX_WIDTH:
                        for frac_name in list(fraction.keys())[1 : -MAX_WIDTH + 1]:
                            del fraction[frac_name]
                    buffer.append((theorem, fraction))
                    return fraction
                for i in range(len(theorem) -1, -1, -1):
                    if isinstance(theorem[i], list):
                        plain_text = ''.join(dfs_print(theorem[i]))
                        if not plain_text.strip():
                            theorem[i] = plain_text
                            continue
                        first_in_this = [i if not isinstance(i, str) else i.split()[0] for i in theorem[i] if not isinstance(i, str) or i.strip()][0]
                        if isinstance(first_in_this, str) and (
                            (first_in_this in ['have', 'haveI'] and not (':' in theorem[i] and ':=' in theorem[i])) or
                            first_in_this in ([i[0] for i in brackets] + ['[']) or
                            first_in_this in tactic_words and ('intro' in first_in_this or 'assume' in first_in_this or (('let' == first_in_this or 'letI' == first_in_this) and 'in' not in theorem[i]) or 'refine' in first_in_this or 'obtain' in first_in_this or 'case' in first_in_this)
                            ):
                            theorem[i] = [''.join(dfs_print(theorem[i]))]
                        for sub_idx, sub_tree in enumerate(theorem[i]):
                            if (isinstance(sub_tree[0], str) and sub_tree[0].strip() == '[') or (isinstance(sub_tree[0][0], str) and sub_tree[0][0].strip() == '['):
                                theorem[i][sub_idx] = ''.join(dfs_print(sub_tree))
                        if isinstance(first_in_this, str) and first_in_this == '⟨' and len(theorem[i]) == 3:
                            theorem[i][1 : 2] = theorem[i][1]
                        postfix = len(theorem)
                        for j in range(len(theorem) - 1, i, -1):
                            if isinstance(theorem[j], str) and (not theorem[j].strip() or theorem[j].strip() in ['end', '}', ',', '⟩', ')', ';', ']']):
                                postfix = j
                                continue
                            else:
                                break
                        if isinstance(first_in_this, str) and first_in_this  == 'calc':
                            calc_answer = dfs_sorry_calc(theorem[i], buffer, tactic_words, tbar, mode=mode)
                            following = dfs_sorry(theorem[i + 1 : postfix], buffer, tactic_words, tbar)
                            for c_answer in calc_answer:
                                for answer in following:
                                    local_answer = ''.join(dfs_print(theorem[ : i]) + [c_answer, answer] + dfs_print(theorem[postfix : ]))
                                    if not local_answer in fraction:
                                        fraction[local_answer] = calc_answer[c_answer] + following[answer]
                        elif (isinstance(first_in_this, str) and 
                            # (first_in_this in tactic_words + ['{', 'by', 'begin', '⟨', 'if'] or ':=' in theorem[i]) and 
                            ' sorry ' in (mask := [' sorry ' if isinstance(line, list) else line for line in theorem[i]])):
                            following = dfs_sorry(theorem[i + 1 : postfix], buffer, tactic_words, tbar)
                            for answer in following:
                                local_answer = ''.join(
                                    dfs_print(theorem[ : i]) + 
                                    mask + 
                                    [answer] + 
                                    dfs_print(theorem[postfix : ])
                                    )
                                if not local_answer in fraction:
                                    fraction[local_answer] = [dfs_sorry(line, buffer, tactic_words, tbar) for line in theorem[i] if isinstance(line, list)] + following[answer]
                        elif len([i for i in theorem if not isinstance(i, str) or i.strip()]) == 1 and isinstance(theorem[0], list):
                            fraction = dfs_sorry(theorem[0], buffer, tactic_words, tbar)
                            buffer.append((theorem, fraction))
                            return fraction
                        if i > 0:
                            local_answer = ''.join(dfs_print(theorem[ : i]) + [' sorry '] + dfs_print(theorem[postfix : ]))
                            # if 'calc' in ''.join(dfs_print(theorem[ : i])):
                            #     raise Exception()
                            if not local_answer in fraction:
                                fraction[local_answer] = [dfs_sorry(theorem[i : postfix], buffer, tactic_words, tbar)]
                        if MAX_WIDTH and len(fraction) > MAX_WIDTH:
                            for frac_name in list(fraction.keys())[1 : -MAX_WIDTH + 1]:
                                del fraction[frac_name]
            elif mode == 'block':
                for i in range(len(theorem) -1, -1, -1):
                    if isinstance(theorem[i], list):
                        plain_text = ''.join(dfs_print(theorem[i]))
                        if not plain_text.strip():
                            theorem[i] = plain_text
                            continue
                        first_in_this = [i if not isinstance(i, str) else i.split()[0] for i in theorem[i] if not isinstance(i, str) or i.strip()][0]
                        if isinstance(first_in_this, str) and (
                            (first_in_this in ['have', 'haveI'] and not (':' in theorem[i] and ':=' in theorem[i])) or
                            first_in_this in ([i[0] for i in brackets] + ['[']) or
                            first_in_this in tactic_words and ('intro' in first_in_this or 'assume' in first_in_this or (('let' == first_in_this or 'letI' == first_in_this) and 'in' not in theorem[i]) or 'refine' in first_in_this or 'obtain' in first_in_this or 'case' in first_in_this)
                            ):
                            theorem[i] = [''.join(dfs_print(theorem[i]))]
                        for sub_idx, sub_tree in enumerate(theorem[i]):
                            if (isinstance(sub_tree[0], str) and sub_tree[0].strip() == '[') or (isinstance(sub_tree[0][0], str) and sub_tree[0][0].strip() == '['):
                                theorem[i][sub_idx] = ''.join(dfs_print(sub_tree))
                        if isinstance(first_in_this, str) and first_in_this == '⟨' and len(theorem[i]) == 3:
                            theorem[i][1 : 2] = theorem[i][1]
                        postfix = len(theorem)
                        for j in range(len(theorem) - 1, i, -1):
                            if isinstance(theorem[j], str) and (not theorem[j].strip() or theorem[j].strip() in ['end', '}', ',', '⟩', ')', ';', ']']):
                                postfix = j
                                continue
                            else:
                                break
                        if isinstance(first_in_this, str) and first_in_this  == 'calc':
                            calc_answer = dfs_sorry_calc(theorem[i], buffer, tactic_words, tbar, mode=mode)
                            following = dfs_sorry(theorem[i + 1 : postfix], buffer, tactic_words, tbar, mode=mode)
                            for c_answer in calc_answer:
                                for answer in following:
                                    local_answer = ''.join(dfs_print(theorem[ : i]) + [c_answer, answer] + dfs_print(theorem[postfix : ]))
                                    if not local_answer in fraction:
                                        fraction[local_answer] = calc_answer[c_answer] + following[answer]
                        elif (isinstance(first_in_this, str) and 
                            first_in_this in ['{', 'from', 'by', 'begin'] and 
                            ' sorry ' in (mask := [' sorry ' if isinstance(line, list) else line for line in theorem[i]])):
                            local_answer = ''.join(dfs_print(theorem[ : i]) + mask + dfs_print(theorem[i + 1 : ]))
                            if not local_answer in fraction:
                                ans = [dfs_sorry(line, buffer, tactic_words, tbar, mode=mode) for line in theorem[i] if isinstance(line, list)]
                                # assert len(ans) == 1
                                fraction[local_answer] = ans
                        elif i > 0 and isinstance(theorem[i - 1], str) and theorem[i - 1].strip() == ':=':
                            local_answer = ''.join(dfs_print(theorem[ : i]) + [' sorry '] + dfs_print(theorem[i + 1 : ]))
                            if not local_answer in fraction and not ('have' in local_answer and local_answer.count(':') < 2):
                                # ans = [dfs_sorry(line, buffer, tactic_words, tbar, mode=mode) for line in theorem[i] if isinstance(line, list)]
                                # assert len(ans) == 1
                                fraction[local_answer] = [dfs_sorry(theorem[i], buffer, tactic_words, tbar, mode=mode)]
                        elif len([i for i in theorem if not isinstance(i, str) or i.strip()]) == 1 and isinstance(theorem[0], list):
                            fraction = dfs_sorry(theorem[0], buffer, tactic_words, tbar, mode=mode)
                            buffer.append((theorem, fraction))
                            return fraction
                        next_level = dfs_sorry(theorem[i], buffer, tactic_words, tbar, mode=mode)
                        following = dfs_sorry(theorem[i + 1 : postfix], buffer, tactic_words, tbar, mode=mode)
                        for ans in next_level:
                            for answer in following:
                                local_answer = ''.join(dfs_print(theorem[ : i]) + [ans, answer] + dfs_print(theorem[postfix : ]))
                                if not local_answer in fraction:
                                    fraction[local_answer] = next_level[ans] + following[answer]
                        if MAX_WIDTH and len(fraction) > MAX_WIDTH:
                            for frac_name in list(fraction.keys())[1 : -MAX_WIDTH + 1]:
                                del fraction[frac_name]
            else:
                raise NotImplementedError
            buffer.append((theorem, fraction))
            if tbar:
                tbar.update()
            return fraction

        theorem = [[]]
        theorem_grammar = GrammarRegistry(src_path / 'lean_syntax/proof_grammar.json', multiline=False)
        theorem_split = theorem_grammar.tokenize_line(text)
        theorem_split = [[line[0], line[1][1:]] for line in theorem_split if len(line) > 1]
        last_isolated_calc_begin = last_calc_begin = -1
        for idx, token in enumerate(theorem_split):
            if 'calc_block' in token[1] and (idx == 0 or 'calc_block' not in theorem_split[idx - 1][1]):
                last_calc_begin = idx
            if idx > 1 and 'isolated_calc_block' in token[1] and 'isolated_calc_block' not in theorem_split[idx - 1][1]:
                last_isolated_calc_begin = idx
            if 'isolated_calc_block' in token[1] and (idx == len(theorem_split) - 1 or 'isolated_calc_block' not in theorem_split[idx + 1][1]):
                assert last_calc_begin != -1 and last_isolated_calc_begin != -1
                base_tags = theorem_split[last_calc_begin][1][ : theorem_split[last_calc_begin][1].index('calc_block')]
                theorem_split[last_calc_begin : idx + 1] = [[line[0], base_tags + line[1]] for line in theorem_grammar.tokenize_line(''.join([i[0] for i in theorem_split[last_calc_begin : idx + 1]]), "calc_rewrite") if len(line) > 1]
        for token in theorem_split:
            flag = theorem[-1]
            for label in token[1]:
                if not flag or flag[-1][0] != label:
                    flag.append([label, []])
                flag = flag[-1][1]
            flag.append(token[0])
        theorem = theorem[0]
        modifiers = [i for i, tree in enumerate(theorem) if tree[0] == 'storage.modifier']
        if remove_modifier:
            theorem = [line for idx, line in enumerate(theorem) if idx not in modifiers]

        if os.path.exists('doc_gen/tactic_words.txt'):
            with open('doc_gen/tactic_words.txt', 'r') as f:
                tactic_words = eval(f.read())
        else:
            with open('tactic_db.json', 'r', encoding='utf8') as f:
                tactics = json.load(f)
            tactic_words = [d.strip() for decl in map(lambda x : x['name'], tactics) for d in decl.split('/') if not d.startswith('#')]
            with open('doc_gen/tactic_words.txt', 'w') as f:
                f.write(str(list(set(tactic_words))))
        proof_blocks = [i for i, tree in enumerate(theorem) if tree[0] == 'proof_block']
        theorem = decoupling(theorem)
        parsed_proof = [dfs_sorry(theorem[idx], [], tactic_words, tbar, MAX_WIDTH=max_width, mode=mode) for idx in proof_blocks]
        init_context = ''.join([' sorry ' if isinstance(line, list) and idx in proof_blocks else ''.join(dfs_print(line)) for idx, line in enumerate(theorem)])
        assert parsed_proof
        return init_context, parsed_proof    

class lean_file_analyser:
    @classmethod
    def parse_export(cls, decls, path=None, only_path=False):
        if only_path:
            assert path is not None
            file_list = [str(i) for i in Path(path).glob('**/*.lean')]
        from typing import NamedTuple, List, Optional
        from collections import Counter, defaultdict, namedtuple
            
        def separate_results(objs):
            file_map = defaultdict(list)
            loc_map = {}
            for obj in objs:
                i_name = obj['filename']
                if 'export_json' in i_name:
                    continue  # this is doc-gen itself
                file_map[i_name].append(obj)
                loc_map[obj['name']] = i_name
                for (cstr_name, tp) in obj['constructors']:
                    loc_map[cstr_name] = i_name
                for (sf_name, tp) in obj['structure_fields']:
                    loc_map[sf_name] = i_name
                if len(obj['structure_fields']) > 0:
                    loc_map[obj['name'] + '.mk'] = i_name
            return file_map, loc_map

        def linkify_efmt(f):
            def go(f):
                if isinstance(f, str):
                    f = f.replace('\n', ' ')
                    # f = f.replace(' ', '&nbsp;')
                    return ''.join(
                        match[4] if match[0] == '' else
                        match[1] + match[2] + match[3]
                        for match in re.findall(r'\ue000(.+?)\ue001(\s*)(.*?)(\s*)\ue002|([^\ue000]+)', f))
                elif f[0] == 'n':
                    return go(f[1])
                elif f[0] == 'c':
                    return go(f[1]) + go(f[2])
                else:
                    raise Exception('unknown efmt object')

            return go(['n', f])
        
        def mk_export_map_entry(decl_name, filename, kind, is_meta, line, args, tp, description, attributes):
            return {'filename': cut_path(filename),
                    'local_filename': str(filename),
                    'kind': kind,
                    'is_meta': is_meta,
                    'line': line,
                    'args': [{key : linkify_efmt(value) if key == 'arg' else value for key, value in item.items()} for item in args],
                    'type': linkify_efmt(tp),
                    'attributes': attributes,
                    'description': description,
                    # 'src_link': library_link(filename, line),
                    # 'docs_link': f'{site_root}{filename.url}#{decl_name}'
                    }

        def mk_export_db(file_map):
            export_db = {}
            for _, decls in file_map.items():
                if only_path and decls[0]['filename'] not in file_list:
                    continue
                for obj in decls:
                    export_db[obj['name']] = mk_export_map_entry(obj['name'], obj['filename'], obj['kind'], obj['is_meta'], obj['line'], obj['args'], obj['type'], obj['doc_string'], obj['attributes'])
                    # export_db[obj['name']]['decl_header_html'] = env.get_template('decl_header.j2').render(decl=obj)
                for (cstr_name, tp) in obj['constructors']:
                    export_db[cstr_name] = mk_export_map_entry(cstr_name, obj['filename'], obj['kind'], obj['is_meta'], obj['line'], [], tp, obj['doc_string'], obj['attributes'])
                for (sf_name, tp) in obj['structure_fields']:
                    export_db[sf_name] = mk_export_map_entry(sf_name, obj['filename'],  obj['kind'], obj['is_meta'], obj['line'], [], tp, obj['doc_string'], obj['attributes'])
            return export_db

        file_map, loc_map = separate_results(decls['decls'])
        for entry in decls['tactic_docs']:
            if len(entry['tags']) == 0:
                entry['tags'] = ['untagged']

        mod_docs = {f: docs for f, docs in decls['mod_docs'].items()}
        # # ensure the key is present for `default.lean` modules with no declarations
        # for i_name in mod_docs:
        #     if 'export_json' in i_name:
        #         continue  # this is doc-gen itself
        #     file_map[i_name]

        # return file_map, loc_map, decls['notes'], mod_docs, decls['instances'], decls['instances_for'], decls['tactic_docs']
        return mk_export_db(file_map)
    
    @classmethod
    def get_decl_source(cls, path, decls):
        def parsing_file(file):
            with open(file) as f:
                lines = f.readlines()
            line_breaks = [0] + sorted(list(set([x[1] for x in files[file]]))) + [len(lines)]
            line_break_switches = {}
            for i, b in enumerate(line_breaks[1 : -1]):
                index = line_breaks.index(b)
                last_b = line_breaks[index - 1]
                if m := re.match(r'.*(/--.*-/\s*)$', ''.join(lines[last_b : b]), re.DOTALL):
                    line_break_switches[b] = b - m.group(1).count('\n')
                else:
                    line_break_switches[b] = b
            files[file] = sorted([(i, line_break_switches[j]) for i, j in files[file]], key=lambda x : (x[1], x[0]))
            line_breaks = sorted(list(set([x[1] for x in files[file]])))
            decl_blocks[cut_paths[file]] = {}
            decl_blocks[cut_paths[file]]['header'] = ''.join(lines[ : line_breaks[0]])
            line_breaks = [0] + line_breaks + [len(lines)]
            decl_blocks[cut_paths[file]]['blocks'] = {}
            for d, i in files[file]:
                index = line_breaks.index(i)
                source = ''.join(lines[line_breaks[index] : line_breaks[index + 1]])
                decl_blocks[cut_paths[file]]['blocks'][d] = (''.join(lines[line_breaks[max(index - 5, 1)] : line_breaks[index]]), source)
                decls[d]['source'] = ''.join([i[0] for i in Tokenizer.parsing_declarations(source)[0][1:]])
                decls[d]['line'] = [j for i, j in files[file] if i == d][0] + 1
        
        files = {}
        cut_paths = {}
        decl_blocks = {}
        for d in decls:
            if decls[d]['local_filename'] not in files:
                files[decls[d]['local_filename']] = []
            files[decls[d]['local_filename']].append((d, decls[d]['line'] - 1))
            cut_paths[decls[d]['local_filename']] = decls[d]['filename']

        for file in files:
            parsing_file(file)

        return decls

    @classmethod
    def get_decl_info(cls, path, only_path=True):
        global lean_path_basis
        global lean_paths
        path = Path(path)
        with open(leaven_path / 'leanpkg.path') as f:
            leanpkg_path = f.read()
        with open(leaven_path / 'leanpkg.path', 'w') as f:
            f.write(leanpkg_path + f'\npath {path}')
        lean_path_basis = [Path(p) for p in json.loads(subprocess.check_output([f_join(leaven_path, 'elan', 'bin', 'lean'), '--path'], cwd=leaven_path).decode())['path']]
        lean_paths = [p.resolve() for p in lean_path_basis if p.exists()]
        file_list = [i for i in path.iterdir() if i.suffix == '.lean'] if path.is_dir() else [path]
        load_file_list = []
        for file in file_list:
            file_cut = cut_path(file)
            load_file_list.append(file_cut)
        with open(src_path / 'entrypoint.lean', 'w') as f:
            f.write('\n'.join([f'import {i}' for i in load_file_list] + \
                              ["import .export_json\nopen_all_locales"]
                              ))
        command = ["lean", "--run", src_path / 'entrypoint.lean']
        result = subprocess.run(command, capture_output=True, text=True, cwd=str(leaven_path))
        # clear temp files
        os.remove(src_path / 'entrypoint.lean')
        with open(leaven_path / 'leanpkg.path', 'w') as f:
            f.write(leanpkg_path)
        return cls.parse_export(json.loads(result.stdout), path=path, only_path=only_path)
    
    @classmethod
    def simplify_lean_code(cls, code):
        import re
        from collections import defaultdict
        lines = code.split('\n')
        simplified_lines = []  # 存储简化后的代码行

        for line in lines:
            ns_match = re.match(r'\s*namespace ([^\s]+)', line)
            end_match = re.match(r'\s*end ([^\s]+)', line)
            if ns_match:
                ns_name = ns_match.group(1)
                # 查找simplified_lines中最后一个非空行
                for i in range(len(simplified_lines) - 1, -1, -1):
                    if simplified_lines[i].strip():
                        last_non_empty_line = simplified_lines[i]
                        break
                else:
                    last_non_empty_line = ''

                end_match = re.match(r'\s*end ' + re.escape(ns_name), last_non_empty_line)
                if end_match:
                    # 如果找到匹配的end语句，删除它
                    del simplified_lines[i]
                else:
                    # 否则，添加当前的namespace语句
                    simplified_lines.append(line)
            elif end_match:
                # 添加当前的end语句
                simplified_lines.append(line)
            else:
                # 对于非namespace和end语句，直接添加到简化后的代码行
                simplified_lines.append(line)

        lines = simplified_lines
        simplified_lines = []  # 存储简化后的代码行

        namespace_stack = []  # 栈用来存储当前的namespace层级
        opened_namespaces = defaultdict(set)  # 用来存储在每个namespace层级中已经打开的命名空间
        universe_namespaces = defaultdict(set)  # 用来存储在每个namespace层级中已经打开的命名空间
        universe_namespaces[None] = set()

        for line in lines:
            ns_match = re.match(r'\s*namespace ([^\s]+)', line)
            end_match = re.match(r'\s*end ([^\s]+)', line)
            open_match = re.match(r'\s*open ([^\s]+)', line)
            universe_match = re.match(r'\s*universes* (.*)', line)

            if ns_match:
                ns_name = ns_match.group(1)
                namespace_stack.append(ns_name)
                simplified_lines.append(line)
            elif end_match:
                if namespace_stack:  # 防止空栈
                    namespace_stack.pop()
                simplified_lines.append(line)
            elif open_match:
                open_ns = open_match.group(1)
                should_add = True
                # 检查是否在当前namespace层级或上层中已经打开了这个命名空间
                for ns in reversed(namespace_stack):
                    if open_ns in opened_namespaces[ns]:
                        should_add = False
                        break
                if should_add:
                    # 如果没有打开过这个命名空间，则添加这个open语句并更新opened_namespaces
                    opened_namespaces[namespace_stack[-1]].add(open_ns) if namespace_stack else None
                    simplified_lines.append(line)
            elif universe_match:
                universes = universe_match.group(1).split()
                universe_to_add = []
                should_add = True
                for universe in universes:
                    for ns in reversed(namespace_stack):
                        if universe in opened_namespaces[ns]:
                            should_add = False
                            break
                    if should_add:
                        ns_to_add = namespace_stack[-1] if namespace_stack else None
                        universe_namespaces[ns_to_add].add(universe)
                        universe_to_add.append(universe)
                if universe_to_add:
                    simplified_lines.append('universes ' + ' '.join(universe_to_add))
            else:
                # 对于非namespace, end和open语句，直接添加到简化后的代码行
                simplified_lines.append(line)

        return re.sub(r'\n\n+', '\n\n', '\n'.join(simplified_lines))
    
    @classmethod
    def auto_complete_code(cls, code_str):
        stack = []
        name_regex = re.compile(r'(?:namespace|section)\s*(.*)')
        for line in code_str.split("\n"):
            if line.startswith("namespace") or line.startswith("section"):
                stack.append(line)
        while len(stack) > 0:
            last_name = stack.pop()
            last_name = name_regex.match(last_name).group(1)
            code_str += "\nend " + last_name
        return code_str
    
    @classmethod
    def parse_ast(cls, file, all_asts, kinds, decls, lines=None):
        def get_fraction(start, end, lines):
            if start[0] < end[0]:
                return lines[start[0] - 1][start[1] : ] + ''.join(lines[start[0] : end[0] - 1]) + lines[end[0] - 1][ : end[1]]
            else:
                return lines[start[0] - 1][start[1] : end[1]]
        
        def get_content(ast, lines, file_comment_span):
            if ast is None:
                return
            if ast['kind'] in ['ident', 'notation', 'nat']:
                if ('content' not in ast or not ast['content']) and 'value' in ast:
                    if isinstance(ast['value'], list):
                        ast['end'][1] += len('.'.join(ast['value']))
                    if ast['kind'] == 'nat':
                        ast['end'][1] += len(ast['value']) + 1
                    else:
                        ast['end'][1] += len(ast['value'])
            if 'children' in ast:
                for c in ast['children']:
                    if c is None:
                        continue
                    get_content(c, lines, file_comment_span)
                    if c['start'] >= file_comment_span and c['start'] < ast['start']:
                        ast['start'] = c['start']
                    if c['end'] > ast['end']:
                        ast['end'] = c['end']
            if ast['start'] >= file_comment_span:
                ast['content'] = get_fraction(ast['start'], ast['end'], lines)

        def get_instance_name(ast):
            if not 'children' in ast and ast['kind'] in ['ident', 'notation'] and ast['value']:
                if isinstance(ast['value'], list):
                    return ast['value']
                elif isinstance(ast['value'], str):
                    return [ast['value']]
                else:
                    raise ValueError
            elif ast['children']:
                for i in ast['children']:
                    if i is not None:
                        return get_instance_name(i)
                raise ValueError
            else:
                raise ValueError
            
        def get_proving_environment(proving_environment):
            stack = []
            for i in proving_environment:
                if i[2] == 'end':
                    while True:
                        last_item = stack.pop()
                        if last_item[2] not in ['namespace', 'section']:
                            continue
                        if end_name := i[3][i[3].find(i[2]) + len(i[2]) : ].strip():
                            closed_name = last_item[3][last_item[3].find(last_item[2]) + len(last_item[2]) : ].strip()
                            assert end_name == closed_name, (end_name, closed_name)
                        break
                else:
                    stack.append(i)
            return stack
        
        if lines is None:
            with open(file, 'r') as f:
                lines = f.readlines()
            if lines[-1].endswith('\n'):
                lines.append('')
        file_comment_regex = re.compile(r'\/-\s*Copyright(?:[^-]|-[^\/])*-\/')
        next_comment_regex = re.compile(r'\/-(?:[^-]|-[^\/])*-\/|--[^\n]*\n')
        # local_proving_environment = {}
        # all_decl_names = []
        decl_fields = {}
        # symbols = {}
        file_comment = file_comment_regex.search(''.join(lines))
        if file_comment:
            file_comment_span = ''.join(lines)[ : file_comment.span()[1]].split('\n')
            file_comment_span = [len(file_comment_span),len(file_comment_span[-1])]
        else:
            file_comment_span = [0,0]
        proving_environment = [(x['start'], x['end'], x['kind'], get_fraction(x['start'], x['end'], lines)) for x in all_asts if x is not None and x['kind'] in ['namespace', 'section', 'end', 'open', 'infix', 'infixl', 'infixr', 'prefix', 'postfix', 'universes', 'variables', 'variable', 'user_command'] and x['start'] >= file_comment_span]

        mdocs = kinds['mdoc'][0]['value'].replace('\r','').replace('> THIS FILE IS SYNCHRONIZED WITH MATHLIB4.\n> Any changes to this file require a corresponding PR to mathlib4.','') if 'mdoc' in kinds else None

        for ast in [i for k, v in kinds.items() if k in ['theorem', 'definition', 'abbreviation', 'instance', 'structure', 'inductive', 'class', 'class_inductive'] for i in v]:
            if ('content' not in ast or
                (ast['kind'] == 'definition' and 'def' not in ast['content']) or
                (ast['kind'] == 'theorem' and 'theorem' not in ast['content'] and 'lemma' not in ast['content']) or
                (ast['kind'] not in ['definition', 'theorem', 'abbreviation'] and ast['kind'] not in ast['content'])):
                # ast = ast
                continue
            get_content(ast, lines, file_comment_span)
            if ast['end'][1] > 0:
                next_comment = next_comment_regex.search(''.join(lines[ast['end'][0] - 1 : ]))
                if next_comment and next_comment.span()[0] < len(lines[ast['end'][0] - 1]) and ast['end'][1] > next_comment.span()[0]:
                    ast['end'][1] = next_comment.span()[0]
                    ast['content'] = get_fraction(ast['start'], ast['end'], lines)
            if (ast['children'][0] is not None and 
                'children' in ast['children'][0] and 
                [i for i in ast['children'][0]['children'] if i is not None and i['kind'] == 'private']):
                continue
            if ast['content'].strip().endswith('.') and ast['children'][6] is not None and not ast['children'][6]['content']:
                continue
            if ast['kind'] == 'instance':
                if ast['children'][3] is not None:
                    ast_decl_list = [get_instance_name(ast['children'][3])]
                else:
                    ast_decl_list = [get_instance_name(ast['children'][5])]
            elif ast['kind'] in ['structure', 'class', 'class_inductive']:
                ast_decl_list = [ast['children'][2]['value']]
            elif ast['kind'] in ['inductive', 'definition', 'abbreviation']:
                if ast['children'][3]['kind'] == "mutuals":
                    ast_decl_list = [get_instance_name(i) for i in ast['children'][3]['children']]
                else:
                    ast_decl_list = [get_instance_name(ast['children'][3])]
            else:
                ast_decl_list = [ast['children'][3]['value']]
            pre_env = cls.simplify_lean_code('\n'.join(i[-1] for i in proving_environment))
            open_namespace_and_section = get_proving_environment([x for x in proving_environment if x[0] <= ast['start']])
            open_namespace = {k : [i[3][i[3].find(k) + len(k) : ].strip() for i in open_namespace_and_section if i[2] == k] for k in ['namespace', 'open']}
            last_namespace = '.'.join(open_namespace['namespace']).split('.') if open_namespace['namespace'] else []
            for ast_decl in ast_decl_list:
                if isinstance(ast_decl, list):
                    if ast_decl[0] == '_root_':
                        ast_decl = ast_decl[1:]
                    else:
                        ast_decl = last_namespace + ast_decl
                else:
                    ast_decl = last_namespace + [ast_decl]
                try:
                    if ast['kind'] == 'instance':
                        get_decl = list(filter(lambda x : 
                                            x.split('.')[ : len(ast_decl) - 1] == ast_decl[ : -1] and 
                                            ast_decl[-1] in x.split('.')[-1] and 
                                            ast['start'][0] <= decls[x]['line'] <= (ast['end'][0] if ast['end'][1] > 0 else (ast['end'][0] - 1)) and
                                            (decls[x]['source'].strip() in ast['content'] or 
                                             ast['content'].strip() in decls[x]['source'] or
                                                all(i['content'].strip() in decls[x]['source'] for i in ast['children'][5:] if i is not None)), decls))
                    elif ast['kind'] in ['structure', 'class', 'class_inductive']:
                        get_decl = list(filter(lambda x : 
                                            x.split('.')[ : len(ast_decl) - 1] == ast_decl[ : -1] and 
                                            ast_decl[-1] in x.split('.')[-1] and 
                                            ast['start'][0] <= decls[x]['line'] <= (ast['end'][0] if ast['end'][1] > 0 else (ast['end'][0] - 1)) and
                                            (decls[x]['source'].strip() in ast['content'] or 
                                             ast['content'].strip() in decls[x]['source'] or
                                                all(i['content'].strip() in decls[x]['source'] for i in ast['children'][5:] if i is not None)
                                                ), decls))
                    else:
                        get_decl = list(filter(lambda x : 
                                            x.split('.') == ast_decl and 
                                            ast['start'][0] <= decls[x]['line'] <= (ast['end'][0] if ast['end'][1] > 0 else (ast['end'][0] - 1)) and
                                            (decls[x]['source'].strip() in ast['content'] or 
                                             ast['content'].strip() in decls[x]['source'] or
                                                all(i['content'].strip() in decls[x]['source'] for i in ast['children'][5:] if i is not None)), 
                                                decls))
                    assert len(get_decl) >= 1
                except AssertionError as e:
                    continue
                except Exception as e:
                    raise e
                if len(get_decl) > 1:
                    last_decl_content = decls[get_decl[0]]['source']
                    for d in get_decl[1 : ]:
                        assert decls[d]['source'] == last_decl_content
                if ast['kind'] in ['structure', 'inductive', 'class', 'class_inductive']:
                    if ast['children'][7] is not None:
                        ast['children'][7]['content'] = get_fraction(ast['children'][7]['start'], ast['end'], lines)
                elif ast['kind'] in ['definition', 'abbreviation'] and ast['children'][5] is not None:
                    ast['children'][5]['content'] = get_fraction(ast['children'][5]['start'], ast['end'], lines)
                else:
                    ast['children'][6]['content'] = get_fraction(ast['children'][6]['start'], ast['end'], lines)
                ast['filename'] = str(file)
                for decl in get_decl:
                    # local_proving_environment[file][decl] = [[i[2], i[3]] for i in open_namespace_and_section]
                    matched_decls = [i for i in decls if i.split('.')[ : len(decl.split('.'))] == decl.split('.') and ast['start'][0] <= decls[i]['line'] <= (ast['end'][0] if ast['end'][1] > 0 else (ast['end'][0] - 1))]
                    last_decl_content = ''
                    for d in matched_decls:
                        assert not last_decl_content or last_decl_content in decls[d]['source'] or decls[d]['source'].strip() in last_decl_content
                        last_decl_content = decls[d]['source'].strip()
                        decls[d]['source'] = ast['content']
                        decls[d]['complete_source'] = cls.auto_complete_code(pre_env + '\n\n' + ast['content'])
                        decls[d]['line'] = ast['start'][0]
                        decls[d]['start_line'] = ast['start']
                        decls[d]['end_line'] = ast['end']
                    decl_fields[decl] = matched_decls
                    # all_decl_names.append((decl, str(file)))
        return decls, decl_fields, mdocs

    @classmethod
    def get_ast(cls, file, lines=None, build_graph=False):
        file = Path(file).resolve()
        graph = None
        if not os.path.exists(file.with_suffix('.ast.json')):
            os.system(' '.join([f_join(leaven_path, 'elan', 'bin', 'lean'), "-M", "20480", "--ast", "--tsast", "--tspp -q ", str(file)]))
        with open(file.with_suffix('.ast.json'), 'r') as f:
            data = json.load(f)
        if lines is None:
            with open(file, 'r') as f:
                lines = f.readlines()
            if lines[-1].endswith('\n'):
                lines.append('')
        os.remove(file.with_suffix('.ast.json'))
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
        if build_graph:
            graph = nx.DiGraph()
            graph.add_node(0, kind=None, start=None, end=None)
            cls.transverse_ast(kinds['file'], graph, 0, None, None)
        return kinds, data['ast'], graph
    
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

    @classmethod
    def document_probing(self, file=None, content=None, lean_server=None):
        if lean_server is None:
            lean_server = LeanEnv(cwd=str(Path('.').resolve()))
        if file is not None:
            with open(file, 'r') as f:
                lines = f.readlines()
        elif content is not None:
            lines = re.split(r'(?<=\n)', content)
        else:
            raise ValueError
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
    
    @classmethod
    def get_training_probes(cls, file=None, content=None, lean_server=None):
        if file is not None:
            file = Path(file).resolve()
            processed_line = cls.document_probing(file=file, lean_server=lean_server)
        elif content is not None:
            processed_line = cls.document_probing(content=content, lean_server=lean_server)
        else:
            raise ValueError
        #  or content is not None
        tactic_states = []
        lemmas = []
        for l, line in processed_line.items():
            for c, column in line.items():
                if 'source' in column and column['source']:
                    if 'state' in column and column['state'] and column['state'].split('⊢')[0]and 'Type' not in column['state'].split('⊢')[0]:
                        tactic_states.append([l, c, column])
                    if 'full_id' in column and column['full_id']:
                        lemmas.append([l, c, column])
        return tactic_states, lemmas
    
    @classmethod
    def get_dependency_graph_within_file(cls, decls, file):
        def position_to_decl(line, column, full_id=None):
            assert len(result := {k : v for k, v in decls.items() if v['local_filename'] == str(file) and v['start_line'] <= [line, column] <= v['end_line'] and k in decl_fields and (full_id is None or k == full_id)}) == 1
            return list(result.keys())[0]

        with open(file, 'r') as f:
            lines = f.readlines()
        if lines[-1].endswith('\n'):
            lines.append('')
        kinds, all_asts, _ = lean_file_analyser.get_ast(file, lines=lines)
        decls, decl_fields, mdocs = lean_file_analyser.parse_ast(file=file, all_asts=all_asts, kinds=kinds, decls=decls, lines=lines)
        tactic_states, lemmas = lean_file_analyser.get_training_probes(file=file)
        graph = nx.DiGraph()
        for line, column, item in lemmas:
            if item['source']['file'] is not None:
                continue
            graph.add_edge(position_to_decl(item['source']['line'], item['source']['column'], full_id=item['full_id']), position_to_decl(line, column))
        return decls, tactic_states, lemmas, graph, decl_fields, mdocs

if __name__ == "__main__":
    path = 'test1.lean'
    graph = lean_file_analyser.get_ast(path)
    tactics, commands, lemmas = lean_file_analyser.get_information(path)

