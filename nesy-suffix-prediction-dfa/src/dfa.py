import os.path
import re
from collections import defaultdict
from sympy import sympify, symbols
import pydot
import networkx as nx
import torch
from torch import nn


class SymbolicDFA:
    def __init__(self, file_path, labels):
        self.totalize_dfa(file_path, labels)
        self.file_path = os.path.join(file_path, 'simpleDFA_final.dot')
        self.graph = nx.MultiDiGraph()
        self.initial_state = None
        self.state_types = {}
        self.label_to_idx = {label: i for i, label in enumerate(labels + ['end'])}
        self.transition_table = {}
        self.parse_dot()

    def parse_dot(self):
        dot_graphs = pydot.graph_from_dot_file(self.file_path)
        dot = dot_graphs[0]

        for edge in dot.get_edges():
            src = edge.get_source()
            dst = edge.get_destination()
            if src == 'init':
                self.initial_state = int(dst)
            else:
                label = edge.get_label().strip('"')
                src, dst = int(src), int(dst)
                self.graph.add_edge(src, dst, key=label, label=label)

                label_idx = self.label_to_idx[label]
                self.transition_table.setdefault(src, {})[label_idx] = dst

        accepting_states = self.extract_accepting_states()
        rejecting_states = self.extract_rejecting_states(accepting_states)
        self.assign_state_types(accepting_states, rejecting_states)

    def extract_accepting_states(self):
        with open(self.file_path, 'r') as f:
            dot_source = f.read()

        accepting_states = set()
        double_circle_lines = re.findall(r'node\s*\[shape\s*=\s*doublecircle];\s*([\d\s;]+)', dot_source)
        for line in double_circle_lines:
            state_idx = line.split(';')[0]
            accepting_states.add(int(state_idx))
        return accepting_states

    def extract_rejecting_states(self, accepting_states):
        reachable = set()
        rev_graph = self.graph.reverse(copy=False)
        for accept in accepting_states:
            reachable.add(accept)
            visited = nx.descendants(rev_graph, accept)
            reachable.update(visited)
        return set(self.graph.nodes) - reachable

    def assign_state_types(self, accepting_states, rejecting_states):
        for state in self.graph.nodes:
            if state in accepting_states:
                self.state_types[state] = 1
            elif state in rejecting_states:
                self.state_types[state] = -1
            else:
                self.state_types[state] = 0

    def totalize_dfa(self, dfa_path, labels):
        with open(os.path.join(dfa_path, 'symbolicDFA.dot'), 'r') as file:
            dot_string = file.read()

        states = set()
        initial_state = None
        temp_accepting_states = set()
        transitions = defaultdict(dict)

        token_symbols = symbols(labels)
        token_map = dict(zip(labels, token_symbols))

        lines = dot_string.split('\n')

        for line in lines:
            if 'doublecircle' in line:
                finals = line.strip().split(';')[1:-1]
                temp_accepting_states.update(int(s.strip()) - 1 for s in finals)
            elif '->' in line:
                if 'init' in line:
                    parts = line.strip().split(' ')
                    initial_state = int(parts[2][:-1]) - 1
                else:
                    parts = line.strip().split(' ')
                    src, dst = int(parts[0]) - 1, int(parts[2]) - 1
                    label = line.strip().split('"')[1]
                    states.add(src)
                    states.add(dst)
                    guard = sympify(a=label, locals=token_map)
                    transitions[src][dst] = self.valid_tokens_for_guard(guard, labels)

        final_rejecting = len(states)
        final_accepting = len(states) + 1

        for state in states:
            if state in temp_accepting_states:
                transitions[state][final_accepting] = ['end']
            else:
                transitions[state][final_rejecting] = ['end']

        states.add(final_rejecting)
        states.add(final_accepting)
        transitions[final_rejecting][final_rejecting] = labels + ['end']
        transitions[final_accepting][final_accepting] = labels + ['end']

        final_dot = self.create_string(initial_state, final_accepting, transitions)
        with open(os.path.join(dfa_path, 'simpleDFA_final.dot'), 'w+') as file:
            file.write(final_dot)

    def valid_tokens_for_guard(self, guard_expr, tokens):
        valid = []
        for token in tokens:
            assignment = {t: False for t in tokens}
            assignment[token] = True
            if bool(guard_expr.subs(assignment)):
                valid.append(token)
        return valid

    def create_string(self, initial_state, final_accepting, transitions):
        intro = """digraph MONA_DFA {
    rankdir = LR;
    center = true;
    size = "7.5,10.5";
    edge [fontname = Courier];
    node [height = .5, width = .5];
    """
        end = f'node [shape = doublecircle]; {final_accepting};'
        start = f'node [shape = circle]; {initial_state};\ninit [shape = plaintext, label = ""];\ninit -> {initial_state};'
        transitions_string = ""
        for src, transitions in transitions.items():
            for dst, edges in transitions.items():
                for edge in edges:
                    transitions_string += f'{src} -> {dst} [label="{edge}"];\n'
        transitions_string += "}"
        return intro + end + '\n' + start + '\n' + transitions_string

class TensorDFA:
    def __init__(self, dfa, device):
        super().__init__()
        self.dfa = dfa
        self.num_states = len(dfa.graph.nodes)
        self.vocab_size = len(dfa.label_to_idx)
        self.device = device
        self.transition_tensor = self.build_transition_tensor()
        self.state_types_tensor = torch.tensor([dfa.state_types[i] for i in range(self.num_states)], dtype=torch.long, device=self.device)

    def build_transition_tensor(self):
        table = torch.full((self.num_states, self.vocab_size), fill_value=-1, dtype=torch.long, device=self.device)
        print(self.dfa.transition_table.items())
        for src, transitions in self.dfa.transition_table.items():
            for symbol_idx, dst in transitions.items():
                table[src, symbol_idx] = dst
        return table

    def simulate(self, input_indices, batch_size):
        seq_len = input_indices.size(1)
        current_states = torch.full((batch_size,), self.dfa.initial_state, dtype=torch.long, device=self.device)
        state_ids = []

        for t in range(seq_len):
            symbol = input_indices[:, t]
            next_states = self.transition_tensor[current_states, symbol].long()
            state_ids.append(next_states)
            current_states = next_states

        state_ids = torch.stack(state_ids, dim=1)
        return state_ids

    def check_satisfiability(self, traces_onehot):
        traces_indices = torch.argmax(traces_onehot, dim=-1)
        state_ids = self.simulate(traces_indices, traces_indices.size(0))
        final_states = state_ids[:, -1]
        is_accepted = (self.state_types_tensor[final_states] == 1).float()
        return is_accepted.mean().item()
