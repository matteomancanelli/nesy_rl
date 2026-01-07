import re
from ltlf2dfa.parser.ltlf import LTLfParser


class Model:
    def __init__(self, root_path, dataset, template_type, template_support):
        self._folder_path = f'{root_path}datasets/{dataset}/model/'
        self._name = f'test_{template_type}_{template_support}'
        #self._name = f'concept_drift_{template_type}_{template_support}'
        self._content = self.read_from_file()
        self._formulas = []

    def to_ltl(self):
        formulas = []
        for row in self._content.split('\n'):
            if row and not row.startswith('#'):
                match = re.match(r'([\w\s-]+)\[([\w\s-]+)(?:, ([\w\s-]+))?]', row)
                if match:
                    declare_constraint = DeclareConstraint(match)
                    ltl_constraint = declare_constraint.to_ltl()
                    if ltl_constraint:
                        formulas.append(ltl_constraint)
                #else:
                #    print(row)
        self._formulas = formulas

    def to_dfa(self, dfa_folder):
        parser = LTLfParser()
        ast = parser(self.get_ltl_formula())
        dot = ast.to_dfa()

        with open(f'{dfa_folder}symbolicDFA.dot', 'w+') as file:
            file.write(dot)

    def read_from_file(self):
        with open(f'{self._folder_path}{self._name}.decl') as f:
            return f.read()

    def write_formula_to_file(self):
        with open(f'{self._folder_path}{self._name}_ltl.txt', 'w') as f:
            f.write(' &\n'.join(self._formulas))

    def get_ltl_formula(self):
        return ' & '.join(self._formulas)


class DeclareConstraint:
    def __init__(self, match):
        self._name = match.group(1)
        self._activation = f"a_{match.group(2).lower().replace(' ', '_').replace('-', '_').replace('.', '_').replace('(',  '_').replace(')', '_')}"
        if match.group(3):
            self._target = f"a_{match.group(3).lower().replace(' ', '_').replace('-', '_').replace('.', '_').replace('(',  '_').replace(')', '_')}"

    def to_ltl(self):
        if self._name == 'Init':
            return f'({self._activation})'
        elif self._name == 'Existence':
            return f'(F({self._activation}))'
        elif self._name == 'Existence2':
            return f'(F({self._activation} & X(F({self._activation}))))'
        elif self._name == 'Existence3':
            return f'(F({self._activation} & X(F({self._activation} & X(F({self._activation}))))))'
        elif self._name == 'Absence':
            return f'(!(F({self._activation})))'
        elif self._name == 'Absence2':
            return f'(!(F({self._activation} & X(F({self._activation})))))'
        elif self._name == 'Absence3':
            return f'(!(F({self._activation} & X(F({self._activation} & X(F({self._activation})))))))'
        elif self._name == 'Exactly1':
            return f'(F({self._activation}) & !(F({self._activation} & X(F({self._activation})))))'
        elif self._name == 'Choice':
            return f'(F({self._activation}) | F({self._target}))'
        elif self._name == "Exclusive Choice":
            return f'((F({self._activation}) & !(F({self._target}))) | (F({self._target}) & !(F({self._activation}))))'
        elif self._name == 'Responded Existence':
            return f'(F({self._activation}) -> F({self._target}))'
        elif self._name == 'Co-Existence':
            return f'((F({self._activation}) -> F({self._target})) & (F({self._target}) -> F({self._activation})))'
        elif self._name == 'Response':
            return f'(G({self._activation} -> F({self._target})))'
        elif self._name == 'Alternate Response':
            return f'(G({self._activation} -> X(!({self._activation}) U {self._target})))'
        elif self._name == 'Chain Response':
            return f'(G({self._activation} -> X({self._target})))'
        elif self._name == 'Precedence':
            return f'((!({self._target}) U {self._activation}) | G(!({self._target})))'
        elif self._name == 'Alternate Precedence':
            return f'((((!{self._target} U {self._activation}) | G(!{self._target})) & G({self._target} ->((!(X({self._activation})) & !(X(!({self._activation})))) | X((!({self._target}) U {self._activation}) | G(!({self._target})))))) & !({self._target}))'
        elif self._name == 'Chain Precedence':
            return f'(G(X({self._target}) -> {self._activation}))'
        elif self._name == 'Succession':
            return f'(G({self._activation} -> F({self._target})) & (!({self._target}) U {self._activation}) | G (!{self._target}))'
        elif self._name == 'Alternate Succession':
            return f'(G({self._activation} -> X(! {self._activation} U {self._target})) & (!({self._target}) U {self._activation}) | G(!{self._target}))'
        elif self._name == 'Chain Succession':
            return f'((G({self._activation} -> X({self._target}))) & (G(X({self._target}) -> {self._activation})))'
        elif self._name == 'Alternate Succession':
            return f'(G({self._activation} -> X(!({self._activation}) U {self._target})) & (!({self._target}) U {self._activation}) | G(!{self._target}))'
        elif self._name == 'Not Co-Existence':
            return f'(!(F({self._activation}) & F({self._target})))'
        elif self._name == 'Not Succession':
            return f'(G({self._activation} -> !(F({self._target}))))'
        elif self._name == 'Not Chain Succession':
            return f'(G({self._activation} -> !(X({self._target}))))'# & (G(X(!({self._target})) -> {self._activation})))'
        else:
            return None
