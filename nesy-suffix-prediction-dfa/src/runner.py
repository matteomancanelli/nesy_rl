import os
from datetime import datetime
import yaml
import torch
from log import Log
from model import Model
from dfa import SymbolicDFA, TensorDFA
from experiment import Experiment
#TODO: da migliorare il passaggio declare -> ltl -> dfa; log noise injection da implementare

def main():
    config = load_config('config.yaml')

    for dataset in config['datasets']:
        full_log = Log(config['root_path'], dataset, 'ordered')
        event_names = full_log.get_event_names()

        dfa_folder_name = f'DFA_{config["template_type"]}_{config["template_support"]}/'
        dfa_folder = os.path.join(config['root_path'], 'datasets', dataset, 'model', dfa_folder_name)
        if not os.path.isdir(dfa_folder):
            os.makedirs(dfa_folder, exist_ok=True)
            declare_model = Model(config['root_path'], dataset, config["template_type"], config["template_support"])
            declare_model.to_ltl()
            declare_model.to_dfa(dfa_folder)
        symbolic_dfa = SymbolicDFA(dfa_folder, event_names)
        dfa = TensorDFA(symbolic_dfa, config['device'])

        os.makedirs(os.path.join(config['root_path'], 'results', dataset), exist_ok=True)

        for noise in config['noise_levels']:
            for alpha in config['alpha_levels']:
                train_log = Log(config['root_path'], dataset, f'train_80_all_{noise}n')
                test_log = Log(config['root_path'], dataset, f'test_20_all')
                train_dataset = train_log.encode(event_names)
                test_dataset = test_log.encode(event_names)

                first_prefix = train_log.get_first_prefix()
                prefixes = [first_prefix, first_prefix + 1, first_prefix + 2]

                experiment = Experiment(config, dataset, prefixes, noise, alpha)
                experiment.run(train_dataset, test_dataset, dfa, event_names + ['end'])
                experiment.save_results()
                experiment.plot_results()

def load_config(config_path):
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)
        config['timestamp'] = datetime.now().strftime('%m%d-%H%M%S')
        config['device'] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        return config

if __name__ == '__main__':
    main()
