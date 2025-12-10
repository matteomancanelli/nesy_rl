import os
import time
import random
from copy import deepcopy
import numpy as np
import pandas as pd
import torch
from modules import LSTM
from training import train
from sampling import sample
from result import Result
from plotting import plot_metrics_as_bars
#TODO: train_ds, test_ds, dfa in self? da vedere dopo aver aggiunto il noise; salavare tracce predette e plot loss


class Experiment:
    def __init__(self, config, dataset_name, prefixes, noise, alpha):
        self.config = config
        self.ds_name = dataset_name
        self.noise = noise
        self.alpha = alpha
        self.prefixes = prefixes
        self.results_df = None
        self.experiment_folder = self.create_experiment_folder()

    def run(self, train_ds, test_ds, tensor_dfa, vocabulary):
        results = []

        for run_id in range(self.config['nr_runs']):
            run_nr = run_id + 1
            run_folder = self.create_run_folder(run_nr)

            set_seed(run_nr * 10)
            rnn = LSTM(len(vocabulary), self.config['hidden_dim'])

            for mode in self.config['modes']:
                mode_results = self.run_mode(rnn, train_ds, test_ds, tensor_dfa, run_folder, run_nr, mode)
                results.extend(mode_results)

        self.results_df = pd.DataFrame(results)
        self.results_df['dataset'] = self.ds_name

    def run_mode(self, rnn, train_ds, test_ds, tensor_dfa, run_folder, run_id, model_type):
        model = deepcopy(rnn).to(self.config['device'])

        start_time = time.perf_counter()
        if model_type == 'baseline':
            train_acc, test_acc, _, _, nr_epochs = train(model, train_ds, test_ds, self.config)
        else:
            train_acc, test_acc, _, _, nr_epochs = train(model, train_ds, test_ds, self.config, tensor_dfa, self.alpha)
        training_time = time.perf_counter() - start_time
        mode_result = Result(run_folder, run_id, model_type, train_acc, test_acc, nr_epochs, training_time)
        export_model(model, run_folder, f'rnn_{model_type}')

        self.test(model, train_ds, test_ds, tensor_dfa, mode_result)
        return mode_result.evaluate_predictions(train_ds, test_ds, tensor_dfa)

    def test(self, model, train_ds, test_ds, tensor_dfa, mode_result):
        for prefix in self.prefixes:
            predictions =  {
                'train_temperature': sample(model, train_ds, prefix, self.config['device'], self.config['temperature'], tensor_dfa),
                'test_temperature': sample(model, test_ds, prefix, self.config['device'], self.config['temperature'], tensor_dfa),
                'train_greedy': sample(model, train_ds, prefix, self.config['device'], 0, tensor_dfa),
                'test_greedy': sample(model, test_ds, prefix, self.config['device'], 0, tensor_dfa)
            }
            mode_result.add_predictions(prefix, predictions)

    def create_experiment_folder(self):
        folder_name = f'{self.config["timestamp"]}_{self.config["template_type"]}_n{self.noise}_a{self.alpha * 100}'
        experiment_folder = os.path.join(self.config['root_path'], 'results', self.ds_name, folder_name)
        os.makedirs(experiment_folder)
        return str(experiment_folder)

    def create_run_folder(self, run_number):
        run_folder = os.path.join(self.experiment_folder, f'run{run_number}')
        for subfolder in ['plots', 'predicted_traces', 'models']:
            os.makedirs(os.path.join(run_folder, subfolder))
        return run_folder

    def save_results(self):
        output_path = os.path.join(self.experiment_folder, 'results.csv')
        self.results_df.to_csv(output_path, index=False)

    def plot_results(self):
        for metric in ['accuracy', 'DL', 'DL scaled', 'sat']:
            filename = os.path.join(self.experiment_folder, f'{metric}_mean.png')
            plot_metrics_as_bars(self.results_df, metric, self.prefixes, filename, errorbar=True)


def export_model(model, run_folder, name):
    torch.save(model.state_dict(), os.path.join(run_folder, 'models', f'{name}.pt'))
    del model
    torch.cuda.empty_cache()

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # set to false for reproducibility, True to boost performance
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    random_state = random.getstate()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return random_state

    #def calculate_results(self, record, run_folder):
    #    with open(os.path.join(run_folder, 'predicted_traces', f'predicted_{model_type}_{strategy}_{prefix}.txt'), mode='w') as file:
    #        file.write(self.log.decode(predicted_test))
