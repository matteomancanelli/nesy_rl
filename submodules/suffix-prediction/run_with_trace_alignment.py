from copy import deepcopy
import json
import numpy as np
import torch
import pathlib

from RNN import LSTM_model
from FiniteStateMachine import DFA
from evaluation import suffix_prediction_with_temperature_with_stop, evaluate_compliance_with_formula, evaluate_DL_distance, greedy_suffix_prediction_with_stop

from utils import *

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"


if __name__ == "__main__":
    # Number of experiments
    N_CONFIGURATIONS = 5
    N_FORMULAS = 5
    N_EXPERIMENTS_PER_FORMULA = 5

    # Parameters for RNN
    NVAR = 3
    HIDDEN_DIM = 100
    TRAIN_RATIO = 0.9
    TEMPERATURE = 0.7
    MAX_NUM_EPOCHS = 4000
    EPSILON = 0.01

    # Experiment parameters initial configuration
    D = 5
    C = 1

    # PARAMETERS TO VARY
    TRACE_LENGTH = 20
    SAMPLE_SIZE_START_VALUE = 250
    SAMPLE_SIZE_INCREMENT = 250
    SAMPLE_SIZE_INCREMENT_ITERATIONS = 4
    PREFIX_LEN_START_VALUE = 5
    PREFIX_LEN_INCREMENT = 5
    PREFIX_LEN_INCREMENT_ITERATIONS = 3

    alphabet = ["c0", "c1", "c2", "end"]

    experiments_date = "2025-01-22_23-53-46"

    configuration_results = {}

    configuration_results[str((D, C))] = {}
    
    formulas = []
    with open(f"./models/results_{experiments_date}/declare_D={D}_C={C}/formulas_dec_infix.txt", "r") as f:
        formulas = [line.strip() for line in f.readlines()]

    for current_sample_size in range(SAMPLE_SIZE_START_VALUE, SAMPLE_SIZE_START_VALUE + SAMPLE_SIZE_INCREMENT * SAMPLE_SIZE_INCREMENT_ITERATIONS, SAMPLE_SIZE_INCREMENT):            
        
        for i_form, formula in enumerate(formulas):
            form_file = write_formula(formula, alphabet)

            if i_form not in configuration_results[str((D, C))]:
                configuration_results[str((D, C))][i_form] = {}
            
            if current_sample_size not in configuration_results[str((D, C))][i_form]:
                configuration_results[str((D, C))][i_form][current_sample_size] = {}
            
            configuration_results[str((D, C))][i_form][current_sample_size]["results"] = {}
            formula_experiment_results = {}
            
            for current_prefix_len in range(PREFIX_LEN_START_VALUE, PREFIX_LEN_START_VALUE + PREFIX_LEN_INCREMENT * PREFIX_LEN_INCREMENT_ITERATIONS, PREFIX_LEN_INCREMENT):
                formula_experiment_results[current_prefix_len] = {
                    # RNN results
                    "test_DL_rnn": [],
                    "test_sat_rnn": [],
                    "test_DL_rnn_ta": [],
                    "test_sat_rnn_ta": [],
                    "test_DL_rnn_bk": [],
                    "test_sat_rnn_bk": [],
                    # RNN Greedy results
                    "test_DL_rnn_greedy": [],
                    "test_sat_rnn_greedy": [],
                    "test_DL_rnn_greedy_ta": [],
                    "test_sat_rnn_greedy_ta": [],
                    "test_DL_rnn_bk_greedy": [],
                    "test_sat_rnn_bk_greedy": []
                }

            for exp in range(N_EXPERIMENTS_PER_FORMULA):
                if i_form!=1 or current_sample_size!=1000:
                    continue

                print(f"--> Configuration: {D}, {C}, {i_form}, {current_sample_size}, {exp}")

                dataset_file_name = f"./datasets/syntetic/datasets_{experiments_date}/dataset_declare_D={D}_C={C}/sample_size_{current_sample_size}/dataset_declare_D={D}_C={C}_{current_sample_size}_{exp}.dat"
                model_file_name = f"./models/results_{experiments_date}/declare_D={D}_C={C}/model_rnn_formula_{i_form}_sample_size_{current_sample_size}_exp_{exp}.pt"

                dfa = DFA(formula, NVAR, "random DNF declare", alphabet)
                deep_dfa = dfa.return_deep_dfa()

                dataset = torch.tensor(np.loadtxt(str(dataset_file_name)))
                dataset = dataset.view(dataset.size(0), -1, NVAR)
                dataset = expand_dataset_with_end_of_trace_symbol(dataset)
                dataset = dataset.float()
                num_traces = dataset.size()[0]

                # Splitting in train and test
                train_dataset = dataset[: int(TRAIN_RATIO * num_traces)]
                test_dataset = dataset[int(TRAIN_RATIO * num_traces) :]

                rnn = LSTM_model(HIDDEN_DIM, NVAR + 1, NVAR + 1)
                rnn_bk = deepcopy(rnn)

                state_dict = torch.load(model_file_name)
                model = deepcopy(rnn).to(device)
                model.load_state_dict(state_dict)
                
                for current_prefix_len in range(PREFIX_LEN_START_VALUE, PREFIX_LEN_START_VALUE + PREFIX_LEN_INCREMENT * PREFIX_LEN_INCREMENT_ITERATIONS, PREFIX_LEN_INCREMENT):
                    test_predicted_traces = suffix_prediction_with_temperature_with_stop(model, test_dataset, current_prefix_len, temperature=TEMPERATURE)

                    test_sat = evaluate_compliance_with_formula(deep_dfa, test_predicted_traces)
                    formula_experiment_results[current_prefix_len]["test_sat_rnn"].append(test_sat)

                    test_DL = evaluate_DL_distance(test_predicted_traces, test_dataset)
                    formula_experiment_results[current_prefix_len]["test_DL_rnn"].append(test_DL)


                for current_prefix_len in range(PREFIX_LEN_START_VALUE, PREFIX_LEN_START_VALUE + PREFIX_LEN_INCREMENT * PREFIX_LEN_INCREMENT_ITERATIONS, PREFIX_LEN_INCREMENT):
                    test_predicted_traces = suffix_prediction_with_temperature_with_stop(model, test_dataset, current_prefix_len, temperature=TEMPERATURE)
                    test_predicted_traces = align_predicted_traces(test_predicted_traces, form_file, alphabet, TRACE_LENGTH)
                    test_predicted_traces = test_predicted_traces.to(device)

                    test_sat = evaluate_compliance_with_formula(deep_dfa, test_predicted_traces)
                    formula_experiment_results[current_prefix_len]["test_sat_rnn_ta"].append(test_sat)

                    test_DL = evaluate_DL_distance(test_predicted_traces, test_dataset)
                    formula_experiment_results[current_prefix_len]["test_DL_rnn_ta"].append(test_DL)

                
                for current_prefix_len in range(PREFIX_LEN_START_VALUE, PREFIX_LEN_START_VALUE + PREFIX_LEN_INCREMENT * PREFIX_LEN_INCREMENT_ITERATIONS, PREFIX_LEN_INCREMENT):
                    test_predicted_traces = greedy_suffix_prediction_with_stop(model, test_dataset, current_prefix_len)

                    test_sat = evaluate_compliance_with_formula(deep_dfa, test_predicted_traces)
                    formula_experiment_results[current_prefix_len]["test_sat_rnn_greedy"].append(test_sat)

                    test_DL = evaluate_DL_distance(test_predicted_traces, test_dataset)
                    formula_experiment_results[current_prefix_len]["test_DL_rnn_greedy"].append(test_DL)
                
                
                for current_prefix_len in range(PREFIX_LEN_START_VALUE, PREFIX_LEN_START_VALUE + PREFIX_LEN_INCREMENT * PREFIX_LEN_INCREMENT_ITERATIONS, PREFIX_LEN_INCREMENT):
                    test_predicted_traces = greedy_suffix_prediction_with_stop(model, test_dataset, current_prefix_len)
                    test_predicted_traces = align_predicted_traces(test_predicted_traces, form_file, alphabet, TRACE_LENGTH)
                    test_predicted_traces = test_predicted_traces.to(device)

                    test_sat = evaluate_compliance_with_formula(deep_dfa, test_predicted_traces)
                    formula_experiment_results[current_prefix_len]["test_sat_rnn_greedy_ta"].append(test_sat)

                    test_DL = evaluate_DL_distance(test_predicted_traces, test_dataset)
                    formula_experiment_results[current_prefix_len]["test_DL_rnn_greedy_ta"].append(test_DL)
                

                for current_prefix_len in range(PREFIX_LEN_START_VALUE, PREFIX_LEN_START_VALUE + PREFIX_LEN_INCREMENT * PREFIX_LEN_INCREMENT_ITERATIONS, PREFIX_LEN_INCREMENT):
                    model_file_name = f"./models/results_{experiments_date}/declare_D={D}_C={C}/model_rnn_bk_formula_{i_form}_sample_size_{current_sample_size}_exp_{exp}_prefix_len_{current_prefix_len}.pt"
                    state_dict = torch.load(model_file_name)
                    model = deepcopy(rnn_bk).to(device)
                    model.load_state_dict(state_dict)

                    test_predicted_traces = suffix_prediction_with_temperature_with_stop(model, test_dataset, current_prefix_len, temperature=TEMPERATURE)

                    # Evaluating compliance with the formula of stochastic sampling
                    test_sat = evaluate_compliance_with_formula(deep_dfa, test_predicted_traces)
                    formula_experiment_results[current_prefix_len]["test_sat_rnn_bk"].append(test_sat)

                    # Evaluating DL distance
                    test_DL = evaluate_DL_distance(test_predicted_traces, test_dataset)
                    formula_experiment_results[current_prefix_len]["test_DL_rnn_bk"].append(test_DL)

                
                for current_prefix_len in range(PREFIX_LEN_START_VALUE, PREFIX_LEN_START_VALUE + PREFIX_LEN_INCREMENT * PREFIX_LEN_INCREMENT_ITERATIONS, PREFIX_LEN_INCREMENT):
                    model_file_name = f"./models/results_{experiments_date}/declare_D={D}_C={C}/model_rnn_bk_formula_{i_form}_sample_size_{current_sample_size}_exp_{exp}_prefix_len_{current_prefix_len}.pt"
                    state_dict = torch.load(model_file_name)
                    model = deepcopy(rnn_bk).to(device)
                    model.load_state_dict(state_dict)

                    test_predicted_traces = greedy_suffix_prediction_with_stop(model, test_dataset, current_prefix_len)

                    test_sat = evaluate_compliance_with_formula(deep_dfa, test_predicted_traces)
                    formula_experiment_results[current_prefix_len]["test_sat_rnn_bk_greedy"].append(test_sat)

                    # Evaluating DL distance
                    test_DL = evaluate_DL_distance(test_predicted_traces, test_dataset)
                    formula_experiment_results[current_prefix_len]["test_DL_rnn_bk_greedy"].append(test_DL)
                
                
            configuration_results[str((D, C))][i_form][current_sample_size]["results"] = formula_experiment_results
            results_config_json_file = pathlib.Path("results", "results_ta.json")
            with open(results_config_json_file, "w+") as f:
                json.dump(configuration_results, f, indent=4)