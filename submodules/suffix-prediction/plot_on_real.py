import pathlib
import json

from matplotlib import pyplot as plt
import numpy as np

def load_all_results():
    results_folder = ""
    results_datetime = ""

    for subfolder in FOLDER.iterdir():
        if not str(subfolder).startswith('results/results'):
            continue

        datetime = str(subfolder).replace("results/results_", "")
        
        if datetime > results_datetime:
            results_datetime = str(subfolder).replace("results/results_", "")
            results_folder = subfolder

    results_file = pathlib.Path(results_folder, "results.json")

    try:
        with open(results_file, "r") as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"File {results_file} not found")

    return results, results_datetime

def summarise_results(results):
    for configuration in results:
        for i_form, formula_dict in results[configuration].items():
            print(f"- Configuration {configuration} formula {i_form}:")                
            try:
                prefix_lengths_dict = formula_dict["results"]
                for prefix_length, prefix_length_dict in prefix_lengths_dict.items():
                    print(f"    - Prefix length {prefix_length}:")
                    print("      - RNN:")
                    print(f"        - Avg. train accuracy: {np.mean(prefix_length_dict['train_acc_rnn'])}")
                    print(f"        - Avg. test accuracy: {np.mean(prefix_length_dict['test_acc_rnn'])}")
                    print(f"        - Avg. train DL distance: {np.mean(prefix_length_dict['train_DL_rnn'])}")
                    print(f"        - Avg. test DL distance: {np.mean(prefix_length_dict['test_DL_rnn'])}")
                    print(f"        - Avg. train satisfaction rate: {np.mean(prefix_length_dict['train_sat_rnn'])}")
                    print(f"        - Avg. test satisfaction rate: {np.mean(prefix_length_dict['test_sat_rnn'])}")
                    print("      - RNN+BK:")
                    print(f"        - Avg. train accuracy: {np.mean(prefix_length_dict['train_acc_rnn_bk'])}")
                    print(f"        - Avg. test accuracy: {np.mean(prefix_length_dict['test_acc_rnn_bk'])}")
                    print(f"        - Avg. train DL distance: {np.mean(prefix_length_dict['train_DL_rnn_bk'])}")
                    print(f"        - Avg. test DL distance: {np.mean(prefix_length_dict['test_DL_rnn_bk'])}")
                    print(f"        - Avg. train satisfaction rate: {np.mean(prefix_length_dict['train_sat_rnn_bk'])}")
                    print(f"        - Avg. test satisfaction rate: {np.mean(prefix_length_dict['test_sat_rnn_bk'])}")
                    print("      - RNN Greedy:")
                    print(f"        - Avg. train accuracy: {np.mean(prefix_length_dict['train_acc_rnn_greedy'])}")
                    print(f"        - Avg. test accuracy: {np.mean(prefix_length_dict['test_acc_rnn_greedy'])}")
                    print(f"        - Avg. train DL distance: {np.mean(prefix_length_dict['train_DL_rnn_greedy'])}")
                    print(f"        - Avg. test DL distance: {np.mean(prefix_length_dict['test_DL_rnn_greedy'])}")
                    print(f"        - Avg. train satisfaction rate: {np.mean(prefix_length_dict['train_sat_rnn_greedy'])}")
                    print(f"        - Avg. test satisfaction rate: {np.mean(prefix_length_dict['test_sat_rnn_greedy'])}")
                    print("      - RNN+BK Greedy:")
                    print(f"        - Avg. train accuracy: {np.mean(prefix_length_dict['train_acc_rnn_bk_greedy'])}")
                    print(f"        - Avg. test accuracy: {np.mean(prefix_length_dict['test_acc_rnn_bk_greedy'])}")
                    print(f"        - Avg. train DL distance: {np.mean(prefix_length_dict['train_DL_rnn_bk_greedy'])}")
                    print(f"        - Avg. test DL distance: {np.mean(prefix_length_dict['test_DL_rnn_bk_greedy'])}")
                    print(f"        - Avg. train satisfaction rate: {np.mean(prefix_length_dict['train_sat_rnn_bk_greedy'])}")
                    print(f"        - Avg. test satisfaction rate: {np.mean(prefix_length_dict['test_sat_rnn_bk_greedy'])}")
            except KeyError:
                print("    - NOT ENOUGH RESULTS YET")
        print("\n")

def plot(configuration_name, configuration_dict, sample_size, metric):
    # Check metric
    assert metric in ["acc", "sat", "DL"]
    if metric == "acc":
        metric_name = "Accuracy"
    elif metric == "sat":
        metric_name = "Satisfiability"
    elif metric == "DL":
        metric_name = "DL distance"

    # Add D= and C= to configuration name
    # Cast configuration name to tuple
    configuration_name = configuration_name.replace(" ", "").replace("(", "").replace(")", "").split(",")
    configuration_name = f"(D={configuration_name[0]}, C={configuration_name[1]})"

    # Join the sample_size metric values of all formulas for each prefix length
    metric_values_for_sample_size_rnn_by_prefix_length = {}
    metric_values_for_sample_size_rnn_bk_by_prefix_length = {}
    metric_values_for_sample_size_rnn_greedy_by_prefix_length = {}
    metric_values_for_sample_size_rnn_bk_greedy_by_prefix_length = {}
    for i_form, formula_dict in configuration_dict.items():
        sample_size_dict = formula_dict[str(sample_size)]
        try:
            prefix_lengths_dict = sample_size_dict["results"]
            for prefix_length, prefix_length_dict in prefix_lengths_dict.items():
                if prefix_length not in metric_values_for_sample_size_rnn_by_prefix_length:
                    metric_values_for_sample_size_rnn_by_prefix_length[prefix_length] = []
                    metric_values_for_sample_size_rnn_bk_by_prefix_length[prefix_length] = []
                    metric_values_for_sample_size_rnn_greedy_by_prefix_length[prefix_length] = []
                    metric_values_for_sample_size_rnn_bk_greedy_by_prefix_length[prefix_length] = []
                metric_values_for_sample_size_rnn_by_prefix_length[prefix_length].extend(prefix_length_dict[f"test_{metric}_rnn"])
                metric_values_for_sample_size_rnn_bk_by_prefix_length[prefix_length].extend(prefix_length_dict[f"test_{metric}_rnn_bk"])
                metric_values_for_sample_size_rnn_greedy_by_prefix_length[prefix_length].extend(prefix_length_dict[f"test_{metric}_rnn_greedy"])
                metric_values_for_sample_size_rnn_bk_greedy_by_prefix_length[prefix_length].extend(prefix_length_dict[f"test_{metric}_rnn_bk_greedy"])
        except KeyError as e:
            print(f"Skipping formula {i_form} for sample size {sample_size} as not enough results yet")
            continue

    # Initialise plot
    fig, ax = plt.subplots()
    bar_width = 0.18
    bar_distance = 0.04
    index = np.arange(len(metric_values_for_sample_size_rnn_by_prefix_length))

    # Plot bars
    rnn_x_values = [np.mean(metric_values) for metric_values in metric_values_for_sample_size_rnn_by_prefix_length.values()]
    rnn_bk_x_values = [np.mean(metric_values) for metric_values in metric_values_for_sample_size_rnn_bk_by_prefix_length.values()]
    rnn_greedy_x_values = [np.mean(metric_values) for metric_values in metric_values_for_sample_size_rnn_greedy_by_prefix_length.values()]
    rnn_bk_greedy_x_values = [np.mean(metric_values) for metric_values in metric_values_for_sample_size_rnn_bk_greedy_by_prefix_length.values()]
    ax.bar(index, rnn_x_values, bar_width, label="RNN", color="skyblue")
    ax.bar(index + bar_width, rnn_bk_x_values, bar_width, label="RNN+BK", color="lightgreen")
    ax.bar((index + bar_distance) + 2 * bar_width, rnn_greedy_x_values, bar_width, label="RNN Greedy", color="dodgerblue")
    ax.bar((index + bar_distance) + 3 * bar_width, rnn_bk_greedy_x_values, bar_width, label="RNN+BK Greedy", color="seagreen")

    # Plot error bars
    rnn_error = [np.std(metric_values) for metric_values in metric_values_for_sample_size_rnn_by_prefix_length.values()]
    rnn_bk_error = [np.std(metric_values) for metric_values in metric_values_for_sample_size_rnn_bk_by_prefix_length.values()]
    rnn_greedy_error = [np.std(metric_values) for metric_values in metric_values_for_sample_size_rnn_greedy_by_prefix_length.values()]
    rnn_bk_greedy_error = [np.std(metric_values) for metric_values in metric_values_for_sample_size_rnn_bk_greedy_by_prefix_length.values()]
    ax.errorbar(index, rnn_x_values, rnn_error, fmt="none", ecolor="black", capsize=5)
    ax.errorbar(index + bar_width, rnn_bk_x_values, rnn_bk_error, fmt="none", ecolor="black", capsize=5)
    ax.errorbar((index + bar_distance) + 2 * bar_width, rnn_greedy_x_values, rnn_greedy_error, fmt="none", ecolor="black", capsize=5)
    ax.errorbar((index + bar_distance) + 3 * bar_width, rnn_bk_greedy_x_values, rnn_bk_greedy_error, fmt="none", ecolor="black", capsize=5)

    # If metric is DL, set y-axis values from 0 to 20
    # if metric == "DL":
    #     ax.set_ylim(0, 20)
    # If metric is acc, set y-axis values from 0 to 0.50
    if metric == "acc":
        ax.set_ylim(0, 0.50)
    # If metric is sat, set y-axis values from 0 to 1
    if metric == "sat":
        ax.set_ylim(0, 1)

    # Set plot labels
    ax.set_xlabel("Prefix length")
    ax.set_ylabel(f"{metric_name}")
    ax.set_title(f"{metric_name} by prefix length for sample size {sample_size}")
    ax.set_xticks((index + bar_distance) + 2 * bar_width - bar_width / 2 - bar_distance / 1.5)
    ax.set_xticklabels(metric_values_for_sample_size_rnn_by_prefix_length.keys())

    # Set legent
    ax.legend()

    # Save plot
    fig.tight_layout()
    fig.savefig(PLOTS_FOLDER / f"{metric}_plots" / f"config_{configuration_name}_sample_size_{sample_size}.png")
    plt.close(fig)


CONFIGS = ["real"]
PREFIX_LENGTHS = [5, 10, 15]

FOLDER = pathlib.Path("results")


if __name__ == "__main__":
    # Load results
    results, datetime = load_all_results()

    PLOTS_FOLDER = pathlib.Path("plots")
    PLOTS_DATETIME_FOLDER = PLOTS_FOLDER / f"plots_{datetime}"
    PLOTS_FOLDER.mkdir(parents=True, exist_ok=True)
    PLOTS_DATETIME_FOLDER.mkdir(parents=True, exist_ok=True)

    # Summarise results
    #summarise_results(results)
    
    # Plot for all configurations as a whole
    for metric in ["acc", "sat", "DL"]:
        print(f"Plotting {metric} plots")

        # Check metric
        if metric == "acc":
            metric_name = "Accuracy %"
        elif metric == "sat":
            metric_name = "Satisfiability %"
        elif metric == "DL":
            metric_name = "DL distance"

        # Create configuration name
        configuration_name = "ALL CONFIGURATIONS"

        # Join the metric values of all configurations for each prefix length
        metric_values_for_sample_size_rnn_by_prefix_length = {}
        metric_values_for_sample_size_rnn_bk_by_prefix_length = {}
        metric_values_for_sample_size_rnn_greedy_by_prefix_length = {}
        metric_values_for_sample_size_rnn_bk_greedy_by_prefix_length = {}
        
        for configuration_dict in results.values():
            for i_form, formula_dict in configuration_dict.items():
                try:
                    prefix_lengths_dict = formula_dict["results"]
                    for prefix_length, prefix_length_dict in prefix_lengths_dict.items():
                        if prefix_length not in metric_values_for_sample_size_rnn_by_prefix_length:
                            metric_values_for_sample_size_rnn_by_prefix_length[prefix_length] = []
                            metric_values_for_sample_size_rnn_bk_by_prefix_length[prefix_length] = []
                            metric_values_for_sample_size_rnn_greedy_by_prefix_length[prefix_length] = []
                            metric_values_for_sample_size_rnn_bk_greedy_by_prefix_length[prefix_length] = []
                        metric_values_for_sample_size_rnn_by_prefix_length[prefix_length].extend(prefix_length_dict[f"test_{metric}_rnn"])
                        metric_values_for_sample_size_rnn_bk_by_prefix_length[prefix_length].extend(prefix_length_dict[f"test_{metric}_rnn_bk"])
                        metric_values_for_sample_size_rnn_greedy_by_prefix_length[prefix_length].extend(prefix_length_dict[f"test_{metric}_rnn_greedy"])
                        metric_values_for_sample_size_rnn_bk_greedy_by_prefix_length[prefix_length].extend(prefix_length_dict[f"test_{metric}_rnn_bk_greedy"])
                except KeyError:
                    print(f"Skipping formula {i_form} for sample size {formula_dict} as not enough results yet")
                    continue
        
        # If metric is accuracy or satisfiability, convert to percentage
        if metric in ["acc", "sat"]:
            for prefix_length in metric_values_for_sample_size_rnn_by_prefix_length:
                metric_values_for_sample_size_rnn_by_prefix_length[prefix_length] = [value * 100 for value in metric_values_for_sample_size_rnn_by_prefix_length[prefix_length]]
                metric_values_for_sample_size_rnn_bk_by_prefix_length[prefix_length] = [value * 100 for value in metric_values_for_sample_size_rnn_bk_by_prefix_length[prefix_length]]
                metric_values_for_sample_size_rnn_greedy_by_prefix_length[prefix_length] = [value * 100 for value in metric_values_for_sample_size_rnn_greedy_by_prefix_length[prefix_length]]
                metric_values_for_sample_size_rnn_bk_greedy_by_prefix_length[prefix_length] = [value * 100 for value in metric_values_for_sample_size_rnn_bk_greedy_by_prefix_length[prefix_length]]

        # Initialise plot
        fig, ax = plt.subplots()
        bar_width = 0.18
        bar_distance = 0.04
        index = np.arange(len(metric_values_for_sample_size_rnn_by_prefix_length))

        # Plot bars
        rnn_x_values = [np.mean(metric_values) for metric_values in metric_values_for_sample_size_rnn_by_prefix_length.values()]
        rnn_bk_x_values = [np.mean(metric_values) for metric_values in metric_values_for_sample_size_rnn_bk_by_prefix_length.values()]
        rnn_greedy_x_values = [np.mean(metric_values) for metric_values in metric_values_for_sample_size_rnn_greedy_by_prefix_length.values()]
        rnn_bk_greedy_x_values = [np.mean(metric_values) for metric_values in metric_values_for_sample_size_rnn_bk_greedy_by_prefix_length.values()]
        ax.bar(index, rnn_x_values, bar_width, label="RNN (random)", color="skyblue")
        ax.bar(index + bar_width, rnn_bk_x_values, bar_width, label="RNN+LTL (random)", color="lightgreen")
        ax.bar((index + bar_distance) + 2 * bar_width, rnn_greedy_x_values, bar_width, label="RNN (greedy)", color="dodgerblue")
        ax.bar((index + bar_distance) + 3 * bar_width, rnn_bk_greedy_x_values, bar_width, label="RNN+LTL (greedy)", color="seagreen")

        # Plot error bars
        rnn_error = [np.std(metric_values) for metric_values in metric_values_for_sample_size_rnn_by_prefix_length.values()]
        rnn_bk_error = [np.std(metric_values) for metric_values in metric_values_for_sample_size_rnn_bk_by_prefix_length.values()]
        rnn_greedy_error = [np.std(metric_values) for metric_values in metric_values_for_sample_size_rnn_greedy_by_prefix_length.values()]
        rnn_bk_greedy_error = [np.std(metric_values) for metric_values in metric_values_for_sample_size_rnn_bk_greedy_by_prefix_length.values()]
        ax.errorbar(index, rnn_x_values, rnn_error, fmt="none", ecolor="black", capsize=5)
        ax.errorbar(index + bar_width, rnn_bk_x_values, rnn_bk_error, fmt="none", ecolor="black", capsize=5)
        ax.errorbar((index + bar_distance) + 2 * bar_width, rnn_greedy_x_values, rnn_greedy_error, fmt="none", ecolor="black", capsize=5)
        ax.errorbar((index + bar_distance) + 3 * bar_width, rnn_bk_greedy_x_values, rnn_bk_greedy_error, fmt="none", ecolor="black", capsize=5)

        # If metric is DL, set y-axis values from 0 to 20
        # if metric == "DL":
        #     ax.set_ylim(0, 20)
        # If metric is acc, set y-axis values from 0 to 0.50
        if metric == "acc":
            ax.set_ylim(0, 100)
        # If metric is sat, set y-axis values from 0 to 1
        if metric == "sat":
            ax.set_ylim(0, 100)

        # Set plot labels
        ax.set_xlabel("Prefix length")
        ax.set_ylabel(f"{metric_name}")
        # ax.set_title(f"{metric_name} by prefix length for sample size {sample_size}")
        ax.set_title(f"{metric_name} by prefix length")
        ax.set_xticks((index + bar_distance) + 2 * bar_width - bar_width / 2 - bar_distance / 1.5)
        ax.set_xticklabels(metric_values_for_sample_size_rnn_by_prefix_length.keys())

        # Set legend
        ax.legend(loc="upper center", ncol=4, fontsize="small", columnspacing=1.0, handletextpad=0.5, handlelength=1.5)

        # Save plot
        fig.tight_layout()
        fig.savefig(PLOTS_FOLDER / f"plots_{datetime}" / f"{metric}_plots.pdf")
        plt.close(fig)
